"""
Agent 5 -- Intelligent Scientific Data Preparation & Analysis Agent.

INVOCATION CONTRACT (do not relax this):
  Agent 5 is constructed ONLY from an Agent4Output whose
  user_chose_preprocessing is True. Agent4Output's own validator
  enforces this, so by the time run_agent5() is called, the "should we
  preprocess" question has already been answered upstream by the user
  choosing "Preprocess" over "Raw Files" at Agent 4. Agent 5 therefore
  NEVER prompts the user for anything, and NEVER asks whether to
  preprocess, how to handle a data quality issue, or how to proceed --
  every decision point in this file is either deterministic Python or
  a single LLM judgment call, never a human prompt.

Agent 5 is not responsible for discovery, source selection,
authentication, or downloading -- those are Agents 3 and 4's job.
Its responsibility begins only once Agent4Output already contains
successfully retrieved files on disk.

PIPELINE (mirrors the system prompt's 8 steps):
  1. phase1_profile_datasets      -- pure Python, per-dataset inspection
  2. phase2_validate_datasets     -- pure Python, checks against the
                                      user's actual scientific request
  3. phase3_build_preprocessing_plan -- ONE LLM call: decides which
                                      steps this specific request needs
                                      (never a fixed pipeline) and
                                      whether the objective is even
                                      achievable given what was found
  4. phase4_execute_preprocessing -- pure Python execution of the plan
                                      (xarray/pandas/rioxarray)
  5. phase5_merge_datasets        -- pure Python, only when the plan
                                      calls for dataset_merging
  6. phase6_run_analysis          -- pure Python statistical/scientific
                                      analysis per expected_scientific_outputs
  7. phase7_assess_quality        -- pure Python, derived from profiling/
                                      validation/execution results
  8. phase8_build_output          -- assembles Agent5Output + writes
                                      cleaned/merged data to the Zarr store
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from groq import Groq

import agent5_config as config
from models.agent5_schemas import (
    Agent4Output,
    Agent5Output,
    Agent5Status,
    AnalysisResult,
    DatasetProfile,
    DatasetValidationResult,
    ExecutedStep,
    PlannedStep,
    PreprocessingPlan,
    PreprocessingStepName,
    ProcessingLogEntry,
    QualityAssessment,
    RetrievedDataset,
    ValidationIssue,
    ValidationSeverity,
)
from models.discovery_schemas import DownloadFormat
from models.retrieval_request import RetrievalRequest
from prompts.agent5_prompt import (
    SYSTEM_PROMPT,
    build_planning_prompt,
    parse_planning_response,
)
from storage import zarr_store
from agent5_units import convert_dataset_variable, resolve_target_units
from agent5_analysis import dispatch_requested_analyses
from agent5_preprocessing_extra import (
    apply_coordinate_normalization,
    apply_crs_transformation,
    apply_derived_variable_computation,
    apply_feature_extraction,
    apply_resampling,
    apply_variable_standardization,
)

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


# ------------------------------------------------------------------ #
# Loading helpers -- format-aware, mechanical                          #
# ------------------------------------------------------------------ #

def _load_as_xarray(dataset: RetrievedDataset) -> Optional[xr.Dataset]:
    """
    Loads a RetrievedDataset's files into an xarray.Dataset where the
    format supports it. CSV is loaded via pandas and converted, since
    CSV has no native gridded structure. Returns None for formats this
    build doesn't yet know how to open (surfaced as a profiling note,
    never a silent skip).
    """
    fmt = dataset.file_format
    paths = dataset.local_file_paths

    try:
        if fmt in (DownloadFormat.netcdf, DownloadFormat.hdf5):
            if len(paths) == 1:
                return xr.open_dataset(paths[0], chunks="auto")
            return xr.open_mfdataset(paths, combine="by_coords", chunks="auto", parallel=True)

        if fmt == DownloadFormat.grib:
            if len(paths) == 1:
                return xr.open_dataset(paths[0], engine="cfgrib", chunks="auto")
            return xr.open_mfdataset(paths, engine="cfgrib", combine="by_coords", chunks="auto", parallel=True)

        if fmt == DownloadFormat.geotiff:
            import rioxarray  # noqa: F401 -- registers the .rio accessor
            arrays = [xr.open_dataarray(p, engine="rasterio") for p in paths]
            return xr.concat(arrays, dim="band").to_dataset(name="value") if len(arrays) > 1 else arrays[0].to_dataset(name="value")

        if fmt == DownloadFormat.csv:
            frames = [pd.read_csv(p) for p in paths]
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            return df.to_xarray()

        if fmt == DownloadFormat.json:
            frames = [pd.read_json(p) for p in paths]
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            return df.to_xarray()

        return None  # Shapefile / Unknown -- not handled by this build yet.

    except Exception:
        return None


# ------------------------------------------------------------------ #
# STEP 1 — Data profiling                                             #
# ------------------------------------------------------------------ #

def phase1_profile_datasets(
    datasets: List[RetrievedDataset],
) -> Tuple[List[DatasetProfile], Dict[str, xr.Dataset]]:
    """
    Returns (profiles, loaded_datasets). loaded_datasets maps
    source_id -> xr.Dataset for every source that could be opened, so
    later phases never have to re-open files from disk.
    """
    profiles: List[DatasetProfile] = []
    loaded: Dict[str, xr.Dataset] = {}

    for ds_meta in datasets:
        ds = _load_as_xarray(ds_meta)

        if ds is None:
            profiles.append(DatasetProfile(
                source_id=ds_meta.source_id,
                file_format=ds_meta.file_format,
                variables_found=[],
                metadata_completeness_score=0.0,
                profiling_notes=(
                    f"Could not open file(s) for format "
                    f"{ds_meta.file_format.value}. This dataset could not "
                    f"be profiled and will be flagged in validation."
                ),
            ))
            continue

        loaded[ds_meta.source_id] = ds

        variables = list(ds.data_vars.keys())
        units_by_var = {
            v: str(ds[v].attrs.get("units", "unknown")) for v in variables
        }
        dims = {str(k): int(v) for k, v in ds.sizes.items()}

        crs = None
        if hasattr(ds, "rio"):
            try:
                crs = str(ds.rio.crs) if ds.rio.crs else None
            except Exception:
                crs = None

        # EFFICIENCY: previously pulled every variable's full array into
        # memory via .values inside a Python loop (defeats dask's lazy
        # chunking on large files, and re-walks the array multiple times
        # for mean/std/isnan separately). Now built as a single dict of
        # lazy xarray reductions and computed together in one pass --
        # stays lazy under dask until the one .compute() call below, and
        # each variable's stats are one dask task graph instead of N.
        numeric_vars = [v for v in variables if np.issubdtype(ds[v].dtype, np.floating)]
        missing_fraction = 0.0
        outlier_count = 0
        if numeric_vars:
            try:
                lazy_stats = {}
                for v in numeric_vars:
                    arr = ds[v]
                    lazy_stats[f"{v}__missing"] = arr.isnull().sum()
                    lazy_stats[f"{v}__total"] = arr.size
                    lazy_stats[f"{v}__mean"] = arr.mean(skipna=True)
                    lazy_stats[f"{v}__std"] = arr.std(skipna=True)

                # One combined compute() instead of one per variable.
                computed = xr.Dataset(
                    {k: v for k, v in lazy_stats.items() if hasattr(v, "compute")}
                ).compute()

                total_cells = sum(lazy_stats[f"{v}__total"] for v in numeric_vars)
                missing_cells = sum(
                    float(computed[f"{v}__missing"].values) for v in numeric_vars
                )
                missing_fraction = (missing_cells / total_cells) if total_cells else 0.0

                for v in numeric_vars:
                    mean = float(computed[f"{v}__mean"].values)
                    std = float(computed[f"{v}__std"].values) or 1.0
                    z_exceed = np.abs((ds[v] - mean) / std) > config.OUTLIER_ZSCORE_THRESHOLD
                    outlier_count += int(z_exceed.sum().compute().values)
            except Exception:
                pass  # profiling is best-effort; never fatal

        duplicate_count = 0
        if "time" in ds.dims:
            try:
                time_vals = pd.to_datetime(ds["time"].values)
                duplicate_count = int(time_vals.duplicated().sum())
            except Exception:
                pass

        metadata_completeness = sum(
            1 for v in variables if ds[v].attrs.get("units")
        ) / len(variables) if variables else 0.0

        profiles.append(DatasetProfile(
            source_id=ds_meta.source_id,
            file_format=ds_meta.file_format,
            variables_found=variables,
            units_by_variable=units_by_var,
            dimensions=dims,
            coordinate_reference_system=crs,
            spatial_resolution=ds_meta.provider_metadata.get("spatial_resolution"),
            temporal_resolution=ds_meta.provider_metadata.get("temporal_resolution"),
            metadata_completeness_score=round(metadata_completeness, 2),
            missing_value_fraction=round(missing_fraction, 4),
            duplicate_record_count=duplicate_count,
            invalid_value_count=0,
            outlier_count=outlier_count,
            profiling_notes=f"Opened successfully as xarray.Dataset with {len(variables)} variable(s).",
        ))

    return profiles, loaded


# ------------------------------------------------------------------ #
# STEP 2 — Data validation                                            #
# ------------------------------------------------------------------ #

def phase2_validate_datasets(
    request: RetrievalRequest,
    datasets: List[RetrievedDataset],
    profiles: List[DatasetProfile],
) -> List[DatasetValidationResult]:
    required_vars = {
        v.variable.lower().strip() for v in request.variables
    } | {
        m.variable_measured.lower().strip() for m in request.measurements
    }

    profile_by_id = {p.source_id: p for p in profiles}
    results: List[DatasetValidationResult] = []

    for ds_meta in datasets:
        profile = profile_by_id.get(ds_meta.source_id)
        issues: List[ValidationIssue] = []

        if profile is None or not profile.variables_found:
            issues.append(ValidationIssue(
                source_id=ds_meta.source_id,
                issue="Dataset could not be opened/profiled.",
                severity=ValidationSeverity.blocking,
            ))
            results.append(DatasetValidationResult(
                source_id=ds_meta.source_id,
                required_variables_present=False,
                missing_required_variables=sorted(required_vars),
                temporal_coverage_sufficient=False,
                spatial_coverage_sufficient=False,
                unit_consistency_ok=False,
                coordinate_validity_ok=False,
                issues=issues,
            ))
            continue

        found_lower = {v.lower().strip() for v in profile.variables_found}
        missing_required = sorted(required_vars - found_lower) if required_vars else []
        vars_present = not missing_required

        if missing_required:
            issues.append(ValidationIssue(
                source_id=ds_meta.source_id,
                issue=f"Required variable(s) not found: {', '.join(missing_required)}",
                severity=ValidationSeverity.significant,
            ))

        temporal_ok = ds_meta.retrieval_status.value != "failed"
        spatial_ok = ds_meta.retrieval_status.value != "failed"
        if ds_meta.retrieval_status.value == "partial":
            issues.append(ValidationIssue(
                source_id=ds_meta.source_id,
                issue=f"Retrieval was only partial: {ds_meta.retrieval_notes}",
                severity=ValidationSeverity.significant,
            ))

        unit_consistency_ok = all(
            u not in ("", "unknown", "None") for u in profile.units_by_variable.values()
        ) if profile.units_by_variable else True
        if not unit_consistency_ok:
            issues.append(ValidationIssue(
                source_id=ds_meta.source_id,
                issue="One or more variables have no declared units -- "
                      "unit conversion will need to assume CF conventions.",
                severity=ValidationSeverity.minor,
            ))

        coordinate_validity_ok = True  # refined once real CRS-checking logic is wired in

        results.append(DatasetValidationResult(
            source_id=ds_meta.source_id,
            required_variables_present=vars_present,
            missing_required_variables=missing_required,
            temporal_coverage_sufficient=temporal_ok,
            spatial_coverage_sufficient=spatial_ok,
            unit_consistency_ok=unit_consistency_ok,
            coordinate_validity_ok=coordinate_validity_ok,
            issues=issues,
        ))

    return results


# ------------------------------------------------------------------ #
# STEP 3 — Build preprocessing plan (the one LLM judgment call)       #
# ------------------------------------------------------------------ #

def phase3_build_preprocessing_plan(
    request: RetrievalRequest,
    profiles: List[DatasetProfile],
    validations: List[DatasetValidationResult],
) -> Tuple[PreprocessingPlan, bool, Optional[str]]:
    """
    Returns (plan, objective_achievable, stop_reason).
    """
    # Deterministic short-circuit: if every dataset has a blocking
    # validation issue, don't even spend an LLM call -- the objective
    # is unreachable regardless of what plan gets proposed.
    if all(v.has_blocking_issue for v in validations):
        return (
            PreprocessingPlan(steps=[], plan_rationale="Not built -- no usable data."),
            False,
            "Every retrieved dataset failed profiling/validation with a "
            "blocking issue (unreadable file or completely missing "
            "required variables). No scientifically defensible "
            "preprocessing plan can be built from what was retrieved.",
        )

    client = _get_groq_client()
    prompt = build_planning_prompt(
        scientific_goal=request.goal,
        expected_scientific_outputs=request.expected_scientific_outputs,
        required_data_fusion=request.required_data_fusion,
        dataset_profiles=profiles,
        validation_results=validations,
    )

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    parsed = parse_planning_response(raw)

    steps = [PlannedStep(**s) for s in parsed["preprocessing_steps"]]
    plan = PreprocessingPlan(
        steps=steps,
        plan_rationale=parsed["plan_rationale"],
    )
    return plan, bool(parsed["objective_achievable"]), parsed.get("stop_reason")


# ------------------------------------------------------------------ #
# STEP 4 — Autonomous preprocessing execution (mechanical)             #
# ------------------------------------------------------------------ #

def _apply_missing_value_handling(ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
    for var in ds.data_vars:
        arr = ds[var]
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        frac_missing = float(np.isnan(arr.values).mean())
        if frac_missing == 0:
            continue
        if frac_missing <= config.MISSING_VALUE_INTERPOLATE_MAX_FRACTION:
            ds[var] = arr.interpolate_na(dim=list(arr.dims)[0], method="linear")
        else:
            ds[var] = arr.fillna(arr.mean(skipna=True))
    return ds, "Linear interpolation for sparse gaps; mean-fill where gaps were extensive."


def _apply_duplicate_removal(ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
    if "time" in ds.dims:
        _, index = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=sorted(index))
    return ds, "Removed duplicate timestamps, keeping first occurrence."


def _apply_invalid_value_filtering(ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
    for var in ds.data_vars:
        arr = ds[var]
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        finite = arr.values[np.isfinite(arr.values)]
        if finite.size == 0:
            continue
        z = np.abs((arr - finite.mean()) / (finite.std() or 1))
        ds[var] = arr.where(z <= config.OUTLIER_ZSCORE_THRESHOLD)
    return ds, f"Masked values beyond {config.OUTLIER_ZSCORE_THRESHOLD} std deviations as invalid."


def _apply_unit_conversion(ds: xr.Dataset, measurements: List) -> Tuple[xr.Dataset, str]:
    """
    Converts each variable to its preferred_unit_or_representation from
    the user's RetrievalRequest measurements, using agent5_units'
    pint-backed conversion. A variable with no declared preferred unit,
    or already in that unit, is left untouched. A genuine dimensional
    mismatch (e.g. requesting a speed unit for a temperature variable)
    raises inside convert_dataset_variable rather than being silently
    skipped -- that's a data-integrity problem, not a missing feature,
    and should surface as a validation-style issue, not disappear.
    """
    variables = list(ds.data_vars.keys())
    target_units = resolve_target_units(measurements, variables)
    if not target_units:
        return ds, "No preferred units specified for any variable in this dataset -- left as retrieved."

    notes = []
    for var, target_unit in target_units.items():
        current_unit = ds[var].attrs.get("units", "unknown")
        if current_unit in ("unknown", "", None):
            notes.append(f"{var}: no declared source unit -- cannot safely convert, left as-is.")
            continue
        try:
            ds, note = convert_dataset_variable(ds, var, current_unit, target_unit)
            notes.append(f"{var}: {note}")
        except ValueError as exc:
            notes.append(f"{var}: conversion FAILED -- {exc}")

    return ds, "; ".join(notes)


def _apply_time_alignment(ds: xr.Dataset, freq: Optional[str] = None) -> Tuple[xr.Dataset, str]:
    if "time" not in ds.dims:
        return ds, "No time dimension present -- skipped."
    freq = freq or "D"
    ds = ds.resample(time=freq).mean()
    return ds, f"Resampled to a uniform '{freq}' time frequency via mean aggregation."


_STEP_EXECUTORS = {
    PreprocessingStepName.missing_value_handling: _apply_missing_value_handling,
    PreprocessingStepName.duplicate_removal: _apply_duplicate_removal,
    PreprocessingStepName.invalid_value_filtering: _apply_invalid_value_filtering,
    PreprocessingStepName.time_alignment: _apply_time_alignment,
    PreprocessingStepName.coordinate_normalization: apply_coordinate_normalization,
    PreprocessingStepName.crs_transformation: apply_crs_transformation,
    PreprocessingStepName.resampling: apply_resampling,
    PreprocessingStepName.variable_standardization: apply_variable_standardization,
    PreprocessingStepName.derived_variable_computation: apply_derived_variable_computation,
    PreprocessingStepName.feature_extraction: apply_feature_extraction,
}
# unit_conversion and spatial_alignment need extra context (measurements /
# a reference dataset respectively) beyond a single `ds` argument, so
# they're handled separately in phase4_execute_preprocessing rather than
# forced into this single-argument dispatch table.


def phase4_execute_preprocessing(
    plan: PreprocessingPlan,
    loaded: Dict[str, xr.Dataset],
    request: RetrievalRequest,
) -> Tuple[Dict[str, xr.Dataset], List[ExecutedStep]]:
    executed: List[ExecutedStep] = []

    for planned in plan.steps:
        # --- unit_conversion: needs request.measurements for target units ---
        if planned.step == PreprocessingStepName.unit_conversion:
            for source_id in planned.applies_to_source_ids:
                if source_id not in loaded:
                    continue
                try:
                    loaded[source_id], summary = _apply_unit_conversion(
                        loaded[source_id], request.measurements
                    )
                    executed.append(ExecutedStep(
                        step=planned.step, applies_to_source_ids=[source_id],
                        method_used=planned.method, result_summary=summary, success=True,
                    ))
                except Exception as exc:
                    executed.append(ExecutedStep(
                        step=planned.step, applies_to_source_ids=[source_id],
                        method_used=planned.method,
                        result_summary=f"Execution failed: {exc}", success=False, warning=str(exc),
                    ))
            continue

        # --- spatial_alignment: aligns every other listed source onto the
        # first listed source's grid (used as the reference) ---
        if planned.step == PreprocessingStepName.spatial_alignment:
            ids = [s for s in planned.applies_to_source_ids if s in loaded]
            if len(ids) < 2:
                executed.append(ExecutedStep(
                    step=planned.step, applies_to_source_ids=planned.applies_to_source_ids,
                    method_used=planned.method,
                    result_summary="Fewer than 2 available sources listed -- nothing to align.",
                    success=False, warning="Insufficient sources.",
                ))
                continue
            reference = loaded[ids[0]]
            for source_id in ids[1:]:
                try:
                    loaded[source_id], summary = apply_spatial_alignment(loaded[source_id], reference)
                    executed.append(ExecutedStep(
                        step=planned.step, applies_to_source_ids=[source_id],
                        method_used=planned.method,
                        result_summary=f"Aligned to '{ids[0]}''s grid: {summary}", success=True,
                    ))
                except Exception as exc:
                    executed.append(ExecutedStep(
                        step=planned.step, applies_to_source_ids=[source_id],
                        method_used=planned.method,
                        result_summary=f"Execution failed: {exc}", success=False, warning=str(exc),
                    ))
            continue

        executor = _STEP_EXECUTORS.get(planned.step)
        if executor is None:
            # interpolation (general, target-grid form) and
            # dataset_merging are handled elsewhere (interpolation has
            # no target grid concept in the current plan schema yet;
            # dataset_merging is phase5's job) -- log honestly rather
            # than silently pretending it ran.
            executed.append(ExecutedStep(
                step=planned.step,
                applies_to_source_ids=planned.applies_to_source_ids,
                method_used=planned.method,
                result_summary="Not yet implemented in this build of Agent 5 -- "
                                "planned but not executed.",
                success=False,
                warning="No executor registered for this step type.",
            ))
            continue

        for source_id in planned.applies_to_source_ids:
            if source_id not in loaded:
                continue
            try:
                loaded[source_id], summary = executor(loaded[source_id])
                executed.append(ExecutedStep(
                    step=planned.step,
                    applies_to_source_ids=[source_id],
                    method_used=planned.method,
                    result_summary=summary,
                    success=True,
                ))
            except Exception as exc:
                executed.append(ExecutedStep(
                    step=planned.step,
                    applies_to_source_ids=[source_id],
                    method_used=planned.method,
                    result_summary=f"Execution failed: {exc}",
                    success=False,
                    warning=str(exc),
                ))

    return loaded, executed


# ------------------------------------------------------------------ #
# STEP 5 — Multi-dataset integration                                   #
# ------------------------------------------------------------------ #

def phase5_merge_datasets(
    plan: PreprocessingPlan,
    loaded: Dict[str, xr.Dataset],
) -> Optional[xr.Dataset]:
    merge_steps = [s for s in plan.steps if s.step == PreprocessingStepName.dataset_merging]
    if not merge_steps:
        return None

    to_merge = []
    for step in merge_steps:
        for source_id in step.applies_to_source_ids:
            if source_id in loaded:
                to_merge.append(loaded[source_id])

    if len(to_merge) < 2:
        return None

    try:
        return xr.merge(to_merge, compat="override", join="outer")
    except Exception:
        # Fall back to a looser merge if coordinates don't align exactly.
        return xr.merge(to_merge, compat="override", join="outer", combine_attrs="drop_conflicts")


# ------------------------------------------------------------------ #
# STEP 6 — Scientific analysis                                        #
# ------------------------------------------------------------------ #

def phase6_run_analysis(
    request: RetrievalRequest,
    loaded: Dict[str, xr.Dataset],
    merged: Optional[xr.Dataset],
) -> List[AnalysisResult]:
    """
    Runs the analysis types actually named in the system prompt
    (statistical summary always; trend/climatology/anomaly/correlation
    when the request's own language calls for them -- see
    agent5_analysis.dispatch_requested_analyses for the keyword
    dispatch). ML/forecasting remains explicitly out of scope unless
    later added on top of an explicit request, per the system prompt.
    """
    dataset_to_analyze = merged if merged is not None else next(iter(loaded.values()), None)
    if dataset_to_analyze is None:
        return []

    return dispatch_requested_analyses(
        dataset_to_analyze,
        goal=request.goal,
        expected_scientific_outputs=request.expected_scientific_outputs,
    )


# ------------------------------------------------------------------ #
# STEP 7 — Quality assessment                                          #
# ------------------------------------------------------------------ #

def phase7_assess_quality(
    profiles: List[DatasetProfile],
    validations: List[DatasetValidationResult],
    executed_steps: List[ExecutedStep],
) -> QualityAssessment:
    completeness = (
        sum(p.metadata_completeness_score for p in profiles) / len(profiles)
        if profiles else 0.0
    )
    successful_steps = [s for s in executed_steps if s.success]
    processing_quality = (
        len(successful_steps) / len(executed_steps) if executed_steps else 1.0
    )

    limitations = []
    for v in validations:
        for issue in v.issues:
            if issue.severity in (ValidationSeverity.significant, ValidationSeverity.blocking):
                limitations.append(f"[{v.source_id}] {issue.issue}")
    for s in executed_steps:
        if not s.success:
            limitations.append(
                f"[{', '.join(s.applies_to_source_ids)}] {s.step.value} did not complete: {s.warning}"
            )

    overall_confidence = round((completeness + processing_quality) / 2, 2)

    return QualityAssessment(
        overall_confidence=overall_confidence,
        completeness_score=round(completeness, 2),
        processing_quality_score=round(processing_quality, 2),
        remaining_limitations=limitations,
        reliability_notes=(
            "Confidence reflects metadata completeness at profiling time "
            "and the fraction of planned preprocessing steps that "
            "executed successfully. It does not independently verify "
            "the retrieved data's scientific accuracy."
        ),
    )


# ------------------------------------------------------------------ #
# Main runner                                                          #
# ------------------------------------------------------------------ #

def run_agent5(
    request: RetrievalRequest,
    agent4_output: Agent4Output,
) -> Agent5Output:
    print("\n" + "=" * 70)
    print("AGENT 5 — INTELLIGENT SCIENTIFIC DATA PREPARATION & ANALYSIS")
    print("=" * 70)

    log: List[ProcessingLogEntry] = []

    # STEP 1
    profiles, loaded = phase1_profile_datasets(agent4_output.retrieved_datasets)
    log.append(ProcessingLogEntry(
        stage="profiling",
        detail=f"Profiled {len(profiles)} dataset(s); {len(loaded)} opened successfully.",
    ))

    # STEP 2
    validations = phase2_validate_datasets(request, agent4_output.retrieved_datasets, profiles)
    log.append(ProcessingLogEntry(
        stage="validation",
        detail=f"Validated {len(validations)} dataset(s) against the scientific request.",
    ))

    # STEP 3
    plan, objective_achievable, stop_reason = phase3_build_preprocessing_plan(
        request, profiles, validations
    )
    log.append(ProcessingLogEntry(
        stage="planning",
        detail=plan.plan_rationale or "Plan built.",
    ))

    if not objective_achievable:
        print(f"\n[Agent 5] STOPPING: {stop_reason}")
        log.append(ProcessingLogEntry(stage="stop", detail=stop_reason or "Objective unreachable."))
        return Agent5Output(
            status=Agent5Status.stopped_objective_unreachable,
            stop_reason=stop_reason,
            dataset_profiles=profiles,
            validation_results=validations,
            preprocessing_plan=plan,
            processing_log=log,
        )

    # STEP 4
    loaded, executed_steps = phase4_execute_preprocessing(plan, loaded, request)
    log.append(ProcessingLogEntry(
        stage="preprocessing",
        detail=f"Executed {len(executed_steps)} step(s) "
               f"({sum(1 for s in executed_steps if s.success)} succeeded).",
    ))

    # STEP 5
    merged = phase5_merge_datasets(plan, loaded)
    if merged is not None:
        log.append(ProcessingLogEntry(
            stage="merging",
            detail=f"Merged {len([s for s in plan.steps if s.step == PreprocessingStepName.dataset_merging])} "
                   f"merge step(s) into a single dataset.",
        ))

    # STEP 6
    analysis_results = phase6_run_analysis(request, loaded, merged)
    log.append(ProcessingLogEntry(
        stage="analysis",
        detail=f"Produced {len(analysis_results)} analysis result(s).",
    ))

    # STEP 7
    quality = phase7_assess_quality(profiles, validations, executed_steps)
    log.append(ProcessingLogEntry(
        stage="quality_assessment",
        detail=f"Overall confidence {quality.overall_confidence:.2f}.",
    ))

    # STEP 8 — write outputs to the canonical Zarr store
    clean_paths: List[str] = []
    if merged is not None:
        clean_paths.append(zarr_store.write_dataset(merged, name="merged_output"))
    else:
        for source_id, ds in loaded.items():
            clean_paths.append(zarr_store.write_dataset(ds, name=f"clean_{source_id}"))

    log.append(ProcessingLogEntry(
        stage="output",
        detail=f"Wrote {len(clean_paths)} cleaned dataset(s) to the Zarr store.",
    ))

    status = (
        Agent5Status.completed
        if not quality.remaining_limitations
        else Agent5Status.completed_with_limitations
    )

    print(f"\n[Agent 5] Done. Status: {status.value}. "
          f"{len(clean_paths)} output dataset(s) written.")

    return Agent5Output(
        status=status,
        dataset_profiles=profiles,
        validation_results=validations,
        preprocessing_plan=plan,
        executed_steps=executed_steps,
        clean_dataset_paths=clean_paths,
        analysis_results=analysis_results,
        quality_assessment=quality,
        processing_log=log,
    )
