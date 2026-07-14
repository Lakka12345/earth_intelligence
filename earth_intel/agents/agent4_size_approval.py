"""
Agent 4 - Size Approval.
"""

from models.agent4_schemas import SizeEstimate, format_bytes

SMALL_DOWNLOAD_BYTES = 500 * 1024 * 1024
LARGE_DOWNLOAD_BYTES = 5 * 1024 * 1024 * 1024


def ask_size_approval(source_name: str, size_estimate: SizeEstimate, running_total_bytes: float) -> bool:
    estimated = size_estimate.estimated_bytes
    projected_total = running_total_bytes + (estimated or 0)

    print(f"\n--- Size check: {source_name} ---")
    print(
        f"  Estimated size: {size_estimate.human_readable}"
        + ("" if size_estimate.is_exact else "  (estimate -- exact size unavailable ahead of download)")
    )
    print(f"  Running total so far (including this source): {format_bytes(projected_total)}")

    if estimated is not None and estimated < SMALL_DOWNLOAD_BYTES:
        print("  Classified as SMALL (< 500 MB): proceeding automatically.")
        return True

    if estimated is not None and estimated <= LARGE_DOWNLOAD_BYTES:
        print("  Classified as MEDIUM (500 MB-5 GB): proceeding unless you cancel.")
        choice = input("  Press Enter to continue, or type 'cancel' to try an alternate source: ").strip().lower()
        return choice not in ("cancel", "c", "no", "n")

    print("  Classified as LARGE or unknown: approval is required before continuing.")
    if estimated is None:
        print("  Exact storage and download time are unavailable from this provider before retrieval.")
    else:
        print(f"  Required local storage: about {format_bytes(estimated)} for this source.")

    choice = input("  Proceed with this source? (yes / no -- I'll try an alternate source instead): ").strip().lower()
    return choice in ("yes", "y")
