"""
Agent 4 — Size Approval.
"""

from models.agent4_schemas import SizeEstimate, format_bytes


def ask_size_approval(source_name: str, size_estimate: SizeEstimate, running_total_bytes: float) -> bool:
    print(f"\n--- Size check: {source_name} ---")
    print(f"  Estimated size: {size_estimate.human_readable}"
          + ("" if size_estimate.is_exact else "  (estimate -- exact size unavailable ahead of download)"))
    print(f"  Running total so far (including this source): {format_bytes(running_total_bytes + (size_estimate.estimated_bytes or 0))}")

    choice = input("  Proceed with this source? (yes / no — I'll try a smaller/alternate source instead): ").strip().lower()
    return choice in ("yes", "y")
