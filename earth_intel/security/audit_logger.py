import json
from datetime import datetime


LOG_FILE = "logs/security_log.jsonl"


def log_security_event(
    event_type: str,
    details: str
):

    log_entry = {

        "timestamp": datetime.now().isoformat(),

        "event_type": event_type,

        "details": details

    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(log_entry)
        )

        f.write("\n")