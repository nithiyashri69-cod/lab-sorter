"""
Lab Booking Request Sorter (Assessment 5)

Reads requests.csv, labs.csv and bookings.csv. Google Gemini (AI Studio)
extracts lab, date, slot, students and summary from each request. Python
checks those details and decides the status. Writes results.csv,
schedule.csv, issues.csv and summary.txt. The tool never writes to
bookings.csv: the lab in-charge confirms every booking.
"""

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime

# ---------------------------------------------------------------- settings
REQUESTS_FILE = "requests.csv"
LABS_FILE = "labs.csv"
BOOKINGS_FILE = "bookings.csv"
PROMPT_FILE = "system_prompt.txt"
OUTPUT_DIR = "output"

SEMESTER_START = date(2026, 10, 1)   # change to your semester's dates
SEMESTER_END = date(2027, 1, 31)

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
MAX_ATTEMPTS = 2                     # first call + one retry
MAX_SUMMARY_WORDS = 12
EXPECTED_KEYS = {"lab", "date", "slot", "students", "summary"}
VALID_SLOTS = {"FN", "AN"}
BANNED_SUMMARY_WORDS = {"urgent", "priority", "approved", "approval",
                        "confirmed", "confirm", "asap"}
FALLBACK_REPLY = {"lab": None, "date": None, "slot": None,
                  "students": None, "summary": "Details could not be read"}
STATUSES = ["needs_more_info", "lab_closed", "over_capacity",
            "conflict", "available"]
RESULT_FIELDS = ["id", "sent", "request", "lab", "date", "slot",
                 "students", "summary", "status", "reason"]


# ------------------------------------------------------------ load files
def load_csv(path, required_columns):
    """Read a CSV into a list of dicts, checking the required columns exist."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = set(required_columns) - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"Error: {path} is missing columns {sorted(missing)}")
            return [{k: (v or "").strip() for k, v in row.items()}
                    for row in reader]
    except FileNotFoundError:
        sys.exit(f"Error: cannot find {path}")


def load_requests(path=REQUESTS_FILE):
    return load_csv(path, ["id", "sent", "request"])


def load_labs(path=LABS_FILE):
    """Return {code: {"name": ..., "capacity": int}}."""
    labs = {}
    for row in load_csv(path, ["code", "name", "capacity"]):
        try:
            capacity = int(row["capacity"])
        except ValueError:
            sys.exit(f"Error: capacity for {row['code']} is not a number")
        labs[row["code"]] = {"name": row["name"], "capacity": capacity}
    return labs


def load_bookings(path=BOOKINGS_FILE):
    """Return a set of (date, slot, lab) tuples already confirmed."""
    rows = load_csv(path, ["date", "slot", "lab", "booked_for"])
    return {(r["date"], r["slot"], r["lab"]) for r in rows}


def load_system_prompt(labs, path=PROMPT_FILE):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text.replace("{{LAB_CODES}}", ", ".join(sorted(labs)))


# ------------------------------------------------------------ LLM call
def call_llm(system_prompt, request_text):
    """Send one request to Gemini (Google AI Studio) and return the raw text.
    The key is read from the GEMINI_API_KEY environment variable."""
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=request_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=300,
            temperature=0,              # same request, same answer
            response_mime_type="application/json",
        ),
    )
    return response.text


def parse_json(raw):
    """Parse the reply. Strip ```json fences if the model adds them."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return json.loads(text)


def is_valid(data, labs):
    """Check the reply has the exact shape the spec asks for.
    Returns (True, "") or (False, reason)."""
    if not isinstance(data, dict):
        return False, "reply is not a JSON object"
    if set(data.keys()) != EXPECTED_KEYS:
        return False, f"wrong keys: {sorted(data.keys())}"

    lab, d = data["lab"], data["date"]
    slot, students, summary = data["slot"], data["students"], data["summary"]

    if lab is not None and lab not in labs:
        return False, f"unknown lab {lab!r}"
    if d is not None and not isinstance(d, str):
        return False, "date is not text"
    if slot is not None and slot not in VALID_SLOTS:
        return False, f"bad slot {slot!r}"
    if students is not None:
        if isinstance(students, bool) or not isinstance(students, int) or students < 1:
            return False, "students is not a positive whole number"
    if not isinstance(summary, str):
        return False, "summary is not text"
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        return False, "summary longer than 12 words"
    if set(re.findall(r"[a-z]+", summary.lower())) & BANNED_SUMMARY_WORDS:
        return False, "summary claims priority or approval"
    return True, ""


def get_details(request_text, system_prompt, labs):
    """Ask the LLM, check the reply, retry once, then fall back.
    Returns (fields, calls_made, used_fallback). Copies only the five fields."""
    problem = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(5)
        user_msg = request_text
        if problem:
            user_msg += (f"\n\n(Your previous reply was rejected: {problem}. "
                         "Reply with only the JSON object.)")
        try:
            data = parse_json(call_llm(system_prompt, user_msg))
        except json.JSONDecodeError:
            problem = "not valid JSON"
            continue
        except Exception as err:          # network, quota, auth, etc.
            problem = f"API error: {err}"
            continue

        ok, why = is_valid(data, labs)
        if ok:
            fields = {k: data[k] for k in EXPECTED_KEYS}
            return fields, attempt, False
        problem = why

    return dict(FALLBACK_REPLY), MAX_ATTEMPTS, True


# ------------------------------------------------------ Python checks
def number_in_text(n, text):
    """True if the number appears as a standalone number in the text."""
    return re.search(rf"(?<!\d)0?{n}(?!\d)", text) is not None


def check_date(date_text, request_text):
    """Check the date is real, inside the semester, and its day number
    appears in the request. Returns (date or None, reason)."""
    if not date_text:
        return None, "no date given"
    try:
        d = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return None, f"'{date_text}' is not a real calendar date"
    if not (SEMESTER_START <= d <= SEMESTER_END):
        return None, f"{d} is outside the semester"
    if not number_in_text(d.day, request_text):
        return None, f"day {d.day} is not written in the request"
    return d, ""


def decide(request_text, details, labs, taken):
    """Apply the status rules in order. Returns (status, reason, date or None).
    `taken` is updated when a slot is held, so later requests see it."""
    lab, slot = details["lab"], details["slot"]
    students = details["students"]

    # 1. Check the facts
    d = None
    if details["date"]:
        d, why = check_date(details["date"], request_text)
        if d is None:
            return "needs_more_info", f"date problem: {why}", None
    if students is not None and not number_in_text(students, request_text):
        return "needs_more_info", f"{students} students not stated in request", None

    # 2. Missing details
    missing = [name for name, val in
               [("lab", lab), ("date", d), ("slot", slot), ("students", students)]
               if val is None]
    if missing:
        return "needs_more_info", "missing " + ", ".join(missing), d

    # 3. Sunday
    if d.weekday() == 6:
        return "lab_closed", f"{d} is a Sunday", d

    # 4. Capacity
    capacity = labs[lab]["capacity"]
    if students > capacity:
        return "over_capacity", f"{students} students but {lab} holds {capacity}", d

    # 5. Conflict (confirmed bookings or earlier requests this run)
    key = (d.isoformat(), slot, lab)
    if key in taken:
        return "conflict", f"{lab} {slot} on {d} is already taken", d

    # 6. Available: hold the slot
    taken.add(key)
    return "available", "slot is free", d


# ------------------------------------------------------ run and report
def process(requests, labs, taken, system_prompt):
    results = []
    total_calls = 0
    fallbacks = 0
    for req in requests:
        details, calls, used_fallback = get_details(req["request"], system_prompt, labs)
        total_calls += calls
        fallbacks += used_fallback

        status, reason, d = decide(req["request"], details, labs, taken)
        results.append({
            "id": req["id"],
            "sent": req["sent"],
            "request": req["request"],
            "lab": details["lab"] or "",
            "date": d.isoformat() if d else (details["date"] or ""),
            "slot": details["slot"] or "",
            "students": "" if details["students"] is None else details["students"],
            "summary": details["summary"],
            "status": status,
            "reason": reason,
        })
        time.sleep(5)
    return results, total_calls, fallbacks


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, results, total_calls, fallbacks):
    status_counts = Counter(r["status"] for r in results)
    lab_counts = defaultdict(Counter)
    for r in results:
        lab_counts[r["lab"] or "(no lab)"][r["status"]] += 1

    lines = [f"Requests processed: {len(results)}", "", "Counts by status:"]
    for s in STATUSES:
        lines.append(f"  {s:<16} {status_counts.get(s, 0)}")
    lines += ["", "Counts by lab:"]
    for lab in sorted(lab_counts):
        parts = ", ".join(f"{s}={n}" for s, n in sorted(lab_counts[lab].items()))
        lines.append(f"  {lab:<14} {parts}")
    lines += ["", f"LLM API calls made: {total_calls}",
              f"Requests that fell back: {fallbacks}"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def write_reports(results, total_calls, fallbacks, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    write_csv(os.path.join(out_dir, "results.csv"), RESULT_FIELDS, results)

    available = [r for r in results if r["status"] == "available"]
    available.sort(key=lambda r: (r["date"], r["slot"], r["lab"]))
    write_csv(os.path.join(out_dir, "schedule.csv"), RESULT_FIELDS, available)

    issues = [r for r in results if r["status"] != "available"]
    write_csv(os.path.join(out_dir, "issues.csv"),
              ["id", "sent", "request", "status", "reason"], issues)

    return write_summary(os.path.join(out_dir, "summary.txt"),
                         results, total_calls, fallbacks)


def main():
    try:
        from google import genai  # noqa: F401
    except ImportError:
        sys.exit("Install the SDK first:  pip install google-genai")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY in your terminal. Never put it in the code.")

    labs = load_labs()
    taken = load_bookings()
    requests = load_requests()

    if "--one" in sys.argv:          # Checkpoint 2: one request end to end
        requests = requests[:1]

    system_prompt = load_system_prompt(labs)
    results, total_calls, fallbacks = process(requests, labs, taken, system_prompt)
    summary = write_reports(results, total_calls, fallbacks)

    print(summary)
    print(f"\nWrote results.csv, schedule.csv, issues.csv and summary.txt to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()