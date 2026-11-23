# Lab Booking Request Sorter

A console tool that reads free-text lab booking requests. Google Gemini (via
Google AI Studio) extracts the lab, date, slot, number of students and a short
summary. Python then checks each request and gives it a status.

The tool never books anything. The lab in-charge confirms every booking.

## Statuses (checked in this order)

1. `needs_more_info`: lab, date, slot or students is missing, or a date or number fails the check
2. `lab_closed`: the date is a Sunday
3. `over_capacity`: more students than the lab holds
4. `conflict`: the lab and slot are already taken, in `bookings.csv` or by an earlier request in the same run
5. `available`: none of the above. The slot is held so that later requests see it as taken

## Files

| File | Purpose |
|---|---|
| `lab_sorter.py` | The program: LLM call, validation, date check, status rules, reports |
| `system_prompt.txt` | The prompt sent to Gemini (Authority, Exemplar, Constraint, Rubric, Metadata) |
| `requests.csv` | 22 booking requests, in the order they were sent |
| `labs.csv` | Lab codes, names and capacities |
| `bookings.csv` | Bookings already confirmed |
| `test_models.py` | Checks which Gemini models work with your key |
| `test_one.py` | Sends one request to Gemini and prints the reply |
| `output/` | Results from the full run |

## Setup

```powershell
py -m pip install google-genai
```

Set your key for the current terminal session. Never put the key in a file:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

Get a key at https://aistudio.google.com/apikey.

## Run

```powershell
py lab_sorter.py
```

The program writes four files to `output/`:

- `results.csv`: every request, with its status and reason
- `schedule.csv`: available requests in date order, for the lab in-charge to confirm
- `issues.csv`: requests that can't be booked as asked
- `summary.txt`: counts by status and by lab

## Notes

- Requests that name a lab without a code (for example "AI Lab") are marked `needs_more_info`. The tool asks for a clear lab code rather than guessing.
- If `Requests that fell back` is above 0, some Gemini calls failed. Run the tool again before using the results.