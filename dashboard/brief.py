"""Plain-English event brief from Gemini.

Rules and statistics decide the tier; the language model only explains the fields it is given, and the
dashboard shows those exact fields beside the brief so every claim can be checked. Events that need review
get a refusal without any API call. The key comes from GEMINI_API_KEY, or a GEMINI_API_KEY= line in the
repo-root .env (keep .env git-ignored).
"""
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]  # tried in order
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FIELDS = ["event_id", "c_object_type", "tier_final", "tier_now", "reason", "drivers", "time_to_tca",
          "risk_latest", "risk_forecast", "risk_hi", "p_critical", "p_critical_storm", "storm_flips",
          "miss_latest_m", "miss_forecast_m", "relative_speed", "pc_diluted", "maneuver_dr_m", "maneuver_dv_mps"]
# risk_lo is never sent: the team decided the lower end of the conformal interval is uninformative (it sits at the
# -30 floor for most events), so only the one-sided 95% upper bound is shown anywhere.
PROMPT = """You help a satellite conjunction analyst read one collision alert.
Using ONLY the JSON fields below, write 3 or 4 plain sentences:
1. why the event has its tier (tier_final), quoting the numbers that put it there;
2. how uncertain the forecast is (risk_hi is a one-sided 95% upper bound on the final collision probability);
3. what the analyst should check next.
Do not say whether to maneuver: that is the analyst's decision. If a field is null, call it unknown; never guess.
Fields named risk* hold log10 of the collision probability Pc: always write Pc itself, e.g. risk -3.2 as
6.3×10⁻⁴, and -30 as "below 10⁻³⁰ (effectively zero)"; never print the log value. p_critical is the probability
the event ends Critical at closest approach. Distances are metres, relative_speed is m/s, time_to_tca is days.

Fields:
{fields}"""


def api_key(root: Path = Path(__file__).resolve().parent.parent) -> str | None:
    if key := os.environ.get("GEMINI_API_KEY", "").strip():
        return key
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    return None


def facts(event: dict) -> dict:
    """The fields sent to the model, JSON-safe: NaN becomes null, numbers are rounded."""
    def clean(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if hasattr(v, "item"):  # numpy scalar
            v = v.item()
        return round(v, 3) if isinstance(v, float) else v
    return {f: clean(event[f]) for f in FIELDS if f in event}


def refusal(event: dict) -> str | None:
    if event.get("needs_review"):
        return (f"No brief: too little evidence to explain a forecast ({event.get('review_reason') or 'see reason'}). "
                "An analyst must review the CDM history.")
    return None


def call_gemini(prompt: str, key: str, model: str, timeout: float = 20) -> str:
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}})
    req = urllib.request.Request(URL.format(model=model), data=body.encode(), method="POST",
                                 headers={"content-type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        reply = json.load(res)
    return reply["candidates"][0]["content"]["parts"][0]["text"].strip()


def explain(fields: dict, key: str) -> str:
    """The brief from the first model that answers; RuntimeError naming each failure when none does."""
    prompt = PROMPT.format(fields=json.dumps(fields, indent=1))
    errors = []
    for model in MODELS:
        try:
            return call_gemini(prompt, key, model)
        except urllib.error.HTTPError as err:
            errors.append(f"{model}: HTTP {err.code} {err.reason}")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as err:
            errors.append(f"{model}: {err}")
    raise RuntimeError("; ".join(errors))
