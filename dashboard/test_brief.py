"""Verification loop for the Gemini brief: run `pytest dashboard -q`. No network: the HTTP layer is faked."""
import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import brief  # noqa: E402

EVENT = {"event_id": 7, "tier_final": "CRITICAL", "risk_latest": np.float64(-3.87654), "p_critical": np.nan,
         "storm_flips": np.bool_(True), "drivers": None, "needs_review": False, "not_a_field": 1}


def test_key_from_environment_beats_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n")
    monkeypatch.setenv("GEMINI_API_KEY", " from-env ")
    assert brief.api_key(tmp_path) == "from-env"


@pytest.mark.parametrize("text, expected", [
    ('OTHER=1\nGEMINI_API_KEY="quoted"\n', "quoted"),
    ("GEMINI_API_KEY=\n", None),  # empty value
    ("# GEMINI_API_KEY is unset\n", None),
])
def test_key_from_dotenv(tmp_path, monkeypatch, text, expected):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(text)
    assert brief.api_key(tmp_path) == expected


def test_no_key_anywhere(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert brief.api_key(tmp_path) is None


def test_facts_are_json_safe_and_limited_to_known_fields():
    sent = brief.facts(EVENT)
    assert sent == {"event_id": 7, "tier_final": "CRITICAL", "risk_latest": -3.877, "p_critical": None,
                    "storm_flips": True, "drivers": None}
    json.dumps(sent)  # numpy types and NaN would fail here


def test_the_uninformative_lower_bound_is_never_sent():
    assert "risk_lo" not in brief.facts({**EVENT, "risk_lo": -30.0, "risk_hi": -22.1})
    assert "95% upper bound" in brief.PROMPT and "90% interval" not in brief.PROMPT
    assert "never print the log value" in brief.PROMPT


def test_refusal_only_when_evidence_is_thin():
    assert brief.refusal(EVENT) is None
    why = brief.refusal({**EVENT, "needs_review": True, "review_reason": "only one CDM"})
    assert why.startswith("No brief") and "only one CDM" in why


def test_falls_back_to_the_next_model(monkeypatch):
    calls = []

    def fake(prompt, key, model, timeout=20):
        calls.append(model)
        if model == brief.MODELS[0]:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        assert '"tier_final": "CRITICAL"' in prompt and "Do not say whether to maneuver" in prompt
        return "brief text"
    monkeypatch.setattr(brief, "call_gemini", fake)
    assert brief.explain(brief.facts(EVENT), "k") == "brief text"
    assert calls == brief.MODELS


def test_every_model_failing_is_an_error_naming_each(monkeypatch):
    def fake(prompt, key, model, timeout=20):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(brief, "call_gemini", fake)
    with pytest.raises(RuntimeError) as err:
        brief.explain({}, "k")
    assert all(m in str(err.value) for m in brief.MODELS)


def test_request_shape_and_reply_parsing(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, key=req.get_header("X-goog-api-key"), body=json.loads(req.data))
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": " hi \n"}]}}]}).encode())
    monkeypatch.setattr(brief.urllib.request, "urlopen", fake_urlopen)
    assert brief.call_gemini("p", "secret", "m1") == "hi"
    assert seen["url"].endswith("/models/m1:generateContent") and "secret" not in seen["url"]  # key in header only
    assert seen["key"] == "secret" and seen["body"]["contents"][0]["parts"][0]["text"] == "p"


def test_empty_reply_is_an_error(monkeypatch):
    monkeypatch.setattr(brief.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b'{"candidates": []}'))
    with pytest.raises(IndexError):
        brief.call_gemini("p", "k", "m")
