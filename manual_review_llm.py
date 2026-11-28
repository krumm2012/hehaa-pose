"""Optional candidate selection for uncertain manual-review evidence."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from http.client import HTTPException
from ipaddress import ip_address
from itertools import islice
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import yaml

from control_panel.settings import load_local_environment_variable
from deepseek_realtime_coach import UrllibDeepSeekTransport


MAX_REVIEW_CASES = 12
_MAX_OUTPUT_TOKENS = 768
_DEFAULT_PROVIDERS = {
    "deepseek": ("deepseek-v4-flash", "https://api.deepseek.com"),
    "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
}
_SYSTEM_PROMPT = (
    "Review uncertain tennis swing evidence. Use only the facts and candidate "
    "evidence supplied for each event. Candidate IDs are scoped to that event. "
    "Choose at most one candidate for each field, and only when its evidence "
    "clearly supports the choice. For contact, interpret offsets relative to the "
    "existing contact frame. Abstain on uncertain fields. Omit events with no "
    "supported choice or give them an empty candidate_ids array. Return only "
    'JSON in this format: {"reviews":[{"event_id":1,"candidate_ids":["stroke:1"]}]}. '
    'If no choice is supported, return {"reviews":[]}.'
)


@dataclass(frozen=True)
class ReviewChoice:
    field: str
    value: Union[str, int]
    evidence: Dict


@dataclass(frozen=True)
class ReviewCase:
    event_id: int
    facts: Dict
    choices: Dict[str, ReviewChoice]


@dataclass(frozen=True)
class ReviewOutcome:
    status: str
    model: Optional[str] = None
    selections: Dict[int, Tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ProviderSettings:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: float


def _section(config: Dict, key: str) -> Dict:
    section = config.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Invalid configuration section: {key}")
    return section


def _text_setting(config: Dict, key: str) -> str:
    value = config.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Invalid configuration value: {key}")
    return value.strip()


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _provider_settings(config_path: Path) -> Optional[_ProviderSettings]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Invalid configuration")
    review = _section(_section(config, "manual_review"), "llm")
    provider = (_text_setting(review, "provider") or "deepseek").lower()
    if provider not in _DEFAULT_PROVIDERS:
        return None

    existing = _section(_section(config, "realtime_swing"), provider)
    default_model, default_base_url = _DEFAULT_PROVIDERS[provider]
    env_path = config_path.resolve().parent.parent / ".env.local"
    api_key_env = (
        _text_setting(review, "api_key_env")
        or _text_setting(existing, "api_key_env")
        or f"{provider.upper()}_API_KEY"
    )
    load_local_environment_variable(env_path, api_key_env)
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        return None

    def provider_setting(key: str, default: str) -> str:
        value = _text_setting(review, key) or _text_setting(existing, key)
        if value:
            return value
        env_name = f"{provider.upper()}_{key.upper()}"
        load_local_environment_variable(env_path, env_name)
        return os.environ.get(env_name, "").strip() or default

    model = provider_setting("model", default_model)
    base_url = provider_setting("base_url", default_base_url)
    url = urlsplit(base_url)
    if (
        not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError("Invalid provider URL")
    if url.scheme != "https" and (url.scheme != "http" or not _is_loopback(url.hostname)):
        raise ValueError("Invalid provider URL scheme")

    raw_timeout = review.get("timeout_seconds", 12)
    if isinstance(raw_timeout, bool):
        raise ValueError("Invalid provider timeout")
    timeout = float(raw_timeout)
    if not math.isfinite(timeout):
        raise ValueError("Invalid provider timeout")
    return _ProviderSettings(provider, model, base_url, api_key, min(20, max(0.2, timeout)))


def _request_payload(cases: Dict[int, ReviewCase], settings: _ProviderSettings) -> Dict:
    evidence = []
    for event_id, case in cases.items():
        choices = {}
        for candidate_id, choice in case.choices.items():
            item = {"field": choice.field, "evidence": choice.evidence}
            if choice.field == "actual_stroke_type":
                item["value"] = choice.value
            choices[candidate_id] = item
        evidence.append({"event_id": event_id, "facts": case.facts, "choices": choices})

    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"cases": evidence}, allow_nan=False)},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if settings.provider == "deepseek":
        payload["max_tokens"] = _MAX_OUTPUT_TOKENS
    else:
        payload["max_completion_tokens"] = _MAX_OUTPUT_TOKENS
    return payload


def _unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return obj


def _parse_response(response: Dict, cases: Dict[int, ReviewCase]) -> Dict[int, Tuple[str, ...]]:
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Invalid completion envelope")
    completion = choices[0]
    if completion.get("finish_reason", "stop") != "stop":
        raise ValueError("Incomplete review")
    content = completion["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Invalid completion content")
    document = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or document.keys() != {"reviews"}:
        raise ValueError("Invalid reviews envelope")
    rows = document["reviews"]
    if not isinstance(rows, list) or len(rows) > len(cases):
        raise ValueError("Invalid reviews list")

    selections = {}
    reviewed = set()
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"event_id", "candidate_ids"}:
            raise ValueError("Invalid review row")
        event_id = row["event_id"]
        if type(event_id) is not int or event_id not in cases or event_id in reviewed:
            raise ValueError("Invalid review event ID")
        reviewed.add(event_id)
        case = cases[event_id]

        candidate_ids = row["candidate_ids"]
        if not isinstance(candidate_ids, list) or len(candidate_ids) > 2:
            raise ValueError("Invalid review candidate list")
        fields = set()
        for candidate_id in candidate_ids:
            if not isinstance(candidate_id, str) or candidate_id not in case.choices:
                raise ValueError("Invalid review candidate ID")
            candidate_field = case.choices[candidate_id].field
            if candidate_field not in ("actual_stroke_type", "contact") or candidate_field in fields:
                raise ValueError("Duplicate or invalid review field")
            fields.add(candidate_field)
        if candidate_ids:
            selections[case.event_id] = tuple(candidate_ids)
    return selections


def review_uncertain_events(cases: Sequence[ReviewCase], config_path: Path) -> ReviewOutcome:
    """Ask the configured provider to select candidates in one bounded batch."""
    candidates = (case for case in cases if case.choices)
    batch = dict(enumerate(islice(candidates, MAX_REVIEW_CASES), start=1))
    if not batch:
        return ReviewOutcome("no_candidates")

    try:
        settings = _provider_settings(config_path)
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError, yaml.YAMLError):
        return ReviewOutcome("not_configured")
    if settings is None:
        return ReviewOutcome("not_configured")

    try:
        payload = _request_payload(batch, settings)
        response = UrllibDeepSeekTransport().complete(
            base_url=settings.base_url,
            api_key=settings.api_key,
            payload=payload,
            timeout=settings.timeout_seconds,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ReviewOutcome("invalid_response", model=settings.model)
    except (OSError, HTTPException, TypeError, ValueError, OverflowError, RecursionError):
        return ReviewOutcome("unavailable", model=settings.model)

    try:
        selections = _parse_response(response, batch)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        return ReviewOutcome("invalid_response", model=settings.model)
    return ReviewOutcome("reviewed", model=settings.model, selections=selections)
