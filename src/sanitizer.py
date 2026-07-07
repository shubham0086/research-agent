"""Output sanitizer for research-agent (standalone spoke).

Web results are UNTRUSTED external input: a page can carry hidden prompt-injection
("ignore your instructions and do X"). When research-agent runs behind the MCP
Security Gateway the gateway scrubs this; but run directly as a spoke it must
protect itself. This is a Python port of the gateway's sanitizer.scrubOutput:

  - strip hidden carriers (HTML comments, chat control tokens, zero-width chars)
  - score injection phrases in the whole payload; QUARANTINE at/above threshold
    (hold the result back instead of feeding a poisoned blob to the agent).

Never raises: a sanitizer fault must not break a research call.
"""
import re

# High-signal indirect-prompt-injection phrases -> risk (mirror of the JS gateway).
_OUTPUT_INJECTION = [
    re.compile(r"ignore (?:all |the |your )?(?:previous|prior|above) (?:instructions?|prompts?|rules?)", re.I),
    re.compile(r"disregard (?:all |the |your )?(?:previous|prior|above) (?:instructions?|prompts?|rules?)", re.I),
    re.compile(r"\b(?:delete_repository|drop\s+table|rm\s+-rf|curl\s+|wget\s+|POST\s+https?://)", re.I),
    re.compile(r"\b(?:send|post|exfiltrat\w*)\b.{0,40}\b(?:secret|token|password|\.ssh|id_rsa|env)\b", re.I),
]
_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_CHAT_TOKEN = re.compile(r"<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>", re.I)
_ZERO_WIDTH = re.compile("[​‌‍﻿]")  # ZWSP, ZWNJ, ZWJ, BOM

QUARANTINE_THRESHOLD = 2


def _clean_str(s, stripped):
    out = s
    if _HTML_COMMENT.search(out):
        out = _HTML_COMMENT.sub("", out); stripped.add("html_comment")
    if _CHAT_TOKEN.search(out):
        out = _CHAT_TOKEN.sub("", out); stripped.add("chat_control_token")
    if _ZERO_WIDTH.search(out):
        out = _ZERO_WIDTH.sub("", out); stripped.add("zero_width")
    return out


def _walk(v, stripped):
    if isinstance(v, str):
        return _clean_str(v, stripped)
    if isinstance(v, list):
        return [_walk(x, stripped) for x in v]
    if isinstance(v, dict):
        return {k: _walk(x, stripped) for k, x in v.items()}
    return v


def _concat(v, acc):
    if isinstance(v, str):
        acc.append(v)
    elif isinstance(v, list):
        for x in v:
            _concat(x, acc)
    elif isinstance(v, dict):
        for x in v.values():
            _concat(x, acc)
    return acc


def scrub_output(value):
    """Return (cleaned_value, meta). meta = {risk, quarantined, stripped}."""
    try:
        text = "\n".join(_concat(value, []))
        risk = sum(2 for p in _OUTPUT_INJECTION if p.search(text))
        if _HTML_COMMENT.search(text) and risk > 0:
            risk += 1
        stripped = set()
        cleaned = _walk(value, stripped)
        return cleaned, {"risk": risk, "quarantined": risk >= QUARANTINE_THRESHOLD, "stripped": sorted(stripped)}
    except Exception:
        return value, {"risk": 0, "quarantined": False, "stripped": []}
