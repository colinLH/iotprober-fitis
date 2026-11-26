"""Contracts shared by the local index reader, the demo and its evaluator."""

import re

RETRIEVAL_SCHEMA_VERSION = "v4-aligned-singletons"
COMMUNITY_MIN_SCORE = 0.70

# Strip acquisition formatting only to test visibility. Never feed this stripped
# text to the encoder: deployed indexes contain the original formatted strings.
_WRAPPERS = re.compile(
    r"\b(?:[a-z][\w-]*-\d+|endpoint\d+|vendor|product|version|subject|issuer|tls|"
    r"(?:as|whois)-[\w-]+|(?:network|organization)-(?:name|handle))\s*:", re.I
)
_NULLS = re.compile(r"\b(?:unknown(?: protocol| port)?|none|null|nan|n/a|n\.a\.)\b|<na>", re.I)
_AS_KEYS = re.compile(r"as-(?:asn|name|bgp_prefix|country_code)\s*:", re.I)


def has_observation(value):
    if value is None:
        return False
    text = _WRAPPERS.sub(" ", _AS_KEYS.sub(" ", str(value)))
    return bool(re.search(r"[^\W_]", _NULLS.sub(" ", text), re.UNICODE))


def view_text(fingerprint, index_cols, detail_cols):
    """Prefer exactly the corpus representation; support partial, split-only uploads."""
    indexed = [str(fingerprint[k]) for k in index_cols if has_observation(fingerprint.get(k))]
    if indexed:
        return "|".join(indexed)
    return "|".join(
        f"{k.split('-', 1)[-1].removesuffix('s')}: {fingerprint[k]}" for k in detail_cols
        if k not in index_cols and has_observation(fingerprint.get(k))
    )


def corpus_texts(frame, index_cols, detail_cols):
    names = [k for k in dict.fromkeys(index_cols + detail_cols) if k in frame.columns]
    return [view_text(dict(zip(names, values)), index_cols, detail_cols)
            for values in frame[names].itertuples(index=False, name=None)]


def verified_example(row):
    """Check actual vector and decision evidence; never infer a hit from the directory label."""
    truth = row.get("true_type")
    if not truth or row.get("predicted_type") != truth or row.get("status") not in (None, "completed"):
        return False
    local = ((row.get("retrieval_evidence") or {}).get("local") or {})
    hits = local.get("similar_devices") or []
    if not hits or not row.get("ip"):
        return False
    first = hits[0]
    score, coverage = first.get("similarity_score"), local.get("confidence_score")
    return bool(first.get("device_type") == truth and first.get("ip") and first["ip"] != row["ip"]
                and first.get("source", "vector_search") == "vector_search" and type(score) in (float, int)
                and type(coverage) in (float, int) and 0 < score <= 1 and 0 < coverage <= 1)
