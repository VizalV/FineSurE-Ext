"""
Utilities for stronger factuality error categorization experiments.
This module is additive and does not modify baseline FineSurE behavior.
"""

import ast
import json
from collections import Counter

ERROR_CATEGORIES = [
    "no error",
    "out-of-context error",
    "entity error",
    "predicate error",
    "circumstantial error",
    "grammatical error",
    "coreference error",
    "linking error",
    "other error",
]

ERROR_CATEGORY_SET = set(ERROR_CATEGORIES)
ERROR_ONLY_CATEGORIES = [
    "out-of-context error",
    "entity error",
    "predicate error",
    "circumstantial error",
    "grammatical error",
    "coreference error",
    "linking error",
    "other error",
]


def normalize_category(category):
    """Normalize model category text to a valid label from the closed taxonomy."""
    if category is None:
        return "other error"

    clean = str(category).strip().lower().replace("_", " ")
    clean = " ".join(clean.split())

    aliases = {
        "no factual error": "no error",
        "factual": "no error",
        "out of context error": "out-of-context error",
        "out-of-context": "out-of-context error",
        "out of context": "out-of-context error",
        "entity mismatch": "entity error",
        "predicate mismatch": "predicate error",
        "time/place error": "circumstantial error",
        "coref error": "coreference error",
        "discourse linking error": "linking error",
    }

    mapped = aliases.get(clean, clean)
    if mapped in ERROR_CATEGORY_SET:
        return mapped
    return "other error"


def build_strict_factcheck_prompt(transcript, sentences):
    """Prompt with stricter decision protocol and schema contract."""
    numbered_sentences = ["[{}] {}".format(i + 1, s) for i, s in enumerate(sentences)]
    joined_sentences = "\n".join(numbered_sentences)

    prompt = """
You are a factuality auditor.

Task:
Given a transcript and a summary, classify EACH summary sentence into exactly one category:
- no error
- out-of-context error
- entity error
- predicate error
- circumstantial error
- grammatical error
- coreference error
- linking error
- other error

Decision rules:
1) If the summary claim is missing from transcript evidence, use out-of-context error.
2) If main entity/argument/attribute is wrong, use entity error.
3) If action/relation/state is wrong, use predicate error.
4) If time/place/circumstance is wrong, use circumstantial error.
5) If reference resolution is wrong, use coreference error.
6) If ordering/causal/discourse links are wrong, use linking error.
7) If grammar makes meaning broken, use grammatical error.
8) If none apply and evidence supports the claim, use no error.

Output format requirements:
- Return ONLY a JSON array.
- The array length MUST equal the number of summary sentences.
- Keep original sentence text in the "sentence" field.
- Use only allowed categories above.
- No markdown, no explanation outside JSON.

Required JSON schema:
[
            {{
                "sentence": "<summary sentence>",
                "reason": "<one sentence evidence-based explanation>",
                "category": "<one allowed category>"
            }}
]

Transcript:
{transcript}

Summary ({count} sentences):
{summary}
""".strip().format(transcript=transcript, count=len(sentences), summary=joined_sentences)

    return prompt


def build_entailment_prompt(transcript, sentences):
        """Stage-1 prompt: decide if each summary sentence is entailed by transcript."""
        numbered_sentences = ["[{}] {}".format(i + 1, s) for i, s in enumerate(sentences)]
        joined_sentences = "\n".join(numbered_sentences)

        prompt = """
You are a strict factuality verifier.

Task:
For each summary sentence, decide whether it is fully supported by the transcript.

Output labels:
- entailed
- not_entailed

Output requirements:
- Return ONLY a JSON array.
- The array length MUST equal the number of summary sentences.
- Keep one object per summary sentence in original order.
- No markdown or extra text.

Schema:
[
    {{
        "sentence": "<summary sentence>",
        "entailment": "entailed|not_entailed",
        "reason": "<brief evidence-based reason>"
    }}
]

Transcript:
{transcript}

Summary ({count} sentences):
{summary}
""".strip().format(transcript=transcript, count=len(sentences), summary=joined_sentences)

        return prompt


def build_error_subtype_prompt(transcript, error_sentences):
        """Stage-2 prompt: classify error subtype for sentences already flagged as not entailed."""
        numbered_sentences = ["[{}] {}".format(i + 1, s) for i, s in enumerate(error_sentences)]
        joined_sentences = "\n".join(numbered_sentences)

        prompt = """
You are a factuality error subtype classifier.

Given transcript and a set of summary sentences that are already known to be NOT entailed,
assign exactly one error subtype for each sentence.

Allowed subtype labels:
- out-of-context error
- entity error
- predicate error
- circumstantial error
- grammatical error
- coreference error
- linking error
- other error

Output requirements:
- Return ONLY a JSON array.
- The array length MUST equal the number of listed sentences.
- Keep one object per sentence in original order.
- Use only allowed subtype labels above.
- No markdown or extra text.

Schema:
[
    {{
        "sentence": "<summary sentence>",
        "category": "<one allowed subtype>",
        "reason": "<brief reason>"
    }}
]

Transcript:
{transcript}

Not-entailed summary sentences ({count}):
{summary}
""".strip().format(transcript=transcript, count=len(error_sentences), summary=joined_sentences)

        return prompt


def build_repair_prompt(previous_output, sentences):
    """Repair prompt used after malformed/non-compliant model output."""
    schema_example = [
        {
            "sentence": sentences[0] if sentences else "",
            "reason": "reason",
            "category": "no error",
        }
    ]

    return """
Your previous response was not valid for the required schema.

Fix the response and return ONLY valid JSON.
Rules:
- JSON array length must be exactly {count}
- keys: sentence, reason, category
- category must be one of: {categories}
- no markdown fences

Previous invalid response:
{invalid}

Example schema:
{example}
""".strip().format(
        count=len(sentences),
        categories=", ".join(ERROR_CATEGORIES),
        invalid=previous_output,
        example=json.dumps(schema_example, ensure_ascii=True),
    )


def _extract_balanced_json_span(text):
    """Extract first balanced JSON list or dict span from text."""
    if text is None:
        return None

    cleaned = str(text).replace("```json", "").replace("```", "").strip()

    for opening, closing in (("[", "]"), ("{", "}")):
        start = cleaned.find(opening)
        if start < 0:
            continue

        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return cleaned[start : idx + 1]

    return None


def parse_factcheck_json(output, expected_num_sentences=None):
    """Parse and validate fact-checking output with strict schema checks."""
    payload = _extract_balanced_json_span(output)
    if payload is None:
        return {
            "success": False,
            "error": "json_not_found",
            "labels": [],
            "categories": [],
            "reasons": [],
            "items": [],
        }

    parsed = None
    parse_error = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(payload)
            break
        except Exception as exc:
            parse_error = str(exc)

    if parsed is None:
        return {
            "success": False,
            "error": "json_parse_failed: {}".format(parse_error),
            "labels": [],
            "categories": [],
            "reasons": [],
            "items": [],
        }

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return {
            "success": False,
            "error": "output_not_list",
            "labels": [],
            "categories": [],
            "reasons": [],
            "items": [],
        }

    categories = []
    reasons = []
    normalized_items = []

    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            return {
                "success": False,
                "error": "item_{}_not_dict".format(idx),
                "labels": [],
                "categories": [],
                "reasons": [],
                "items": [],
            }

        if "category" not in item:
            return {
                "success": False,
                "error": "item_{}_missing_category".format(idx),
                "labels": [],
                "categories": [],
                "reasons": [],
                "items": [],
            }

        normalized_category = normalize_category(item.get("category"))
        reason = str(item.get("reason", "")).strip()
        sentence = str(item.get("sentence", "")).strip()

        categories.append(normalized_category)
        reasons.append(reason)
        normalized_items.append(
            {
                "sentence": sentence,
                "reason": reason,
                "category": normalized_category,
            }
        )

    if expected_num_sentences is not None and len(normalized_items) != expected_num_sentences:
        return {
            "success": False,
            "error": "length_mismatch_expected_{}_got_{}".format(
                expected_num_sentences, len(normalized_items)
            ),
            "labels": [],
            "categories": [],
            "reasons": [],
            "items": [],
        }

    labels = [0 if category == "no error" else 1 for category in categories]

    return {
        "success": True,
        "error": "",
        "labels": labels,
        "categories": categories,
        "reasons": reasons,
        "items": normalized_items,
    }


def parse_entailment_json(output, expected_num_sentences=None):
    """Parse stage-1 entailment output and return binary labels (0 entailed, 1 not_entailed)."""
    payload = _extract_balanced_json_span(output)
    if payload is None:
        return {
            "success": False,
            "error": "json_not_found",
            "labels": [],
            "entailment": [],
            "reasons": [],
        }

    parsed = None
    parse_error = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(payload)
            break
        except Exception as exc:
            parse_error = str(exc)

    if parsed is None:
        return {
            "success": False,
            "error": "json_parse_failed: {}".format(parse_error),
            "labels": [],
            "entailment": [],
            "reasons": [],
        }

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return {
            "success": False,
            "error": "output_not_list",
            "labels": [],
            "entailment": [],
            "reasons": [],
        }

    entailment = []
    reasons = []
    labels = []

    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            return {
                "success": False,
                "error": "item_{}_not_dict".format(idx),
                "labels": [],
                "entailment": [],
                "reasons": [],
            }

        value = str(item.get("entailment", "")).strip().lower().replace("-", "_")
        if value in ("entailed", "supported", "yes"):
            normalized = "entailed"
            label = 0
        elif value in ("not_entailed", "not entailed", "unsupported", "no"):
            normalized = "not_entailed"
            label = 1
        else:
            return {
                "success": False,
                "error": "invalid_entailment_label_at_{}".format(idx),
                "labels": [],
                "entailment": [],
                "reasons": [],
            }

        entailment.append(normalized)
        labels.append(label)
        reasons.append(str(item.get("reason", "")).strip())

    if expected_num_sentences is not None and len(labels) != expected_num_sentences:
        return {
            "success": False,
            "error": "length_mismatch_expected_{}_got_{}".format(expected_num_sentences, len(labels)),
            "labels": [],
            "entailment": [],
            "reasons": [],
        }

    return {
        "success": True,
        "error": "",
        "labels": labels,
        "entailment": entailment,
        "reasons": reasons,
    }


def normalize_error_only_category(category):
    """Normalize subtype category and force non-no-error class in stage-2."""
    normalized = normalize_category(category)
    if normalized == "no error":
        return "other error"
    if normalized not in ERROR_ONLY_CATEGORIES:
        return "other error"
    return normalized


def aggregate_self_consistency(parsed_runs):
    """Majority-vote aggregation across multiple parsed runs."""
    if not parsed_runs:
        return {
            "labels": [],
            "categories": [],
            "confidence": [],
        }

    num_sentences = len(parsed_runs[0]["categories"])
    agg_categories = []
    agg_confidence = []

    for sent_idx in range(num_sentences):
        sent_votes = [run["categories"][sent_idx] for run in parsed_runs]
        count = Counter(sent_votes)
        top_category, top_freq = count.most_common(1)[0]
        agg_categories.append(top_category)
        agg_confidence.append(float(top_freq) / float(len(sent_votes)))

    agg_labels = [0 if category == "no error" else 1 for category in agg_categories]
    return {
        "labels": agg_labels,
        "categories": agg_categories,
        "confidence": agg_confidence,
    }
