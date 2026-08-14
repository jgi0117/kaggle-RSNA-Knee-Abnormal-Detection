import json
from .constants import LABEL_COLS


def build_prompt(report: str, guidance_context: str) -> str:
    schema = {k: 0 for k in LABEL_COLS}

    return f"""
You are classifying knee MRI radiology reports for a fixed 12-target dataset.

The report may be written in English, Spanish, French, German, Dutch, Greek,
Bulgarian, Turkish, or another language. Interpret the medical meaning in the
original language. Do not output a translation.

Use the retrieved medical guidance below as the primary clinical reference.
The report itself is the evidence for the patient; the guidance defines how to
interpret the finding.

Important:
- Handle negation carefully.
- Distinguish definite findings from suspected/equivocal findings.
- Distinguish degeneration/sprain from a definite tear when the guidance does.
- Respect anatomical compartment and laterality within the knee.
- Do not label a target positive just because a related term appears.
- If the report and guidance do not support a positive label, output 0.
- Output JSON only. No explanation.

RETRIEVED MEDICAL GUIDANCE:
{guidance_context}

KNEE MRI REPORT:
{report}

Return exactly these keys with integer values 0 or 1:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
