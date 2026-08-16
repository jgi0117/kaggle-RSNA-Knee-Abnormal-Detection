def build_translation_prompt(
    report: str,
) -> str:
    return f"""
You are translating a knee MRI radiology report into English.

Translate the RADIOLOGY REPORT into clear medical English.

IMPORTANT:
- Preserve the original clinical meaning exactly.
- Preserve all positive and negative findings.
- Preserve uncertainty, including expressions such as possible, suspected,
  favored, cannot exclude, or rule out.
- Preserve anatomical locations and laterality.
- Preserve severity, grading, measurements, and numerical values.
- Do not summarize.
- Do not add interpretations.
- Do not infer diagnoses that are not stated.
- Do not omit findings.
- If the report is already in English, return it unchanged.
- Output ONLY the translated radiology report.

RADIOLOGY REPORT:
{report}
""".strip()


def build_target_prompt(
    report: str,
    target: str,
    guidance: str,
) -> str:
    return f"""
Classify the knee MRI report for ONLY the target below.

TARGET:
{target}

GUIDANCE:
{guidance}

REPORT:
{report}

RULES:
- Use only the REPORT as patient-specific evidence.
- Follow the target-specific GUIDANCE.
- Respect anatomy, negation, uncertainty, and explicit normal findings.
- Do not infer the target from related abnormalities alone.
- When Findings and Impression conflict, judge the report as a whole.

Think carefully about the report and the target-specific guidance before deciding.

After reasoning, provide the final answer as exactly one JSON object:
{{"target": "{target}", "label": 0}}
or
{{"target": "{target}", "label": 1}}

Do not include any text after the final JSON object.
""".strip()
