def build_translation_prompt(report: str) -> str:
    return f"""
Translate the following knee MRI radiology report into English.

Requirements:
- Preserve all medical findings and anatomical locations.
- Preserve negations such as "no tear", "intact", and "normal".
- Preserve severity and diagnostic terms.
- Do not summarize.
- Do not add or remove findings.
- Return only the English translation.

REPORT:
{report}
""".strip()


def build_target_prompt(
    report: str,
    target: str,
    guidance: str,
) -> str:
    return f"""
You are classifying one knee MRI report for one binary target.

TARGET:
{target}

TARGET-SPECIFIC GUIDANCE:
{guidance}

MRI REPORT:
{report}

Think carefully but concisely.

Focus only on:
1. decisive evidence supporting the target,
2. decisive evidence against the target,
3. direct conflicts between Findings and Impression/Conclusion.

Rules:
- Do not restate the full report.
- Do not restate the full guidance.
- Do not discuss unrelated abnormalities.
- Prefer specific anatomical findings over vague indirect clues.
- If Findings and Impression conflict, use the target-specific guidance to decide which evidence should take priority.
- Classify only from the report and target-specific guidance.

After reasoning, output exactly one JSON object:

{{"target": "{target}", "label": 0}}

or

{{"target": "{target}", "label": 1}}

Do not output any text after the final JSON object.
""".strip()