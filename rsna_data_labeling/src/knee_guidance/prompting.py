import json


def build_target_prompt(
    report: str,
    target: str,
    guidance: str,
) -> str:

    return f"""
You are classifying abnormalities from a knee MRI radiology report.

This same radiology report is evaluated independently for 12 predefined targets.
For this call, evaluate ONLY the target specified below.

Each target has its own target-specific medical guidance.
The guidance is general medical knowledge used only to interpret the report.

IMPORTANT:
- The MEDICAL GUIDANCE is NOT evidence about this patient.
- Only statements in the RADIOLOGY REPORT may be used as patient evidence.
- Do not classify the target as positive merely because positive findings or disease terminology appear in the guidance.
- Ignore findings related only to other targets unless they provide direct evidence for the current TARGET.
- Pay careful attention to negation, uncertainty, anatomical location, and differential diagnoses.
- Interpret reports written in languages other than English according to their clinical meaning.

TARGET:
{target}

MEDICAL GUIDANCE:
{guidance}

RADIOLOGY REPORT:
{report}

TASK:
Determine whether the RADIOLOGY REPORT supports the presence of the TARGET.

Return exactly one JSON object:

{{"target": "{target}", "label": 0}}

or

{{"target": "{target}", "label": 1}}

Do not provide explanations or additional text.
""".strip()