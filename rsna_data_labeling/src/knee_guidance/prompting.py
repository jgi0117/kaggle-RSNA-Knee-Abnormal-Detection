import json


def build_target_prompt(
    report: str,
    target: str,
    guidance: str,
) -> str:
    output_schema = {
        "target": target,
        "label": 0,
    }

    return f"""
You are an expert musculoskeletal radiologist interpreting a knee MRI radiology report.

Your task is to classify ONE target abnormality.

TARGET:
{target}

MEDICAL GUIDANCE:
The following guidance is a compact summary of external radiology/medical literature.
Use it only as clinical interpretation context. Do not invent competition-specific rules.

{guidance}

RADIOLOGY REPORT:
{report}

INSTRUCTIONS:
- The report may be written in English or another language. Interpret its medical meaning in the original language.
- The radiology report is the evidence for this patient.
- The medical guidance defines the relevant clinical/radiologic concepts.
- Determine whether the report supports the presence of the TARGET abnormality.
- Handle negation, uncertainty, anatomical compartment, and differential diagnoses carefully.
- Do not mark a finding positive merely because a related word appears.
- If the target is explicitly absent, normal, or not supported by the report, return label 0.
- If the report supports the target abnormality according to the supplied guidance, return label 1.
- Do not output reasoning, translation, markdown, or extra text.

Return JSON only in exactly this shape:
{json.dumps(output_schema, ensure_ascii=False)}
""".strip()
