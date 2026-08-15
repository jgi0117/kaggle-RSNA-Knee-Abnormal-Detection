def build_translation_prompt(
    report: str,
) -> str:
    return f"""
You are translating a knee MRI radiology report into English.

Translate the RADIOLOGY REPORT into clear medical English.

IMPORTANT:
- Preserve the original clinical meaning exactly.
- Preserve all positive and negative findings.
- Preserve uncertainty, such as "possible", "suspected", "cannot exclude", or equivalent expressions.
- Preserve anatomical locations and laterality.
- Preserve severity and grading information.
- Preserve measurements and numerical values.
- Do not summarize.
- Do not add interpretations.
- Do not infer diagnoses that are not stated in the original report.
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
You are classifying abnormalities from a knee MRI radiology report.

This same radiology report is evaluated independently for 12 predefined targets.
For this call, evaluate ONLY the target specified below.

Each target has its own target-specific medical guidance.
The guidance provides general medical knowledge for interpreting the report.
It is not patient-specific evidence.

TARGET:
{target}

MEDICAL GUIDANCE:
{guidance}

RADIOLOGY REPORT:
{report}

CLASSIFICATION PRINCIPLES:

1. Use only the RADIOLOGY REPORT as evidence about this patient.
   The MEDICAL GUIDANCE must only help interpret the meaning and clinical significance
   of findings stated in the report.

2. Do not classify the TARGET as positive simply because there is any minor abnormality
   related to the target.

3. Consider the overall strength and clinical significance of the evidence.
   Distinguish a definite, meaningful abnormality from a subtle, minimal, low-grade,
   equivocal, degenerative, or incidental finding.

4. Severity-related language must be interpreted in context rather than by rigid keyword rules.
   Terms describing a small amount, mild degree, low grade, subtle signal change,
   intrasubstance/interstitial change, degeneration, or limited abnormality may represent
   findings below the level of a clinically meaningful target abnormality.

5. However, do NOT automatically assign label 0 based on words such as
   "small", "mild", "partial", or "low-grade".
   A finding may still be positive when the report clearly describes a genuine and
   clinically meaningful abnormality for the TARGET.

6. Conversely, do not automatically assign label 1 simply because words such as
   "tear", "edema", "effusion", "chondrosis", "sprain", or "contusion" appear.
   Interpret their severity, certainty, anatomical relevance, and surrounding context.

7. Give greater weight to:
   - definite diagnostic statements,
   - clinically meaningful structural abnormalities,
   - moderate-to-severe or high-grade abnormalities,
   - findings emphasized in the Impression or Conclusion,
   - multiple consistent findings supporting the same diagnosis.

8. Give less weight to:
   - minimal or trace abnormalities,
   - subtle isolated signal changes,
   - low-grade changes without clear structural abnormality,
   - equivocal or suspected findings,
   - incidental findings,
   - abnormalities explicitly described as degenerative when they do not establish
     the TARGET itself.

9. Pay careful attention to:
   - negation,
   - uncertainty,
   - severity and grade,
   - anatomical location,
   - the distinction between related but different abnormalities,
   - conflicts between Findings and Impression.

10. If Findings and Impression differ, interpret the report as a whole.
    Prefer the more definitive diagnostic conclusion when supported by the report,
    but do not ignore explicit contradictory evidence.

11. Do not use findings from other targets as evidence for the current TARGET unless
    they directly support the diagnosis of the current TARGET.

TASK:

Determine whether the RADIOLOGY REPORT provides sufficiently definite and
clinically meaningful evidence for the TARGET.

Use the MEDICAL GUIDANCE to interpret the report, but do not apply rigid
keyword-based rules.

Return exactly one JSON object:

{{"target": "{target}", "label": 0}}

or

{{"target": "{target}", "label": 1}}

Do not provide explanations or additional text.
""".strip()