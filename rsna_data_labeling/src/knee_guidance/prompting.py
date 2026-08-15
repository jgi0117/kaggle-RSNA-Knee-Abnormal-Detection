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
You are classifying a knee MRI radiology report for one predefined target.

The same radiology report is evaluated independently for 12 different targets.
For this call, evaluate ONLY the TARGET below.

The TARGET GUIDANCE defines what evidence should and should not count for
this specific target. Apply it carefully.

TARGET:
{target}

TARGET GUIDANCE:
{guidance}

RADIOLOGY REPORT:
{report}

GENERAL RULES:

1. Use only the RADIOLOGY REPORT as patient-specific evidence.

2. The TARGET GUIDANCE defines the interpretation boundary for this target.
   Do not treat examples or medical knowledge in the guidance as evidence
   that the patient has the abnormality.

3. Evaluate the exact target, not merely a related abnormality.
   A nearby anatomical finding or clinically associated condition is not
   automatically evidence for the target.

4. Pay careful attention to:
   - anatomical location,
   - structural involvement,
   - negation,
   - uncertainty,
   - acute versus degenerative findings,
   - traumatic versus non-traumatic findings,
   - Findings versus Impression/Conclusion.

5. Do not classify by isolated keywords.
   Interpret the complete statement in which a finding appears.

6. When a report explicitly states that the target structure is normal,
   intact, preserved, or without tear, this is strong negative evidence
   unless another part of the report clearly provides stronger contradictory
   evidence.

7. When the Findings and Impression differ, interpret the report as a whole.
   A definitive Impression or Conclusion should receive substantial weight,
   but explicit contradictory structural findings must not be ignored.

8. Do not apply one universal severity threshold to all targets.
   Different targets have different diagnostic boundaries.
   Follow the TARGET GUIDANCE for the current target.

9. Findings belonging to another target must not be transferred to the
   current target unless the TARGET GUIDANCE specifically indicates that
   they are valid supporting evidence.

TASK:

Determine whether the RADIOLOGY REPORT provides sufficient evidence for
the TARGET according to the TARGET GUIDANCE.

Return exactly one JSON object:

{{"target": "{target}", "label": 0}}

or

{{"target": "{target}", "label": 1}}

Do not provide explanations or additional text.
""".strip()