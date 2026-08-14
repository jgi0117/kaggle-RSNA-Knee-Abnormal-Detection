from .constants import LABEL_COLS
from .guidance import GuidanceStore
from .llm import QwenTargetClassifier
from .prompting import build_target_prompt


class KneeGuidanceClassifier:
    def __init__(
        self,
        guidance_dir,
        model_name="Qwen/Qwen3-8B",
    ):
        self.guidance_store = GuidanceStore(
            guidance_dir
        )

        self.llm = QwenTargetClassifier(
            model_name=model_name,
        )

    def classify_report(
        self,
        report: str,
    ):
        predictions = {}
        raw_outputs = {}

        for target in LABEL_COLS:
            guidance = self.guidance_store.get(
                target
            )

            prompt = build_target_prompt(
                report=report,
                target=target,
                guidance=guidance,
            )

            label, raw_output = self.llm.predict(
                prompt=prompt,
                target=target,
            )

            predictions[target] = label
            raw_outputs[target] = raw_output

        return predictions, raw_outputs