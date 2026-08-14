from .constants import LABEL_COLS
from .guidance import GuidanceStore
from .llm import MistralTargetClassifier


class KneeGuidanceClassifier:
    def __init__(
        self,
        guidance_dir: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit: bool = True,
    ):
        self.guidance = GuidanceStore(guidance_dir)

        self.llm = MistralTargetClassifier(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
        )

    def predict_target(
        self,
        report: str,
        target: str,
    ):
        guidance = self.guidance.get(target)

        return self.llm.predict(
            report=report,
            target=target,
            guidance=guidance,
        )

    def predict_report(
        self,
        report: str,
    ):
        predictions = {}
        raw_outputs = {}

        for target in LABEL_COLS:
            label, raw = self.predict_target(
                report=report,
                target=target,
            )

            predictions[target] = label
            raw_outputs[target] = raw

        return predictions, raw_outputs
