from tqdm.auto import tqdm

from .constants import LABEL_COLS
from .guidance import GuidanceStore
from .llm import MedGemmaTargetClassifier
from .prompting import build_target_prompt


class KneeGuidanceClassifier:
    def __init__(self, guidance_dir, model_name="google/medgemma-1.5-4b-it"):
        self.guidance_store = GuidanceStore(guidance_dir)
        self.llm = MedGemmaTargetClassifier(model_name=model_name)

    def classify_translated_report(self, translated_report: str, show_progress: bool = False):
        predictions = {}
        raw_outputs = {}
        failures = {}
        targets = LABEL_COLS
        if show_progress:
            targets = tqdm(LABEL_COLS, desc="Targets", unit="target", leave=False)
        for target in targets:
            guidance = self.guidance_store.get(target)
            prompt = build_target_prompt(
                report=translated_report,
                target=target,
                guidance=guidance,
            )
            try:
                label, raw_output = self.llm.predict(prompt=prompt, target=target)
                predictions[target] = label
                raw_outputs[target] = raw_output
            except Exception as exc:
                predictions[target] = None
                raw_outputs[target] = None
                failures[target] = f"{type(exc).__name__}: {exc}"
        return predictions, raw_outputs, failures

    def unload(self):
        self.llm.unload()
