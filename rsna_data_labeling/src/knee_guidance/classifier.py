from tqdm.auto import tqdm

from .constants import LABEL_COLS
from .guidance import GuidanceStore
from .llm import QwenTargetClassifier
from .prompting import (
    build_target_prompt,
    build_translation_prompt,
)


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

    def translate_report(
        self,
        report: str,
    ) -> str:
        translation_prompt = (
            build_translation_prompt(
                report=report,
            )
        )

        translated_report = (
            self.llm.translate(
                prompt=translation_prompt,
            )
        )

        return translated_report

    def classify_report(
        self,
        report: str,
        show_progress: bool = False,
    ):
        translated_report = (
            self.translate_report(
                report=report,
            )
        )

        predictions = {}
        raw_outputs = {}

        targets = LABEL_COLS

        if show_progress:
            targets = tqdm(
                LABEL_COLS,
                desc="Targets",
                unit="target",
                leave=False,
            )

        for target in targets:
            guidance = (
                self.guidance_store.get(
                    target
                )
            )

            prompt = build_target_prompt(
                report=translated_report,
                target=target,
                guidance=guidance,
            )

            label, raw_output = (
                self.llm.predict(
                    prompt=prompt,
                    target=target,
                )
            )

            predictions[target] = label
            raw_outputs[target] = (
                raw_output
            )

        return (
            predictions,
            raw_outputs,
            translated_report,
        )