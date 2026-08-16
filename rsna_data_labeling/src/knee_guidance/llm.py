import gc
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class QwenTargetClassifier:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        max_input_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_4bit_config(),
            device_map="auto",
            dtype=torch.float16,
        )

        self.model.eval()

        print(
            "Model memory: "
            f"{self.model.get_memory_footprint() / 1024**3:.2f} GB"
        )

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def translate(self, prompt: str) -> str:
        translated_report = self._generate(
            prompt=prompt,
            max_new_tokens=2048,
        ).strip()

        if not translated_report:
            raise ValueError(
                "Translation output is empty."
            )

        return translated_report

    def predict(
        self,
        prompt: str,
        target: str,
    ):
        raw_output = self._generate(
            prompt=prompt,
            max_new_tokens=4096,
        )

        label = self._parse_label(
            raw_output=raw_output,
            expected_target=target,
        )

        return label, raw_output

    @staticmethod
    def _parse_label(
        raw_output: str,
        expected_target: str,
    ) -> int:
        if not raw_output.strip():
            raise ValueError(
                "Classification output is empty."
            )

        matches = re.findall(
            r"\{[^{}]*\}",
            raw_output,
            flags=re.DOTALL,
        )

        for candidate in reversed(matches):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if "label" not in data:
                continue

            if (
                "target" in data
                and str(data["target"]) != expected_target
            ):
                continue

            try:
                label = int(data["label"])
            except (TypeError, ValueError):
                continue

            if label in (0, 1):
                return label

        fallback_matches = list(
            re.finditer(
                r'"label"\s*:\s*([01])',
                raw_output,
                flags=re.IGNORECASE,
            )
        )

        if fallback_matches:
            return int(
                fallback_matches[-1].group(1)
            )

        raise ValueError(
            "Could not parse classification output: "
            f"{raw_output!r}"
        )

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        clear_gpu_memory()


class QwenTranslator(QwenTargetClassifier):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_input_tokens: int = 8192,
    ):
        super().__init__(
            model_name=model_name,
            max_input_tokens=max_input_tokens,
        )
