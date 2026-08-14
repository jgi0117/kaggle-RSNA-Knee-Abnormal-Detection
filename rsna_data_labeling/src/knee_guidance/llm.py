import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class QwenTargetClassifier:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_input_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        print(
            f"Loading tokenizer: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(
            f"Loading model: {model_name}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            dtype=torch.float16,
        )

        self.model.eval()

        print(
            f"Model memory: "
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
            enable_thinking=False,
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
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        raw_output = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return raw_output

    def translate(
        self,
        prompt: str,
    ) -> str:
        translated_report = self._generate(
            prompt=prompt,
            max_new_tokens=2048,
        )

        translated_report = translated_report.strip()

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
            max_new_tokens=64,
        )

        label = self._parse_label(
            raw_output=raw_output,
        )

        return label, raw_output

    @staticmethod
    def _parse_label(
        raw_output: str,
    ) -> int:
        match = re.search(
            r"\{.*?\}",
            raw_output,
            flags=re.DOTALL,
        )

        if match is None:
            raise ValueError(
                f"JSON not found: {raw_output}"
            )

        data = json.loads(
            match.group(0)
        )

        label = int(
            data["label"]
        )

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label: {label}"
            )

        return label