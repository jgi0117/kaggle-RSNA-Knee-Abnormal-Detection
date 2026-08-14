import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .prompting import build_target_prompt


class MistralTargetClassifier:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit: bool = True,
        max_input_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

    @staticmethod
    def _extract_json(text: str):
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    @torch.inference_mode()
    def predict(
        self,
        report: str,
        target: str,
        guidance: str,
    ):
        prompt = build_target_prompt(
            report=report,
            target=target,
            guidance=guidance,
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        raw_output = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        parsed = self._extract_json(raw_output)

        if parsed is None:
            return None, raw_output

        try:
            label = int(parsed["label"])
        except (KeyError, TypeError, ValueError):
            return None, raw_output

        if label not in (0, 1):
            return None, raw_output

        return label, raw_output
