import gc
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def build_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def clear_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QwenTranslator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_input_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        print(
            f"Loading Qwen tokenizer: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
            )
        )

        print(
            f"Loading Qwen translation model: "
            f"{model_name}"
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=(
                    build_4bit_config()
                ),
                device_map="auto",
                torch_dtype=torch.float16,
            )
        )

        self.model.eval()

        print(
            f"Qwen model memory: "
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

        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        raw_output = (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        return raw_output

    def translate(
        self,
        prompt: str,
    ) -> str:
        translated_report = (
            self._generate(
                prompt=prompt,
                max_new_tokens=2048,
            )
            .strip()
        )

        if not translated_report:
            raise ValueError(
                "Translation output is empty."
            )

        return translated_report

    def unload(self):
        if hasattr(
            self,
            "model",
        ):
            del self.model

        if hasattr(
            self,
            "tokenizer",
        ):
            del self.tokenizer

        clear_gpu_memory()


class MedGemmaTargetClassifier:
    def __init__(
        self,
        model_name: str = (
            "google/medgemma-1.5-4b-it"
        ),
    ):
        self.model_name = model_name

        print(
            f"Loading MedGemma processor: "
            f"{model_name}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name,
            )
        )

        print(
            f"Loading MedGemma classification model: "
            f"{model_name}"
        )

        # MedGemma 4B is small enough to run without
        # 4-bit quantization on a suitable Kaggle GPU.
        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        )

        self.model.eval()

        print(
            f"MedGemma model memory: "
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
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = (
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        inputs = inputs.to(
            self.model.device
        )

        input_length = (
            inputs["input_ids"]
            .shape[-1]
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=(
                    self.processor
                    .tokenizer
                    .pad_token_id
                ),
                eos_token_id=(
                    self.processor
                    .tokenizer
                    .eos_token_id
                ),
            )

        generated_tokens = outputs[0][
            input_length:
        ]

        raw_output = (
            self.processor.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        if not raw_output:
            print(
                "[WARNING] "
                "MedGemma returned empty output. "
                f"input_tokens={input_length}, "
                f"generated_tokens="
                f"{len(generated_tokens)}"
            )

        return raw_output

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

        return (
            label,
            raw_output,
        )

    @staticmethod
    def _parse_label(
        raw_output: str,
    ) -> int:
        raw_output = (
            raw_output.strip()
        )

        if not raw_output:
            raise ValueError(
                "MedGemma generated "
                "an empty response."
            )

        # Primary path:
        # parse the requested JSON object.
        match = re.search(
            r"\{.*?\}",
            raw_output,
            flags=re.DOTALL,
        )

        if match is not None:
            try:
                data = json.loads(
                    match.group(0)
                )

                label = int(
                    data["label"]
                )

                if label not in (
                    0,
                    1,
                ):
                    raise ValueError(
                        f"Invalid label: "
                        f"{label}"
                    )

                return label

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                pass

        # Fallback:
        # MedGemma may return "0", "1",
        # "label: 0", or similar despite the
        # JSON instruction.
        label_patterns = [
            r'"label"\s*:\s*([01])',
            r"\blabel\s*[:=]\s*([01])\b",
            r"^\s*([01])\s*$",
        ]

        for pattern in label_patterns:
            label_match = re.search(
                pattern,
                raw_output,
                flags=(
                    re.IGNORECASE
                    | re.MULTILINE
                ),
            )

            if label_match is not None:
                return int(
                    label_match.group(1)
                )

        raise ValueError(
            "Could not parse MedGemma "
            f"classification output: "
            f"{raw_output!r}"
        )

    def unload(self):
        if hasattr(
            self,
            "model",
        ):
            del self.model

        if hasattr(
            self,
            "processor",
        ):
            del self.processor

        clear_gpu_memory()