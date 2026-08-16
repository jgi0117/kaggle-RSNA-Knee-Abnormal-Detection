import gc
import json
import re
from typing import Callable

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
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", max_input_tokens: int = 8192):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        print(f"Loading Qwen tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading Qwen translation model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=build_4bit_config(),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"Qwen model memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB")

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
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
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def translate(self, prompt: str) -> str:
        translated_report = self._generate(prompt=prompt, max_new_tokens=2048).strip()
        if not translated_report:
            raise ValueError("Translation output is empty.")
        return translated_report

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        clear_gpu_memory()


class MedGemmaTargetClassifier:
    def __init__(self, model_name: str = "google/medgemma-1.5-4b-it"):
        self.model_name = model_name
        print(f"Loading MedGemma processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Loading MedGemma classification model: {model_name}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print(f"MedGemma model memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB")

    def _build_prefix_allowed_tokens_fn(
        self,
        prompt_length: int,
        candidate_token_ids: list[list[int]],
    ) -> Callable:
        eos_token_id = self.processor.tokenizer.eos_token_id
        sequences = [ids + [eos_token_id] for ids in candidate_token_ids]

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor):
            generated = input_ids[prompt_length:].tolist()
            allowed = set()
            for sequence in sequences:
                if len(generated) <= len(sequence) and sequence[:len(generated)] == generated:
                    if len(generated) < len(sequence):
                        allowed.add(sequence[len(generated)])
            return list(allowed) if allowed else [eos_token_id]

        return prefix_allowed_tokens_fn

    def _generate_constrained(self, prompt: str, target: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[-1]

        candidates = [
            json.dumps({"target": target, "label": 0}, ensure_ascii=False),
            json.dumps({"target": target, "label": 1}, ensure_ascii=False),
        ]
        candidate_token_ids = [
            self.processor.tokenizer.encode(candidate, add_special_tokens=False)
            for candidate in candidates
        ]
        prefix_allowed_tokens_fn = self._build_prefix_allowed_tokens_fn(
            prompt_length=prompt_length,
            candidate_token_ids=candidate_token_ids,
        )
        max_candidate_tokens = max(len(ids) for ids in candidate_token_ids)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_candidate_tokens + 1,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][prompt_length:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

    def predict(self, prompt: str, target: str):
        raw_output = self._generate_constrained(prompt=prompt, target=target)
        label = self._parse_label(raw_output=raw_output, expected_target=target)
        return label, raw_output

    @staticmethod
    def _parse_label(raw_output: str, expected_target: str) -> int:
        raw_output = raw_output.strip()
        if not raw_output:
            raise ValueError("MedGemma generated an empty response.")
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", raw_output, flags=re.DOTALL)
            if match is None:
                raise ValueError(f"JSON not found in MedGemma output: {raw_output!r}")
            data = json.loads(match.group(0))
        if "label" not in data:
            raise ValueError(f"Missing 'label' in MedGemma output: {raw_output!r}")
        if "target" in data and str(data["target"]) != expected_target:
            raise ValueError(
                f"Unexpected target in MedGemma output: {data['target']!r} "
                f"(expected {expected_target!r})"
            )
        label = int(data["label"])
        if label not in (0, 1):
            raise ValueError(f"Invalid label: {label}")
        return label

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        clear_gpu_memory()
