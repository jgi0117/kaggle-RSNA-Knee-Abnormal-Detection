import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .constants import LABEL_COLS
from .prompting import build_prompt


class MistralLabeler:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit=True,
        max_input_tokens=8192,
    ):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @staticmethod
    def _parse_json(text: str):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None

        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        clean = {}
        for label in LABEL_COLS:
            if label not in obj:
                return None
            try:
                value = int(obj[label])
            except Exception:
                return None
            if value not in (0, 1):
                return None
            clean[label] = value
        return clean

    @torch.inference_mode()
    def predict(self, report: str, guidance_context: str, max_new_tokens=256):
        prompt = build_prompt(report, guidance_context)

        # Mistral v0.3에서 system-role 차이를 피하기 위해 single user message 사용
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_json(raw), raw
