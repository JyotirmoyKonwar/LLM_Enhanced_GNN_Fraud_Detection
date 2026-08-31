from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TextPair:
    causal: str
    residual: str


class GemmaEnhancer:
    def __init__(self, model_name: str, device: str, max_input_tokens: int, max_new_tokens: int, lora_rank: int, lora_alpha: int, lora_dropout: float):
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=["q_proj", "v_proj"], bias="none",
        )).to(device)
        self.device, self.max_input_tokens, self.max_new_tokens = device, max_input_tokens, max_new_tokens

    def _prompt(self, text: str, causal: bool) -> str:
        mode = "discriminative features related to the node's fraud label" if causal else "generic background information unrelated to the node's fraud label"
        return f"Extract one concise sentence containing {mode}. Do not predict the label. Text: {text}\nAnswer:"

    @torch.no_grad()
    def generate(self, texts: list[str]) -> list[TextPair]:
        pairs = []
        for text in texts:
            outputs = []
            for causal in (True, False):
                inputs = self.tokenizer(self._prompt(text[:4000], causal), return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.device)
                tokens = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
                generated = tokens[0, inputs.input_ids.shape[1]:]
                outputs.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip().splitlines()[0])
            pairs.append(TextPair(*outputs))
        return pairs

    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens).to(self.device)
        hidden = self.model.base_model.model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)