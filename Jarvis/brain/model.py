import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MAX_NEW_TOKENS, MODEL_ID, SYSTEM_PROMPT


class JarvisBrain:
    def __init__(self):
        print("Loading JARVIS brain...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto"
        )

        print("JARVIS brain online.")

    def ask(self, system_prompt, user_prompt, max_tokens=700):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=True
            )

        generated = output[0, inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

    def think(self, user_prompt, max_tokens=MAX_NEW_TOKENS):
        return self.ask(SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
