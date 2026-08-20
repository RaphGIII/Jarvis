import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MAX_NEW_TOKENS, MODEL_ID, SYSTEM_PROMPT


class JarvisBrain:
    def __init__(self, model_id=MODEL_ID):
        print("Loading JARVIS brain...")

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )

        print("JARVIS brain online.")

    def ask(self, system_prompt, user_prompt, max_tokens=700, temperature=0.2, top_p=None):
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
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
            }
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

    def think(self, user_prompt, max_tokens=MAX_NEW_TOKENS):
        return self.ask(SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)

    def think_coding(self, user_prompt, max_tokens=900, temperature=0.6, top_p=0.9):
        return self.ask(SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
