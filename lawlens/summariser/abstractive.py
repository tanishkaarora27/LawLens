"""Optional abstractive rewrite of each cited paragraph with a pretrained seq2seq model.

OFF by default. The reported evaluation uses the extractive pipeline, because extractive
points are verbatim and can be machine-checked against the source paragraph. Abstractive
points are fluent but can drift from the source, so the UI labels them "paraphrased".
"""

from __future__ import annotations

from .textutils import chunk_text

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


class AbstractiveCompressor:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def rewrite(self, paragraph_text: str, max_new_tokens: int = 70) -> str:
        outs = []
        for chunk in chunk_text(paragraph_text, 350):          # keeps each input under the 1024-token limit
            enc = self.tok(chunk, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=3, no_repeat_ngram_size=3
                )
            outs.append(self.tok.decode(ids[0], skip_special_tokens=True).strip())
        return " ".join(outs)
