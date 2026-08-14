import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class E5Embedder:
    """
    Minimal multilingual E5 embedder implemented with transformers only.
    No sentence-transformers dependency required.
    """

    def __init__(
        self,
        model_name="intfloat/multilingual-e5-small",
        device=None,
        batch_size=32,
        max_length=512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(self, texts, kind="passage"):
        if isinstance(texts, str):
            texts = [texts]

        prefix = "query: " if kind == "query" else "passage: "
        texts = [prefix + str(x) for x in texts]

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            emb = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
