from .retriever import EmbeddingGuidanceRetriever
from .llm import MistralLabeler


class KneeGuidanceRAG:
    def __init__(
        self,
        index_dir: str,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit=True,
        top_k_per_target=2,
        max_context_chars=16000,
    ):
        self.retriever = EmbeddingGuidanceRetriever(
            index_dir
        )

        self.labeler = MistralLabeler(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
        )

        self.top_k_per_target = top_k_per_target
        self.max_context_chars = max_context_chars

    def predict(self, report: str):

        context, retrieved = (
            self.retriever.retrieve_all_targets(
                report,
                top_k_per_target=self.top_k_per_target,
                max_context_chars=self.max_context_chars,
            )
        )

        pred, raw = self.labeler.predict(
            report,
            context,
        )

        return {
            "prediction": pred,
            "raw_output": raw,
            "context": context,
            "retrieved": retrieved,
        }
