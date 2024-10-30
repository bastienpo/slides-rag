import dspy
from src.slides_rag.mistral import MistralLM

lm = MistralLM(model="ministral-3b-2410")

dspy.configure(lm=lm)


class RetrievalGeneration(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


class SlidesRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(RetrievalGeneration)

    def forward(self, question: str):
        return self.generate_answer(question=question)
