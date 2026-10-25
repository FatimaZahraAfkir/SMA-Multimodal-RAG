from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def evaluate_rag(dataset):
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    return result