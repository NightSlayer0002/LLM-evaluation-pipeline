from utils import load_json, save_json
from llm import generate_llm_response
from evaluators import score_relevance_completeness, score_hallucination



CHAT_PATH = "data/chat.json"
CONTEXT_PATH = "data/context.json"
OUTPUT_PATH = "output/evaluation_report.json"


def run_evaluation_pipeline():
    # Load inputs
    chat_data = load_json(CHAT_PATH)
    context_data = load_json(CONTEXT_PATH)

    # Extract last user message
    user_query = chat_data["messages"][-1]["content"]

    # Extract context chunks from provided structure
    context_chunks = context_data["data"]["vector_data"]

    # Generate LLM response
    llm_output = generate_llm_response(user_query)

    # Evaluate
    relevance_score = score_relevance_completeness(
        llm_output["response"], context_chunks
    )

    hallucination_score = score_hallucination(
        llm_output["response"], context_chunks
    )

    # Final report
    report = {
        "user_query": user_query,
        "llm_response": llm_output["response"],
        "relevance_score": relevance_score,
        "hallucination_score": hallucination_score,
        "latency_seconds": llm_output["latency_seconds"],
        "estimated_cost_usd": llm_output["estimated_cost_usd"]
    }

    save_json(report, OUTPUT_PATH)
    print("Evaluation completed. Report saved to output/evaluation_report.json")


if __name__ == "__main__":
    run_evaluation_pipeline()
