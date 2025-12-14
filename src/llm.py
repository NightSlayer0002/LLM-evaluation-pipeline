import time


def generate_llm_response(user_query: str) -> dict:
    """
    Simulates an LLM call.
    Returns response text, latency, and estimated cost.
    """

    start_time = time.time()

    # Simulated LLM output (for assignment purpose)
    response_text = (
        "Several affordable hotels near Malpani Infertility Clinic include "
        "Hotel Raj Palace, Hotel Blue Nile, and Hotel Woodland. "
        "These hotels offer basic amenities and are budget-friendly."
    )

    latency = time.time() - start_time

    # Very rough cost estimation
    token_estimate = len(response_text.split())
    cost_usd = (token_estimate / 1000) * 0.002  # approximate GPT pricing

    return {
        "response": response_text,
        "latency_seconds": round(latency, 3),
        "estimated_cost_usd": round(cost_usd, 6)
    }
