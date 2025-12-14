from sentence_transformers import SentenceTransformer, util


# Load embedding model once (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Generates embedding for a given text.
    """
    return model.encode(text, convert_to_tensor=True)


def score_relevance_completeness(response: str, context_chunks: list) -> float:
    """
    Measures how relevant and complete the response is
    compared to the entire retrieved context.
    """

    # combined_context = " ".join(chunk["text"] for chunk in context_chunks)
    texts = [
        chunk["text"]
        for chunk in context_chunks
        if isinstance(chunk, dict) and "text" in chunk and chunk["text"]
    ]

    combined_context = " ".join(texts)


    response_embedding = embed_text(response)
    context_embedding = embed_text(combined_context)

    similarity = util.cos_sim(response_embedding, context_embedding).item()
    return round(similarity, 3)


def score_hallucination(response: str, context_chunks: list) -> float:
    """
    Measures hallucination by checking if the response
    is supported by any context chunk.
    """

    response_embedding = embed_text(response)

    similarities = []
    for chunk in context_chunks:
        if not isinstance(chunk, dict):
            continue
        if "text" not in chunk or not chunk["text"]:
            continue

        chunk_embedding = embed_text(chunk["text"])
        similarity = util.cos_sim(response_embedding, chunk_embedding)
        similarities.append(similarity.item())
    max_similarity = max(similarities)
    hallucination_score = 1 - max_similarity

    return round(hallucination_score, 3)
    if not similarities:
        return 0.0

