import numpy as np
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads and returns a Sentence Transformer model.

    Args:
        model_name (str): The name of the Sentence Transformer model to load.

    Returns:
        SentenceTransformer: The loaded model, or None if an error occurs.
    """
    try:
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

def get_embedding(text, model):
    """
    Generates an embedding vector for the given text using the provided model.

    Args:
        text (str): The input text to embed.
        model (SentenceTransformer): The loaded Sentence Transformer model.

    Returns:
        numpy.ndarray: The embedding vector, or None if an error occurs or model is invalid.
    """
    if not isinstance(model, SentenceTransformer):
        print("Error: Invalid model provided. Please load a model using load_embedding_model().")
        return None
    if not isinstance(text, str) or not text.strip():
        print("Error: Input text must be a non-empty string.")
        return None

    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

if __name__ == "__main__":
    print("Testing embedding utilities...")

    # Test model loading
    model = load_embedding_model()

    if model:
        # Test embedding generation
        sample_sentence = "This is a test sentence for embedding generation."
        embedding_vector = get_embedding(sample_sentence, model)

        if embedding_vector is not None:
            print(f"Sample sentence: '{sample_sentence}'")
            print(f"Embedding vector shape: {embedding_vector.shape}")
            print(f"First 5 elements of embedding: {embedding_vector[:5]}")
        else:
            print("Failed to generate embedding for the sample sentence.")

        # Test with invalid input
        print("\nTesting with invalid input to get_embedding:")
        invalid_embedding = get_embedding("", model) # Empty string
        if invalid_embedding is None:
            print("Correctly handled empty string input.")

        invalid_embedding_model = get_embedding(sample_sentence, "not_a_model")
        if invalid_embedding_model is None:
            print("Correctly handled invalid model input.")

    else:
        print("Failed to load the embedding model. Cannot run further tests.")
