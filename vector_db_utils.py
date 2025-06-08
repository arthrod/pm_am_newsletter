import chromadb
import shutil
import os
import numpy as np # For dummy data generation

# --- ChromaDB Utility Functions ---

def init_chroma_client(db_path="./chroma_db"):
    """
    Initializes and returns a ChromaDB PersistentClient.

    Args:
        db_path (str): The path to the ChromaDB database directory.

    Returns:
        chromadb.PersistentClient: The initialized client, or None if an error occurs.
    """
    try:
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        client = chromadb.PersistentClient(path=db_path)
        print(f"ChromaDB client initialized at path: {db_path}")
        return client
    except Exception as e:
        print(f"Error initializing ChromaDB client at {db_path}: {e}")
        return None

def get_or_create_collection(client, collection_name, metadata_config={"hnsw:space": "cosine"}):
    """
    Gets an existing ChromaDB collection or creates it if it doesn't exist.

    Args:
        client (chromadb.PersistentClient): The ChromaDB client.
        collection_name (str): The name of the collection.
        metadata_config (dict, optional): Configuration for the collection's metadata,
                                          e.g., to specify the distance metric.
                                          Defaults to {"hnsw:space": "cosine"}.

    Returns:
        chromadb.api.models.Collection.Collection: The collection object, or None if an error occurs.
    """
    if not client:
        print("Error: ChromaDB client is not initialized.")
        return None
    try:
        collection = client.get_or_create_collection(name=collection_name, metadata=metadata_config)
        print(f"Collection '{collection_name}' accessed/created successfully.")
        return collection
    except Exception as e:
        print(f"Error getting or creating collection '{collection_name}': {e}")
        return None

def add_article_embedding(collection, embedding, metadata, article_id):
    """
    Adds a single article's embedding, metadata, and ID to the collection.

    Args:
        collection (chromadb.api.models.Collection.Collection): The ChromaDB collection.
        embedding (list of float): The embedding vector for the article.
        metadata (dict): Metadata associated with the article (e.g., title, publish_date).
        article_id (str): A unique ID for the article.

    Returns:
        bool: True if addition was successful, False otherwise.
    """
    if not collection:
        print("Error: Collection is not valid.")
        return False
    if not embedding or not isinstance(embedding, list):
        print("Error: Embedding must be a non-empty list.")
        return False
    if not metadata or not isinstance(metadata, dict):
        print("Error: Metadata must be a non-empty dictionary.")
        return False
    if not article_id or not isinstance(article_id, str):
        print("Error: Article ID must be a non-empty string.")
        return False

    try:
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[article_id]
        )
        # print(f"Article ID '{article_id}' added to collection '{collection.name}'.")
        return True
    except Exception as e:
        print(f"Error adding article ID '{article_id}' to collection '{collection.name}': {e}")
        return False

def query_similar_articles(collection, query_embedding, n_results=5, include_fields=['metadatas', 'distances']):
    """
    Queries the collection for articles similar to the given query_embedding.

    Args:
        collection (chromadb.api.models.Collection.Collection): The ChromaDB collection.
        query_embedding (list of float): The embedding vector to query against.
        n_results (int, optional): The number of similar results to return. Defaults to 5.
        include_fields (list of str, optional): List of fields to include in the results
                                               (e.g., 'metadatas', 'documents', 'distances', 'embeddings').
                                               Defaults to ['metadatas', 'distances'].

    Returns:
        dict: A dictionary containing the query results, or None if an error occurs.
    """
    if not collection:
        print("Error: Collection is not valid.")
        return None
    if not query_embedding or not isinstance(query_embedding, list):
        print("Error: Query embedding must be a non-empty list.")
        return None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include_fields
        )
        # print(f"Query executed successfully on collection '{collection.name}'.")
        return results
    except Exception as e:
        print(f"Error querying collection '{collection.name}': {e}")
        return None

# --- Basic Test Block ---

if __name__ == "__main__":
    print("--- Testing ChromaDB Utility Functions ---")
    test_db_path = "./chroma_test_db_dir" # Changed name to avoid conflict if old one exists
    test_collection_name = "test_article_collection"

    # 1. Initialize client
    client = init_chroma_client(db_path=test_db_path)

    if client:
        # 2. Get or create collection
        collection = get_or_create_collection(client, test_collection_name)

        if collection:
            # 3. Add a dummy embedding
            # Generate a dummy embedding (e.g., 384 dimensions for all-MiniLM-L6-v2)
            dummy_embedding_1 = np.random.rand(384).tolist()
            dummy_metadata_1 = {'publish_date': '2023-01-01', 'title': 'Test Article 1', 'url': 'http://example.com/article1'}
            dummy_id_1 = "article_test_1"

            dummy_embedding_2 = np.random.rand(384).tolist() # Slightly different
            dummy_metadata_2 = {'publish_date': '2023-01-02', 'title': 'Test Article 2 Similar', 'url': 'http://example.com/article2'}
            dummy_id_2 = "article_test_2"

            # Make a very different embedding
            dummy_embedding_3 = (np.random.rand(384) * 10).tolist()
            dummy_metadata_3 = {'publish_date': '2023-01-03', 'title': 'Test Article 3 Different', 'url': 'http://example.com/article3'}
            dummy_id_3 = "article_test_3"


            print(f"\nAttempting to add article: {dummy_id_1}")
            add_success_1 = add_article_embedding(collection, dummy_embedding_1, dummy_metadata_1, dummy_id_1)
            print(f"Addition successful: {add_success_1}")

            print(f"\nAttempting to add article: {dummy_id_2}")
            add_success_2 = add_article_embedding(collection, dummy_embedding_2, dummy_metadata_2, dummy_id_2)
            print(f"Addition successful: {add_success_2}")

            print(f"\nAttempting to add article: {dummy_id_3}")
            add_success_3 = add_article_embedding(collection, dummy_embedding_3, dummy_metadata_3, dummy_id_3)
            print(f"Addition successful: {add_success_3}")

            # Verify count
            print(f"Number of items in collection: {collection.count()}")

            if add_success_1 and add_success_2 and add_success_3:
                # 4. Query for similar articles (using dummy_embedding_1 as query)
                print(f"\nQuerying for articles similar to '{dummy_id_1}'...")
                # In a real scenario, query_embedding would come from a new article's summary
                query_embedding_for_test = dummy_embedding_1

                similar_articles = query_similar_articles(collection, query_embedding_for_test, n_results=3)

                if similar_articles:
                    print("\nQuery Results:")
                    if similar_articles.get('ids'):
                        for i, article_id in enumerate(similar_articles['ids'][0]):
                            print(f"  Result {i+1}:")
                            print(f"    ID: {article_id}")
                            if similar_articles.get('metadatas') and similar_articles['metadatas'][0][i]:
                                print(f"    Metadata: {similar_articles['metadatas'][0][i]}")
                            if similar_articles.get('distances') and similar_articles['distances'][0][i] is not None:
                                print(f"    Distance: {similar_articles['distances'][0][i]:.4f}")
                            print("-" * 20)
                    else:
                        print("No 'ids' found in query results.")
                else:
                    print("Query failed or returned no results.")
            else:
                print("\nSkipping query test due to failure in adding dummy articles.")

            # 5. Clean up (optional, but good for testing)
            try:
                print(f"\nAttempting to delete collection: {test_collection_name}")
                client.delete_collection(name=test_collection_name)
                print(f"Collection '{test_collection_name}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting collection '{test_collection_name}': {e}")

        # Attempt to remove the database directory
        # Note: PersistentClient might keep file locks, making rmtree difficult immediately
        # For robust cleanup, client re-initialization or more advanced handling might be needed
        # if os.path.exists(test_db_path):
        #     try:
        #         print(f"Attempting to remove database directory: {test_db_path}")
        #         # You might need to ensure the client is fully shut down or reset if using PersistentClient
        #         # client.reset() # Resets the entire database! Use with caution.
        #         shutil.rmtree(test_db_path)
        #         print(f"Test database directory '{test_db_path}' removed successfully.")
        #     except Exception as e:
        #         print(f"Error removing test database directory '{test_db_path}': {e}")
        #         print("This might be due to file locks. Manual deletion might be required.")
    else:
        print("ChromaDB client initialization failed. Cannot run tests.")

    print("\n--- Testing Complete ---")
