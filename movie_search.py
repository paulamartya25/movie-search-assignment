# movie_search.py

# Import necessary libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Data and Model Loading ---
# This part runs only once when the module is first imported.
print("Loading model and data for the first time...")

# Load the movies.csv dataset
df = pd.read_csv('movies.csv')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the embeddings for the 'plot' column
plot_embeddings = model.encode(df['plot'].tolist())

print("Model and data loaded successfully.")

# --- Search Function ---
def search_movies(query, top_n=5):
    """
    Searches for movies based on a query using semantic similarity.
    """
    # 1. Encode the search query
    query_embedding = model.encode([query])

    # 2. Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]

    # 3. Get top_n indices
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]

    # 4. Create a results DataFrame
    results_df = df.iloc[top_n_indices].copy()
    results_df['similarity'] = similarities[top_n_indices]

    return results_df