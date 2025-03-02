import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import openai

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Semantic URL Matcher")

st.sidebar.header("Settings")
use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Requires API Key)", value=False)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", disabled=not use_openai)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

st.subheader("Upload Your Data")
broken_file = st.file_uploader("Upload Broken URLs CSV", type="csv")
working_file = st.file_uploader("Upload Working URLs CSV", type="csv")

if broken_file and working_file:
    broken_df = pd.read_csv(broken_file)
    working_df = pd.read_csv(working_file)

    # Expecting a single column in both CSVs named 'URL'
    if 'URL' not in broken_df.columns or 'URL' not in working_df.columns:
        st.error("CSV files must contain a column named 'URL'.")
    else:
        broken_urls = broken_df['URL'].tolist()
        working_urls = working_df['URL'].tolist()

        # Select embedding model
        if use_openai and openai_api_key:
            openai.api_key = openai_api_key
            def get_embedding(text):
                response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
                return response['data'][0]['embedding']
        else:
            def get_embedding(text):
                return sbert_model.encode(text, convert_to_numpy=True)

        # Compute embeddings
        st.text("Computing embeddings, this may take a moment...")
        broken_embeddings = np.array([get_embedding(url) for url in broken_urls])
        working_embeddings = np.array([get_embedding(url) for url in working_urls])

        # Compute similarity scores
        matches = []
        for i, b_emb in enumerate(broken_embeddings):
            similarities = [1 - cosine(b_emb, w_emb) for w_emb in working_embeddings]
            top_matches = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:3]

            for rank, (index, score) in enumerate(top_matches):
                if score >= similarity_threshold:
                    matches.append({
                        "Broken URL": broken_urls[i],
                        "Match Rank": rank + 1,
                        "Matched URL": working_urls[index],
                        "Similarity Score": score
                    })

        results_df = pd.DataFrame(matches)

        if not results_df.empty:
            st.subheader("Top Matches")
            st.dataframe(results_df)

            # Download CSV option
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Matches CSV", csv_data, "semantic_url_matches.csv", "text/csv")
        else:
            st.warning("No matches found above the similarity threshold.")

st.sidebar.info("Ensure CSVs have a single column named 'URL'. The tool will compute semantic matches.")
