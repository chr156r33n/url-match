import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import openai
import json
import os
from google.cloud import translate_v2 as translate

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Semantic URL Matcher with Google Cloud Translation")

st.sidebar.header("Settings")
use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Requires API Key)", value=False)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", disabled=not use_openai)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

# File uploader for Google Cloud service account JSON
google_credentials = st.sidebar.file_uploader("Upload Google Cloud Service Account JSON", type="json")

# Option to manually provide source language (optional)
manual_source_language = st.sidebar.text_input("Source Language Code (optional, e.g., 'fr' for French)")

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

        # Set up Google Cloud Translation if credentials are provided
        translated_broken_urls = broken_urls  # Default to original URLs
        if google_credentials:
            st.text("Authenticating with Google Cloud Translation API...")
            credentials_path = "/tmp/google_credentials.json"
            with open(credentials_path, "wb") as f:
                f.write(google_credentials.getvalue())
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

            translate_client = translate.Client()

            def translate_to_english(text):
                try:
                    translate_params = {"q": text, "target": "en"}
                    if manual_source_language:
                        translate_params["source"] = manual_source_language  # Use user-provided language if available
                    result = translate_client.translate(**translate_params)
                    translated_text = result.get("translatedText", text)

                    if translated_text == text:
                        st.warning(f"Translation unchanged for: {text}")

                    return translated_text
                except Exception as e:
                    st.error(f"Translation failed for '{text}': {str(e)}")
                    return text  # Return original if translation fails

            # Translate broken URLs if needed
            st.text("Translating non-English URLs, please wait...")
            translated_broken_urls = []
            translation_progress = st.progress(0)
            for i, url in enumerate(broken_urls):
                translated_text = translate_to_english(url)
                translated_broken_urls.append(translated_text)
                st.write(f"Original: {url} -> Translated: {translated_text}")
                translation_progress.progress((i + 1) / len(broken_urls))
            st.success("Translation completed!")

        # Compute embeddings
        st.text("Computing embeddings for broken URLs, please wait...")
        embedding_progress = st.progress(0)
        broken_embeddings = []
        for i, url in enumerate(translated_broken_urls):
            broken_embeddings.append(get_embedding(url))
            embedding_progress.progress((i + 1) / len(translated_broken_urls))
        st.success("Embeddings for broken URLs completed!")

        st.text("Computing embeddings for working URLs, please wait...")
        embedding_progress = st.progress(0)
        working_embeddings = []
        for i, url in enumerate(working_urls):
            working_embeddings.append(get_embedding(url))
            embedding_progress.progress((i + 1) / len(working_urls))
        st.success("Embeddings for working URLs completed!")

        broken_embeddings = np.array(broken_embeddings)
        working_embeddings = np.array(working_embeddings)

        # Compute similarity scores
        st.text("Calculating similarity scores...")
        matches = []
        similarity_progress = st.progress(0)
        for i, b_emb in enumerate(broken_embeddings):
            similarities = [1 - cosine(b_emb, w_emb) for w_emb in working_embeddings]
            top_matches = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:3]

            for rank, (index, score) in enumerate(top_matches):
                if score >= similarity_threshold:
                    matches.append({
                        "Original Broken URL": broken_urls[i],
                        "Translated Broken URL": translated_broken_urls[i] if google_credentials else "Not Translated",
                        "Match Rank": rank + 1,
                        "Matched URL": working_urls[index],
                        "Similarity Score": score
                    })
            similarity_progress.progress((i + 1) / len(broken_embeddings))

        st.success("Similarity calculation completed!")

        results_df = pd.DataFrame(matches)

        if not results_df.empty:
            st.subheader("Top Matches")
            st.dataframe(results_df)

            # Download CSV option
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Matches CSV", csv_data, "semantic_url_matches.csv", "text/csv")
        else:
            st.warning("No matches found above the similarity threshold.")

st.sidebar.info("Ensure CSVs have a single column named 'URL'. Upload a Google Cloud Service Account JSON to enable translation.")
