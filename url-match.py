aaimport streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import re
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
match_product_flag = st.sidebar.checkbox("Match /product/ URLs with /products/ URLs", value=False)  # New flag

# File uploader for Google Cloud service account JSON
google_credentials = st.sidebar.file_uploader("Upload Google Cloud Service Account JSON", type="json")

# Option to manually provide source language (optional)
manual_source_language = st.sidebar.text_input("Source Language Code (optional, e.g., 'fr' for French)")

st.subheader("Upload Your Data")
broken_file = st.file_uploader("Upload Broken URLs CSV", type="csv")
working_file = st.file_uploader("Upload Working URLs CSV", type="csv")

# Function to preprocess URLs for better translation
def preprocess_url(url):
    try:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path  # Get only the URL path
        path = urllib.parse.unquote(path)  # Decode URL-encoded characters
        path = path.replace("-", " ").replace("_", " ").replace("/", " ")  # Convert hyphens, underscores, and slashes to spaces
        path = re.sub(r'[^a-zA-ZÃ-ÃÃ-Ã¶Ã¸-Ã¿\s]', '', path)  # Remove special characters and numbers
        path = path.strip()  # Remove leading/trailing spaces
        return path if path else url  # Return cleaned path or original if empty
    except Exception as e:
        return url  # Fallback to original URL if an error occurs

def should_match_product(broken_url, working_url):
    if match_product_flag:
        # Match broken URLs with "/product/" to working URLs with "/products/"
        if "/product/" in broken_url and "/products/" in working_url:
            return True
        # Additionally, match URLs with "/products/" to "/product/" URLs
        if "/products/" in broken_url and "/product/" in working_url:
            return True
        return False
    return True  # Always match if the flag is not enabled

if broken_file and working_file:
    broken_df = pd.read_csv(broken_file)
    working_df = pd.read_csv(working_file)

    # Expecting a single column in both CSVs named 'URL'
    if 'URL' not in broken_df.columns or 'URL' not in working_df.columns:
        st.error("CSV files must contain a column named 'URL'.")
    else:
        broken_urls = broken_df['URL'].tolist()
        working_urls = working_df['URL'].tolist()

        # Preprocess URLs before translation
        cleaned_broken_urls = [preprocess_url(url) for url in broken_urls]

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
        translated_broken_urls = cleaned_broken_urls  # Default to processed URLs
        if google_credentials:
            st.text("Authenticating with Google Cloud Translation API...")
            credentials_path = "/tmp/google_credentials.json"
            with open(credentials_path, "wb") as f:
                f.write(google_credentials.getvalue())
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

            translate_client = translate.Client()

            def translate_to_english(text):
                try:
                    if manual_source_language:
                        result = translate_client.translate(text, target_language="en", source_language=manual_source_language)
                    else:
                        result = translate_client.translate(text, target_language="en")

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
            for i, text in enumerate(cleaned_broken_urls):
                translated_text = translate_to_english(text)
                translated_broken_urls.append(translated_text)
                st.write(f"Original: {broken_urls[i]} -> Cleaned: {text} -> Translated: {translated_text}")
                translation_progress.progress((i + 1) / len(cleaned_broken_urls))
            st.success("Translation completed!")

# Compute embeddings for broken URLs
st.text("Computing embeddings for broken URLs, please wait...")
embedding_progress = st.progress(0)
broken_embeddings = []
matching_broken_urls = []  # Store URLs that are actually processed
for i, url in enumerate(translated_broken_urls):
    matching_urls = [w_url for w_url in working_urls if should_match_product(url, w_url)]  # Apply product matching logic
    if matching_urls:  # Only process if there is a valid match
        broken_embeddings.append(get_embedding(url))
        matching_broken_urls.append(url)  # Add this URL to the list of matched URLs
    else:
        broken_embeddings.append(None)  # Skip if no match found
    embedding_progress.progress((i + 1) / len(translated_broken_urls))
st.success("Embeddings for broken URLs completed!")

# Compute embeddings for working URLs
st.text("Computing embeddings for working URLs, please wait...")
embedding_progress = st.progress(0)
working_embeddings = []
matching_working_urls = []  # Store URLs that are actually processed
for i, url in enumerate(working_urls):
    # Only process if there is a corresponding matched broken URL
    matching_urls = [b_url for b_url in translated_broken_urls if should_match_product(b_url, url)]  
    if matching_urls:  # Only process if there is a valid match
        working_embeddings.append(get_embedding(url))
        matching_working_urls.append(url)  # Add this URL to the list of matched URLs
    else:
        working_embeddings.append(None)  # Skip if no match found
    embedding_progress.progress((i + 1) / len(working_urls))
st.success("Embeddings for working URLs completed!")

# Filter out None values in embeddings for broken and working URLs
broken_embeddings = [emb for emb in broken_embeddings if emb is not None]
working_embeddings = [emb for emb in working_embeddings if emb is not None]
broken_urls = [url for url, emb in zip(matching_broken_urls, broken_embeddings) if emb is not None]
working_urls = [url for url, emb in zip(matching_working_urls, working_embeddings) if emb is not None]

# Ensure we don't have misaligned lists for similarity calculation
if len(broken_embeddings) != len(working_embeddings):
    st.error("Mismatch in the number of embeddings for broken and working URLs. Please check your data.")
else:
    # Compute similarity scores
    st.text("Calculating similarity scores...")
    matches = []
    similarity_progress = st.progress(0)
    for i, b_emb in enumerate(broken_embeddings):
        similarities = [1 - cosine(b_emb, w_emb) for w_emb in working_embeddings]
        top_matches = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:3]

        if not top_matches or max([score for _, score in top_matches]) < similarity_threshold:
            matches.append({
                "Original Broken URL": broken_urls[i],
                "Cleaned Broken Text": cleaned_broken_urls[i],
                "Translated Broken Text": translated_broken_urls[i] if google_credentials else "Not Translated",
                "Match Rank": "No Match",
                "Matched URL": "No Match Found",
                "Similarity Score": "N/A"
            })
        else:
            for rank, (index, score) in enumerate(top_matches):
                if score >= similarity_threshold:
                    matches.append({
                        "Original Broken URL": broken_urls[i],
                        "Cleaned Broken Text": cleaned_broken_urls[i],
                        "Translated Broken Text": translated_broken_urls[i] if google_credentials else "Not Translated",
                        "Match Rank": rank + 1,
                        "Matched URL": working_urls[index],
                        "Similarity Score": score
                    })
        similarity_progress.progress((i + 1) / len(broken_embeddings))

    st.success("Similarity calculation completed!")

    results_df = pd.DataFrame(matches)

    if not results_df.empty:
        st.subheader("Top Matches (Including Non-Matches)")
        st.dataframe(results_df)

        # Download CSV option
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Matches CSV", csv_data, "semantic_url_matches.csv", "text/csv")
    else:
        st.warning("No matches found.")
