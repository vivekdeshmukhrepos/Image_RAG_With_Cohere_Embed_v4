import streamlit as st
import requests
import os
import io
import base64
from PIL import Image
import numpy as np
import cohere
from google import genai
from Config import Configuration as Config

# --- Constants ---
MAX_PIXELS = Config.MAX_PIXELS  # Maximum allowed pixels for image resizing to control memory usage
IMAGE_FOLDER = Config.Image_Folder_Name  # Folder to store downloaded images

# --- Initialize API clients ---
co = cohere.ClientV2(api_key=Config.Cohere_API_key)  # Cohere client for embeddings
client = genai.Client(api_key=Config.Gemini_API_key)  # Google Gemini client for LLM responses

# --- Streamlit page configuration ---
st.set_page_config(page_title="Image Q&A with Cohere & Gemini", layout="wide")

# --- Custom CSS for app theming ---
custom_css = """
<style>
/* Expander styling */
.streamlit-expanderHeader {
    font-size: 16px;
    font-weight: bold;
    color: #336699;
}
</style>
"""
# Inject custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# --- Sidebar with model information ---
with st.sidebar:
    st.header("ℹ️ Model Overview")
    # Expandable section for Cohere Embed-4 model details
    with st.expander("Cohere Embed-4", expanded=True):
        st.markdown("""
        - Advanced multimodal model for enterprise search.
        - Combines text and images for accurate retrieval.
        - Handles complex images like charts without preprocessing.
        """)
    # Expandable section for Google Gemini 2.5 Flash model details
    with st.expander("Google Gemini 2.5 Flash", expanded=True):
        st.markdown("""
        - Efficient multimodal processing for text and images.
        - Balances speed and precision for real-time applications.
        """)

# --- Helper functions ---
@st.cache_data(show_spinner=False)
def download_image(url: str, filename: str) -> str:
    """
    Download an image from a URL and save it locally.
    If the image already exists locally, skip downloading.
    
    Args:
        url (str): URL of the image to download.
        filename (str): Local filename to save the image.
    
    Returns:
        str: Path to the saved image file.
    """
    os.makedirs(IMAGE_FOLDER, exist_ok=True)  # Ensure image folder exists
    path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
    return path

def resize_image(pil_img: Image.Image) -> Image.Image:
    """
    Resize an image if its pixel count exceeds MAX_PIXELS.
    Maintains aspect ratio and uses high-quality resizing.
    
    Args:
        pil_img (PIL.Image.Image): Input PIL image.
    
    Returns:
        PIL.Image.Image: Resized image if needed, else original.
    """
    width, height = pil_img.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        new_size = (int(width * scale), int(height * scale))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    return pil_img

def image_to_base64(img_path: str) -> str:
    """
    Convert an image file to a base64-encoded string suitable for API input.
    Resizes the image before encoding to control size.
    
    Args:
        img_path (str): Path to the image file.
    
    Returns:
        str: Base64-encoded image string with data URI prefix.
    """
    img = Image.open(img_path)
    img = resize_image(img)
    buffered = io.BytesIO()
    img_format = img.format or "PNG"
    img.save(buffered, format=img_format)
    b64_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{img_format.lower()};base64,{b64_str}"

@st.cache_data(show_spinner=False)
def embed_images(image_urls: dict) -> (list, np.ndarray):
    """
    Download images and compute embeddings using Cohere Embed-v4.0.
    Caches results to avoid redundant API calls.
    
    Args:
        image_urls (dict): Dictionary mapping image filenames to URLs.
    
    Returns:
        tuple: (list of image paths, numpy array of embeddings)
    """
    paths, embeddings = [], []
    for name, url in image_urls.items():
        path = download_image(url, name)
        paths.append(path)
        # Prepare input document with base64 image for embedding API
        input_doc = {"content": [{"type": "image", "image": image_to_base64(path)}]}
        resp = co.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            inputs=[input_doc],
        )
        emb = np.array(resp.embeddings.float[0])
        embeddings.append(emb)
    return paths, np.vstack(embeddings)

def embed_query(text: str) -> np.ndarray:
    """
    Compute the embedding vector for a given text query using Cohere Embed-v4.0.
    
    Args:
        text (str): Query text.
    
    Returns:
        np.ndarray: Embedding vector of the query.
    """
    resp = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[text],
    )
    return np.array(resp.embeddings.float[0])

def find_most_relevant_image(question: str, img_paths: list, img_embeds: np.ndarray) -> (str, float):
    """
    Find the most relevant image to a question by comparing embeddings.
    Uses cosine similarity between query embedding and image embeddings.
    
    Args:
        question (str): User question.
        img_paths (list): List of image file paths.
        img_embeds (np.ndarray): Array of image embeddings.
    
    Returns:
        str: Path to the most relevant image.
    """
    query_emb = embed_query(question)
    similarities = np.dot(query_emb, img_embeds.T) # Cosine similarity via dot product (assumes normalized vectors)
    best_idx = np.argmax(similarities)
    max_sim = similarities[best_idx]
    return img_paths[best_idx], max_sim

def generate_answer(question: str, img_path: str) -> str:
    """
    Generate an answer to the question based on the content of the image.
    Uses Google Gemini 2.5 Flash multimodal LLM.
    
    Args:
        question (str): User question.
        img_path (str): Path to the relevant image.
    
    Returns:
        str: Generated answer text.
    """
    prompt = [
        f"Answer the question based on the image. Exclude markdown and provide context.\nQuestion: {question}",
        Image.open(img_path)
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt
    )
    return response.text

# --- Main app UI ---

# Title of the app
st.header("🔍Visual Search with Cohere Embed V4 & Gemini 2.5")

# Text input for user question with a default value
question = st.text_input("Ask a question about the uploaded images:", value="What is the Gross profit for Nike?")

if question:
    with st.spinner("Processing your query..."):
        # Download images and compute embeddings (cached for speed)
        image_paths, image_embeddings = embed_images(Config.Images)

        # Find the image most relevant to the question
        matched_img_path, max_similarity  = find_most_relevant_image(question, image_paths, image_embeddings)

        # Define a threshold for "no relevant image found"
        SIMILARITY_THRESHOLD = 0.4  # Adjust this threshold based on your embedding scale
        
        if max_similarity < SIMILARITY_THRESHOLD:
            st.warning("⚠️ Sorry, I couldn't find any relevant image for your question.")
            st.stop()  # Stop further processing
        else:
            # Generate a detailed answer based on the matched image
            answer_text = generate_answer(question, matched_img_path)

            st.markdown("### Relevant Image")
            # Layout: show image on left, answer on right
            st.image(matched_img_path, caption="Most relevant image", use_container_width=True)

            st.markdown("### Answer")
            st.info(answer_text)