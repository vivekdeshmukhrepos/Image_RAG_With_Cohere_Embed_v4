
# 🧠 Visual Q&A with Cohere's Embed v4 & Google Gemini

## 📌 Problem Statement

In many enterprise contexts, important information is stored in complex images such as financial charts, dashboards, infographics, or visual reports. Extracting relevant insights from these visual documents in response to natural language questions is time-consuming and often manual.

This application solves the problem by:

- **Identifying** the most relevant image from a collection using **Cohere Embed-v4 multimodal embeddings** and  
- **Answering questions** about the image using **Google Gemini 2.5 Flash**, a powerful multimodal LLM.

## 🚀 Features

-   Multimodal **image embedding and search** using Cohere's Embed-v4 model.
-   Intelligent **question answering** based on visual context using Google Gemini.
-   Optimized image handling (resizing, caching, embedding).
-   Simple, elegant **Streamlit UI** with model insights and real-time feedback.
    
 
## 🧰 Tech Stack

-   [Streamlit](https://streamlit.io/) – UI and frontend
-   [Cohere Embed-v4](https://cohere.com/blog/embed-4) – Image and text embedding
-   [Google Gemini 2.5 Flash](https://ai.google.dev/) – Multimodal LLM
-   [PIL (Pillow)](https://pillow.readthedocs.io/) – Image processing
-   [NumPy](https://numpy.org/) – Vector math and similarity
-   [Requests](https://pypi.org/project/requests/) – Image downloading
    



## 📁 Project Structure
```
│   Config.py
│   app.py
│   README.md
│   requirements.txt
│
├───Images
│       alphabet.png
│       nike.png
```
## ⚙️ Setup Instructions

1. Clone the repository
    ``` https://github.com/vivekdeshmukhrepos/Image_RAG_With_Cohere_Embed_v4.git```

2. Initialize Cohere and Gemini API Keys in 
    ```Config.py```
3. Install dependencies
    ```pip install -r requirements.txt```

## 🧪 Usage
1. Start the app ```streamlit run .\app.py```
2. Ask a question related to your image like:

    ```"What is the gross profit for Nike?"```
3. The app will:
    - Download and embed the images (if not cached),
    - Find the most relevant image using cosine similarity,
    - Generate an answer using Gemini 2.5 Flash based on that image.

## 📌 Notes
1. Adjust ```SIMILARITY_THRESHOLD``` in ```app.py``` to make the image search more or less strict.
2. API calls to Cohere and Gemini may be rate-limited based on your plan.
3. Ideal for dashboards, infographics, and structured visual documents.
4. Embedded images are cached locally and resized dynamically to stay under memory-safe limits for API processing.