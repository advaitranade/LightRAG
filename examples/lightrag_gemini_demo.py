# pip install -q -U google-generativeai sentence-transformers
# Note: Use 'google-generativeai', not 'google-genai' for the modern client

import os
import numpy as np
import google.generativeai as genai  # --- FIX: Use the full library ---
from google.generativeai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or Render environment.")

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)

# --- FIX: Initialize models ONCE, globally ---
# 1. Configure and initialize the Gemini model globally
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 2. Initialize the embedding model globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# ---------------------------------------------


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    
    # --- FIX: Use the globally initialized model ---
    # No need to create a new client here

    # 2. Combine prompts (your logic is good)
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model using the ASYNCHRONOUS method
    try:
        # --- FIX: Use the async 'generate_content_async' ---
        response = await gemini_model.generate_content_async(
            contents=[combined_prompt],
            generation_config=types.GenerationConfig(max_output_tokens=500, temperature=0.1),
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return f"Error: {e}"


async def embedding_func(texts: list[str]) -> np.ndarray:
    
    # --- FIX: Use the globally initialized model ---
    # model = SentenceTransformer("all-MiniLM-L6-v2") # <-- Remove this
    
    # 'model.encode' is synchronous (blocking). We must run it in a separate
    # thread pool to avoid blocking the asyncio event loop.
    
    # --- FIX: Use asyncio.to_thread for blocking calls ---
    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(
        None,  # Use the default thread pool executor
        lambda: embedding_model.encode(texts, convert_to_numpy=True)
    )
    return embeddings


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,      # This model (all-MiniLM-L6-v2) has 384 dimensions
            max_token_size=8192,  # This is fine
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    print("Initializing RAG...")
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    
    # --- FIX: Create a dummy story.txt file for the demo ---
    file_path = "story.txt"
    try:
        with open(file_path, "w") as file:
            file.write("A brave knight named Sir Reginald lived in a castle. ")
            file.write("He had a loyal dragon named Sparky. ")
            file.write("Together, they defended the kingdom from a grumpy giant. ")
            file.write("The main theme is about friendship and courage.")
        
        with open(file_path, "r") as file:
            text = file.read()
        
        print("Inserting text...")
        rag.insert(text)
        print("Text inserted.")

        print("Querying...")
        response = rag.query(
            query="What is the main theme of the story?",
            param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
        )

        print("\n--- Response ---")
        print(response)
        print("------------------\n")
        
    finally:
        # Clean up the dummy file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    main()
