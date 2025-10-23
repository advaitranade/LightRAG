import os
import asyncio
import google.generativeai as genai
from lightrag.core.types import ModelClient, ModelOutput, LLMCompletionInput
from lightrag.core.lightrag import LightRAG
from lightrag.core.embedder import EmbeddingFunc
from lightrag.core.types import QueryParam
from lightrag.components.retriever import LLMRetriever
from lightrag.components.generator import LLMGenerator
from lightrag.utils.logger import TrivialLogger

# --- This is the new part for Gemini ---

# 1. Configure the Gemini API client
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in Render.")

genai.configure(api_key=api_key)

class GeminiClient(ModelClient):
    """A custom ModelClient for the Gemini API."""
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__()
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def call(self, input: LLMCompletionInput, **kwargs) -> ModelOutput:
        """
        Calls the Gemini API.
        The 'input' object contains the prompt string.
        """
        try:
            # input.prompt is the final prompt string
            response = self.model.generate_content(input.prompt, **kwargs)
            return ModelOutput(output=response.text)
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ModelOutput(output=None, error=str(e))

    async def acall(self, input: LLMCompletionInput, **kwargs) -> ModelOutput:
        """
        Async version of the call.
        """
        try:
            response = await self.model.generate_content_async(input.prompt, **kwargs)
            return ModelOutput(output=response.text)
        except Exception as e:
            print(f"Error calling Gemini (async): {e}")
            return ModelOutput(output=None, error=str(e))

# 2. Define the LLM function LightRAG will use
# This function matches the signature LightRAG expects
def gemini_llm_func(model_name: str = "gemini-1.5-flash", **kwargs) -> GeminiClient:
    return GeminiClient(model_name=model_name, **kwargs)

# --- End of new Gemini part ---

# --- Standard LightRAG setup (from other examples) ---
# NOTE: This setup assumes you also have an embedding model.
# You may need to replace this with your actual embedding function.
# For simplicity, this example uses a placeholder.
# If you use an OpenAI embedder, configure its API key in Render as well.

def get_mock_embedding_func(dim: int = 768):
    def embed_func(texts: list[str]) -> list[list[float]]:
        # A mock embedding function.
        # REPLACE THIS with your actual embedding model (e.g., from OpenAI or HuggingFace)
        print(f"Mock embedding {len(texts)} texts...")
        import numpy as np
        return [np.random.rand(dim).tolist() for _ in texts]

    return EmbeddingFunc(embedding_dim=dim, func=embed_func)

async def initialize_rag():
    """Initializes the LightRAG instance with the Gemini model."""
    print("Initializing RAG with Gemini...")
    
    # 3. Pass the new function here:
    rag = LightRAG(
        llm_model_func=gemini_llm_func,  # <--- THIS IS THE KEY CHANGE
        embedding_func=get_mock_embedding_func(), # Replace with your embedder
        working_dir="lightrag_gemini_data",
        logger=TrivialLogger(),
    )
    
    await rag.initialize_storages()
    # You might need this from other examples
    # from lightrag.core.pipeline import initialize_pipeline_status
    # await initialize_pipeline_status()
    return rag

async def main():
    rag = await initialize_rag()

    # Insert some data
    documents = [
        "The capital of France is Paris.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "Mars is the fourth planet from the Sun."
    ]
    await rag.ainsert(documents)
    print("Documents inserted.")

    # Query using Gemini
    query = "What is the capital of France?"
    print(f"\nQuerying: {query}")
    
    param = QueryParam(mode="naive", user_prompt="Answer the question based only on the provided context.")
    response = await rag.aquery(query, param=param)
    
    print("\nGemini Response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
