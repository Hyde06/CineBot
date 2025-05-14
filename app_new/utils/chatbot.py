import os
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class MovieChatbot:
    def __init__(self):
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.client = InferenceClient(
            token=self.hf_api_key
        )
        self.data_loader = None
        self.embeddings = None
        self.model = None
        self.chunk_metadata = []
        
    def initialize(self, data_loader):
        """Initialize the chatbot with data and models"""
        self.data_loader = data_loader
        # Initialize the sentence transformer model
        print("Initializing Sentence Transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load embeddings and metadata"""
        try:
            # Get the directory where the current file is located
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            app_dir = current_dir.parent
            data_path = app_dir / "app_data"
            embeddings_path = data_path / "movie_embeddings.npy"
            metadata_path = data_path / "movie_chunk_metadata.json"
            
            print(f"Loading embeddings from: {embeddings_path}")
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
                
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            
            print(f"Loading chunk metadata from: {metadata_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
                
            with open(metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            print(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
            
        except Exception as e:
            print(f"Error in _load_embeddings: {str(e)}")
            raise
            
    def _search_movies(self, query, top_k=5):
        """Search for relevant movies using semantic similarity"""
        try:
            # Encode the query
            query_embedding = self.model.encode(query)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Get top k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Get movie details from metadata
            results = []
            for idx in top_indices:
                movie_info = self.chunk_metadata[idx]
                results.append({
                    'title': movie_info['title'],
                    'year': movie_info['year'],
                    'imdb_id': movie_info['imdb_id'],
                    'rating': movie_info['rating'],
                    'genres': movie_info['genres'],
                    'similarity_score': float(similarities[idx])
                })
            
            print("\n=== RAG Search Results ===")
            for movie in results:
                print(f"Found: {movie['title']} ({movie['year']}) - Score: {movie['similarity_score']:.3f}")
            print("========================\n")
            
            return results
        except Exception as e:
            print(f"Error in _search_movies: {str(e)}")
            return []
        
    def _query_llm(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Query the Hugging Face API"""
        if not self.hf_api_key:
            print("Error: HUGGINGFACE_API_KEY not found in environment variables")
            return None
            
        try:
            print(f"Debug - Sending request to Hugging Face API")
            
            # Format the prompt with context if available
            if context:
                context_str = ""
                for movie in context:
                    context_str += f"Title: {movie['title']} ({movie['year']})\n"
                    context_str += f"Genres: {movie['genres']}\n"
                    context_str += f"Rating: {movie['rating']:.1f}/10\n\n"
                
                formatted_prompt = f"<|user|>\nYou are a helpful movie information assistant. Use the following movie information to answer the question:\n\n{context_str}\n\nQuestion: {prompt}\n<|assistant|>\n"
                print("\n=== RAG Context Being Used ===")
                print(context_str)
                print("=============================\n")
            else:
                formatted_prompt = f"<|user|>\nYou are a helpful movie information assistant. Answer the following question about movies: {prompt}\n<|assistant|>\n"
                print("\n=== No RAG Context Available ===")
                print("Using direct question answering mode")
                print("===============================\n")
            
            # Generate response
            response = self.client.text_generation(
                prompt=formatted_prompt,
                model="microsoft/Phi-3.5-mini-instruct",
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            print(f"Debug - Response received")
            return response.strip()
            
        except Exception as e:
            print(f"Error querying Hugging Face API: {str(e)}")
            return None
            
    def _generate_response(self, user_input):
        """Generate a response using RAG with the LLM"""
        try:
            # Search for relevant movies
            search_results = self._search_movies(user_input)
            
            if not search_results:
                return "I couldn't find any movies matching your query. Could you try rephrasing your question?"
            
            # Get response from LLM with context
            response = self._query_llm(user_input, search_results)
            
            if response is None:
                # Fallback to template-based response if LLM fails
                response = "I found some information about your query:\n\n"
                for movie in search_results:
                    response += f"• {movie['title']} ({movie['year']})\n"
                    response += f"  Genres: {movie['genres']}\n"
                    response += f"  Rating: {movie['rating']:.1f}/10\n\n"
                response += "Is there anything specific about these movies you'd like to know?"
            
            return response
            
        except Exception as e:
            print(f"Error in _generate_response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again." 