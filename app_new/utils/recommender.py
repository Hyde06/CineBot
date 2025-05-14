import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import streamlit as st
from pathlib import Path
import os
import faiss

class MovieRecommender:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.st_index = None
        
    def initialize(self):
        """Initialize the recommender system"""
        if self.st_index is None:
            self._initialize_st_index()
            
    def _initialize_st_index(self):
        """Initialize FAISS index for ST embeddings"""
        if self.data_loader.st_embeddings is not None:
            embedding_dim = self.data_loader.st_embeddings.shape[1]
            # Create index for cosine similarity
            self.st_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
            # Add normalized embeddings to the index
            self.st_index.add(self.data_loader.st_embeddings.astype(np.float32))
            
    def _get_st_recommendations(self, st_idx, n_recommendations=5):
        """Get recommendations using sentence transformer embeddings with FAISS"""
        if st_idx is None:
            return []
        
        try:
            # Get the embedding for the movie and reshape for FAISS
            query_vector = self.data_loader.st_embeddings[st_idx].reshape(1, -1).astype(np.float32)
            
            # Search using FAISS
            similarities, indices = self.st_index.search(query_vector, n_recommendations + 1)
            
            # Filter out the query movie and get recommendations
            recommendations = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx != st_idx:  # Skip the query movie
                    # Convert index to IMDB ID
                    imdb_id = self.data_loader.st_idx_to_imdb.get(str(idx))
                    if imdb_id:
                        # Verify the movie exists in metadata
                        movie = self.data_loader.get_movie_by_id(imdb_id)
                        if movie is not None:
                            # Convert similarity to a proper score between 0 and 1
                            score = float(similarity)
                            recommendations.append({
                                'imdb_id': imdb_id,
                                'similarity': score
                            })
            
            return recommendations[:n_recommendations]
        except Exception as e:
            st.error(f"Error in ST recommendations: {str(e)}")
            return []
        
    def _get_svd_recommendations(self, svd_idx, n_recommendations=5):
        """Get recommendations using SVD embeddings"""
        if self.data_loader.svd_embeddings is None or svd_idx is None:
            return []
            
        try:
            # Check if the index is within bounds
            if svd_idx >= len(self.data_loader.svd_embeddings):
                return []
                
            # Get the movie embedding and ensure it's a 2D array
            movie_embedding = np.asarray(self.data_loader.svd_embeddings[svd_idx]).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(movie_embedding, self.data_loader.svd_embeddings)[0]
            
            # Get top N recommendations (excluding the input movie)
            top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in top_indices:
                try:
                    # Get movieId using integer index
                    movie_id = self.data_loader.svd_idx_to_movie_id[idx]
                    imdb_id = self.data_loader.get_imdb_id(movie_id)
                    if imdb_id:
                        movie = self.data_loader.get_movie_by_id(imdb_id)
                        if movie is not None:
                            score = float(similarities[idx])
                            recommendations.append({
                                'imdb_id': imdb_id,
                                'similarity': score
                            })
                except (KeyError, ValueError) as e:
                    continue
            
            return recommendations[:n_recommendations]
        except Exception as e:
            st.error(f"Error in SVD recommendations: {str(e)}")
            return []
        
    def get_recommendations(self, movie_title: str, n_recommendations: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get recommendations from both ST and SVD models"""
        try:
            # Find the movie
            movie = self.data_loader.movie_metadata[
                self.data_loader.movie_metadata['title'].str.lower() == movie_title.lower()
            ]
            
            if movie.empty:
                st.warning(f"No movie found with title: {movie_title}")
                return [], []
                
            imdb_id = movie.iloc[0]['imdb_id']
            
            # Get ST recommendations
            st_idx = self.data_loader.get_st_movie_idx(imdb_id)
            st_recommendations = []
            if st_idx is not None:
                st_recs = self._get_st_recommendations(st_idx, n_recommendations)
                for rec in st_recs:
                    movie = self.data_loader.get_movie_by_id(rec['imdb_id'])
                    if movie is not None:
                        st_recommendations.append({
                            'title': movie['title'],
                            'year': movie['release_year'],
                            'genres': movie['genres'],
                            'plot': movie['plot_summary'],
                            'cast': movie['cast'],
                            'director': movie['directors'],
                            'similarity': rec['similarity']
                        })
            
            # Get SVD recommendations
            movie_id = self.data_loader.get_movie_id(imdb_id)
            svd_recommendations = []
            if movie_id is not None:
                svd_idx = self.data_loader.get_svd_movie_idx(movie_id)
                if svd_idx is not None:
                    svd_recs = self._get_svd_recommendations(svd_idx, n_recommendations)
                    for rec in svd_recs:
                        movie = self.data_loader.get_movie_by_id(rec['imdb_id'])
                        if movie is not None:
                            svd_recommendations.append({
                                'title': movie['title'],
                                'year': movie['release_year'],
                                'genres': movie['genres'],
                                'plot': movie['plot_summary'],
                                'cast': movie['cast'],
                                'director': movie['directors'],
                                'similarity': rec['similarity']
                            })
            
            return st_recommendations, svd_recommendations
            
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return [], [] 