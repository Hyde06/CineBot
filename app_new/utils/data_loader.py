import pandas as pd
import numpy as np
from pathlib import Path
import json
import streamlit as st

class DataLoader:
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent
        self.data_dir = self.app_dir / "app_data"
        self.movie_metadata = None
        self.ratings_metadata = None
        self.st_embeddings = None
        self.svd_embeddings = None
        self.st_idx_to_imdb = {}
        self.svd_idx_to_movie_id = {}
        self.movie_mapping = None
        
    def load_metadata(self):
        """Load movie and ratings metadata"""
        try:
            # Load movie mapping first
            mapping_path = self.data_dir / "movie_mapping.csv"
            if mapping_path.exists():
                self.movie_mapping = pd.read_csv(mapping_path)
            else:
                st.error(f"Movie mapping file not found at {mapping_path}")
                return
                
            # Load movie metadata
            movie_metadata_path = self.data_dir / "movie_metadata.csv"
            if movie_metadata_path.exists():
                self.movie_metadata = pd.read_csv(movie_metadata_path)
            else:
                st.error(f"Movie metadata file not found at {movie_metadata_path}")
                
            # Load ratings metadata
            ratings_metadata_path = self.data_dir / "ratings_metadata.csv"
            if ratings_metadata_path.exists():
                self.ratings_metadata = pd.read_csv(ratings_metadata_path)
            else:
                st.error(f"Ratings metadata file not found at {ratings_metadata_path}")
                
            # Load ST embeddings
            st_embeddings_path = self.data_dir / "movie_embeddings.npy"
            if st_embeddings_path.exists():
                self.st_embeddings = np.load(str(st_embeddings_path))
                
                # Load ST index to IMDB mapping
                chunk_metadata_path = self.data_dir / "movie_chunk_metadata.json"
                if chunk_metadata_path.exists():
                    with open(chunk_metadata_path, 'r') as f:
                        chunk_metadata = json.load(f)
                        for idx, chunk in enumerate(chunk_metadata):
                            self.st_idx_to_imdb[str(idx)] = chunk['imdb_id']
            else:
                st.error(f"ST embeddings file not found at {st_embeddings_path}")
                
            # Load SVD embeddings
            svd_embeddings_path = self.data_dir / "svd_embeddings.npy"
            if svd_embeddings_path.exists():
                self.svd_embeddings = np.load(str(svd_embeddings_path))
                
                # Create SVD index to movie ID mapping
                if self.ratings_metadata is not None:
                    unique_movie_ids = self.ratings_metadata['movieId'].unique()
                    for idx, movie_id in enumerate(unique_movie_ids):
                        self.svd_idx_to_movie_id[idx] = movie_id
            else:
                st.error(f"SVD embeddings file not found at {svd_embeddings_path}")
                
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            
    def get_movie_by_id(self, imdb_id):
        """Get movie details by IMDB ID"""
        if self.movie_metadata is not None:
            movie = self.movie_metadata[self.movie_metadata['imdb_id'] == imdb_id]
            if not movie.empty:
                return movie.iloc[0].to_dict()
        return None
        
    def get_movie_id(self, imdb_id):
        """Get movieId from IMDB ID using movie_mapping.csv"""
        if self.movie_mapping is not None:
            mapping = self.movie_mapping[self.movie_mapping['imdb_id'] == imdb_id]
            if not mapping.empty:
                return mapping.iloc[0]['movieId']
        return None
        
    def get_imdb_id(self, movie_id):
        """Get IMDB ID from movieId using movie_mapping.csv"""
        if self.movie_mapping is not None:
            mapping = self.movie_mapping[self.movie_mapping['movieId'] == movie_id]
            if not mapping.empty:
                return mapping.iloc[0]['imdb_id']
        return None
        
    def get_st_movie_idx(self, imdb_id):
        """Get ST index for a movie by IMDB ID"""
        for idx, stored_imdb_id in self.st_idx_to_imdb.items():
            if stored_imdb_id == imdb_id:
                return int(idx)
        return None
        
    def get_svd_movie_idx(self, movie_id):
        """Get SVD index for a movie by movieId"""
        for idx, stored_movie_id in self.svd_idx_to_movie_id.items():
            if stored_movie_id == movie_id:
                return idx
        return None 