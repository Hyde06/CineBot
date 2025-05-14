# Movie Assistant

An AI-powered movie chatbot and recommendation system that combines natural language processing with collaborative filtering to provide movie information and personalized recommendations.

## Features

- 🤖 **Movie Chat**: Ask questions about any movie and get detailed information
- 🎯 **Smart Recommendations**: Get personalized movie recommendations using both content-based and collaborative filtering approaches
- 📊 **Comprehensive Data**: Access information about thousands of movies including plots, cast, ratings, and more

## Prerequisites

- Python 3.8 or higher
- Hugging Face API key (get it from [Hugging Face](https://huggingface.co/settings/tokens))
- Jupyter Notebook (for data preparation)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-assistant.git
cd movie-assistant
```

2. Install dependencies:
```bash
pip install -r app_new/requirements.txt
```

3. Set up your Hugging Face API key:
   - Create a `.env` file in the `app_new` directory
   - Add your API key:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here
   ```

4. Prepare the required data files:
   - Create the `app_data` directory in `app_new` if it doesn't exist:
   ```bash
   mkdir -p app_new/app_data
   ```

   - Run the Jupyter notebooks in the `notebooks` directory in the following order:
     1. `1_data_preparation.ipynb`: Creates `movie_metadata.csv` and `ratings_metadata.csv`
     2. `2_create_chunks.ipynb`: Generates `movie_chunk_metadata.json`
     3. `3_generate_embeddings.ipynb`: Creates `st_embeddings.npy` and `svd_embeddings.npy`

   - After running the notebooks, ensure the following files are in `app_new/app_data/`:
     - `movie_metadata.csv`: Contains movie information (title, year, genres, etc.)
     - `ratings_metadata.csv`: Contains user ratings data
     - `movie_chunk_metadata.json`: Contains processed movie chunks for RAG
     - `st_embeddings.npy`: Sentence Transformer embeddings for content-based recommendations
     - `svd_embeddings.npy`: SVD embeddings for collaborative filtering

   Note: The notebooks will automatically save the generated files to the correct location.

## Running the Application

1. Navigate to the app directory:
```bash
cd app_new
```

2. Run the Streamlit app:
```bash
streamlit run Home.py
```

3. Open your browser and go to `http://localhost:8501`

## Usage

1. **Chat Interface**:
   - Ask questions about movies
   - Get information about plots, cast, ratings, etc.
   - Example questions:
     - "Tell me about Inception"
     - "Who directed The Dark Knight?"
     - "What are the main themes in Interstellar?"

2. **Recommendations**:
   - Search for a movie
   - Get both content-based and collaborative filtering recommendations
   - View detailed information about recommended movies

## Data Preparation Details

The application requires several data files that are generated using Jupyter notebooks. Here's what each notebook does:

1. **1_data_preparation.ipynb**:
   - Processes raw movie data
   - Creates movie metadata with titles, years, genres, etc.
   - Generates ratings metadata for collaborative filtering

2. **2_create_chunks.ipynb**:
   - Processes movie descriptions and metadata
   - Creates chunks for the RAG system
   - Generates movie_chunk_metadata.json

3. **3_generate_embeddings.ipynb**:
   - Creates Sentence Transformer embeddings for content-based recommendations
   - Generates SVD embeddings for collaborative filtering
   - Saves embeddings in the correct format

## Note

This application requires a Hugging Face API key to function. The API key is used to access the language model for generating responses. Make sure to keep your API key secure and never commit it to version control.

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 