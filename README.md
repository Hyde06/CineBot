# Movie Assistant

An AI-powered movie chatbot and recommendation system that combines natural language processing with collaborative filtering to provide movie information and personalized recommendations.

## Features

- 🤖 **Movie Chat**: Ask questions about any movie and get detailed information
- 🎯 **Smart Recommendations**: Get personalized movie recommendations using both content-based and collaborative filtering approaches
- 📊 **Comprehensive Data**: Access information about thousands of movies including plots, cast, ratings, and more

## Prerequisites

- Python 3.8 or higher
- Hugging Face API key (get it from [Hugging Face](https://huggingface.co/settings/tokens))

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

4. Download the required data files:
   - Place the following files in the `app_new/app_data` directory:
     - `movie_metadata.csv`
     - `ratings_metadata.csv`
     - `movie_chunk_metadata.json`
     - `st_embeddings.npy`
     - `svd_embeddings.npy`

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

## Note

This application requires a Hugging Face API key to function. The API key is used to access the language model for generating responses. Make sure to keep your API key secure and never commit it to version control.

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 