import streamlit as st
from pathlib import Path
import sys

# Add the app directory to the Python path
app_dir = str(Path(__file__).parent)
if app_dir not in sys.path:
    sys.path.append(app_dir)

from utils.data_loader import DataLoader
from utils.chatbot import MovieChatbot
from utils.recommender import MovieRecommender

# Page config
st.set_page_config(
    page_title="Movie Assistant",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'data_loader' not in st.session_state:
    with st.spinner('Loading movie data...'):
        st.session_state.data_loader = DataLoader()
        st.session_state.data_loader.load_metadata()

if 'chatbot' not in st.session_state:
    with st.spinner('Initializing chatbot...'):
        st.session_state.chatbot = MovieChatbot()
        st.session_state.chatbot.initialize(st.session_state.data_loader)

if 'recommender' not in st.session_state:
    with st.spinner('Setting up recommendations...'):
        st.session_state.recommender = MovieRecommender(st.session_state.data_loader)
        st.session_state.recommender.initialize()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chat", "Recommendations"])

# Main content
if page == "Home":
    st.title("Welcome to Movie Assistant")
    
    st.markdown("""
    ### Your AI-Powered Movie Companion
    
    Welcome to Movie Assistant, your intelligent guide to the world of cinema! This application combines 
    advanced AI technology with comprehensive movie data to provide you with an enhanced movie discovery 
    and information experience.
    
    #### Features:
    
    **🤖 Movie Chat**
    - Ask questions about any movie
    - Get detailed information about plots, cast, and production details
    - Discover interesting facts and trivia
    
    **🎯 Smart Recommendations**
    - Get personalized movie recommendations based on your preferences
    - Explore similar movies using both content-based and collaborative filtering
    - Find your next favorite film with AI-powered suggestions
    
    Use the navigation menu to explore these features and start your movie journey!
    """)
    
elif page == "Chat":
    st.title("Movie Chat")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about movies!"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot._generate_response(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

elif page == "Recommendations":
    st.title("Movie Recommendations")
    
    # Movie search
    search_query = st.text_input("Search for a movie:", "")
    
    if search_query:
        try:
            with st.spinner("Searching for movies..."):
                # Search for movies
                matching_movies = st.session_state.data_loader.movie_metadata[
                    st.session_state.data_loader.movie_metadata['title'].str.contains(search_query, case=False, na=False)
                ]
            
            st.write(f"Found {len(matching_movies)} matching movies")
            
            if len(matching_movies) > 0:
                # Create a list of movie options with titles and years
                movie_options = [f"{row['title']} ({row['release_year']})" for _, row in matching_movies.iterrows()]
                selected_movie = st.selectbox("Select a movie:", movie_options)
                
                if selected_movie:
                    # Extract the movie title from the selected option
                    movie_title = selected_movie.split(" (")[0]
                    
                    # Get recommendations
                    with st.spinner("Finding similar movies..."):
                        st_recommendations, svd_recommendations = st.session_state.recommender.get_recommendations(movie_title)
                    
                    # Display ST recommendations
                    st.subheader("Content-Based Recommendations (ST)")
                    if st_recommendations:
                        cols = st.columns(3)
                        for i, movie in enumerate(st_recommendations):
                            with cols[i % 3]:
                                st.markdown(f"### {movie['title']} ({movie['year']})")
                                st.markdown(f"**Genres:** {movie['genres']}")
                                st.markdown(f"**Similarity Score:** {movie['similarity']:.2f}")
                                st.markdown(f"**Director:** {movie['director']}")
                    else:
                        st.warning("No content-based recommendations found.")
                    
                    # Display SVD recommendations
                    st.subheader("Collaborative Filtering Recommendations (SVD)")
                    if svd_recommendations:
                        cols = st.columns(3)
                        for i, movie in enumerate(svd_recommendations):
                            with cols[i % 3]:
                                st.markdown(f"### {movie['title']} ({movie['year']})")
                                st.markdown(f"**Genres:** {movie['genres']}")
                                st.markdown(f"**Similarity Score:** {movie['similarity']:.2f}")
                                st.markdown(f"**Director:** {movie['director']}")
                    else:
                        st.warning("No collaborative filtering recommendations found.")
            else:
                st.warning("No movies found matching your search.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Debug information:")
            st.write(f"Data loader state: {st.session_state.data_loader is not None}")
            st.write(f"Movie metadata shape: {st.session_state.data_loader.movie_metadata.shape if st.session_state.data_loader.movie_metadata is not None else 'None'}") 