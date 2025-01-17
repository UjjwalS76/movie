import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set page config
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Check for API key in secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY is missing from secrets!")
    st.stop()

# Initialize API settings
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    OPENAI_API_BASE = "https://api.perplexity.ai"
    OPENAI_MODEL = "llama-3.1-sonar-small-128k-online"
    
    # Display configuration (remove in production)
    st.sidebar.write("Configuration:")
    st.sidebar.write(f"API Base: {OPENAI_API_BASE}")
    st.sidebar.write(f"Model: {OPENAI_MODEL}")
except Exception as e:
    st.error(f"Error setting environment variables: {str(e)}")
    st.stop()

# Movie data
movies = [
    {
        "title": "Krrish",
        "plot": "A poor but big-hearted man takes orphans into his home. After discovering his scientist father's invisibility device, he rises to the occasion and fights to save his children and all of India from the clutches of a greedy gangster",
        "year": 2006,
        "director": "Rakesh Roshan",
        "rating": 7.1,
        "genre": "science fiction"
    },
    {
        "title": "Rang De Basanti",
        "plot": "The story of six young Indians who assist an English woman to film a documentary on the freedom fighters from their past, and the events that lead them to relive the long-forgotten saga of freedom",
        "year": 2006,
        "director": "Rakeysh Omprakash Mehra",
        "rating": 9.1,
        "genre": "drama"
    },
    {
        "title": "Life in a Metro",
        "plot": "A depressed wealthy businessman finds his life changing after he meets a spunky and care-free young woman",
        "year": 2007,
        "director": "Anurag Basu",
        "rating": 6.8,
        "genre": "romance"
    },
    {
        "title": "Ghoomer",
        "plot": "A schoolteacher's world turns upside down when he realizes that his former student, who is now a world-famous artist, may have plagiarized his work",
        "year": 2023,
        "director": "R. Balki",
        "rating": 7.8,
        "genre": "drama"
    },
    {
        "title": "DDLJ",
        "plot": "A man returns to his country in order to marry his childhood sweetheart and proceeds to create misunderstanding between the families",
        "year": 1995,
        "director": "Aditya Chopra",
        "rating": 8.1,
        "genre": "romance"
    },
    {
        "title": "LOC Kargil",
        "plot": "The story of an Indian army officer guarding a picket alone in the Kargil conflict between India and Pakistan",
        "year": 2003,
        "director": "J.P. Dutta",
        "rating": 7.9,
        "genre": "war"
    },
    {
        "title": "Sholay",
        "plot": "Three young men from different parts of India arrive in Mumbai, seeking fame and fortune",
        "year": 1975,
        "director": "Ramesh Sippy",
        "rating": 8.2,
        "genre": "action"
    },
    {
        "title": "Maine Pyar Kiya",
        "plot": "A simple man from a village falls in love with his new neighbor. He enlists the help of his musical-theater friends to woo the lovely girl-next-door away from her music teacher",
        "year": 1990,
        "director": "Sooraj Barjatya",
        "rating": 7.7,
        "genre": "musical"
    },
    {
        "title": "Bajrangi Bhaijaan",
        "plot": "A young mute girl from Pakistan loses herself in India with no way to head back. A devoted man undertakes the task to get her back to her homeland and unite her with her family",
        "year": 2015,
        "director": "Kabir Khan",
        "rating": 8.0,
        "genre": "drama"
    },
    {
        "title": "3 Idiots",
        "plot": "Three idiots embark on a quest for a lost buddy. This journey takes them on a hilarious and meaningful adventure through memory lane and gives them a chance to relive their college days",
        "year": 2009,
        "director": "Rajkumar Hirani",
        "rating": 9.4,
        "genre": "comedy"
    }
]

# Initialize LLM
@st.cache_resource
def initialize_llm():
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=OPENAI_MODEL,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            openai_api_base=OPENAI_API_BASE
        )
        st.sidebar.success("✓ LLM initialized")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

# Create recommendation chain
@st.cache_resource
def create_recommendation_chain():
    llm = initialize_llm()
    
    # Template for movie recommendations
    template = """Based on the following user query, recommend relevant movies from the provided list.
    Consider the following criteria:
    - Movie plots
    - Genres
    - Directors
    - Ratings
    - Release years

    User Query: {query}

    Available Movies (as Python list): {movies}

    Return the titles of relevant movies as a comma-separated list.
    If no movies match the criteria, return "No matches found."

    Relevant Movies:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"query": RunnablePassthrough(), "movies": lambda _: str(movies)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("""
This app helps you find movies based on your preferences. You can:
- Search by rating (e.g., "movies rated above 8")
- Search by genre (e.g., "drama movies")
- Search by director (e.g., "movies by Rajkumar Hirani")
- Search by plot elements (e.g., "movies about college life")
- Combine multiple criteria!
""")

# Initialize chain
chain = create_recommendation_chain()

# User input
user_query = st.text_input("What kind of movie are you looking for?", 
                          placeholder="E.g., 'I want to watch a drama movie rated above 8'")

if user_query:
    with st.spinner("Searching for movies..."):
        try:
            results = chain.invoke(user_query)
            
            if results and results != "No matches found.":
                st.subheader("Here are your movie recommendations:")
                movie_titles = [title.strip() for title in results.split(",")]
                
                for title in movie_titles:
                    # Find the movie in our data
                    movie = next((m for m in movies if m["title"].lower() == title.lower()), None)
                    if movie:
                        with st.expander(f"📽️ {movie['title']} ({movie['year']})"):
                            st.markdown(f"""
                            **Plot**: {movie['plot']}
                            
                            **Director**: {movie['director']}  
                            **Genre**: {movie['genre']}  
                            **Rating**: ⭐ {movie['rating']}/10
                            """)
            else:
                st.warning("No movies found matching your criteria. Try adjusting your search!")
                
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")
            st.info("Please try rephrasing your query or use simpler search criteria.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain and Perplexity AI")
