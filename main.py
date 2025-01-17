import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Set page config
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Check and display secrets status
required_secrets = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL"]
missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]

if missing_secrets:
    st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
    st.stop()

# Initialize API settings
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    
    # Display configuration (remove in production)
    st.sidebar.write("Configuration:")
    st.sidebar.write(f"API Base: {st.secrets['OPENAI_API_BASE']}")
    st.sidebar.write(f"Model: {st.secrets['OPENAI_MODEL']}")
except Exception as e:
    st.error(f"Error setting environment variables: {str(e)}")
    st.stop()

# Movie data
docs = [
    Document(
        page_content="A poor but big-hearted man takes orphans into his home. After discovering his scientist father's invisibility device, he rises to the occasion and fights to save his children and all of India from the clutches of a greedy gangster",
        metadata={"year": 2006, "director": "Rakesh Roshan", "rating": 7.1, "genre": "science fiction"},
    ),
    Document(
        page_content="The story of six young Indians who assist an English woman to film a documentary on the freedom fighters from their past, and the events that lead them to relive the long-forgotten saga of freedom",
        metadata={"year": 2006, "director": "Rakeysh Omprakash Mehra", "rating": 9.1, "genre": "drama"},
    ),
    Document(
        page_content="A depressed wealthy businessman finds his life changing after he meets a spunky and care-free young woman",
        metadata={"year": 2007, "director": "Anurag Basu", "rating": 6.8, "genre": "romance"},
    ),
    Document(
        page_content="A schoolteacher's world turns upside down when he realizes that his former student, who is now a world-famous artist, may have plagiarized his work",
        metadata={"year": 2023, "director": "R. Balki", "rating": 7.8, "genre": "drama"},
    ),
    Document(
        page_content="A man returns to his country in order to marry his childhood sweetheart and proceeds to create misunderstanding between the families",
        metadata={"year": 1995, "director": "Aditya Chopra", "rating": 8.1, "genre": "romance"},
    ),
    Document(
        page_content="The story of an Indian army officer guarding a picket alone in the Kargil conflict between India and Pakistan",
        metadata={"year": 2003, "director": "J.P. Dutta", "rating": 7.9, "genre": "war"},
    ),
    Document(
        page_content="Three young men from different parts of India arrive in Mumbai, seeking fame and fortune",
        metadata={"year": 1975, "director": "Ramesh Sippy", "rating": 8.2, "genre": "action"},
    ),
    Document(
        page_content="A simple man from a village falls in love with his new neighbor. He enlists the help of his musical-theater friends to woo the lovely girl-next-door away from her music teacher",
        metadata={"year": 1990, "director": "Sooraj Barjatya", "rating": 7.7, "genre": "musical"},
    ),
    Document(
        page_content="A young mute girl from Pakistan loses herself in India with no way to head back. A devoted man undertakes the task to get her back to her homeland and unite her with her family",
        metadata={"year": 2015, "director": "Kabir Khan", "rating": 8.0, "genre": "drama"},
    ),
    Document(
        page_content="Three idiots embark on a quest for a lost buddy. This journey takes them on a hilarious and meaningful adventure through memory lane and gives them a chance to relive their college days",
        metadata={"year": 2009, "director": "Rajkumar Hirani", "rating": 9.4, "genre": "comedy"},
    ),
]

# Initialize components
@st.cache_resource
def initialize_retriever():
    try:
        embeddings = OpenAIEmbeddings()
        st.sidebar.success("✓ Embeddings initialized")
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        st.stop()
        
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.sidebar.success("✓ Vector store initialized")
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        st.stop()
    
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie.",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating",
            description="A 1-10 rating for the movie",
            type="float"
        ),
    ]
    
    document_content_description = "Brief summary of a movie"
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=st.secrets["OPENAI_MODEL"],
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            openai_api_base=st.secrets["OPENAI_API_BASE"]
        )
        st.sidebar.success("✓ LLM initialized")
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()
    
    try:
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vectorstore,
            document_content_description,
            metadata_field_info,
        )
        st.sidebar.success("✓ Retriever initialized")
        return retriever
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        st.stop()

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

# Initialize retriever
retriever = initialize_retriever()

# User input
user_query = st.text_input("What kind of movie are you looking for?", 
                          placeholder="E.g., 'I want to watch a drama movie rated above 8'")

if user_query:
    with st.spinner("Searching for movies..."):
        try:
            results = retriever.invoke(user_query)
            
            if results:
                st.subheader("Here are your movie recommendations:")
                for doc in results:
                    with st.expander(f"📽️ {doc.metadata.get('director', 'Unknown Director')} - {doc.metadata.get('year', 'Unknown Year')}"):
                        st.markdown(f"""
                        **Plot**: {doc.page_content}
                        
                        **Genre**: {doc.metadata.get('genre', 'N/A')}  
                        **Rating**: ⭐ {doc.metadata.get('rating', 'N/A')}/10
                        """)
            else:
                st.warning("No movies found matching your criteria. Try adjusting your search!")
                
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")
            st.info("Please try rephrasing your query or use simpler search criteria.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain and Perplexity AI")
