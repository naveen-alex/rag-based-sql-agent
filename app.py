import streamlit as st
from dotenv import load_dotenv
import os
from rag_logic import create_sql_agent

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="SQL Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize SQL Agent
@st.cache_resource
def initialize_sql_agent():
    """Initialize the SQL Agent with Qdrant configuration"""
    qdrant_config = {
        'url': os.getenv('QDRANT_URL'),
        'collection_name': os.getenv('QDRANT_COLLECTION')
    }
    
    groq_api_key = os.getenv('GROQ_API_KEY')  # Updated variable name
    
    if not groq_api_key or not qdrant_config['url'] or not qdrant_config['collection_name']:
        st.error("Please configure QDRANT_URL, QDRANT_COLLECTION, and GROQ_API_KEY in the .env file")  # Updated error message
        st.stop()
    
    try:
        return create_sql_agent(qdrant_config, groq_api_key)  # Updated parameter name
    except Exception as e:
        st.error(f"Failed to initialize SQL Agent: {str(e)}")
        st.stop()

# SQL Agent function
def sql_agent_response(user_query):
    """
    Process user query using RAG-based SQL agent
    """
    try:
        agent = initialize_sql_agent()
        result = agent.process_query(user_query)
        
        if result['success']:
            response = f"""
**Query Understanding:**
I found relevant tables: {', '.join(result['relevant_tables'])}

**Generated SQL:**
```sql
{result['sql_query']}
```

**Results:**
{result['results']}
            """
        else:
            response = f"❌ **Error:** {result['error']}"
        
        return response
    
    except Exception as e:
        return f"❌ **System Error:** {str(e)}"

# App title and description
st.title("🤖 SQL Agent Chat Interface")
st.markdown("Ask questions about your database and get SQL-powered insights!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = sql_agent_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*SQL Agent powered by Streamlit, PostgreSQL, and Gemini AI*")
# Footer
st.markdown("---")
st.markdown("*SQL Agent powered by Streamlit, PostgreSQL, and Gemini AI*")
