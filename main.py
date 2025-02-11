import streamlit as st
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import YouTubeSearchTool
from youtubesearchpython import VideosSearch

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key is missing! Please add it to the .env file.")
    st.stop()

# Define global variables
TEMP = 0.65  # Default temperature for OpenAI LLM

# Ensure ChromaDB directory exists
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# Initialize Embedding Function
embedding_function = OpenAIEmbeddings()

# Create SQLite Database
def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT
            )
        """)

def read_research_table():
    with sqlite3.connect("MASTER.db") as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM Research", conn)
            return df
        except Exception:
            return pd.DataFrame(columns=["research_id", "user_input", "introduction", "quant_facts", "publications", "books", "ytlinks"])

def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks):
    with sqlite3.connect("MASTER.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks))
        conn.commit()

# Function to generate research content
def generate_research(userInput):
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=TEMP)
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()

    tools = [
        Tool(name="Wikipedia Research Tool", func=wiki.run, description="Research Wikipedia articles."),
        Tool(name="DuckDuckGo Search Tool", func=DDGsearch.run, description="Search the internet for information."),
        Tool(name="YouTube Search Tool", func=YTsearch.run, description="Find relevant YouTube videos.")
    ]

    # Add previous research retrieval tool if available
    if st.session_state.embeddings_db:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.embeddings_db.as_retriever())
        tools.append(Tool(name="Previous Research Tool", func=qa.run, description="Retrieve past research."))

    memory = ConversationBufferMemory(memory_key="chat_history")

    runAgent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

    st.subheader("User Input:")
    st.write(userInput)

    st.subheader("Introduction:")
    with st.spinner("Generating Introduction..."):
        intro = runAgent(f'Write an academic introduction about {userInput}')
        st.write(intro['output'])

    st.subheader("Quantitative Facts:")
    with st.spinner("Fetching Statistical Facts..."):
        quantFacts = runAgent(f'Generate 3-5 quantitative facts about {userInput}')
        st.write(quantFacts['output'])

    st.subheader("Recent Publications:")
    with st.spinner("Finding Research Papers..."):
        papers = runAgent(f'Find 2-3 recent academic papers on {userInput}')
        st.write(papers['output'])

    st.subheader("Recommended Books:")
    with st.spinner("Finding Books..."):
        readings = runAgent(f'List 5 books about {userInput}')
        st.write(readings['output'])

    st.subheader("YouTube Links:")
    with st.spinner("Finding YouTube Videos..."):
        search = VideosSearch(userInput, limit=5)
        ytlinks = "\n".join([f"{i+1}. {video['title']} - https://www.youtube.com/watch?v={video['id']}" for i, video in enumerate(search.result()['result'])])
        st.write(ytlinks)

    # Save research data
    insert_research(userInput, intro['output'], quantFacts['output'], papers['output'], readings['output'], ytlinks)

# Define Streamlit app
def main():
    st.set_page_config(page_title="AI Research Assistant")

    create_research_db()
    st.session_state.setdefault("embeddings_db", None)

    # Initialize vector database
    try:
        st.session_state.embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")

    st.title("GPT-4 Research Assistant")
    st.caption("Powered by OpenAI, LangChain, ChromaDB, SQLite, YouTube, Wikipedia, DuckDuckGo")

    with st.sidebar:
        st.subheader("LLM Settings")
        global TEMP
        TEMP = st.slider("Temperature", 0.0, 1.0, 0.65)

    deploy_tab, prev_tab = st.tabs(["Generate Research", "Previous Research"])

    with deploy_tab:
        userInput = st.text_area("Enter Topic:")
        if st.button("Generate Report") and userInput:
            generate_research(userInput)

    with prev_tab:
        research_df = read_research_table()
        st.dataframe(research_df)
        selected_input = st.selectbox("Select Previous Research", research_df["user_input"].unique() if not research_df.empty else [])

        if st.button("Show Research") and selected_input:
            selected_df = research_df[research_df["user_input"] == selected_input].iloc[0]
            st.write(f"**Introduction:** {selected_df['introduction']}")
            st.write(f"**Quantitative Facts:** {selected_df['quant_facts']}")
            st.write(f"**Publications:** {selected_df['publications']}")
            st.write(f"**Books:** {selected_df['books']}")
            st.write(f"**YouTube Links:** {selected_df['ytlinks']}")

if __name__ == '__main__':
    main()
