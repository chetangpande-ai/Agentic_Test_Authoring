import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import pprint

# Load environment variables
load_dotenv()

# Initialize AI model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Load documents from ./data folder
loader = DirectoryLoader("./data", glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
new_docs = text_splitter.split_documents(documents=docs)

# Create a vector database from document chunks
db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Pull RAG prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Define a custom prompt for Test Automation Engineers
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Test Automation Engineer. "
     "Your role is to create API Test scripts in Java language based on the reference methods mentioned in ApiClient class reusable methods. "
     "Do not create pom.xml or associated dependencies. "
     "Do not create verbose/text output. "
     "Create modular scripts as per user instructions."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Helper function to format documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | model
    | StrOutputParser()
)

# ------------------- STREAMLIT USER INTERFACE -------------------

st.set_page_config(page_title="AI API Test Script Generator")
st.title("AI API Test Script Generator")

# Input text box for user query
user_query = st.text_area(
    "Enter your API test query:",
    "Create API Test, for endpoint https://api.restful-api.dev/objects, Operation is GET call. Validate status code as 200",
    height=150
)

# Button to generate the test script
if st.button("Generate Test Script"):
    if user_query.strip():
        with st.spinner("Generating script..."):
            result = rag_chain.invoke(user_query)
        # Display result
        st.subheader("Generated Java API Test Script")
        st.code(result, language="java")
    else:
        st.warning("Please enter a query to generate the test script.")
