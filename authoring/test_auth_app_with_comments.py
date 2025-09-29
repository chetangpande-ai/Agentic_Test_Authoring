from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
import pprint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# Initialize AI model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Load all text files from the ./data directory
loader = DirectoryLoader("./data", glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

# Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
new_docs = text_splitter.split_documents(documents=docs)

# Optional: create a list of document text for debugging/reference
doc_string = [doc.page_content for doc in new_docs]

# Create a vector database from document chunks
db = Chroma.from_documents(new_docs, embeddings)

# Create a retriever to fetch the top 3 relevant document chunks
retriever = db.as_retriever(search_kwargs={"k": 3})

# Pull a RAG prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Define a custom prompt for Test Automation Engineers
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Test Automation Developer. "
     "You role is to create API Test scripts in Java language based on the reference methods mentioned in ApiClient class reusable methods. "
     "Do not create pom.xml and associated dependencies or the base APIClient library. "
     "Do not create any verbose/text output. "
     "Create modular scripts as per the details given by the user."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Helper function to format documents into a single string for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain
# Steps:
# 1. Retriever fetches relevant document chunks
# 2. format_docs combines them into a single string
# 3. custom_prompt formats the prompt for the AI
# 4. model generates the output
# 5. StrOutputParser converts AI output into a clean string
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | model
    | StrOutputParser()
)

# Run the chain with a specific user query
result = rag_chain.invoke(
    "Create API Test, for endpoint https://api.restful-api.dev/objects, "
    "Operation is GET call. Validate status code as 200"
)

# Print the generated Java API test script
pprint.pprint(result)
