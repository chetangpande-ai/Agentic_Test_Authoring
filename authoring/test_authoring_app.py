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

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

loader=DirectoryLoader("./data",glob="./*.txt",loader_cls=TextLoader)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
chunk_size=500,
chunk_overlap=50
)

new_docs=text_splitter.split_documents(documents=docs)
doc_string=[doc.page_content for doc in new_docs]
db=Chroma.from_documents(new_docs,embeddings)

retriever=db.as_retriever(search_kwargs={"k": 3})
#retriever.invoke("create API Test for endpoint https://api.restful-api.dev/objects, its GET call. validate status code as 200")


prompt = hub.pull("rlm/rag-prompt")


custom_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Test Automation Engineer. You role is to create API Test scripts in Java language based on the reference methods mentioned in  ApiClient class  reusable methods. "
    "DO not create pom.xml and associated dependencies and the base APIClient library"
    "Do not create any verbose/text as output"
    "Create modular script as per the details given by user "),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | model
    | StrOutputParser()
)


result=rag_chain.invoke("Create API Test, for endpoint https://api.restful-api.dev/objects,  Operation is GET call. Validate status code as 200")
pprint.pprint(result)