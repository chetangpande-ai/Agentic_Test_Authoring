# API Test Script Generator using LangChain, Google Gemini, and HuggingFace Embeddings

This project demonstrates how to create a **modular API test script generator** using LangChain, Google Gemini AI, and HuggingFace embeddings. The generator reads text documents from a local directory, processes them into chunks, stores them in a vector database, and uses a RAG (Retrieval-Augmented Generation) pipeline to create Java-based API test scripts based on user prompts.


## Features

- Load and process multiple text documents from a directory.
- Split documents into manageable chunks for vector storage.
- Use HuggingFace embeddings to create semantic vector representations.
- Store and retrieve documents using **Chroma** vector database.
- Integrate Google Gemini AI for generating context-aware API test scripts.
- Custom prompt designed specifically for **Test Automation Engineers**.
- Outputs modular, reusable Java API test scripts for endpoints.


## Process
Load documents from the data directory.

Split documents into chunks of 500 characters (with 50-character overlap).

Generate embeddings using HuggingFace.

Store documents in Chroma vector database.

Retrieve relevant context based on your query.

Use Google Gemini AI with a custom prompt to generate Java API test scripts.

Print the generated script to the console.

Customization
Directory Loader: Change DirectoryLoader("./data", glob="./*.txt", loader_cls=TextLoader) to point to a different folder or file type.

Chunk Size: Adjust chunk_size and chunk_overlap in RecursiveCharacterTextSplitter for finer control.

Retriever: Modify search_kwargs={"k": 3} to retrieve more or fewer documents for context.

Prompt: Customize custom_prompt to adapt the generated scripts to your project standards.

## Example Output
Running the script with the query:

python
Copy code
"Create API Test, for endpoint https://api.restful-api.dev/objects, Operation is GET call. Validate status code as 200"
will return a Java API test script following your reusable methods in ApiClient class.

Notes
This script does not create pom.xml or any dependencies; it focuses purely on generating modular API test scripts.

Ensure your API keys are valid and have access to Google Gemini and HuggingFace models.

Designed for Test Automation Engineers who want to quickly generate API test scripts.