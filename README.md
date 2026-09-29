# ResearchFlow

ResearchFlow is an AI-powered research assistant that uses Retrieval-Augmented Generation (RAG) and agentic workflows to answer questions grounded in uploaded research documents.

Instead of relying only on an LLM's existing knowledge, ResearchFlow retrieves relevant information from local PDFs and provides that context to an AI agent, allowing it to generate responses grounded in the source material.

## Features

- Ingest and process research PDFs
- Split documents into retrieval-friendly chunks
- Generate semantic embeddings using Google Gemini
- Store and retrieve document embeddings with Chroma
- Search documents using semantic similarity
- Expose document retrieval as a LangChain tool
- Allow an AI agent to decide when document retrieval is necessary
- Maintain short-term conversational state using LangGraph memory
- Preserve source and page metadata for grounded responses

## Tech Stack

- Python
- LangChain
- LangGraph
- Google Gemini
- Google Generative AI Embeddings
- Chroma
- PyPDFLoader
- RecursiveCharacterTextSplitter

## Architecture

ResearchFlow follows a RAG-based pipeline:

```text
Research PDFs
      |
      v
Document Loader
      |
      v
Text Chunking
      |
      v
Gemini Embeddings
      |
      v
Chroma Vector Database
      |
      v
Semantic Retriever
      |
      v
rag_search Tool
      |
      v
LangChain / LangGraph Agent
      |
      v
Grounded Research Response
```

## Document Ingestion

Research documents are loaded from a directory using `DirectoryLoader` and `PyPDFLoader`.

The documents are then divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Example configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

Chunking allows the retrieval system to locate specific sections of a document rather than passing entire PDFs to the model.

## Embeddings and Vector Search

Document chunks are converted into vector embeddings using Google's embedding model through:

```python
GoogleGenerativeAIEmbeddings
```

The vectors are stored in a persistent Chroma vector database.

The vector store is exposed as a retriever configured to return the most relevant document chunks:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)
```

## RAG Search Tool

ResearchFlow exposes the retrieval system to the AI agent through a custom LangChain tool:

```python
@tool("rag_search")
```

The tool searches uploaded research documents and returns information including:

- Page content
- Source document
- Page number

The agent can invoke this tool when a question may be answered using the uploaded research material.

This allows retrieval to become part of the agent's reasoning workflow instead of being executed unconditionally for every question.

## Agent Memory

ResearchFlow uses LangGraph short-term memory with `InMemorySaver`.

This allows the agent to maintain conversational state within a thread and use previous messages as context during follow-up questions.

## Example

Example research question:

```text
What is agent misalignment?
```

ResearchFlow searches the uploaded documents for relevant passages and uses the retrieved context to construct a grounded response.

An example response may reference information such as:

```text
(ai report.pdf, p. 0)
(ai report.pdf, p. 1)
```

This makes it easier to trace generated research answers back to the underlying documents.

## Project Structure

A typical project structure looks like:

```text
researchflow/
│
├── main.py
│
├── rag/
│   ├── loader.py
│   ├── vectorstore.py
│   └── ...
│
├── tools/
│   ├── rag_tool.py
│   └── ...
│
├── documents/
│   └── research PDFs
│
├── chroma_db/
│
├── requirements.txt
└── README.md
```

The exact structure may vary as the project evolves.

## Setup

Clone the repository:

```bash
git clone <your-repository-url>
cd researchflow
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add the required API credentials:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit the `.env` file or API keys to GitHub.

## Run

After adding research documents and configuring the environment:

```bash
python3 main.py
```

Enter a research question when prompted.

## What I Learned

Building ResearchFlow helped me understand how modern AI applications combine LLMs with external knowledge systems.

In particular, I gained hands-on experience with:

- Retrieval-Augmented Generation
- Semantic search
- Vector databases
- Embedding models
- Document chunking
- LLM tool calling
- Agent workflows
- Conversational state
- Grounding AI responses in private documents

The project also demonstrated the difference between directly prompting an LLM and building an agent capable of retrieving external information when needed.
