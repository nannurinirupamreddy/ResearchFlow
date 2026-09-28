from langchain.tools import tool
from rag.vectorstore import search_documents

description = """
    Search the uploaded research documents for information relevant to the query.
    Use this tool when the user's question may be answered by the ingested PDFs
    or other private research documents.
"""

@tool('rag_search', description=description)
def rag_search(query: str):
    results = search_documents(query)

    if not results:
        return "No relevant information was found in the uploaded documents."

    formatted_results = []

    for doc in results:
        formatted_results.append(
            f"Page Content: {doc.page_content}\n"
            f"Source: {doc.metadata.get('source')}\n"
            f"Page: {doc.metadata.get('page') + 1 if doc.metadata.get('page') is not None else 'unknown'}"
        )

    return "\n\n---\n\n".join(formatted_results)