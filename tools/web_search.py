from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

description = """
    Search the web for current information relevant to a research question.
    Use this tool when up-to-date or external information is needed.
"""

client = TavilyClient()

@tool('web_search', description=description)
def web_search(query: str):
    response =  client.search(query=query, max_results=5, search_depth='advanced')

    results = response.get("results", [])

    if not results:
        return "No relevant web results were found."

    formatted_results = []

    for result in results:
        formatted_results.append(
            f"Title: {result.get('title')}\n"
            f"URL: {result.get('url')}\n"
            f"Content: {result.get('content')}"
        )

    return "\n\n---\n\n".join(formatted_results)