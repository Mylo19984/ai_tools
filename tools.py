import os
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests
from tavily import TavilyClient

session = requests.Session()


def tavily_search_tool(
    query: str,
    max_results: int = 5,
    include_images: bool = False,
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
    **kwargs: Any,
) -> list[dict]:
    """
    Search the web using Tavily and return a list of structured results.

    Args:
        query: The search query.
        max_results: Number of results to return.
        include_images: Whether to include image URLs.
        topic: Optional Tavily topic filter: "general", "news", or "finance".
        time_range: Optional Tavily time filter: "day", "week", "month", or "year".
        **kwargs: Additional optional Tavily search parameters.

    Returns:
        A list of dictionaries with title/content/url fields, plus image entries when requested.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")

   # api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")
    client = TavilyClient(api_key=api_key)
                          #, api_base_url=api_base_url) if api_base_url else TavilyClient(api_key=api_key)

    try:
        search_kwargs = {
            "query": query,
            "max_results": max_results,
            "include_images": include_images,
        }
        if topic is not None:
            search_kwargs["topic"] = topic
        if time_range is not None:
            search_kwargs["time_range"] = time_range
        search_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        response = client.search(**search_kwargs)

        results: list[dict] = []
        for item in response.get("results", []):
            result = {
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
            }
            if item.get("score") is not None:
                result["score"] = item["score"]
            results.append(result)

        if include_images:
            for image_url in response.get("images", []):
                results.append({"image_url": image_url})

        return results
    except Exception as exc:
        return [{"error": str(exc)}]


claude_tavily_tool = {
    "name": "tavily_search_tool",
    "description": "Performs a web search using the Tavily API and returns concise, structured results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or keywords to find relevant web content.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
                "default": 5,
            },
            "include_images": {
                "type": "boolean",
                "description": "Whether to include image results in the output.",
                "default": False,
            },
            "topic": {
                "type": "string",
                "description": "Optional topic filter: general, news, or finance.",
            },
            "time_range": {
                "type": "string",
                "description": "Optional time filter: day, week, month, or year.",
            },
        },
        "required": ["query"],
    },
}

# OpenAI-compatible alias for environments that expect the chat-completions tool schema.
tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic filter: general, news, or finance.",
                },
                "time_range": {
                    "type": "string",
                    "description": "Optional time filter: day, week, month, or year.",
                },
            },
            "required": ["query"],
        },
    },
}

def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches arXiv for research papers matching the given query.
    """
    clean_query = query.strip()
    if not clean_query:
        return [{"error": "Query must not be empty."}]

    url = f"https://export.arxiv.org/api/query?search_query=all:{clean_query}&start=0&max_results={max_results}"

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return [{"error": str(exc)}]

    try:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results: list[dict] = []
        for entry in root.findall("atom:entry", ns):
            title_node = entry.find("atom:title", ns)
            title = (title_node.text or "").strip() if title_node is not None else ""

            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
                if author.find("atom:name", ns) is not None and author.find("atom:name", ns).text
            ]

            published_node = entry.find("atom:published", ns)
            published = published_node.text[:10] if published_node is not None and published_node.text else ""

            id_node = entry.find("atom:id", ns)
            url_abstract = id_node.text if id_node is not None and id_node.text else ""

            summary_node = entry.find("atom:summary", ns)
            summary = (summary_node.text or "").strip() if summary_node is not None else ""

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "url": url_abstract,
                    "summary": summary,
                    "link_pdf": link_pdf,
                }
            )

        return results
    except Exception as exc:
        return [{"error": f"Parsing failed: {str(exc)}"}]


claude_arxiv_tool = {
    "name": "arxiv_search_tool",
    "description": "Searches arXiv for research papers and returns structured metadata for each result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research topic or keywords to search for on arXiv.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of arXiv results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

# OpenAI-compatible alias for environments that expect the chat-completions tool schema.
arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches arXiv for research papers and returns structured metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic or keywords to search for on arXiv.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of arXiv results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}



__all__ = [
    "tavily_search_tool",
    "claude_tavily_tool",
    "tavily_tool_def",
    "arxiv_search_tool",
    "claude_arxiv_tool",
    "arxiv_tool_def",
]
