import os
from typing import Any, Optional

from tavily import TavilyClient


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

    api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")
    client = TavilyClient(api_key=api_key, api_base_url=api_base_url) if api_base_url else TavilyClient(api_key=api_key)

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

__all__ = ["tavily_search_tool", "claude_tavily_tool", "tavily_tool_def"]