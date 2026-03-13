"""Web tools for ClawScope."""

from __future__ import annotations

import os
import re

import httpx
from loguru import logger


async def web_search(query: str) -> str:
    """
    Search the web for information.

    Uses Brave Search API if available, falls back to DuckDuckGo.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Try Brave Search API
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if brave_api_key:
        return await _brave_search(query, brave_api_key)

    # Fallback to DuckDuckGo
    return await _duckduckgo_search(query)


async def _brave_search(query: str, api_key: str) -> str:
    """Search using Brave Search API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 5},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:5]:
            results.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Description: {item.get('description', '')}\n"
            )

        if not results:
            return "No search results found"

        return "\n---\n".join(results)

    except Exception as e:
        logger.error(f"Brave search error: {e}")
        return f"Search error: {str(e)}"


async def _duckduckgo_search(query: str) -> str:
    """Fallback search using DuckDuckGo HTML."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()

        # Simple extraction
        html = response.text
        results = []

        # Extract result snippets (simplified)
        pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html)[:5]

        for url, title in matches:
            results.append(f"Title: {title}\nURL: {url}\n")

        if not results:
            return "No search results found"

        return "\n---\n".join(results)

    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return f"Search error: {str(e)}"


async def web_fetch(url: str) -> str:
    """
    Fetch content from a URL.

    Args:
        url: URL to fetch

    Returns:
        Page content (text extracted from HTML)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type:
            # Extract text from HTML
            html = response.text
            text = _extract_text_from_html(html)
            return text[:16000] if len(text) > 16000 else text

        elif "application/json" in content_type:
            return response.text[:16000]

        elif "text/" in content_type:
            return response.text[:16000]

        else:
            return f"Fetched {len(response.content)} bytes of {content_type}"

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code}: {str(e)}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')

    # Clean whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


__all__ = ["web_search", "web_fetch"]
