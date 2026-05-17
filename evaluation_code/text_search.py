"""
Text Search Module (Question-Focused Summarization)

Pipeline: SerpAPI search -> Jina Reader (parallel) -> LLM question-focused summarization

Configuration via environment variables:
  SERPAPI_KEY          - SerpAPI API key (required for real search)
  SUMMARIZER_BASE_URL  - LLM API base URL (OpenAI-compatible, default: https://api.openai.com/v1)
  SUMMARIZER_API_KEY   - LLM API key for summarization
  SUMMARIZER_MODEL     - LLM model name (default: gpt-4o)
"""

import os
import re
import asyncio
import aiohttp
from serpapi import GoogleSearch
from openai import OpenAI
from typing import Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Configuration via environment variables
SERPAPI_KEY = os.environ.get('SERPAPI_KEY', '')
SUMMARIZER_BASE_URL = os.environ.get('SUMMARIZER_BASE_URL', 'https://api.openai.com/v1')
SUMMARIZER_API_KEY = os.environ.get('SUMMARIZER_API_KEY', '')
SUMMARIZER_MODEL = os.environ.get('SUMMARIZER_MODEL', 'gpt-4o')


def serpapi_search(text_query: str, api_key: str = None, num_results: int = 5) -> List[Dict]:
    """
    Execute a web search using the SerpAPI official library.

    Args:
        text_query: Search query string
        api_key: SerpAPI API key (falls back to SERPAPI_KEY env var)
        num_results: Number of results to return

    Returns:
        List of search result dicts with keys: link, title, snippet
    """
    if api_key is None:
        api_key = SERPAPI_KEY

    if not api_key:
        print('Warning: SERPAPI_KEY is not set. Please set the SERPAPI_KEY environment variable.')
        return []

    try:
        print(f"Searching via SerpAPI: {text_query[:50]}...")

        search = GoogleSearch({
            "q": text_query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-CN",
            "num": num_results
        })

        results = search.get_dict()
        organic_results = results.get('organic_results', [])

        formatted_results = []
        for result in organic_results[:num_results]:
            formatted_results.append({
                'link': result.get('link', ''),
                'title': result.get('title', 'No title'),
                'snippet': result.get('snippet', '')
            })

        print(f"Search successful, got {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        print(f"[SerpAPI Error] Search failed: {e}")
        return []


async def _jina_fetch_async(session: aiohttp.ClientSession, url: str, timeout: int = 30) -> str:
    """Asynchronously fetch full webpage content via Jina Reader."""
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                return await response.text()
    except asyncio.TimeoutError:
        print(f"[Jina Reader] Timeout: {url}")
    except Exception as e:
        print(f"[Jina Reader] Failed to fetch content: {e}")
    return ""


async def _jina_batch_async(urls: List[str], timeout: int = 30) -> List[str]:
    """Fetch multiple URLs in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [_jina_fetch_async(session, url, timeout) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, str) else "" for r in results]


def jina_reader_batch(urls: List[str], timeout: int = 30) -> List[str]:
    """
    Fetch multiple URLs in parallel (synchronous wrapper).

    Args:
        urls: List of URLs
        timeout: Timeout per request in seconds

    Returns:
        List of page content strings (in same order as input URLs)
    """
    return asyncio.run(_jina_batch_async(urls, timeout))


def _clean_urls(text: str) -> str:
    """Remove URLs from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text


def _fallback_extract(content: str, max_length: int = 2000) -> str:
    """Fallback: truncate content when summarization fails."""
    cleaned = _clean_urls(content)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length] + "..."


def summarize_with_question(content: str, original_question: str) -> str:
    """
    Use an LLM to generate a question-focused summary of webpage content.

    Completeness is prioritized: relevant legal information should be preserved
    in full, including exact article numbers, statute names, and specific conditions.

    Args:
        content: Webpage content to summarize
        original_question: The user's original question

    Returns:
        Summarized text focused on information relevant to the question
    """
    try:
        client = OpenAI(
            base_url=SUMMARIZER_BASE_URL,
            api_key=SUMMARIZER_API_KEY
        )

        prompt = f"""你是一个法律信息提取专家。请从以下网页内容中提取与用户问题相关的法律信息要点。

用户问题：
{original_question}

要求：
1. 只提取与问题直接相关的法律信息（法条、规定、案例、解释等）
2. 包括：
   - 相关的法律条文和法规名称
   - 具体的法律规定和要求
   - 适用范围、条件、例外情况
   - 相关的法律后果或责任
3. 不要分析问题、不要给出结论、不要添加自己的推理
4. 保持客观中立，直接提取法律事实
5. 去除所有网页 URL 链接
6. 用 5-10 句话简明总结
7. 如果内容与问题无关，直接说明"无有效信息"

网页内容：
{content}
""".strip()

        rsp = client.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192
        )

        result = rsp.choices[0].message.content.strip()
        result = _clean_urls(result)

        if not result or result in ['', '无', '空', '[无有效信息]', '无有效信息']:
            return ""

        return result

    except Exception as e:
        print(f"[Summarize Error] {e}")
        return _fallback_extract(content)


def summarize_batch(
    contents: List[str],
    original_question: str,
    max_workers: int = 10
) -> List[str]:
    """
    Summarize multiple content pieces in parallel.

    Args:
        contents: List of content strings to summarize
        original_question: The user's original question
        max_workers: Maximum parallel threads (default: 10)

    Returns:
        List of summaries in same order as input
    """
    def _summarize_one(content):
        return summarize_with_question(content, original_question)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_summarize_one, contents))
    return results


def call_text_search(
    text_query: str,
    original_question: str = None,
) -> Tuple[str, Dict]:
    """
    Execute the full search pipeline: search -> fetch content -> summarize.

    Pipeline:
    1. Use SerpAPI to search for relevant webpages
    2. Use Jina Reader to fetch full content from each result URL (parallel)
    3. Use an LLM to generate question-focused summaries (parallel)
    4. Return formatted results

    Args:
        text_query: Search query string
        original_question: The user's original question (for targeted summarization).
                           Falls back to text_query if not provided.

    Returns:
        tool_returned_str: Formatted search result string
        tool_stat: Execution status dict with keys: success, num_results
    """
    if original_question is None:
        original_question = text_query

    try:
        print(f"[Text Search] 开始搜索: {text_query}")
        print(f"[Text Search] 原始问题: {original_question[:100]}...")

        # Step 1: SerpAPI 搜索
        search_results = serpapi_search(text_query)

        if not search_results:
            return (
                "[文本搜索结果] 未找到相关的法律信息。",
                {"success": False, "num_results": 0}
            )

        results_to_process = search_results[:5]
        urls = [r.get('link', '') for r in results_to_process]
        snippets = [r.get('snippet', '') for r in results_to_process]

        print(f"[Text Search] 并行获取 {len(urls)} 个网页内容...")

        # Step 2: 并行获取网页内容
        full_contents = jina_reader_batch(urls, timeout=30)

        # 准备需要摘要的内容
        contents_to_summarize = []
        for i, content in enumerate(full_contents):
            if content:
                contents_to_summarize.append(content[:8000])
            else:
                contents_to_summarize.append(snippets[i])

        print(f"[Text Search] 并行生成 {len(contents_to_summarize)} 个问题导向的摘要...")

        # Step 3: 并行生成摘要
        summaries = summarize_batch(contents_to_summarize, original_question)

        # Step 4: 过滤空摘要并格式化结果
        result_texts = []
        for i, summary in enumerate(summaries):
            if summary and summary.strip():
                result_texts.append(f"{i}: {summary}\n")

        tool_stat = {
            "success": True,
            "num_results": len(result_texts)
        }

        if result_texts:
            tool_returned_str = (
                "[文本搜索结果] 以下是与您的问题相关的法律信息摘要：\n\n"
                + "\n".join(result_texts)
            )
        else:
            tool_returned_str = "[文本搜索结果] 未找到与问题直接相关的法律信息。"

        print(f"[Text Search] 搜索完成，共 {len(result_texts)} 条有效结果")

    except Exception as e:
        print(f"[Text Search Error] 搜索过程中遇到错误: {e}")
        tool_stat = {"success": False, "num_results": 0}
        tool_returned_str = "[文本搜索结果] 搜索过程中遇到错误，请基于已有信息回答。"

    return tool_returned_str, tool_stat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试文本搜索功能")
    parser.add_argument("--query", type=str, default="房东不退押金 法律规定",
                        help="搜索查询")
    parser.add_argument("--question", type=str,
                        default="如果我的房东不退还押金，我该怎么办？有什么法律依据？",
                        help="原始问题")
    args = parser.parse_args()

    print("=" * 80)
    print("测试: 问题导向的搜索与摘要")
    print("=" * 80)

    result_str, stat = call_text_search(args.query, args.question)
    print(result_str)
    print(f"\nStatus: {stat}")
