#!/usr/bin/env python3
"""
Prompt definitions for the evaluation script.

The user prompt prefix is prepended to each question before sending to the model.
It should match the format used during SFT training.
"""

USER_PROMPT_PREFIX = """你是一位专业的法律助手,擅长通过深度搜索准确回答法律问题。

工作方式:
1. 先在 <think></think> 内推理分析,若判断已有知识不足以准确回答,则进行搜索
2. 如需查找法律依据,用 <search>查询</search> 调用搜索,结果在 <information></information> 返回
3. 可多次搜索,直到信息充分
4. 最后在 <answer></answer> 内给出简洁答案

问题:"""


def format_user_message(question: str) -> str:
    """
    Format the user message by prepending the prompt prefix to the question.

    Args:
        question: The user's question

    Returns:
        Formatted user message string
    """
    return USER_PROMPT_PREFIX + question
