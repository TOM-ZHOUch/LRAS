#!/usr/bin/env python3
"""
Utility functions for data loading and answer extraction.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON data file.

    Supports two formats:
    1. JSON array: [{...}, {...}]
    2. JSONL: one JSON object per line

    Args:
        file_path: Path to the data file

    Returns:
        List of sample dicts
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Try loading as a JSON array first
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"Loaded JSON array: {len(data)} samples")
            return data
        elif isinstance(data, dict):
            for key in ['data', 'samples', 'questions', 'items']:
                if key in data and isinstance(data[key], list):
                    print(f"Loaded from key '{key}': {len(data[key])} samples")
                    return data[key]
            raise ValueError("Unsupported JSON format: could not find a data list.")
    except json.JSONDecodeError:
        pass

    # Try loading as JSONL
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}")

    if samples:
        print(f"Loaded JSONL format: {len(samples)} samples")
        return samples

    raise ValueError(f"Unable to parse data file: {file_path}")


def load_lawbench(data_path: str) -> List[Dict[str, Any]]:
    """
    Load LawBench dataset.

    Args:
        data_path: Path to LawBench zero_shot directory or a single file

    Returns:
        List of sample dicts
    """
    samples = []
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    json_files = [data_path] if data_path.is_file() else sorted(data_path.glob("*.json"))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            prompt = f"{item['instruction']}\n{item['question']}"
            samples.append({
                'benchmark': 'lawbench',
                'question': item['question'],
                'task': json_file.stem,
                'prompt': prompt,
                'answer': item['answer'],
                'raw_data': item
            })

    print(f"Loaded LawBench: {len(samples)} samples, {len(json_files)} tasks")
    return samples


def load_lexeval(data_path: str) -> List[Dict[str, Any]]:
    """
    Load LexEval dataset.

    Args:
        data_path: Path to LexEval data directory or a single file

    Returns:
        List of sample dicts
    """
    samples = []
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    json_files = [data_path] if data_path.is_file() else sorted(data_path.glob("*.json"))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    prompt = f"{item['instruction']}{item['input']}"
                    samples.append({
                        'benchmark': 'lexeval',
                        'question': item['input'],
                        'task': json_file.stem,
                        'prompt': prompt,
                        'answer': item['answer'],
                        'raw_data': item
                    })

    print(f"Loaded LexEval: {len(samples)} samples, {len(json_files)} tasks")
    return samples


def load_disc_law(data_path: str) -> List[Dict[str, Any]]:
    """
    Load DISC-Law legal multiple-choice dataset.

    Args:
        data_path: Path to disc_law directory (containing *_converted.json files)
                   or a single JSON file

    Returns:
        List of sample dicts
    """
    samples = []
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if data_path.is_file():
        json_files = [data_path]
    else:
        json_files = sorted(data_path.glob("*_converted.json"))

    for json_file in json_files:
        if json_file.name == "all_converted.json":
            continue

        task_name = json_file.stem.replace('_converted', '')

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            prompt = f"{item['instruction']}\n{item['question']}"
            samples.append({
                'benchmark': 'disc_law',
                'task': task_name,
                'question': item['question'],
                'prompt': prompt,
                'answer': item['answer'],
                'raw_data': item
            })

        task_count = len([s for s in samples if s['task'] == task_name])
        print(f"  {task_name}: {task_count} samples")

    valid_files = [f for f in json_files if f.name != 'all_converted.json']
    print(f"Loaded DISC-Law: {len(samples)} total samples, {len(valid_files)} tasks")
    return samples


def _extract_mcq_letter(text: str) -> str:
    """Extract a multiple-choice answer letter from text."""
    if not text:
        return ""
    patterns = [
        r"<answer>\s*([A-Za-z]+)\s*</answer>",
        r"(?:final answer|answer)[:\s]+([A-Za-z]+)",
        r"(?:answer is|select)[:\s]+([A-Za-z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def load_mcq_rollout(data_path: str) -> List[Dict[str, Any]]:
    """
    Load MCQ rollout merged dataset.

    Compatible with two row formats:
    1. Rewritten questions: question/explanation
    2. Rollout results: prompt/responses/meta (meta contains question and gold_answer)

    Args:
        data_path: Path to the JSONL data file

    Returns:
        List of sample dicts
    """
    raw_data = load_json_data(data_path)

    samples = []
    skipped = 0
    for idx, item in enumerate(raw_data):
        meta = item.get('meta') or {}

        question = item.get('question') or meta.get('question')
        prompt = item.get('prompt') or question

        answer = (
            meta.get('gold_answer')
            or meta.get('ground_truth')
            or item.get('answer')
            or _extract_mcq_letter(item.get('explanation', ''))
        )
        if isinstance(answer, str):
            answer = _extract_mcq_letter(answer) or answer.strip().upper()

        if not question or not prompt:
            skipped += 1
            print(f"Warning: Sample {idx} has no question field, skipping.")
            continue

        task_name = (
            meta.get('task_id')
            or item.get('subject')
            or meta.get('dataset')
            or 'mcq_rollout'
        )

        samples.append({
            'benchmark': 'mcq_rollout',
            'task': str(task_name),
            'question': question,
            'prompt': prompt,
            'answer': answer,
            'raw_data': item
        })

    if skipped:
        print(f"Skipped {skipped} samples with missing fields.")
    print(f"Loaded MCQ rollout: {len(samples)} samples")
    return samples


def load_custom_data(file_path: str, task_name: str = "custom") -> List[Dict[str, Any]]:
    """
    Load custom-format data with automatic field detection.

    Supported field names (auto-detected):
    - Instruction: instruction (optional, prepended to the question)
    - Question: question, prompt, query, text, input
    - Answer: answer, label, ground_truth

    Args:
        file_path: Path to the data file
        task_name: Task identifier

    Returns:
        List of sample dicts
    """
    raw_data = load_json_data(file_path)

    samples = []
    for idx, item in enumerate(raw_data):
        original_question = None
        for key in ['question', 'prompt', 'query', 'text', 'input']:
            if key in item:
                original_question = item[key]
                break

        if not original_question:
            print(f"Warning: Sample {idx} has no question field, skipping.")
            continue

        full_prompt = original_question
        if 'instruction' in item and item['instruction']:
            full_prompt = f"{item['instruction']}{original_question}"

        answer = None
        for key in ['answer', 'label', 'ground_truth']:
            if key in item:
                answer = item[key]
                break

        samples.append({
            'benchmark': 'custom',
            'task': task_name,
            'question': original_question,
            'prompt': full_prompt,
            'answer': answer,
            'raw_data': item
        })

    print(f"Loaded custom data: {len(samples)} samples")
    return samples


def extract_answer_from_response(response: str) -> str:
    """
    Extract the final answer from the model's response.

    Looks for content inside <answer>...</answer> tags.
    Falls back to the full response if no tag is found.

    Args:
        response: The model's full response text

    Returns:
        Extracted answer string
    """
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def has_search_tag(text: str) -> bool:
    """Check whether the text contains a <search>...</search> tag."""
    return '<search>' in text and '</search>' in text


def extract_search_query(text: str) -> str:
    """
    Extract the search query from a model response.

    Args:
        text: Model response text

    Returns:
        Search query string, or empty string if not found
    """
    match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
