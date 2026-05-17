#!/usr/bin/env python3
"""
Legal QA Model Evaluation Script (Question-Focused Search & Summarization)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
import requests

from prompts import format_user_message
from utils import (
    load_lawbench,
    load_lexeval,
    load_disc_law,
    load_mcq_rollout,
    load_custom_data,
    extract_answer_from_response,
    has_search_tag,
    extract_search_query
)

# 全局配置（用于多进程）
GLOBAL_CONFIG = {}


def init_worker(config: Dict):
    """初始化 worker 进程的全局配置。"""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config


def call_llm_sync(
    messages: List[Dict[str, str]],
    base_url: str,
    model_name: str,
    api_key: str = None,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    同步调用 LLM API。

    Args:
        messages: 消息列表
        base_url: API base URL
        model_name: 模型名称
        api_key: API 密钥（可选）
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        max_retries: 最大重试次数

    Returns:
        API 响应字典
    """
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'raw_response': result
                }
            else:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(min(2 ** attempt, 30))
                    continue
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                import time
                time.sleep(min(2 ** attempt, 30))
                continue
            return {'success': False, 'error': '请求超时'}
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(min(2 ** attempt, 30))
                continue
            return {'success': False, 'error': str(e)}

    return {'success': False, 'error': '超过最大重试次数'}


def execute_search(
    query: str,
    original_question: str,
    enable_real_search: bool = True
) -> tuple:
    """
    执行搜索（问题导向的摘要模式）。

    Args:
        query: 搜索查询
        original_question: 用户的原始问题（用于生成针对性摘要）
        enable_real_search: 是否启用真实搜索

    Returns:
        (搜索结果文本, 统计信息字典)
    """
    if not enable_real_search:
        return (
            f"""[模拟搜索结果]
查询：{query}
问题：{original_question}

注：这是模拟的搜索结果。本次评测主要测试模型是否学会了使用 <search> 和 <answer> 标签的 workflow。
如果模型能够正确使用标签格式，说明训练效果良好。""",
            {"success": True, "num_results": 1, "mode": "mock"}
        )

    try:
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from text_search import call_text_search
    except ImportError as e:
        print(f"⚠ 警告: text_search 模块导入失败: {e}，使用模拟搜索")
        return (
            f"""[模拟搜索结果]
查询：{query}
问题：{original_question}

注：这是模拟的搜索结果（text_search 模块导入失败）。""",
            {"success": True, "num_results": 1, "mode": "mock"}
        )

    try:
        print(f"[搜索] 查询={query[:50]}... | 问题={original_question[:50]}...")
        result_text, stat = call_text_search(query, original_question)
        stat['mode'] = 'question_focused'
        return result_text, stat

    except Exception as e:
        print(f"[搜索错误] {e}")
        return (
            "[搜索失败] 搜索服务暂时不可用，请基于已有知识回答。",
            {"success": False, "num_results": 0, "mode": "question_focused", "error": str(e)}
        )


def run_multi_turn_inference(
    original_question: str,
    initial_prompt: str,
    sample_id: str,
    config: Dict
) -> Dict[str, Any]:
    """
    执行多轮推理（支持问题导向的搜索 workflow）。

    Args:
        original_question: 原始问题（用于搜索摘要）
        initial_prompt: 初始问题
        sample_id: 样本 ID
        config: 配置字典

    Returns:
        推理结果字典
    """
    formatted_prompt = format_user_message(initial_prompt)
    messages = [{"role": "user", "content": formatted_prompt}]

    conversation_history = []
    search_history = []
    final_answer = None

    max_rounds = config.get('max_rounds', 10)
    enable_real_search = config.get('enable_real_search', False)

    for round_num in range(1, max_rounds + 1):
        result = call_llm_sync(
            messages=messages,
            base_url=config['base_url'],
            model_name=config['model_name'],
            api_key=config.get('api_key'),
            max_tokens=config.get('max_tokens', 16384),
            temperature=config.get('temperature', 0.6)
        )

        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', '未知错误'),
                'conversation_history': conversation_history,
                'search_history': search_history,
                'rounds': round_num
            }

        assistant_response = result['content']

        conversation_history.append({
            'round': round_num,
            'messages': messages.copy(),
            'response': assistant_response
        })

        if '<answer>' in assistant_response and '</answer>' in assistant_response:
            final_answer = extract_answer_from_response(assistant_response)
            return {
                'success': True,
                'final_answer': final_answer,
                'conversation_history': conversation_history,
                'search_history': search_history,
                'rounds': round_num,
                'full_response': assistant_response
            }

        if has_search_tag(assistant_response):
            search_query = extract_search_query(assistant_response)

            if search_query:
                search_result, search_stat = execute_search(
                    search_query,
                    original_question,
                    enable_real_search
                )

                search_history.append({
                    'round': round_num,
                    'query': search_query,
                    'original_question': original_question,
                    'result': search_result,
                    'stat': search_stat
                })

                messages.append({"role": "assistant", "content": assistant_response})
                messages.append({
                    "role": "user",
                    "content": f"<information>\n{search_result}\n</information>"
                })

                continue

        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({
            "role": "user",
            "content": "请使用<search>...</search>进行搜索，或使用<answer>...</answer>提供最终答案。"
        })

    return {
        'success': False,
        'error': f'达到最大轮数限制（{max_rounds}）',
        'conversation_history': conversation_history,
        'search_history': search_history,
        'rounds': max_rounds,
        'final_answer': conversation_history[-1]['response'] if conversation_history else ''
    }


def evaluate_sample(args_tuple) -> Optional[Dict[str, Any]]:
    """
    评测单个样本（用于多进程）。

    Args:
        args_tuple: (index, sample, config, completed_ids, detail_dir, checkpoint_file)

    Returns:
        评测结果字典
    """
    index, sample, config, completed_ids, detail_dir, checkpoint_file = args_tuple

    task_name = sample.get('task', 'unknown')
    sample_id = f"{task_name}_{index}"

    if sample_id in completed_ids:
        return None

    try:
        result = run_multi_turn_inference(
            original_question=sample['question'],
            initial_prompt=sample['prompt'],
            sample_id=sample_id,
            config=config
        )

        output = {
            'sample_id': sample_id,
            'benchmark': sample.get('benchmark', 'unknown'),
            'task': task_name,
            'index': index,
            'prompt': sample['prompt'],
            'ground_truth': sample.get('answer', ''),
            'prediction': result.get('final_answer', ''),
            'success': result['success'],
            'rounds': result.get('rounds', 0),
            'num_searches': len(result.get('search_history', [])),
            'error': result.get('error', None),
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

        detail_file = Path(detail_dir) / f"{sample_id}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sample': sample,
                'result': output,
                'conversation_history': result.get('conversation_history', []),
                'search_history': result.get('search_history', [])
            }, f, ensure_ascii=False, indent=2)

        return output

    except Exception as e:
        print(f"[错误] 样本 {sample_id} 处理失败: {e}")
        return {
            'sample_id': sample_id,
            'success': False,
            'error': str(e)
        }


def load_completed_ids(checkpoint_file: Path) -> set:
    """加载已完成的样本 ID（用于断点续存）。"""
    if not checkpoint_file.exists():
        return set()

    completed = set()
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                completed.add(item.get('sample_id', ''))
            except:
                continue

    if completed:
        print(f"✓ 已完成 {len(completed)} 个样本，将跳过")
    return completed


def main():
    parser = argparse.ArgumentParser(
        description='法律问答模型评测脚本（问题导向的搜索摘要版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：

1. 评测 LawBench（模拟搜索）：
   python eval_sft.py \\
       --data-type lawbench \\
       --data-path /path/to/lawbench/zero_shot \\
       --base-url http://localhost:8000/v1 \\
       --model-name your_model_name \\
       --experiment-name lawbench_test

2. 评测自定义数据（启用真实搜索）：
   python eval_sft.py \\
       --data-type custom \\
       --data-path /path/to/data.json \\
       --base-url http://localhost:8000/v1 \\
       --model-name your_model_name \\
       --experiment-name custom_test \\
       --enable-real-search

3. Debug 模式（只测试 2 个样本）：
   python eval_sft.py \\
       --data-type custom \\
       --data-path /path/to/data.json \\
       --base-url http://localhost:8000/v1 \\
       --model-name your_model_name \\
       --experiment-name debug \\
       --debug

4. 评测前 100 条：
   python eval_sft.py \\
       --data-type mcq_rollout \\
       --data-path /path/to/data.jsonl \\
       --base-url http://localhost:8000/v1 \\
       --model-name your_model_name \\
       --experiment-name test_100 \\
       --limit 100

启用真实搜索时需设置以下环境变量：
   SERPAPI_KEY          - SerpAPI API 密钥
   SUMMARIZER_BASE_URL  - 摘要 LLM 的 API base URL（OpenAI 兼容）
   SUMMARIZER_API_KEY   - 摘要 LLM 的 API 密钥
   SUMMARIZER_MODEL     - 摘要 LLM 的模型名称（默认：gpt-4o）
        """
    )

    # 数据参数
    parser.add_argument(
        '--data-type',
        type=str,
        required=True,
        choices=['lawbench', 'lexeval', 'disc_law', 'mcq_rollout', 'custom'],
        help='数据类型'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='数据路径（文件或目录）'
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default='custom',
        help='任务名称（仅用于 custom 类型）'
    )

    # 模型参数
    parser.add_argument(
        '--base-url',
        type=str,
        required=True,
        help='模型 API base URL（OpenAI 兼容）'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='模型名称'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='模型 API 密钥（可选）'
    )

    # 搜索配置
    parser.add_argument(
        '--enable-real-search',
        action='store_true',
        help='启用真实网络搜索（SerpAPI + Jina Reader + LLM 摘要）'
    )

    # 实验配置
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='实验名称（用于输出文件命名）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='输出目录（默认：./outputs）'
    )

    # 推理参数
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=10,
        help='最大对话轮数（默认：10）'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=16384,
        help='最大生成 token 数（默认：16384）'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='温度参数（默认：0.6）'
    )

    # 并发参数
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='并发进程数（默认：CPU 核数）'
    )

    # 其他
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug 模式：只评测 2 个样本'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='仅评测前 N 个样本（默认：全部）'
    )

    args = parser.parse_args()

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / f"{args.experiment_name}_checkpoint.jsonl"
    detail_dir = output_dir / f"{args.experiment_name}_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(checkpoint_file)

    print(f"\n{'='*80}")
    print(f"加载数据")
    print(f"{'='*80}")
    print(f"数据类型: {args.data_type}")
    print(f"数据路径: {args.data_path}")

    if args.data_type == 'lawbench':
        samples = load_lawbench(args.data_path)
    elif args.data_type == 'lexeval':
        samples = load_lexeval(args.data_path)
    elif args.data_type == 'disc_law':
        samples = load_disc_law(args.data_path)
    elif args.data_type == 'mcq_rollout':
        samples = load_mcq_rollout(args.data_path)
    elif args.data_type == 'custom':
        samples = load_custom_data(args.data_path, args.task_name)
    else:
        raise ValueError(f"不支持的数据类型: {args.data_type}")

    if not samples:
        print("错误：没有加载到任何样本")
        return

    if args.debug:
        samples = samples[:2]
        print(f"\n🔧 DEBUG 模式：只评测前 2 个样本\n")
        for i, s in enumerate(samples):
            print(f"样本 {i+1}: {s['prompt'][:100]}...")
            print(f"答案: {s.get('answer', 'N/A')}\n")
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须为正整数")
        samples = samples[:args.limit]
        print(f"\n🔧 LIMIT 模式：只评测前 {len(samples)} 个样本\n")

    pending_samples = [
        (i, s) for i, s in enumerate(samples)
        if f"{s.get('task', 'unknown')}_{i}" not in completed_ids
    ]

    config = {
        'base_url': args.base_url,
        'model_name': args.model_name,
        'api_key': args.api_key,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_rounds': args.max_rounds,
        'enable_real_search': args.enable_real_search,
    }

    print(f"\n{'='*80}")
    print(f"开始评测")
    print(f"{'='*80}")
    print(f"总样本数: {len(samples)}")
    print(f"待评测: {len(pending_samples)}")
    print(f"已完成: {len(samples) - len(pending_samples)}")
    search_mode = '模拟搜索'
    if config['enable_real_search']:
        search_mode = "真实搜索（SerpAPI + Jina Reader + LLM 摘要）"
    print(f"搜索模式: {search_mode}")
    print(f"最大轮数: {args.max_rounds}")
    print(f"进程数: {args.num_workers or cpu_count()}")
    print(f"{'='*80}\n")

    if not pending_samples:
        print("所有样本已完成！")
        return

    eval_args = [
        (idx, sample, config, completed_ids, str(detail_dir), str(checkpoint_file))
        for idx, sample in pending_samples
    ]

    num_workers = args.num_workers or cpu_count()

    results = []
    from tqdm import tqdm

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(evaluate_sample, eval_args),
            total=len(eval_args),
            desc=f"评测-{args.experiment_name}"
        ):
            if result:
                results.append(result)

    total = len(results)
    success = sum(1 for r in results if r.get('success', False))
    failed = total - success

    avg_rounds = sum(r.get('rounds', 0) for r in results) / total if total > 0 else 0
    avg_searches = sum(r.get('num_searches', 0) for r in results) / total if total > 0 else 0

    print(f"\n{'='*80}")
    print(f"评测完成")
    print(f"{'='*80}")
    print(f"总样本数: {total}")
    print(f"成功: {success} ({success/total*100:.1f}%)" if total > 0 else "成功: 0")
    print(f"失败: {failed} ({failed/total*100:.1f}%)" if total > 0 else "失败: 0")
    print(f"平均轮数: {avg_rounds:.2f}")
    print(f"平均搜索次数: {avg_searches:.2f}")
    print(f"\n结果文件:")
    print(f"  - Checkpoint: {checkpoint_file}")
    print(f"  - 详细日志: {detail_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
