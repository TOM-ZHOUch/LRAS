<h1 align="center">⚖️ LRAS: Advanced Legal Reasoning with Agentic Search</h1>

<p align="center">
  ─────────────  ✦  ─────────────
</p>

<p align="center">
  <b>The official repository for</b><br>
  <b>"LRAS: Advanced Legal Reasoning with Agentic Search"</b>
</p>

<p align="center">
  📑 <a href="https://arxiv.org/abs/2601.07296"><b>Paper</b></a> &nbsp;•&nbsp;
  🤗 <a href="https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-4B">LRAS-RL-Qwen3-4B</a> &nbsp;•&nbsp;
  🤗 <a href="https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-8B">LRAS-RL-Qwen3-8B</a> &nbsp;•&nbsp;
  🤗 <a href="https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-14B">LRAS-RL-Qwen3-14B</a>
</p>

---

## Timeline

- **[2026/05/18]** 🔧 We released the **evaluation code** for LRAS! See [Evaluation](#evaluation) below for usage.

- **[2026/05/10]** 🚀 We released model checkpoints across multiple scales: [LRAS-RL-Qwen3-4B](https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-4B), [LRAS-RL-Qwen3-8B](https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-8B), and [LRAS-RL-Qwen3-14B](https://huggingface.co/Edwinzz/LRAS-RL-Qwen3-14B)! Dataset will be released soon, stay tuned!

- **[2026/01/12]** 🎉 Our paper *"LRAS: Advanced Legal Reasoning with Agentic Search"* is now available on arXiv!

---

## Evaluation

### File Structure

```
evaluation_code/
├── eval.py           # Main evaluation script
├── text_search.py    # Search pipeline (SerpAPI + Jina Reader + LLM summarizer)
├── utils.py          # Data loading utilities
├── prompts.py        # Prompt template
└── requirements.txt
```

### Installation

```bash
pip install -r requirements.txt
```

### Step 1 — Deploy the Model

Use [vLLM](https://github.com/vllm-project/vllm) to serve any LRAS model checkpoint as an OpenAI-compatible API:

```bash
vllm serve Edwinzz/LRAS-RL-Qwen3-8B \
    --served-model-name lras-8b \
    --port 8000
```

Any OpenAI-compatible endpoint works. Replace the model name and port as needed.

### Step 2 — Run Evaluation

**Without web search (mock mode):**

```bash
python eval.py \
    --data-type lawbench \
    --data-path /path/to/data \
    --base-url http://localhost:8000/v1 \
    --model-name lras-8b \
    --experiment-name my_experiment
```

**With real web search:**

Set the following environment variables first:

```bash
export SERPAPI_KEY="your_serpapi_key"
export SUMMARIZER_BASE_URL="https://api.openai.com/v1"
export SUMMARIZER_API_KEY="your_openai_api_key"
export SUMMARIZER_MODEL="gpt-4o"
```

Then run with `--enable-real-search`:

```bash
python eval.py \
    --data-type custom \
    --data-path /path/to/data.json \
    --base-url http://localhost:8000/v1 \
    --model-name lras-8b \
    --experiment-name my_experiment \
    --enable-real-search
```

### Supported Datasets

| `--data-type` | Description |
|---|---|
| `lawbench` | [LawBench](https://github.com/open-compass/LawBench) zero-shot tasks |
| `lexeval` | [LexEval](https://github.com/CSHaitao/LexEval) legal benchmark |
| `disc_law` | [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM) MCQ dataset |
| `mcq_rollout` | Custom MCQ rollout JSONL format |
| `custom` | Generic JSON/JSONL (auto-detects question/answer fields) |

### Output

Results are saved to `./outputs/` by default:

```
outputs/
├── {experiment_name}_checkpoint.jsonl   # Per-sample results (supports resume)
└── {experiment_name}_details/           # Full conversation & search history per sample
```

If evaluation is interrupted, simply re-run the same command — completed samples are automatically skipped.
