import os
import json
import sys
import asyncio
import platform
import concurrent.futures
from pathlib import Path
from typing import AsyncGenerator

BASE_MODEL_MLX = "mlx-community/Qwen2.5-7B-Instruct-4bit"
TRAIN_ITERATIONS = 300

_mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_model_cache: dict = {}  # adapter_path -> (model, tokenizer)


def _app_data_dir() -> Path:
    return Path(os.environ.get("APP_DATA_DIR", str(Path(__file__).parent.parent / "data")))


def _adapter_dir(notebook_id: str) -> Path:
    d = _app_data_dir() / "adapters" / notebook_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapter_exists(notebook_id: str) -> bool:
    return (_adapter_dir(notebook_id) / "adapter_config.json").exists()


def delete_adapter(notebook_id: str) -> None:
    import shutil
    d = _adapter_dir(notebook_id)
    adapter_path = str(d)
    _model_cache.pop(adapter_path, None)
    if d.exists():
        shutil.rmtree(d)


def _write_training_data(notebook_id: str, chunks: list[str], sources: list[str]) -> Path:
    data_dir = _adapter_dir(notebook_id)
    records = [
        {"text": f"[Source: {src}]\n{chunk}"}
        for chunk, src in zip(chunks, sources)
    ]
    split_at = max(1, len(records) * 9 // 10)
    train_recs = records[:split_at]
    valid_recs = records[split_at:] if split_at < len(records) else records[-1:]
    for name, recs in [("train.jsonl", train_recs), ("valid.jsonl", valid_recs)]:
        with open(data_dir / name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return data_dir


async def train_notebook(
    notebook_id: str,
    chunks: list[str],
    sources: list[str],
) -> AsyncGenerator[dict, None]:
    if platform.system() != "Darwin":
        yield {"type": "error", "message": "LoRA training is currently supported on macOS (Apple Silicon) only."}
        return

    try:
        import mlx_lm  # type: ignore  # noqa: F401
    except ImportError:
        yield {"type": "error", "message": "mlx-lm is not installed. Run: pip install mlx-lm"}
        return

    yield {"type": "progress", "message": "Preparing training data…", "percent": 3}
    data_dir = _write_training_data(notebook_id, chunks, sources)
    adapter_dir = _adapter_dir(notebook_id)

    # Evict cached model so the new adapter loads fresh after training
    _model_cache.pop(str(adapter_dir), None)

    yield {
        "type": "progress",
        "message": "Launching trainer — first run downloads ~4 GB of model files…",
        "percent": 5,
    }

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", BASE_MODEL_MLX,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--num-layers", "8",
        "--batch-size", "2",
        "--num-iterations", str(TRAIN_ITERATIONS),
        "--learning-rate", "1e-4",
        "--steps-per-report", "10",
        "--val-batches", "1",
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    while True:
        raw = await process.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if "Iter" in line and ":" in line:
            try:
                step = int(line.split("Iter")[1].split(":")[0].strip())
                pct = 5 + int((step / TRAIN_ITERATIONS) * 90)
                yield {"type": "progress", "message": f"Training… step {step} / {TRAIN_ITERATIONS}", "percent": pct}
            except (ValueError, IndexError):
                pass
        elif any(k in line.lower() for k in ("download", "fetch", "loading")):
            yield {"type": "progress", "message": line[:100], "percent": 6}

    rc = await process.wait()
    if rc != 0:
        yield {"type": "error", "message": f"Training failed (exit {rc}). Open macOS Console and filter for LocalNotebookLM for details."}
        return

    yield {"type": "done", "message": "Training complete! This notebook now uses your fine-tuned model.", "percent": 100}


def _load_model_sync(adapter_path: str):
    if adapter_path not in _model_cache:
        from mlx_lm import load  # type: ignore
        model, tokenizer = load(BASE_MODEL_MLX, adapter_path=adapter_path)
        _model_cache[adapter_path] = (model, tokenizer)
    return _model_cache[adapter_path]


def _generate_sync(adapter_path: str, prompt: str) -> str:
    from mlx_lm import generate  # type: ignore
    model, tokenizer = _load_model_sync(adapter_path)
    return generate(model, tokenizer, prompt=prompt, max_tokens=1024, temp=0.3, verbose=False)


async def generate_with_adapter(notebook_id: str, prompt: str) -> str:
    adapter_path = str(_adapter_dir(notebook_id))
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_mlx_executor, _generate_sync, adapter_path, prompt)
