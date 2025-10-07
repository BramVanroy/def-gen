import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON text, raising a ValueError on failure with context.

    Args:
        text: The JSON string to parse.

    Returns:
        Parsed JSON object.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(
            f"Failed to parse JSON: {json_exc.msg} at pos {json_exc.pos}; snippet={snippet!r}"
        ) from json_exc


def enable_quiet_mode():
    # --- 1) FORCE env vars (override anything that’s there) ---
    env = {
        "TORCH_CPP_LOG_LEVEL": "ERROR",  # PyTorch/Gloo C++ logs
        "TORCH_DISTRIBUTED_DEBUG": "OFF",
        "NCCL_DEBUG": "WARN",
        "TRANSFORMERS_VERBOSITY": "error",
        "PYTHONWARNINGS": "ignore",
        # vLLM / Ray
        "RAY_LOG_TO_STDERR": "0",
        "RAY_BACKEND_LOG_LEVEL": "ERROR",
        "RAY_DEDUP_LOGS": "1",
        # Optional: keep Ray from spamming log files
        "RAY_LOG_TO_FILE": "0",
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_CONFIGURE_LOGGING": "0",
    }
    os.environ.update(env)
    # Also export to any subprocesses that might spawn after imports:
    for k, v in env.items():
        try:
            os.putenv(k, v)
        except Exception:
            pass

    # --- 2) Warn if called too late (after imports) ---
    late = [m for m in ("torch", "transformers", "vllm", "ray") if m in sys.modules]
    if late:
        # You can raise here if you prefer strictness.
        print(
            f"[quiet-mode] Warning: called after imports: {late}. "
            "Some C++/init-time logs may have already been enabled.",
            file=sys.stderr,
        )

    # --- 3) Python warnings + logging clamps (handles libs that change levels later) ---
    warnings.filterwarnings("ignore")
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("transformers", "huggingface_hub", "datasets", "torch", "vllm", "ray"):
        logging.getLogger(name).handlers.clear()  # drop pre-added noisy handlers
        logging.getLogger(name).propagate = True
        logging.getLogger(name).setLevel(logging.ERROR)

    # Transformers explicit API
    try:
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:
        pass

    # --- 4) Optional: silence nanobind exit spew only at shutdown (stderr) ---
    try:
        import atexit

        def _silence_c_stderr_on_exit():
            import os
            import sys

            try:
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, sys.stderr.fileno())
                os.close(devnull_fd)
            except Exception:
                pass

        atexit.register(_silence_c_stderr_on_exit)
    except Exception:
        pass
