import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.manual_seed = lambda *args, **kwargs: None
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *args, **kwargs: None,
        device_count=lambda: 0,
    )
    sys.modules["torch"] = torch_stub


if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    datasets_stub.concatenate_datasets = lambda datasets: datasets[0]
    sys.modules["datasets"] = datasets_stub


if "llm_annotator" not in sys.modules:
    llm_stub = types.ModuleType("llm_annotator")

    class _Annotator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def annotate_dataset(self, **kwargs):
            return kwargs["dataset"]

        def destroy(self):
            return None

    class _Client:
        def __init__(self, model: str, **kwargs):
            self.model = model
            self.kwargs = kwargs

    class _RuntimeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    llm_stub.Annotator = _Annotator
    llm_stub.OpenAIClient = _Client
    llm_stub.ClaudeClient = _Client
    llm_stub.VLLMClient = _Client
    llm_stub.VLLMOfflineClient = _Client
    llm_stub.OpenAIRuntimeOptions = _RuntimeOptions
    llm_stub.ClaudeRuntimeOptions = _RuntimeOptions
    llm_stub.VLLMServerRuntimeOptions = _RuntimeOptions
    llm_stub.VLLMRuntimeOptions = _RuntimeOptions
    sys.modules["llm_annotator"] = llm_stub
