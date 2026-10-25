"""The model the CPU suites build their schedulers for (tests/test_make_room.py, tests/test_trace.py)."""

import json

import pytest


@pytest.fixture(scope="session")
def model(tmp_path_factory) -> str:
    """A model directory with only a config: the scheduler reads the model's configuration, never its weights."""
    path = tmp_path_factory.mktemp("model")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }
    (path / "config.json").write_text(json.dumps(config))
    return str(path)
