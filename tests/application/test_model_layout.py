"""Exercise real HF checkpoint admission on CPU, without fake weight files."""

import pytest

from recipes.inputs import inspect_model_layout, model_artifact

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")


@pytest.fixture
def model_directory(tmp_path):
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen2ForCausalLM(config)
    model.save_pretrained(tmp_path)
    return tmp_path


def test_real_checkpoint_uses_meta_storage(model_directory):
    initialized = torch.cuda.is_initialized()
    layout = inspect_model_layout(model_artifact(model_directory))
    assert layout["storage"] == "meta"
    assert layout["parameters"] > 0
    assert layout["tensors"] > 0
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("mutation", ["missing", "shape", "extra"])
def test_checkpoint_layout_rejects_corruption(model_directory, mutation):
    path = model_directory / "model.safetensors"
    weights = safetensors.load_file(path)
    name = next(iter(weights))
    if mutation == "missing":
        del weights[name]
    elif mutation == "shape":
        weights[name] = weights[name][:-1].contiguous()
    else:
        weights["foreign.weight"] = torch.zeros(2)
    safetensors.save_file(weights, path)
    with pytest.raises(ValueError, match="tensor layout differs"):
        inspect_model_layout(model_artifact(model_directory))
