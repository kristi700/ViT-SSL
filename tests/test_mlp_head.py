import torch
import pytest

from vit_core.mlp_head import MLPHead


@pytest.fixture(scope="module")
def params():
    """Test parameters"""
    return {"batch_size": 8, "d_model": 128, "num_classes": 10}


@pytest.fixture(scope="module")
def mlp_head_model(params):
    """Provides an instance of the MLPHead model."""
    model = MLPHead(d_model=params["d_model"], num_classes=params["num_classes"])
    model.eval()
    return model


@pytest.fixture
def sample_input(params):
    """Provides sample input tensor (batch of CLS token embeddings)."""
    return torch.rand(params["batch_size"], params["d_model"])


def test_mlp_head_forward_shape(mlp_head_model, sample_input, params):
    """Tests the output shape of the forward pass."""
    output = mlp_head_model(sample_input)
    expected_shape = (params["batch_size"], params["num_classes"])
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
    assert output.dtype == sample_input.dtype


def test_mlp_head_forward_value_change(mlp_head_model, sample_input):
    """Tests if the forward pass actually changes the values (not identity)."""
    input_clone = sample_input.clone()
    output = mlp_head_model(sample_input)

    assert torch.equal(input_clone, sample_input), "Input tensor modified in-place"

    assert not torch.allclose(
        output, input_clone[:, : output.shape[1]], atol=1e-5
    ), "Output is too close to input; MLP head might be an identity function or have zero weights."


def test_mlp_head_batch_independence(params):
    """Tests if processing items individually matches batch processing."""

    model = MLPHead(d_model=params["d_model"], num_classes=params["num_classes"])
    model.eval()

    batch_input = torch.rand(params["batch_size"], params["d_model"])

    output_batch = model(batch_input)

    outputs_individual = []
    for i in range(params["batch_size"]):
        individual_input = batch_input[i : i + 1]
        output_ind = model(individual_input)
        outputs_individual.append(output_ind)

    output_concat = torch.cat(outputs_individual, dim=0)

    assert output_concat.shape == output_batch.shape
    assert torch.allclose(
        output_batch, output_concat, atol=1e-6
    ), "Batch processing result differs from processing items individually."
