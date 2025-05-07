import torch
import pytest

from vit_core.encoder_block import EncoderBlock

@pytest.fixture(scope="module")
def params():
    """Test parameters"""
    return {
        "batch_size": 4,
        "seq_len": 50,
        "d_model": 64,
        "num_heads": 8,
        "mlp_dim": 128,
        "dropout": 0.1,
    }


@pytest.fixture(scope="module")
def encoder_block_model(params):
    """Provides an instance of the EncoderBlock model."""

    model = EncoderBlock(
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        mlp_dim=params["mlp_dim"],
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def sample_input(params):
    """Provides sample input tensor."""
    return torch.rand(params["batch_size"], params["seq_len"], params["d_model"])


def test_encoder_block_forward_shape(encoder_block_model, sample_input, params):
    """Tests the output shape of the forward pass."""
    output = encoder_block_model(sample_input)
    assert (
        output.shape == sample_input.shape
    ), f"Expected output shape {sample_input.shape}, but got {output.shape}"
    assert output.dtype == sample_input.dtype


def test_encoder_block_forward_value_change(encoder_block_model, sample_input):
    """Tests if the forward pass actually changes the values (not identity)."""
    input_clone = sample_input.clone()
    output = encoder_block_model(sample_input)

    assert torch.equal(input_clone, sample_input), "Input tensor modified in-place"

    assert not torch.equal(
        output, input_clone
    ), "Output is identical to input; block might be an identity function."


def test_encoder_block_dropout_behavior(params, sample_input):
    """Tests if dropout behaves differently in train vs eval mode."""

    block_with_dropout = EncoderBlock(
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        mlp_dim=params["mlp_dim"],
        dropout=params["dropout"],
    )
    assert params["dropout"] > 0.0, "Dropout rate must be > 0 for this test"

    block_with_dropout.eval()
    output_eval = block_with_dropout(sample_input)

    block_with_dropout.train()
    output_train1 = block_with_dropout(sample_input)
    output_train2 = block_with_dropout(sample_input)

    assert not torch.allclose(
        output_eval, output_train1, atol=1e-7
    ), "Eval and Train modes produced the same output with dropout > 0."

    assert not torch.allclose(
        output_train1, output_train2, atol=1e-7
    ), "Two forward passes in train mode produced the same output with dropout > 0."
    
    block_with_dropout.eval()
    output_eval_again = block_with_dropout(sample_input)
    assert torch.allclose(
        output_eval, output_eval_again, atol=1e-7
    ), "Two forward passes in eval mode produced different outputs."
