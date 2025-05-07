import torch
import pytest

from vit_core.feed_forward import FeedForwardBlock

D_MODEL = 64
D_FF = 128
SEQ_LEN = 10
BATCH_SIZE = 4

@pytest.fixture
def ffn_model() -> FeedForwardBlock:
    """Provides an instance of the FeedForwardBlock module."""
    return FeedForwardBlock(d_model=D_MODEL, d_ff=D_FF)

@pytest.fixture
def sample_input_tensor() -> torch.Tensor:
    """Provides a sample input tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

def test_ffn_output_shape(ffn_model: FeedForwardBlock, sample_input_tensor: torch.Tensor):
    """
    Tests if the output tensor shape is identical to the input tensor shape.
    """
    ffn_model.eval() 
    output = ffn_model(sample_input_tensor)
    expected_shape = sample_input_tensor.shape
    assert output.shape == expected_shape, f"Output shape mismatch: Expected {expected_shape}, got {output.shape}"

def test_ffn_batch_independence(ffn_model: FeedForwardBlock, sample_input_tensor: torch.Tensor):
    """
    Tests if processing items in a batch gives the same result as
    processing them individually (demonstrates position-wise application).
    """
    ffn_model.eval() 
    
    output_batch = ffn_model(sample_input_tensor)
    
    outputs_individual = []
    for i in range(BATCH_SIZE):
        individual_input = sample_input_tensor[i].unsqueeze(0)
        individual_output = ffn_model(individual_input)
        
        outputs_individual.append(individual_output)

    output_concat = torch.cat(outputs_individual, dim=0)

    assert torch.allclose(output_batch, output_concat, atol=1e-6), "Batch processing differs significantly from individual processing."