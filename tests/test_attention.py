import torch
import pytest

from vit_core.attention import ScaledDotProductAttention, MultiHeadedAttention

def test_scaled_dot_product_attention_output_shape():
    """
    Tests if the output tensor has the expected shape (batch_size, seq_len_q, d_v).
    """
    batch_size = 4
    seq_len_q = seq_len_k = 10
    d_k = d_v = 10
    
    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_k, d_v)

    output = ScaledDotProductAttention(query, key, value)

    expected_shape = (batch_size, seq_len_q, d_k)
    assert output.shape == expected_shape, f"Output shape mismatch: Expected {expected_shape}, got {output.shape}"

def test_scaled_dot_product_attention_batching():
    """
    Tests if processing items in a batch gives the same result as
    processing them individually and concatenating.
    """
    batch_size = 3
    seq_len_q = seq_len_k = 6
    d_k = d_v = 10

    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_k, d_v)

    output_batch = ScaledDotProductAttention(query, key, value)

    outputs_individual = []
    for i in range(batch_size):
        q_i = query[i:i+1]
        k_i = key[i:i+1]
        v_i = value[i:i+1]

        output_i = ScaledDotProductAttention(q_i, k_i, v_i)
        outputs_individual.append(output_i)

    output_concat = torch.cat(outputs_individual, dim=0)

    assert torch.allclose(output_batch, output_concat, atol=1e-6), \
        "Batch processing differs from individual processing."
    
# MHA TESTS

D_MODEL = 64
NUM_HEADS = 8
SEQ_LEN_Q = 10
SEQ_LEN_K = 12
BATCH_SIZE = 4

@pytest.fixture
def mha_model():
    """Provides an instance of the MultiHeadedAttention module."""
    return MultiHeadedAttention(d_model=D_MODEL, num_heads=NUM_HEADS)

@pytest.fixture
def sample_data():
    """Provides sample Q, K, V tensors."""
    query = torch.randn(BATCH_SIZE, SEQ_LEN_Q, D_MODEL)
    key = torch.randn(BATCH_SIZE, SEQ_LEN_K, D_MODEL)
    value = torch.randn(BATCH_SIZE, SEQ_LEN_K, D_MODEL)
    return query, key, value

@pytest.fixture
def sample_data_self_attn():
    """Provides sample Q, K, V tensors for self-attention (same seq_len)."""
    seq_len = 10
    query = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    key = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    value = torch.randn(BATCH_SIZE, seq_len, D_MODEL)
    return query, key, value

def test_mha_output_shape_no_mask(mha_model, sample_data):
    """
    Tests the output shape of MHA without any mask.
    """
    query, key, value = sample_data
    output = mha_model(query, key, value)

    assert output.shape == (BATCH_SIZE, SEQ_LEN_Q, D_MODEL), f"Expected shape {(BATCH_SIZE, SEQ_LEN_Q, D_MODEL)}, but got {output.shape}"
    assert output.dtype == query.dtype, "Output dtype should match input dtype"