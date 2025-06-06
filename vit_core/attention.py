import torch
from torch import nn


def ScaledDotProductAttention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    return_attn: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention

    Args:
        - query: matrix of shape (batch_size, seq_len, d_k)
        - key: matrix of shape (batch_size, seq_len, d_k)
        - value: matrix of shape (batch_size, seq_len, d_v)
    Output:
        - context matrix after attention applied
    """
    attention_score = torch.matmul(query, torch.transpose(key, -2, -1))
    scaled_attention_score = attention_score / torch.sqrt(torch.tensor(query.shape[-1]))
    normalized_scores = torch.softmax(scaled_attention_score, dim=-1)
    context_matrix = torch.matmul(normalized_scores, value)
    if return_attn:
        return context_matrix, normalized_scores
    else:
        return context_matrix, None


class MultiHeadedAttention(nn.Module):
    """
    Vanilla MultiHeadedAttention module.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            - d_model: input embeddings dimensionality
            - num_heads: number of heads for the attention
        Output:
            - context matrix after paralell attentions applied
        """
        super().__init__()

        assert (
            d_model % num_heads == 0
        ), f"d_model({d_model}) must be cleanly divisible by num_heads({num_heads})!"
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads  # d_k = d_v this way
        self.num_heads = num_heads

        # Weight matricies for linear projection
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)

        self.final_linear = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            - query: Query tensor (batch_size, seq_len_q, d_model)
            - key: Key tensor (batch_size, seq_len_k, d_model)
            - value: Value tensor (batch_size, seq_len_v, d_model) (Note: seq_len_k == seq_len_v)
            - mask (Optional): Mask to exclude tokens (e.g., padding mask, look-ahead mask).
        Output:
            - context matrix after attention applied (batch_size, seq_len_q, d_model)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        q_proj = self.w_query(query)
        k_proj = self.w_key(key)
        v_proj = self.w_value(value)

        q_heads = q_proj.view(
            batch_size, seq_len_q, self.num_heads, self.d_k
        ).transpose(1, 2)
        k_heads = k_proj.view(
            batch_size, seq_len_k, self.num_heads, self.d_k
        ).transpose(1, 2)
        v_heads = v_proj.view(
            batch_size, seq_len_k, self.num_heads, self.d_v
        ).transpose(1, 2)

        context, attn_probs = ScaledDotProductAttention(
            q_heads, k_heads, v_heads, return_attn
        )
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, query.shape[1], self.d_model)
        )

        mha_context = self.final_linear(context)
        return mha_context, attn_probs
