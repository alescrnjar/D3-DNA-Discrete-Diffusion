import torch
import torch.nn.functional as F

def vanilla_attention(qkv, mask=None):
    # Assume qkv is packed as (batch_size, seq_len, 3 * hidden_dim)
    batch_size, seq_len, _ = qkv.size()
    hidden_dim = qkv.size(-1) // 3
    # Unpack QKV
    q, k, v = qkv.chunk(3, dim=-1)
    # Reshape to (batch_size, num_heads, seq_len, head_dim)
    num_heads = 8  # Adjust this based on your model’s configuration
    head_dim = hidden_dim // num_heads
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    # Apply mask if provided
    if mask is not None:
        #scores = scores.masked_fill(mask == 0, float(‘-inf’))
        scores = scores.masked_fill(mask == 0, float("-inf"))
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    # Compute output
    output = torch.matmul(attn_weights, v)
    # Reshape output
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    return output

if __name__=='__main__':
    # Replace FlashAttention call with vanilla attention
    # Instead of:
    # output = flash_attn_varlen_qkvpacked_func(qkv, ...)
    output = vanilla_attention(qkv, mask)