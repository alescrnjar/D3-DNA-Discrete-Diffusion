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
    
    qkv=torch.rand((58880, 3, 12, 64), dtype=torch.bfloat16) 

    cu_seqlens=torch.tensor([    0,   230,   460,   690,   920,  1150,  1380,  1610,  1840,  2070,
         2300,  2530,  2760,  2990,  3220,  3450,  3680,  3910,  4140,  4370,
         4600,  4830,  5060,  5290,  5520,  5750,  5980,  6210,  6440,  6670,
         6900,  7130,  7360,  7590,  7820,  8050,  8280,  8510,  8740,  8970,
         9200,  9430,  9660,  9890, 10120, 10350, 10580, 10810, 11040, 11270,
        11500, 11730, 11960, 12190, 12420, 12650, 12880, 13110, 13340, 13570,
        13800, 14030, 14260, 14490, 14720, 14950, 15180, 15410, 15640, 15870,
        16100, 16330, 16560, 16790, 17020, 17250, 17480, 17710, 17940, 18170,
        18400, 18630, 18860, 19090, 19320, 19550, 19780, 20010, 20240, 20470,
        20700, 20930, 21160, 21390, 21620, 21850, 22080, 22310, 22540, 22770,
        23000, 23230, 23460, 23690, 23920, 24150, 24380, 24610, 24840, 25070,
        25300, 25530, 25760, 25990, 26220, 26450, 26680, 26910, 27140, 27370,
        27600, 27830, 28060, 28290, 28520, 28750, 28980, 29210, 29440, 29670,
        29900, 30130, 30360, 30590, 30820, 31050, 31280, 31510, 31740, 31970,
        32200, 32430, 32660, 32890, 33120, 33350, 33580, 33810, 34040, 34270,
        34500, 34730, 34960, 35190, 35420, 35650, 35880, 36110, 36340, 36570,
        36800, 37030, 37260, 37490, 37720, 37950, 38180, 38410, 38640, 38870,
        39100, 39330, 39560, 39790, 40020, 40250, 40480, 40710, 40940, 41170,
        41400, 41630, 41860, 42090, 42320, 42550, 42780, 43010, 43240, 43470,
        43700, 43930, 44160, 44390, 44620, 44850, 45080, 45310, 45540, 45770,
        46000, 46230, 46460, 46690, 46920, 47150, 47380, 47610, 47840, 48070,
        48300, 48530, 48760, 48990, 49220, 49450, 49680, 49910, 50140, 50370,
        50600, 50830, 51060, 51290, 51520, 51750, 51980, 52210, 52440, 52670,
        52900, 53130, 53360, 53590, 53820, 54050, 54280, 54510, 54740, 54970,
        55200, 55430, 55660, 55890, 56120, 56350, 56580, 56810, 57040, 57270,
        57500, 57730, 57960, 58190, 58420, 58650, 58880], device='cuda:0',
       dtype=torch.int32) 
    seq_len=230

    # Replace FlashAttention call with vanilla attention
    # Instead of:
    # output = flash_attn_varlen_qkvpacked_func(qkv, ...)
    output = vanilla_attention(qkv, mask) #output.shape=torch.Size([58880, 12, 64])