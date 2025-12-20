import torch

# Simple batch maker for the reverse-sequence toy task used by the Transformer demo.
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
min_len = 3
max_len = 5
vocab_size = 10
# pad用于填充，bos 作为句首信号，eos作为句尾信号
pad_id, bos_id, eos_id = 0, 1, 2

# 1) Randomly decide each sentence length.
lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)

# 2) Create source tensor padded with pad_id.
S = lengths.max().item()
src = torch.full((batch_size, S), pad_id, device=device, dtype=torch.long)
for i, L in enumerate(lengths.tolist()):
    src[i, :L] = torch.randint(3, vocab_size, (L,), device=device)

# 3) Reverse each non-pad portion to build target tokens.
tgt_tokens = []
for i, L in enumerate(lengths.tolist()):
    seq = src[i, :L].flip(0)
    tgt_tokens.append(seq)

# 4) Build tgt with BOS/EOS and padding to a fixed length.
T = max_len + 2  # extra 2 for BOS and EOS
tgt = torch.full((batch_size, T), pad_id, device=device, dtype=torch.long)
for i, seq in enumerate(tgt_tokens):
    full = torch.cat([
        torch.tensor([bos_id], device=device),
        seq,
        torch.tensor([eos_id], device=device),
    ])
    tgt[i, :full.numel()] = full

# 5) Shift for teacher forcing.
tgt_in = tgt[:, :-1]
tgt_out = tgt[:, 1:]

print("lengths:", lengths)
print("src:", src)
print("tgt_in:", tgt_in)
print("tgt_out:", tgt_out)
