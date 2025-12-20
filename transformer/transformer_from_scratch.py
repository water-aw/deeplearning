import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1, L, d_model)


def make_causal_mask(T: int, device):
    # True means "mask out"
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        # q: (B, Tq, d_model), k/v: (B, Tk, d_model)
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape

        Q = self.Wq(q)  # (B, Tq, d_model)
        K = self.Wk(k)  # (B, Tk, d_model)
        V = self.Wv(v)  # (B, Tk, d_model)

        # reshape to heads
        Q = Q.view(B, Tq, self.num_heads, self.d_head).transpose(1, 2)  # (B, h, Tq, d_head)
        K = K.view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)  # (B, h, Tk, d_head)
        V = V.view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)  # (B, h, Tk, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, h, Tq, Tk)

        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) with True at PAD positions
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        if attn_mask is not None:
            # attn_mask: (Tq, Tk) or broadcastable to (B, h, Tq, Tk), True means mask out
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, h, Tq, Tk)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, h, Tq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # (B, Tq, d_model)
        out = self.Wo(out)  # (B, Tq, d_model)

        if need_weights:
            return out, attn
        return out, None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None, need_weights=False):
        # Pre-LN style (更稳定，也更常用)
        a, w = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                             key_padding_mask=src_key_padding_mask, need_weights=need_weights)
        x = x + self.drop(a)
        f = self.ffn(self.norm2(x))
        x = x + self.drop(f)
        return x, w


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_attn_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None, need_weights=False):
        # masked self-attn
        a1, w1 = self.self_attn(self.norm1(y), self.norm1(y), self.norm1(y),
                                attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask, need_weights=need_weights)
        y = y + self.drop(a1)

        # cross-attn: Q from decoder, K/V from encoder memory
        a2, w2 = self.cross_attn(self.norm2(y), memory, memory,
                                 key_padding_mask=src_key_padding_mask, need_weights=need_weights)
        y = y + self.drop(a2)

        # ffn
        f = self.ffn(self.norm3(y))
        y = y + self.drop(f)

        return y, (w1, w2)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.1, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def make_padding_mask(self, ids):
        # True for PAD
        return ids.eq(self.pad_id)

    def encode(self, src_ids, need_weights=False):
        src_pad = self.make_padding_mask(src_ids)  # (B, S)
        x = self.pos(self.src_emb(src_ids))  # (B, S, d_model)
        weights = []
        for layer in self.enc_layers:
            x, w = layer(x, src_key_padding_mask=src_pad, need_weights=need_weights)
            if need_weights:
                weights.append(w)
        return x, src_pad, weights

    def decode(self, tgt_ids, memory, src_pad_mask, need_weights=False):
        tgt_pad = self.make_padding_mask(tgt_ids)  # (B, T)
        y = self.pos(self.tgt_emb(tgt_ids))  # (B, T, d_model)
        T = tgt_ids.size(1)
        causal = make_causal_mask(T, tgt_ids.device)  # (T, T), True means mask out
        # broadcast causal to (1, 1, T, T) works with masked_fill
        causal = causal[None, None, :, :]

        weights = []
        for layer in self.dec_layers:
            y, (w_self, w_cross) = layer(
                y, memory,
                tgt_attn_mask=causal,
                tgt_key_padding_mask=tgt_pad,
                src_key_padding_mask=src_pad_mask,
                need_weights=need_weights,
            )
            if need_weights:
                weights.append((w_self, w_cross))
        return y, weights

    def forward(self, src_ids, tgt_ids_in, need_weights=False):
        # 把源句子（src）编码成记忆（memory）
        memory, src_pad_mask, enc_w = self.encode(src_ids, need_weights=need_weights)
        # 根据记忆和目标输入（tgt_in），解码出特征（y）
        y, dec_w = self.decode(tgt_ids_in, memory, src_pad_mask, need_weights=need_weights)
        # 最后映射到词表，计算概率（logits）
        logits = self.lm_head(y)  # (B, T, vocab)
        return logits, (enc_w, dec_w)

    @torch.no_grad()
    def greedy_generate(self, src_ids, bos_id, eos_id, max_len=64):
        self.eval()
        memory, src_pad_mask, _ = self.encode(src_ids, need_weights=False)

        B = src_ids.size(0)
        out = torch.full((B, 1), bos_id, device=src_ids.device, dtype=torch.long)

        for _ in range(max_len - 1):
            y, _ = self.decode(out, memory, src_pad_mask, need_weights=False)
            next_logits = self.lm_head(y[:, -1])  # (B, vocab)
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            out = torch.cat([out, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return out


# -------- toy task: reverse a sequence --------
def make_batch(batch_size, min_len, max_len, vocab_size, pad_id, bos_id, eos_id, device):
    # tokens in [3..vocab_size-1], reserve 0 pad, 1 bos, 2 eos
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    S = lengths.max().item()
    src = torch.full((batch_size, S), pad_id, device=device, dtype=torch.long)
    for i, L in enumerate(lengths.tolist()):
        src[i, :L] = torch.randint(3, vocab_size, (L,), device=device)

    # target is reversed source (excluding pads)
    tgt_tokens = []
    for i, L in enumerate(lengths.tolist()):
        seq = src[i, :L].flip(0)
        tgt_tokens.append(seq)
    T = max_len + 2
    tgt = torch.full((batch_size, T), pad_id, device=device, dtype=torch.long)
    for i, seq in enumerate(tgt_tokens):
        full = torch.cat([torch.tensor([bos_id], device=device), seq, torch.tensor([eos_id], device=device)])
        tgt[i, :full.numel()] = full

    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    return src, tgt_in, tgt_out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 词表大小
    vocab_size = 50
    # pad 填充标记，用于对齐和占位
    # bos 句首标记，作为启动信号
    # eos 句尾标记，作为终止信号
    pad_id, bos_id, eos_id = 0, 1, 2

    # d_model 是隐藏层维度
    model = Transformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.1, pad_id=pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(1, 801):
        src, tgt_in, tgt_out = make_batch(
            batch_size=64, min_len=4, max_len=16,
            vocab_size=vocab_size, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, device=device
        )
        logits, _ = model(src, tgt_in, need_weights=False)  # (B, T-1, vocab)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_out.reshape(-1), ignore_index=pad_id)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    # inspect attention on one sample
    src, tgt_in, tgt_out = make_batch(1, 6, 10, vocab_size, pad_id, bos_id, eos_id, device)
    logits, (enc_w, dec_w) = model(src, tgt_in, need_weights=True)
    pred = logits.argmax(dim=-1)

    print("src:", src[0].tolist())
    print("tgt_out:", tgt_out[0].tolist())
    print("pred:", pred[0].tolist())

    # Example: decoder self-attn weights from last layer, head 0
    last_self_attn = dec_w[-1][0]  # (B, h, T, T)
    print("decoder self-attn (head0) shape:", last_self_attn[0, 0].shape)

    # greedy generate
    gen = model.greedy_generate(src, bos_id=bos_id, eos_id=eos_id, max_len=24)
    print("generated:", gen[0].tolist())


if __name__ == "__main__":
    main()
