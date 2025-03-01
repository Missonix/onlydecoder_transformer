import torch
import torch.nn as nn
from torch.nn import functional as F
import textwrap

# 超参数
block_size = 256  # 序列长度
n_embedding = 512  # 嵌入维度
num_heads = 8     # 注意力头数
head_size = n_embedding // num_heads
n_layers = 8      # 层数
dropout_value = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表
def load_vocab(file_name="combined_train.txt"):
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

# 编码解码函数
def encode(s, stoi):
    return [stoi[c] for c in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

# 模型定义
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embedding)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, n_embedding * 4),
            nn.ReLU(),
            nn.Linear(n_embedding * 4, n_embedding),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedding, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size  # 保存词表大小
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        # 添加特殊token嵌入
        self.special_token_embedding = nn.Embedding(4, n_embedding)  # [用户], [AI], [对话结束], 其他
        
        self.blocks = nn.Sequential(*[Block(n_embedding, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 组合嵌入
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # 检测特殊标记
        is_special = (idx >= self.vocab_size - 4)  # 使用self.vocab_size
        special_emb = self.special_token_embedding(torch.clamp(idx - (self.vocab_size-4), 0, 3))
        
        # 合并普通token和特殊token的嵌入
        x = torch.where(is_special.unsqueeze(-1), special_emb, tok_emb) + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, prompt, max_new_tokens, temperature=0.8, top_k=50):
        idx = torch.tensor([encode(prompt, stoi)], device=device)
        
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # 获取预测
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # 只保留top_k个选项
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 如果生成了结束标记，就停止生成
            if idx_next.item() == self.vocab_size - 2:  # [对话结束] 标记
                break
                
            idx = torch.cat([idx, idx_next], dim=1)
            
        return decode(idx[0].tolist(), itos)

def load_model(model_path):
    """加载预训练模型"""
    print(f"Loading model from {model_path}")
    stoi, itos = load_vocab()
    model = LanguageModel(len(stoi)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, stoi, itos

def chat(model, prompt, max_tokens=100):
    """聊天函数"""
    response = model.generate(prompt, max_new_tokens=max_tokens)
    return response

if __name__ == "__main__":
    # 加载模型
    model_path = 'models/transformer_model_best.pth'
    model, stoi, itos = load_model(model_path)
    
    print("模型加载完成，开始对话（输入 'quit' 退出）")
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break
            
        # 构造输入格式
        prompt = f"[用户] {user_input}\n[AI]"
        
        # 生成回复
        response = chat(model, prompt)
        
        # 提取AI回复部分
        try:
            ai_response = response.split("[AI]")[1].split("[用户]")[0].strip()
        except:
            ai_response = response
            
        print("\nAI:", textwrap.fill(ai_response, width=70))

