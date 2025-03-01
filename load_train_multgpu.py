import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import datetime
import math
from torch.optim import lr_scheduler

'''
Decoder-only Transformer模型

数据预处理 + token embedding + position embedding

最终输入模型的数据是个三维的巨大tensor数据，可以理解为一个空间长方形
长是batch_size也就是有多少句话，
高是词向量的维度，
宽是训练的每句话的长度也就是序列长度

掩码矩阵用来遮住后面的字符，防止模型看到后面的字符以及未来的内容


注意力矩阵 

残差网络避免梯度消失

'''

# 超参数
batch_size = 128  # 增大批量大小以获得更稳定的梯度
block_size = 256 # 一句话的字符串长度 训练时太长的序列可能导致内存问题

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embedding = 512  # 保持原有维度

num_heads = 8

head_size = n_embedding // num_heads # 每个注意力头的维度 如query、key、value的维度

n_layers = 8      # 改回原来的层数

learning_rate = 5e-4  # 适当提高学习率

max_iters = 200000  # 训练轮次

dropout_value = 0.2  # 增强正则化

eval_iters = 100 # 评测次数

eval_interval = 200 # 评测间隔


torch.manual_seed(1337)
file_name = "combined_train.txt"


'''数据预处理'''
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

'''构造词典'''
chars = sorted(list(set(text))) # 用set去重提取字符，用list封装，用sorted排序
vocab_size = len(chars) # 词表长度 即 字典里存了多少个不同的字(token)

print(vocab_size)
    
'''构造字符与整数间的投影'''
stoi = { ch:i for i,ch in enumerate(chars) } # 字符:数字
itos = { i:ch for i,ch in enumerate(chars) } # 数字:字符
encode = lambda s: [stoi[c] for c in s] # 把字符串转换成列表
decode = lambda l: ''.join([itos[i] for i in l]) # 把列表转换成字符串


'''划分数据集'''
data = torch.tensor(encode(text), dtype=torch.long) # 用长整数表示字符
n = int(0.9*len(data)) # 90%的数据用于训练
train_data =data[:n] # 训练集
val_data = data[n:] # 测试集



print(f"文件{file_name}读取完成")

'''
划分batch
完成了将字符串数据转化成列表数据，并随机划分batch
此后训练模型时，将每个batch的数据输入模型
bacth=64 说明要构成一个64句话并排的列表矩阵 供之后模型一次并行学习64句话
此时 数据是 bacth_size * block_size 的二维数据
即 一个由 64句话 每句话有256个字符 构成的二维tensor
'''
# 修改get_batch函数，增加对话感知
def get_batch(split, per_gpu_batch, device):
    data = train_data if split == "train" else val_data
    indices = []
    
    # 优先选择包含对话标记的起始位置
    for _ in range(per_gpu_batch):
        while True:
            # 随机选择起始点
            idx = random.randint(0, len(data) - block_size - 1)
            # 检查是否包含对话开始标记
            segment = decode(data[idx:idx+32].tolist())  # 检查前32个字符
            if "[用户]" in segment or random.random() < 0.3:  # 30%概率强制选择对话
                indices.append(idx)
                break

    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return x.to(device), y.to(device)

'''损失评测'''
@torch.no_grad()
def estimate_loss(model, device, per_gpu_batch):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            with torch.no_grad():
                xb, yb = get_batch(split, per_gpu_batch, device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


'''head类 完成基础的attention注意力机制'''
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False) # 线性变换层
        self.query = nn.Linear(n_embedding, head_size, bias=False) # 线性变换层
        self.value = nn.Linear(n_embedding, head_size, bias=False) # 线性变换层

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 不可训练的结构，三角掩码矩阵(约等于常量)
        '''
        三角掩码矩阵
        什么时候用：股票预测 续写下文
        什么时候不用： 翻译 总结文章大意
        '''
        self.dropout = nn.Dropout(dropout_value) # 随机去掉(归零)一些值，增加网络稳定性，减少对某些值的依赖性

    def forward(self, x):
        B, T, C = x.shape # B 是 batch_size  T 是 block_size  C 是 n_embedding
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        '''核心 将k和q点乘注意力矩阵 (B, T, T)'''
        wei = q@k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.to(device)

        '''下三角掩码矩阵填充 把掩码矩阵为0的地方填充为负无穷'''
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # 对注意力矩阵进行softmax
        wei = self.dropout(wei) # 随机去掉(归零)一些值，增加网络稳定性，减少对某些值的依赖性

        v = self.value(x) # 对输入的embedding做线性变换
        out = wei @ v # 注意力矩阵与输入的embedding相乘 (B, T, head_size)
        return out

'''多头注意力'''
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embedding) # head_size*num_heads其实就等于n_embedding
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 把每个注意力头的输出拼接起来
        out = self.dropout(self.proj(out))
        return out
    

'''线性前馈网络'''
class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, n_embedding*4),
            nn.ReLU(),
            nn.Linear(n_embedding*4, n_embedding),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.net(x)


'''残差网络'''
class Block(nn.Module):
    def __init__(self, n_embedding, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, head_size) # 自注意力 多头注意力
        self.feed_forward = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding, eps=1e-5)
        self.ln2 = nn.LayerNorm(n_embedding, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout_value)
        self.dropout2 = nn.Dropout(dropout_value)

    def forward(self, x):
        x = x + self.dropout1(self.self_attention(self.ln1(x)))
        x = x + self.dropout2(self.feed_forward(self.ln2(x)))
        return x


'''多级残差网络transformer'''
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        
        # 添加对话标记专用嵌入
        self.special_token_embedding = nn.Embedding(4, n_embedding)  # [用户], [AI], [对话结束], 其他
        
        # 改进的注意力机制
        self.blocks = nn.Sequential(
            *[Block(n_embedding, num_heads) for _ in range(n_layers)],
            nn.Dropout(dropout_value)
        )
        
        self.ln_f = nn.LayerNorm(n_embedding) # 最后的归一化层
        self.lm_head = nn.Linear(n_embedding, vocab_size) # 最后的线性变换层


    def forward(self, idx, targets=None):
        ''' 
        idx 是输入数据，targets是目标数据(仅在训练时用到)
        idx 是二维数据，长是batch_size 宽是block_size 即 几句话 * 每句话几个字
        '''
        B, T = idx.shape # B 是 batch_size  T 是 block_size 数据是token(整数)形式
        # 组合嵌入
        token_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # 检测特殊标记
        is_special = (idx >= vocab_size - 4)  # 最后4个ID留给特殊标记
        special_emb = self.special_token_embedding(torch.clamp(idx - (vocab_size-4), 0, 3))
        
        x = torch.where(is_special.unsqueeze(-1), special_emb, token_emb) + pos_emb

        x = self.blocks(x) # 残差多头注意力
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 摊平 把 B句话 每句话有T各自的二维表 转化成 一个全是字符的向量 维度(B*T, C)
            targets = targets.view(B*T) # 摊平 维度(B*T)
            loss = F.cross_entropy(logits, targets) # 交叉熵计算损失
        else:
            loss = None
        return logits ,loss 
    
    '''生成文本'''
    def generate(self, prompt, max_new_tokens, temperature=0.9, top_k=50):
        # 添加对话生成逻辑
        if "[用户]" in prompt:
            context = self._prepare_dialog_context(prompt)
        else:
            context = torch.tensor([encode(prompt)], device=device)
        
        for _ in range(max_new_tokens):
            logits, _ = self(context[:, -block_size:])
            # 采样策略
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # 限制候选词
            top_probs, top_indices = torch.topk(probs, top_k)
            idx_next = torch.multinomial(top_probs, num_samples=1)
            context = torch.cat([context, top_indices.gather(-1, idx_next)], dim=1)
            
            # 检测对话结束标记
            if top_indices[0,0] == stoi.get('[对话结束]', -1):
                break
        return decode(context[0].tolist())
    
    def _prepare_dialog_context(self, text):
        # 将自然语言提示转换为带标记的格式
        encoded = []
        for line in text.split('\n'):
            if line.startswith('用户:'):
                encoded += encode(f"[用户] {line[3:].strip()}\n")
            elif line.startswith('AI:'):
                encoded += encode(f"[AI] {line[3:].strip()}\n")
            else:
                encoded += encode(line+"\n")
        return torch.tensor([encoded[-block_size:]], device=device)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

def add_gradient_noise(model, scale=0.1):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * scale
                param.grad.add_(noise)

def main(rank, world_size):
    try:
        # 修改初始化设置
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'  # 换一个端口

         # 等待所有进程就绪
        print(f"Process {rank}: Starting initialization")
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=1)
        )
        
        print(f"Process {rank}: Process group initialized")

        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        dist.barrier()
        print(f"Process {rank}: Passed first barrier")
        
        # 创建模型实例
        model = LanguageModel().to(device)
        
        # 加载预训练模型（只在主进程加载）
        checkpoint_path = 'models/transformer_model_iter_18000.pth'  # 替换为你的模型路径
        
        # 初始化最佳验证损失
        best_val_loss = float('inf')
        
        # 如果加载了检查点，从检查点获取最佳验证损失
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
            # 从检查点获取迭代次数和最佳损失
            start_iter = checkpoint.get('iter', 0)
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming from iteration {start_iter}, best val loss: {best_val_loss:.4f}")
        
        # 等待主进程加载完成
        dist.barrier()
        
        # 广播最佳验证损失到所有进程
        best_val_loss = torch.tensor(best_val_loss, device=device)
        dist.broadcast(best_val_loss, src=0)
        best_val_loss = best_val_loss.item()
        
        # 广播模型参数给所有进程
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
            
        # 转换为DDP模型
        model = DDP(model, device_ids=[rank])
        print(f"Process {rank}: Model initialized and synchronized")

        # 每个GPU的batch_size
        per_gpu_batch = batch_size // world_size
        
        # 使用分层学习率和权重衰减
        param_groups = []
        # embedding层使用较小的学习率
        param_groups.append({
            'params': [p for n, p in model.named_parameters() if 'embedding' in n],
            'lr': learning_rate * 0.5,
            'weight_decay': 0.01
        })
        # 注意力层使用基础学习率
        param_groups.append({
            'params': [p for n, p in model.named_parameters() 
                      if 'self_attention' in n and 'embedding' not in n],
            'lr': learning_rate,
            'weight_decay': 0.1
        })
        # 其他层使用较大的学习率
        param_groups.append({
            'params': [p for n, p in model.named_parameters() 
                      if not any(x in n for x in ['embedding', 'self_attention'])],
            'lr': learning_rate * 1.5,
            'weight_decay': 0.1
        })
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # 3. 使用更激进的学习率调度
        def get_lr_scheduler(optimizer):
            def lr_lambda(step):
                warmup_steps = 1000  # 减少预热步数
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                
                # 使用更激进的余弦退火
                progress = float(step - warmup_steps) / float(max_iters - warmup_steps)
                # 添加周期性重启
                num_cycles = 10
                progress = progress * num_cycles
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (progress % 1.0))))
            
            return lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = get_lr_scheduler(optimizer)
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler()

        # 训练前同步所有进程
        dist.barrier()
        print(f"Process {rank}: All processes ready for training")

        # 4. 改进训练循环
        for iter in range(max_iters):
            try:
                optimizer.zero_grad()
                
                # 使用梯度累积来增加等效批量大小
                accumulation_steps = 2
                total_loss = 0
                
                for _ in range(accumulation_steps):
                    # 获取训练数据
                    xb, yb = get_batch('train', per_gpu_batch, device)
                    
                    # 使用混合精度训练
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        _, loss = model(xb, yb)
                        loss = loss / accumulation_steps
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    total_loss += loss.item()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                
                # 学习率调度
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                
                # 每1000步进行一次学习率重启
                if iter > 0 and iter % 1000 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 2  # 重启时提高学习率
                
                # 评估和保存
                if iter % eval_interval == 0:
                    losses = estimate_loss(model, device, per_gpu_batch)
                    if rank == 0:
                        print(f"step {iter}: train loss {losses['train']:.4f}, "
                              f"val loss {losses['val']:.4f}, lr {current_lr:.2e}")
                        
                        # 保存检查点
                        if losses['val'] < best_val_loss * 0.9995:  # 更激进的保存策略
                            best_val_loss = losses['val']
                            torch.save({
                                'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'iter': iter,
                                'loss': best_val_loss,
                                'best_val_loss': best_val_loss
                            }, 'models/transformer_model_best.pth')
                
            except Exception as e:
                print(f"Process {rank} failed during iteration {iter} with error: {str(e)}")
                raise e

        # 训练结束后生成文本（只在主进程）
        if rank == 0:
            model.eval()
            context = "[用户] 请介绍一下自己\n[AI]"
            generated = model.module.generate(context, max_new_tokens=100, temperature=0.8)
            print("\nGenerated text:\n", textwrap.fill(generated, width=70))

    except Exception as e:
        print(f"Process {rank} failed with error: {str(e)}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e

def test_communication(rank, model, world_size):
    """测试进程间通信"""
    try:
        # 创建一个测试tensor
        test_tensor = torch.ones(1).to(rank)
        if rank == 0:
            test_tensor *= 2
        
        # 使用all_reduce测试通信
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        # 检查结果
        # 当 world_size = 4 时:
        # rank 0 贡献了 2.0，其他三个进程各贡献了 1.0
        # 所以最终结果应该是 2.0 + 1.0 + 1.0 + 1.0 = 5.0
        expected = torch.ones(1).to(rank) * (world_size + 1.0)  # 修改这里的计算逻辑
        
        success = torch.allclose(test_tensor, expected)
        print(f"Process {rank}: Communication test {'successful' if success else 'failed'}")
        print(f"Process {rank}: Expected {expected.item()}, got {test_tensor.item()}")  # 添加详细的调试信息
        return success
    except Exception as e:
        print(f"Process {rank}: Communication test failed with error: {str(e)}")
        return False


'''加载模型的函数'''
def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['loss'], checkpoint['epoch']

# 运行
if __name__ == "__main__":
    if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
    # 获取可用的GPU数量
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    # 设置world_size为可用GPU数量，最多2个
    world_size = min(4, n_gpus)
    print(f"Using {world_size} GPUs")
    
    # 清理环境变量
    if 'MASTER_ADDR' in os.environ:
        del os.environ['MASTER_ADDR']
    if 'MASTER_PORT' in os.environ:
        del os.environ['MASTER_PORT']
    
    # 启动分布式训练
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)



