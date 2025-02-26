import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap # 自动换行

'''
第三步 增加token embedding 和 postition embedding

数据预处理 + token embedding + position embedding

最终输入模型的数据是个三维的巨大tensor数据，可以理解为一个空间长方形
长是batch_size也就是有多少句话，
高是词向量的维度，
宽是训练的每句话的长度也就是序列长度
'''

# 超参数
batch_size = 32 # 一次训练多少句话
block_size = 8 # 一句话的字符串长度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embedding = 32 # 嵌入后的维度

learning_rate = 3e-4 # 学习率

max_iters = 10000 # 训练次数

eval_iters = 200 # 评测次数

eval_interval = int(max_iters/10) # 评测间隔


torch.manual_seed(1337)
file_name = "news.txt"


# 数据预处理
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# 有序又不重复的列表
chars = sorted(list(set(text)))
vocab_size = len(chars) # 词表长度 即 字典里存了多少个不同的字

# print(vocab_size)
    
# 字符与整数间的投影
stoi = { ch:i for i,ch in enumerate(chars) } # 字符:数字

itos = { i:ch for i,ch in enumerate(chars) } # 数字:字符

encode = lambda s: [stoi[c] for c in s] # 把字符串转换成列表
decode = lambda l: ''.join([itos[i] for i in l]) # 把列表转换成字符串

data = torch.tensor(encode(text), dtype=torch.long) # 用长整数表示字符

# print(encode("数据资源登记平台"))
# print(decode([114, 126, 153, 23]))
# print(data)

# 训练集测试集分离
n = int(0.9*len(data)) # 90%的数据用于训练
train_data =data[:n] # 训练集
val_data = data[n:] # 测试集

# print(train_data)
# print(val_data)

print(f"文件{file_name}读取完成")

'''
完成了将字符串数据转化成列表数据，并划分了batch
此后训练模型时，将每个batch的数据输入模型
bacth=32 说明要构成一个32句话的列表 供之后模型一次并行学习32句话
此时 数据是bacth_size*block_size的二维数据
即 一个由 32句话 每句话有8个字符 构成的二维tensor
'''
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix]) # 输入模型的batch
    y = torch.stack([data[i+1: i+block_size+1] for i in ix]) # 输入模型的batch
    x, y = x.to(device), y.to(device)
    return x, y

# 损失评测
@torch.no_grad() # 下面的函数 不做梯度计算
def estimate_loss(model):
    out = {}
    model.eval() # 把模型转化为eval模式
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters) # 建立一个初始值为0的容器，存储loss值
        for k in range(eval_iters):
            x, y = get_batch(split) # split是一个字符串， 用来控制get_batch()函数的行为
            logits, loss = model(x, y) # model的输入值一个是index， 一个是target
            losses[k] = loss.item()
        out[split] = losses.mean() # out是含两个元素的字典，一个是train，一个是val，每个元素对应一个loss的平均值
    model.train() # 把模型转化为train模式
    return out




'''简易的模型 实现one-hot编码 随机生成下文内容'''
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding).to(device)
        self.fc1 = nn.Linear(n_embedding, 100)
        self.fc2 = nn.Linear(100, vocab_size)



    def forward(self, idx, targets=None):
        ''' 
        idx 是输入数据，targets是目标数据(仅在训练时用到)
        idx 是二维数据，长是batch_size 宽是block_size 即 几句话 * 每句话几个字
        '''
        B, T = idx.shape # B 是 batch_size  T 是 block_size 数据是token(整数)形式
        token_embed = self.token_embedding_table(idx) # 词嵌入

        position_idx = torch.arange(T).to(device)
        position_embd = self.position_embedding_table(position_idx)

        x = token_embed + position_embd # 输入的嵌入是 词嵌入 + 位置嵌入  维度(B, T, n_embedding)
        logits = torch.relu(self.fc1(x)) # 输入的嵌入 经过全连接层 维度(B, T, vocab_size)
        logits = self.fc2(logits)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 摊平 把 B句话 每句话有T各自的二维表 转化成 一个全是字符的向量 维度(B*T, C)
            targets = targets.view(B*T) # 摊平 维度(B*T)
            loss = F.cross_entropy(logits, targets) # 交叉熵计算损失
        else:
            loss = None
        return logits ,loss 
    
    def genrate(self, token_sequ, max_new_tokens):
        '''token_sequ是已知的上文，max_new_tokens是续写的长度'''
        for _ in range(max_new_tokens):
            '''迭代生成，每次生成一个字符放在序列最后，去掉序列内第一个字符'''
            token_input = token_sequ[:, -block_size:] # 取序列最后block_size个字符给一个新变量token_input(注意：不会影响token_sequ)
            logits, loss =self.forward(token_input)
            '''
            比如输入一句话(序列)：'你好我是ai聊' 那么模型并行预测每个字的下一个字
            比如 '你' 预测-> '好'
            '好' 预测-> '我'
            ...
            最后一个 '聊' 预测-> '天'
            所有最后真正我们需要的只有 '天' 只取最后一个
            '''
            logits = logits[:, -1, :]# 只取序列最后一个字符的logits(概率分布的向量)
            probs = F.softmax(logits, dim=-1) # 对最后一个字符的概率分布(向量维度)进行softmax得到准确概率 把 概率分布向量 变成 one-hot向量
            next_token = torch.multinomial(probs, num_samples=1).to(device) # 把 one-hot向量 -> 整数token
            token_sequ = torch.cat((token_sequ, next_token), dim=1) # 将新字符添加到序列中
        new_tokens = token_sequ[:, -max_new_tokens:] # 去掉序列内第一个字符
        return new_tokens

def main():
    print(f"训练内容:{file_name}")

    model = LanguageModel() # 实例化模型
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # 打印模型有多少参数

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 优化器

    # 训练循环
    for i in range(max_iters):
        if i % eval_iters == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train") # 拿到训练数据
        logits, loss = model(xb, yb) # 前馈运算
        optimizer.zero_grad(set_to_none=True) # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

    print("训练结束，开始生成内容")
    max_new_tokens = 70
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)

    # 上文内容 建立一个形状正确全是0的张量
    context = torch.zeros((1, block_size), dtype=torch.long, device=device) # (B, T) B=1 T=block_size
    context[0, :] = val_data[start_idx: start_idx+block_size] # 把上文内容赋值给context
    context_str = decode(context[0].tolist()) # 把上文内容转成字符串 一维张量
    wrapped_context_str = textwrap.fill(context_str, width=70) # 自动换行

    # 真正的下文内容
    real_next_token = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
    real_next_token[0, :] = val_data[start_idx+block_size: start_idx+block_size+max_new_tokens] # 把下文内容赋值给real_next_token
    real_next_token_str = decode(real_next_token[0].tolist()) # 把下文内容转成字符串 一维张量
    wrapped_real_next_token_str = textwrap.fill(real_next_token_str, width=70) # 自动换行

    generated_tokens = model.genrate(context, max_new_tokens=max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=70)

    print(f"上文内容: {wrapped_context_str}")
    print(f"下文内容: {wrapped_real_next_token_str}")
    print(f"生成内容: {wrapped_generated_str}")

# 运行
main()



