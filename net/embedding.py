import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap # 自动换行

'''
第一步 完成数据预处理

数据预处理 + token embedding + position embedding
# embedding 其实就是把你给的一个数字列表，随机生成向量，
# 在后面模型训练的时候，会学习到词与词之间的相似性关系，并且会调整这些向量的值

最终输入模型的数据是个三维的巨大tensor数据，可以理解为一个空间长方形
长是batch_size也就是有多少句话，
高是词向量的维度，
宽是训练的每句话的长度也就是序列长度
'''

# 超参数
batch_size = 3 # 一次训练多少句话
block_size = 16 # 一句话的字符串长度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embedding = 3 # 嵌入后的维度

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

# x_list = x.tolist() # tensor转成列表
# for str_list in x_list:
#     decode_str = decode(str_list) # 转成字符串
#     print(decode_str) # 看看一个batch输入了哪几个句子

'''上面实现了 一句话字符串 -> 列表数据 的映射 
数据是 一个由 32句话 每句话有8个字符 构成的二维tensor

下面词嵌入token embedding：
    列表数据 -> 向量数据 的映射
将每个字符嵌入成一个向量
最终得到

输入模型的数据 是个三维的巨大tensor数据，可以理解为一个空间长方体
长是 batch_size 即 这个数据内 有多少句话，
宽是 block_size 即 每句话有多少个数(之前将字符串转化成了数字) 也就是序列长度
高是 n_embedding 即 词向量的维度 每个字 嵌入的向量维度
'''
# token_embedding_table = nn.Embedding(vocab_size, n_embedding)
# token_embedding_table = token_embedding_table.to(device)
# x = x.to(device)
# embd = token_embedding_table(x)
# print(embd)

# '''位置嵌入position embedding'''
# position_embedding_table = nn.Embedding(block_size, n_embedding)
# position_embedding_table = position_embedding_table.to(device)
# position_idx = torch.arange(block_size).to(device)
# position_embd = position_embedding_table(position_idx)
# print(position_embd)

