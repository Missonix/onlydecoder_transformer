import json
import torch

file_name = "combined_train.txt"

'''
读取文件
创建词表
保存词表

读取词表
编码解码
转tensor
'''

def read_file(file_name):
    '''读取文件
    创建词表
    保存词表'''
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()

    '''构造词典'''
    chars = sorted(list(set(text))) # 用set去重提取字符，用list封装，用sorted排序
    vocab_size = len(chars) # 词表长度 即 字典里存了多少个不同的字(token)

    # 保存token词典
    with open('token_dict.json', 'w') as f:
        json.dump(chars, f)
    return vocab_size
    

class TokenDict:
    def __init__(self, file_name):
        self.file_name = file_name
        # 初始化时加载词表并创建映射字典
        self.chars = self.get_token_dict()
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def get_token_dict(self):
        '''读取词表'''
        with open(self.file_name, 'r') as f:
            return json.load(f)
    
    def get_vocab_size(self):
        '''获取词表长度'''
        return len(self.chars)

    def encode(self, s):
        '''编码字符串'''
        return [self.stoi[c] for c in s]

    def decode(self, l):
        '''解码列表'''
        return ''.join([self.itos[i] for i in l])

    def get_tensor(self, text):
        '''读取原字符串 转tensor'''
        data = torch.tensor(self.encode(text), dtype=torch.long) # 用长整数表示字符
        return data


if __name__ == "__main__":
    # 先创建词表文件
    read_file(file_name)  # 新增这行来生成词表文件
    
    token_dict = TokenDict('token_dict.json')
    # 读取测试文件内容
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 测试编码解码
    encoded = token_dict.encode(text[:10])  # 编码前10个字符
    decoded = token_dict.decode(encoded)
    dicttensor = token_dict.get_tensor(text[:10])  # 转换前10个字符为tensor
    
    print("原始文本:", text[:10])
    print("编码结果:", encoded)
    print("解码结果:", decoded)
    print("Tensor:", dicttensor)
