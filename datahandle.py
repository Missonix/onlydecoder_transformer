import pandas as pd
import re

def clean_text(text):
    """清理文本，去除多余的空白字符和特殊字符"""
    if pd.isna(text):  # 处理空值
        return ""
    # 去除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符，但保留基本标点
    text = re.sub(r'[^\u4e00-\u9fff\w\s,.!?，。！？、]', '', text)
    return text.strip()

def process_news_csv(csv_path, output_path):
    """处理新闻CSV文件并保存为txt格式"""
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 打开输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 遍历每一行
            total_rows = len(df)
            for idx, row in df.iterrows():
                if idx % 1000 == 0:  # 每处理1000条打印一次进度
                    print(f"处理进度: {idx}/{total_rows}")
                
                # 清理标题和内容
                title = clean_text(row['title'])
                content = clean_text(row['content'])
                
                # 如果标题和内容都不为空，则写入文件
                if title and content:
                    # 写入格式：标题[SEP]内容[EOS]
                    f.write(f"{title}[SEP]{content}[EOS]\n")
        
        print(f"处理完成！数据已保存至: {output_path}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    # 设置输入输出路径
    csv_path = "RenMin_Daily.csv"  # CSV文件路径
    output_path = "datasets/news_data.txt"  # 输出的txt文件路径
    
    # 处理数据
    process_news_csv(csv_path, output_path)

    # with open("news_data.txt", "r", encoding="utf-8") as f:
    #     for line in f:
    #         print(line)
    #         break