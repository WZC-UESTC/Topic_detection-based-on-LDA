# download_bbc_dataset.py
"""
自动下载 BBC News 数据集
来源: http://mlg.ucd.ie/datasets/bbc.html
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path


def download_bbc_dataset(output_dir="./test/datasets/bbc"):
    """
    下载并解压 BBC News 数据集
    
    BBC News 数据集包含:
    - 5个主题: business, entertainment, politics, sport, tech
    - 共 2,225 篇文档
    """
    
    dataset_url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    dataset_name = "bbc-fulltext.zip"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("📥 BBC News 数据集下载工具")
    print("="*60)
    print(f"\n数据集信息:")
    print(f"  - 来源: {dataset_url}")
    print(f"  - 主题: business, entertainment, politics, sport, tech")
    print(f"  - 文档数: 约 2,225 篇")
    print(f"  - 保存位置: {output_dir}")
    
    zip_path = os.path.join(output_dir, dataset_name)
    
    # 检查是否已下载
    if os.path.exists(zip_path):
        print(f"\n✅ 数据集已存在: {zip_path}")
        return extract_bbc_dataset(zip_path, output_dir)
    
    # 下载数据集
    print(f"\n开始下载数据集...")
    print("(文件大小约 5.5MB，请稍候...)")
    
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 下载完成: {zip_path}")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return None
    
    return extract_bbc_dataset(zip_path, output_dir)


def extract_bbc_dataset(zip_path, output_dir):
    """解压 BBC 数据集"""
    print(f"\n📂 解压数据集中...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ 解压完成")
        
        # 清理 zip 文件
        os.remove(zip_path)
        
        bbc_dir = os.path.join(output_dir, "bbc")
        return count_bbc_documents(bbc_dir)
        
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return None


def count_bbc_documents(bbc_dir):
    """统计 BBC 数据集的文档数量"""
    print(f"\n📊 数据集统计:")
    
    topic_stats = {}
    total_docs = 0
    
    for topic in os.listdir(bbc_dir):
        topic_path = os.path.join(bbc_dir, topic)
        if os.path.isdir(topic_path):
            doc_count = len([f for f in os.listdir(topic_path) if f.endswith('.txt')])
            topic_stats[topic] = doc_count
            total_docs += doc_count
    
    print(f"  - 总文档数: {total_docs}")
    for topic, count in topic_stats.items():
        print(f"    {topic}: {count} 篇")
    
    return bbc_dir, topic_stats, total_docs


def load_bbc_documents(bbc_dir, samples_per_topic=None, topics=None):
    """
    加载 BBC 数据集文档
    
    参数:
        bbc_dir: BBC 数据集目录
        samples_per_topic: 每个主题采样数量（None表示全部）
        topics: 指定要加载的主题列表（None表示全部）
    """
    documents = []
    doc_names = []
    doc_labels = []
    
    if topics is None:
        topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    import random
    random.seed(42)  # 固定随机种子，保证可重复
    
    for topic in topics:
        topic_path = os.path.join(bbc_dir, topic)
        if not os.path.exists(topic_path):
            print(f"警告: 主题目录不存在 - {topic_path}")
            continue
        
        files = [f for f in os.listdir(topic_path) if f.endswith('.txt')]
        
        # 采样
        if samples_per_topic and samples_per_topic < len(files):
            files = random.sample(files, samples_per_topic)
        
        for file in files:
            file_path = os.path.join(topic_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    doc_names.append(f"{topic}/{file}")
                    doc_labels.append(topic)
            except Exception as e:
                print(f"读取失败 {file_path}: {e}")
    
    return documents, doc_names, doc_labels


if __name__ == "__main__":
    # 测试下载
    result = download_bbc_dataset()
    if result:
        print("\n✅ 数据集准备完成！")