# test_with_bbc.py
"""
使用 BBC News 数据集测试你的主题检测系统
直接调用你的 main.py 中的核心功能
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from collections import Counter

# 导入你的核心模块
from topic_model_core import TopicModelCore
from download_bbc_dataset import download_bbc_dataset, load_bbc_documents


class BBCDatasetTester:
    """BBC 数据集测试器"""
    
    def __init__(self, output_dir="./test/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def run_test(self, samples_per_topic=20, num_topics=5, num_words=10):
        """
        运行 BBC 数据集测试
        
        参数:
            samples_per_topic: 每个主题采样多少篇文档
            num_topics: 检测的话题数量
            num_words: 每个话题显示的关键词数量
        """
        print("\n" + "="*70)
        print("🎯 BBC News 数据集主题检测测试")
        print("="*70)
        
        # 1. 准备数据集
        print("\n📥 步骤1: 准备 BBC News 数据集")
        print("-"*50)
        
        result = download_bbc_dataset()
        if not result:
            print("❌ 数据集准备失败")
            return
        
        bbc_dir, stats, total = result
        
        # 2. 加载文档
        print(f"\n📖 步骤2: 加载文档 (每个主题 {samples_per_topic} 篇)")
        print("-"*50)
        
        documents, doc_names, doc_labels = load_bbc_documents(
            bbc_dir, 
            samples_per_topic=samples_per_topic
        )
        
        print(f"✅ 成功加载 {len(documents)} 篇文档")
        
        # 显示主题分布
        label_counts = Counter(doc_labels)
        print(f"\n📊 主题分布:")
        for label, count in sorted(label_counts.items()):
            print(f"   {label}: {count} 篇")
        
        # 3. 创建模型并运行分析
        print(f"\n🔄 步骤3: 运行主题检测 (话题数={num_topics})")
        print("-"*50)
        
        model = TopicModelCore()
        
        start_time = time.time()
        
        # 运行完整流程
        topics, doc_topic_dist = model.run_pipeline(
            docs=documents,
            num_topics=num_topics,
            num_words=num_words,
            max_iter=50
        )
        
        elapsed_time = time.time() - start_time
        
        # 4. 显示检测到的话题
        print(f"\n📌 步骤4: 检测到的话题")
        print("-"*50)
        
        for topic in topics:
            words_str = ", ".join(topic['words'])
            print(f"\n   话题 {topic['index']+1}:")
            print(f"   关键词: {words_str}")
        
        # 5. 评估准确率
        print(f"\n📈 步骤5: 准确率评估")
        print("-"*50)
        
        predicted_topics = np.argmax(doc_topic_dist, axis=1)
        
        # 计算最优映射
        accuracy = self._calculate_accuracy(predicted_topics, doc_labels)
        
        print(f"\n   测试结果:")
        print(f"   ⏱️  运行时间: {elapsed_time:.2f} 秒")
        print(f"   ✅ 准确率: {accuracy:.2%}")
        
        # 6. 显示详细分配（前20个文档）
        print(f"\n📋 文档分配详情 (前20个):")
        print("-"*50)
        
        # 找到话题到主题的映射
        mapping = self._get_topic_to_label_mapping(predicted_topics, doc_labels)
        
        for i in range(min(20, len(documents))):
            pred_topic = predicted_topics[i]
            mapped_label = mapping.get(pred_topic, "未知")
            is_correct = "✅" if mapped_label == doc_labels[i] else "❌"
            print(f"   {doc_names[i][:40]:40} → 话题{pred_topic+1} ({mapped_label}) 真实:{doc_labels[i]} {is_correct}")
        
        # 7. 保存结果
        self._save_results(topics, accuracy, elapsed_time, samples_per_topic, num_topics)
        
        return topics, accuracy
    
    def _calculate_accuracy(self, predicted_topics, true_labels):
        """计算准确率（通过最优话题到主题映射）"""
        from collections import defaultdict
        
        # 构建映射表
        mapping = defaultdict(lambda: defaultdict(int))
        for pred, true in zip(predicted_topics, true_labels):
            mapping[pred][true] += 1
        
        # 贪心匹配：每个预测话题映射到最常见的真实标签
        topic_to_label = {}
        used_labels = set()
        
        # 按匹配度排序
        matches = []
        for pred, counts in mapping.items():
            if counts:
                best_label = max(counts, key=counts.get)
                matches.append((counts[best_label], pred, best_label))
        
        matches.sort(reverse=True)
        
        for _, pred, label in matches:
            if label not in used_labels:
                topic_to_label[pred] = label
                used_labels.add(label)
        
        # 计算准确率
        correct = sum(1 for pred, true in zip(predicted_topics, true_labels) 
                     if topic_to_label.get(pred) == true)
        
        return correct / len(true_labels) if true_labels else 0.0
    
    def _get_topic_to_label_mapping(self, predicted_topics, true_labels):
        """获取话题到主题的映射"""
        from collections import defaultdict
        
        mapping = defaultdict(lambda: defaultdict(int))
        for pred, true in zip(predicted_topics, true_labels):
            mapping[pred][true] += 1
        
        topic_to_label = {}
        for pred, counts in mapping.items():
            if counts:
                topic_to_label[pred] = max(counts, key=counts.get)
        
        return topic_to_label
    
    def _save_results(self, topics, accuracy, elapsed_time, samples_per_topic, num_topics):
        """保存测试结果"""
        report_path = os.path.join(self.output_dir, f"bbc_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 BBC News 数据集测试报告\n\n")
            f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📋 测试配置\n\n")
            f.write("| 参数 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 每个主题采样数 | {samples_per_topic} |\n")
            f.write(f"| 检测话题数 | {num_topics} |\n")
            f.write(f"| 每个话题词数 | {len(topics[0]['words']) if topics else 0} |\n\n")
            
            f.write("## 📈 测试结果\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 运行时间 | {elapsed_time:.2f} 秒 |\n")
            f.write(f"| **准确率** | **{accuracy:.2%}** |\n\n")
            
            f.write("## 🎯 检测到的话题\n\n")
            for topic in topics:
                f.write(f"### 话题 {topic['index']+1}\n\n")
                f.write("| 关键词 | 权重 |\n")
                f.write("|--------|------|\n")
                for word, weight in zip(topic['words'][:10], topic['weights'][:10]):
                    f.write(f"| {word} | {weight:.4f} |\n")
                f.write("\n")
            
            # 添加结论
            f.write("## 💡 结论\n\n")
            if accuracy > 0.7:
                f.write("✅ **模型表现优秀**！准确率超过70%，可以很好地识别BBC新闻的主题。\n")
            elif accuracy > 0.5:
                f.write("⚠️ **模型表现中等**。建议增加训练文档或调整话题数量。\n")
            else:
                f.write("❌ **模型需要优化**。建议检查文档质量或调整参数。\n")
        
        print(f"\n📄 测试报告已保存: {report_path}")
    
    def run_multiple_configs(self):
        """运行多种配置组合测试"""
        print("\n" + "="*70)
        print("🔬 多配置对比测试")
        print("="*70)
        
        configs = [
            {"samples": 10, "topics": 3, "name": "小样本 + 3主题"},
            {"samples": 20, "topics": 5, "name": "中样本 + 5主题"},
            {"samples": 50, "topics": 5, "name": "大样本 + 5主题"},
            {"samples": 30, "topics": 8, "name": "中样本 + 8主题"},
        ]
        
        results_summary = []
        
        for config in configs:
            print(f"\n{'='*50}")
            print(f"测试配置: {config['name']}")
            print(f"{'='*50}")
            
            # 重新加载数据（不同采样）
            result = download_bbc_dataset()
            if not result:
                continue
            
            bbc_dir, stats, total = result
            
            documents, doc_names, doc_labels = load_bbc_documents(
                bbc_dir, 
                samples_per_topic=config["samples"]
            )
            
            if len(documents) == 0:
                continue
            
            model = TopicModelCore()
            start_time = time.time()
            
            topics, doc_topic_dist = model.run_pipeline(
                docs=documents,
                num_topics=config["topics"],
                num_words=10,
                max_iter=50
            )
            
            elapsed = time.time() - start_time
            
            predicted = np.argmax(doc_topic_dist, axis=1)
            accuracy = self._calculate_accuracy(predicted, doc_labels)
            
            results_summary.append({
                "name": config["name"],
                "samples": config["samples"],
                "topics": config["topics"],
                "accuracy": accuracy,
                "time": elapsed
            })
            
            print(f"\n   准确率: {accuracy:.2%}")
            print(f"   耗时: {elapsed:.2f}秒")
        
        # 打印对比结果
        print("\n" + "="*70)
        print("📊 多配置测试对比结果")
        print("="*70)
        print(f"\n{'配置':<20} {'样本数':<10} {'话题数':<10} {'准确率':<12} {'耗时':<10}")
        print("-"*70)
        for r in results_summary:
            print(f"{r['name']:<20} {r['samples']:<10} {r['topics']:<10} {r['accuracy']:.2%}       {r['time']:.2f}s")
        
        # 找出最佳配置
        best = max(results_summary, key=lambda x: x['accuracy'])
        print(f"\n🏆 最佳配置: {best['name']} (准确率: {best['accuracy']:.2%})")


def main():
    """主函数"""
    tester = BBCDatasetTester()
    
    print("\n" + "="*70)
    print("🚀 BBC News 数据集自动化测试")
    print("="*70)
    print("\n请选择测试模式:")
    print("  1. 快速测试 (每个主题20篇)")
    print("  2. 完整测试 (每个主题50篇)")
    print("  3. 多配置对比测试")
    print("  4. 自定义测试")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        tester.run_test(samples_per_topic=20, num_topics=5)
    elif choice == '2':
        tester.run_test(samples_per_topic=50, num_topics=5)
    elif choice == '3':
        tester.run_multiple_configs()
    elif choice == '4':
        samples = int(input("每个主题采样数量: ") or "20")
        topics = int(input("检测话题数量: ") or "5")
        tester.run_test(samples_per_topic=samples, num_topics=topics)
    else:
        print("无效选择，运行默认测试")
        tester.run_test(samples_per_topic=20, num_topics=5)


if __name__ == "__main__":
    main()