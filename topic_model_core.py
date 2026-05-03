# topic_model_core.py
"""
主题检测核心模块 - 从 main.py 中提取的核心功能
可被 GUI 和测试脚本共用
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string


class TopicModelCore:
    """主题检测核心引擎（从你的 main.py 提取）"""
    
    def __init__(self):
        self.documents = []
        self.doc_names = []
        self.processed_docs = []
        self.vectorizer = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.feature_names = None
        self.topics = None
        
    def load_documents(self, docs, names=None):
        """加载文档"""
        self.documents = docs
        if names:
            self.doc_names = names
        else:
            self.doc_names = [f"doc_{i+1}.txt" for i in range(len(docs))]
    
    def preprocess_documents(self):
        """预处理文档（与你的 main.py 中相同）"""
        self.processed_docs = []
        
        # 获取停用词
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        # 初始化词形还原器
        try:
            lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet', quiet=True)
            lemmatizer = WordNetLemmatizer()
        
        # 下载punkt
        try:
            word_tokenize("test")
        except:
            nltk.download('punkt', quiet=True)
        
        for doc in self.documents:
            # 转换为小写
            doc = doc.lower()
            # 移除数字
            doc = re.sub(r'\d+', '', doc)
            # 移除标点
            doc = doc.translate(str.maketrans('', '', string.punctuation))
            # 分词
            tokens = word_tokenize(doc)
            # 过滤
            filtered_tokens = []
            for token in tokens:
                if len(token) > 2 and token not in stop_words:
                    token = lemmatizer.lemmatize(token)
                    filtered_tokens.append(token)
            # 重新组合
            self.processed_docs.append(' '.join(filtered_tokens))
        
        return self.processed_docs
    
    def vectorize_documents(self):
        """向量化文档"""
        num_docs = len(self.processed_docs)
        
        if num_docs < 3:
            min_df = 1
            max_df = 1.0
        else:
            min_df = 2
            max_df = 0.8
        
        self.vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=min_df,
            stop_words='english',
            max_features=1000
        )
        
        self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return self.doc_term_matrix
    
    def train_lda(self, num_topics=5, max_iter=50):
        """训练LDA模型"""
        n_topics = min(num_topics, len(self.processed_docs))
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.lda_model.fit(self.doc_term_matrix)
        return self.lda_model
    
    def extract_topics(self, num_words=10):
        """提取话题"""
        num_words = min(num_words, len(self.feature_names))
        self.topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-num_words-1:-1]
            top_features = [self.feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            
            self.topics.append({
                'index': topic_idx,
                'words': top_features,
                'weights': top_weights
            })
        
        return self.topics
    
    def get_doc_topic_distribution(self):
        """获取文档-话题分布"""
        return self.lda_model.transform(self.doc_term_matrix)
    
    def run_pipeline(self, docs, num_topics=5, num_words=10, max_iter=50):
        """运行完整流程"""
        self.load_documents(docs)
        self.preprocess_documents()
        self.vectorize_documents()
        self.train_lda(num_topics, max_iter)
        self.extract_topics(num_words)
        return self.topics, self.get_doc_topic_distribution()