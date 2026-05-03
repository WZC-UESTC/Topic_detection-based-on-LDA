import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from datetime import datetime
import numpy as np

# 使用scikit-learn代替gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# NLTK用于文本预处理
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

class TopicModelingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("文本主题检测系统")
        self.root.geometry("1200x800")
        
        # 存储文档的列表
        self.documents = []      # 原始文档
        self.doc_names = []      # 文档名称
        self.processed_docs = [] # 预处理后的文档（字符串列表）
        self.vectorizer = None   # 向量化器
        self.doc_term_matrix = None  # 文档-词矩阵
        self.lda_model = None    # LDA模型
        self.feature_names = None # 特征名称（词）
        self.topics = None       # 检测到的话题
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建所有界面元素"""
        # 标题
        title_label = ttk.Label(
            self.root, 
            text="文本主题检测系统——子琛课设", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=10)
        
        # 工具栏
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(toolbar, text="加载文档", command=self.load_documents).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="加载文件夹", command=self.load_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="开始分析", command=self.start_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="清空所有", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        
        # 参数设置
        ttk.Label(toolbar, text="话题数量:").pack(side=tk.LEFT, padx=(20, 2))
        self.topic_var = tk.StringVar(value="5")
        ttk.Spinbox(toolbar, from_=2, to=20, textvariable=self.topic_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(toolbar, text="每个话题词数:").pack(side=tk.LEFT, padx=(20, 2))
        self.words_var = tk.StringVar(value="10")
        ttk.Spinbox(toolbar, from_=5, to=20, textvariable=self.words_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：文档区域
        left_frame = ttk.LabelFrame(main_frame, text="文档列表", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 文档列表
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.doc_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.doc_listbox.yview)
        self.doc_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.doc_listbox.bind('<<ListboxSelect>>', self.on_document_select)
        
        # 文档内容预览
        content_frame = ttk.LabelFrame(left_frame, text="文档内容预览", padding=5)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.content_text = tk.Text(content_frame, wrap=tk.WORD, height=10)
        content_scroll = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=content_scroll.set)
        
        self.content_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右侧：结果显示区域
        right_frame = ttk.LabelFrame(main_frame, text="话题分析结果", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.result_text = tk.Text(right_frame, wrap=tk.WORD, font=('Courier', 10))
        result_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # 状态栏
        self.status_bar = ttk.Label(
            self.root, 
            text="就绪", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_documents(self):
        """加载单个或多个文档"""
        files = filedialog.askopenfilenames(
            title="选择文档",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if files:
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents.append(content)
                        self.doc_names.append(os.path.basename(file_path))
                except Exception as e:
                    messagebox.showerror("错误", f"无法加载文件 {file_path}\n{str(e)}")
            
            self.update_document_list()
            self.update_status(f"已加载 {len(self.documents)} 个文档")
            messagebox.showinfo("成功", f"成功加载 {len(self.documents)} 个文档")
    
    def load_folder(self):
        """加载整个文件夹"""
        folder = filedialog.askdirectory(title="选择文件夹")
        
        if folder:
            count = 0
            for file in os.listdir(folder):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.documents.append(content)
                            self.doc_names.append(file)
                            count += 1
                    except Exception as e:
                        print(f"无法加载 {file}: {str(e)}")
            
            self.update_document_list()
            self.update_status(f"已从文件夹加载 {count} 个文档")
            if count > 0:
                messagebox.showinfo("成功", f"成功加载 {count} 个文档")
            else:
                messagebox.showwarning("警告", "文件夹中没有找到txt文件")
    
    def start_analysis(self):
        """开始分析（在后台线程中运行）"""
        if len(self.documents) < 2:
            messagebox.showwarning("警告", "至少需要2个文档才能进行分析！")
            return
        
        # 禁用按钮，显示进度
        self.progress.start()
        self.update_status("正在分析中，请稍候...")
        
        # 创建新线程进行分析
        thread = threading.Thread(target=self.analyze_documents)
        thread.daemon = True
        thread.start()
    
    def analyze_documents(self):
        """分析文档（在后台运行）"""
        try:
            # 步骤1：预处理文档
            self.update_status("正在预处理文档...")
            self.preprocess_documents()
            
            # 步骤2：向量化文档
            self.update_status("正在向量化文档...")
            self.vectorize_documents()
            
            # 步骤3：训练LDA模型
            self.update_status("正在训练LDA模型...")
            self.train_lda()
            
            # 步骤4：提取话题
            self.update_status("正在提取话题...")
            self.extract_topics()
            
            # 步骤5：显示结果
            self.root.after(0, self.display_results)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("错误", f"分析失败：{msg}"))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.update_status("分析完成"))
    
    def preprocess_documents(self):
        """预处理文档"""
        self.processed_docs = []
        
        # 下载NLTK数据（如果需要）
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        # 初始化词形还原器
        try:
            lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
        
        # 下载punkt（如果需要）
        try:
            word_tokenize("test")
        except:
            nltk.download('punkt')
        
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
                # 移除短词和停用词
                if len(token) > 2 and token not in stop_words:
                    # 词形还原
                    token = lemmatizer.lemmatize(token)
                    filtered_tokens.append(token)
            
            # 重新组合成文本（scikit-learn需要字符串格式）
            self.processed_docs.append(' '.join(filtered_tokens))
    
    def vectorize_documents(self):
        """向量化文档"""
        num_docs = len(self.processed_docs)
        print(f"文档数量: {num_docs}")
        
        # 根据文档数量智能选择参数
        if num_docs < 3:
            # 只有2-3个文档
            min_df_value = 1
            max_df_value = 1.0
            max_features_value = 500
            print("使用超宽松参数（文档极少）")
        elif num_docs < 10:
            # 3-9个文档
            min_df_value = 1
            max_df_value = 0.9
            max_features_value = 800
            print("使用宽松参数（文档较少）")
        else:
            # 10个以上文档
            min_df_value = 2
            max_df_value = 0.8
            max_features_value = 1000
            print("使用标准参数")
        
        try:
            self.vectorizer = CountVectorizer(
                max_df=max_df_value,
                min_df=min_df_value,
                stop_words='english',
                max_features=max_features_value
            )
            
            self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            print(f"✓ 向量化成功！")
            print(f"  - 词汇表大小: {len(self.feature_names)}")
            print(f"  - 文档-词矩阵形状: {self.doc_term_matrix.shape}")
            
        except ValueError as e:
            print(f"向量化失败，使用最终备选方案: {str(e)}")
            # 最宽松的备选方案
            self.vectorizer = CountVectorizer(
                min_df=1,              # 只要出现一次就保留
                max_df=1.0,            # 不限制最大频率
                stop_words='english',  # 仍然使用停用词
                token_pattern=r'(?u)\b\w+\b'  # 更宽松的词匹配规则
            )
            
            self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f"✓ 使用备选方案成功！")
            print(f"  - 词汇表大小: {len(self.feature_names)}")
    
    def train_lda(self):
        """训练LDA模型"""
        num_topics = int(self.topic_var.get())
        
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=50,
            learning_method='online',
            learning_offset=50.0,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        
        # 训练模型
        self.lda_model.fit(self.doc_term_matrix)
    
    def extract_topics(self):
        """提取话题"""
        num_words = int(self.words_var.get())
        self.topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # 获取该话题中权重最高的词
            top_features_indices = topic.argsort()[:-num_words-1:-1]
            top_features = [self.feature_names[i] for i in top_features_indices]
            top_weights = [topic[i] for i in top_features_indices]
            
            self.topics.append({
                'index': topic_idx,
                'words': top_features,
                'weights': top_weights
            })
    
    def display_results(self):
        """显示分析结果"""
        self.result_text.delete(1.0, tk.END)
        
        # 显示基本信息
        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "                话题检测结果\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")
        
        self.result_text.insert(tk.END, f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.result_text.insert(tk.END, f"文档数量：{len(self.documents)}\n")
        self.result_text.insert(tk.END, f"话题数量：{len(self.topics)}\n")
        self.result_text.insert(tk.END, f"词汇表大小：{len(self.feature_names)}\n\n")
        
        # 显示每个话题
        for topic in self.topics:
            self.result_text.insert(tk.END, f"\n📌 话题 {topic['index']}:\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            
            words_with_weights = zip(topic['words'], topic['weights'])
            for word, weight in words_with_weights:
                # 格式化显示
                self.result_text.insert(tk.END, f"   {word:<15} {weight:.4f}\n")
        
        # 显示每个文档的主要话题
        self.result_text.insert(tk.END, "\n\n" + "=" * 60 + "\n")
        self.result_text.insert(tk.END, "              文档-话题分布\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # 获取文档的话题分布
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        
        for i, dist in enumerate(doc_topic_dist):
            if i < len(self.doc_names):
                main_topic = np.argmax(dist)
                confidence = dist[main_topic]
                
                self.result_text.insert(tk.END, f"📄 {self.doc_names[i]}\n")
                self.result_text.insert(tk.END, f"   主要话题：话题 {main_topic} (置信度：{confidence:.2%})\n")
                
                # 显示所有话题的概率
                self.result_text.insert(tk.END, "   所有话题概率：\n")
                for t_idx, prob in enumerate(dist):
                    self.result_text.insert(tk.END, f"     话题 {t_idx}: {prob:.2%}\n")
                self.result_text.insert(tk.END, "-" * 40 + "\n")
    
    def clear_all(self):
        """清空所有文档"""
        if messagebox.askyesno("确认", "确定要清空所有文档吗？"):
            self.documents = []
            self.doc_names = []
            self.processed_docs = []
            self.vectorizer = None
            self.doc_term_matrix = None
            self.lda_model = None
            self.feature_names = None
            self.topics = None
            
            self.doc_listbox.delete(0, tk.END)
            self.content_text.delete(1.0, tk.END)
            self.result_text.delete(1.0, tk.END)
            self.update_status("已清空")
    
    def update_document_list(self):
        """更新文档列表显示"""
        self.doc_listbox.delete(0, tk.END)
        for name in self.doc_names:
            self.doc_listbox.insert(tk.END, name)
    
    def on_document_select(self, event):
        """当选择文档时显示内容"""
        selection = self.doc_listbox.curselection()
        if selection:
            index = selection[0]
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(1.0, self.documents[index])
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def run(self):
        """运行程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TopicModelingApp()
    app.run()