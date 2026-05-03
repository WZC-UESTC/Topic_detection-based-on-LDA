import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from datetime import datetime
import numpy as np

# 使用scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# NLTK用于文本预处理
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

# 可视化相关
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TopicModelingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("文本主题检测系统")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')
        
        # 存储文档的列表
        self.documents = []
        self.doc_names = []
        self.all_doc_names = []
        self.processed_docs = []
        self.vectorizer = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.feature_names = None
        self.topics = None
        
        # 设置样式
        self.setup_styles()
        
        # 创建界面
        self.create_menu()
        self.create_widgets()
        
        # 缓存变量
        self._filter_timer = None
        self._plot_cache = None
    
    def setup_styles(self):
        """设置现代样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#e67e22',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'white': '#ffffff',
            'gray': '#95a5a6',
            'teal': '#1abc9c'
        }
        
        style.configure('Title.TLabel', 
                       font=('微软雅黑', 20, 'bold'),
                       foreground=self.colors['primary'],
                       background='#f5f5f5')
        
        style.configure('Card.TLabelframe',
                       relief='flat',
                       borderwidth=1,
                       background=self.colors['white'])
        style.configure('Card.TLabelframe.Label',
                       font=('微软雅黑', 11, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['white'])
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载文档", command=self.load_documents, accelerator="Ctrl+O")
        file_menu.add_command(label="加载文件夹", command=self.load_folder, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="导出结果", command=self.export_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit, accelerator="Ctrl+Q")
        
        analyze_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analyze_menu)
        analyze_menu.add_command(label="开始分析", command=self.start_analysis, accelerator="F5")
        analyze_menu.add_command(label="清空所有", command=self.clear_all, accelerator="Ctrl+Del")
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="刷新图表", command=self.refresh_charts)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
        # 绑定快捷键
        self.root.bind('<Control-o>', lambda e: self.load_documents())
        self.root.bind('<Control-O>', lambda e: self.load_folder())
        self.root.bind('<Control-s>', lambda e: self.export_results())
        self.root.bind('<F5>', lambda e: self.start_analysis())
        self.root.bind('<Control-Delete>', lambda e: self.clear_all())
    
    def create_widgets(self):
        """创建所有界面元素"""
        
        # ========== 顶部工具栏 ==========
        top_toolbar = tk.Frame(self.root, bg=self.colors['primary'], height=60)
        top_toolbar.pack(fill=tk.X, side=tk.TOP)
        top_toolbar.pack_propagate(False)
        
        # 标题
        title_label = tk.Label(top_toolbar, 
                               text="📊 文本主题检测系统——子琛综合课设",
                               font=('微软雅黑', 16, 'bold'),
                               bg=self.colors['primary'],
                               fg='white')
        title_label.pack(side=tk.LEFT, padx=20)
        
        # 按钮区域
        button_frame = tk.Frame(top_toolbar, bg=self.colors['primary'])
        button_frame.pack(side=tk.RIGHT, padx=20)
        
        self.btn_analyze = tk.Button(button_frame,
                                     text="🚀 开始分析",
                                     command=self.start_analysis,
                                     font=('微软雅黑', 10, 'bold'),
                                     bg=self.colors['teal'],
                                     fg='white',
                                     padx=20, pady=5,
                                     relief='flat',
                                     cursor='hand2')
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        self.btn_load = tk.Button(button_frame,
                                  text="📂 加载文档",
                                  command=self.load_documents,
                                  font=('微软雅黑', 10),
                                  bg=self.colors['secondary'],
                                  fg='white',
                                  padx=15, pady=5,
                                  relief='flat',
                                  cursor='hand2')
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = tk.Button(button_frame,
                                   text="🗑️ 清空",
                                   command=self.clear_all,
                                   font=('微软雅黑', 10),
                                   bg=self.colors['warning'],
                                   fg='white',
                                   padx=15, pady=5,
                                   relief='flat',
                                   cursor='hand2')
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # ========== 参数栏 ==========
        param_frame = tk.Frame(self.root, bg=self.colors['white'], height=40)
        param_frame.pack(fill=tk.X, side=tk.TOP, padx=20, pady=(10, 0))
        param_frame.pack_propagate(False)
        
        param_inner = tk.Frame(param_frame, bg=self.colors['white'])
        param_inner.pack(fill=tk.BOTH, expand=True, padx=15)
        
        # 话题数量
        tk.Label(param_inner, text="📈 话题数量：", 
                font=('微软雅黑', 10),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(side=tk.LEFT)
        
        self.topic_var = tk.StringVar(value="5")
        topic_spinbox = tk.Spinbox(param_inner, from_=2, to=20, 
                                   textvariable=self.topic_var,
                                   width=6, font=('微软雅黑', 10),
                                   bg='white', relief='solid', bd=1)
        topic_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        # 每个话题词数
        tk.Label(param_inner, text="📝 每个话题词数：", 
                font=('微软雅黑', 10),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(side=tk.LEFT)
        
        self.words_var = tk.StringVar(value="10")
        words_spinbox = tk.Spinbox(param_inner, from_=5, to=20,
                                   textvariable=self.words_var,
                                   width=6, font=('微软雅黑', 10),
                                   bg='white', relief='solid', bd=1)
        words_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        # 文档统计
        self.stats_label = tk.Label(param_inner, 
                                    text="📄 已加载 0 个文档",
                                    font=('微软雅黑', 10),
                                    bg=self.colors['white'],
                                    fg=self.colors['success'])
        self.stats_label.pack(side=tk.RIGHT)
        
        # ========== 主内容区域 ==========
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 选项卡1：文档管理
        docs_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(docs_tab, text="📁 文档管理")
        
        # 选项卡2：话题分析
        topics_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(topics_tab, text="📌 话题分析")
        
        # 选项卡3：可视化
        viz_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(viz_tab, text="📊 可视化")
        
        # ========== 文档管理界面 ==========
        self.create_docs_tab(docs_tab)
        
        # ========== 话题分析界面 ==========
        self.create_topics_tab(topics_tab)
        
        # ========== 可视化界面 ==========
        self.create_viz_tab(viz_tab)
        
        # ========== 底部状态栏 ==========
        status_frame = tk.Frame(self.root, bg=self.colors['gray'], height=28)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_bar = tk.Label(status_frame,
                                   text="✅ 就绪",
                                   font=('微软雅黑', 9),
                                   bg=self.colors['gray'],
                                   fg='white',
                                   anchor=tk.W,
                                   padx=10)
        self.status_bar.pack(fill=tk.BOTH, expand=True)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=120)
        self.progress.pack(side=tk.RIGHT, padx=10, pady=4)
    
    def create_docs_tab(self, parent):
        """创建文档管理选项卡"""
        # 左侧文档列表
        left_frame = tk.Frame(parent, bg=self.colors['white'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 搜索框
        search_frame = tk.Frame(left_frame, bg=self.colors['white'])
        search_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(search_frame, text="🔍 搜索：", 
                font=('微软雅黑', 9),
                bg=self.colors['white']).pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        # 使用 trace_add 替代 trace（修复警告）
        self.search_var.trace_add('write', lambda *args: self.filter_documents())
        
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                font=('微软雅黑', 9),
                                bg='#f8f9fa', relief='solid', bd=1)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 文档列表
        list_frame = tk.Frame(left_frame, bg=self.colors['white'])
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.doc_listbox = tk.Listbox(list_frame,
                                      font=('微软雅黑', 10),
                                      bg='#f8f9fa',
                                      fg='#2c3e50',
                                      selectbackground='#3498db',
                                      selectforeground='white',
                                      relief='flat',
                                      highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.doc_listbox.yview)
        self.doc_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.doc_listbox.bind('<<ListboxSelect>>', self.on_document_select)
        
        # 右侧内容预览
        right_frame = tk.Frame(parent, bg=self.colors['white'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(right_frame, text="📄 文档内容预览",
                font=('微软雅黑', 10, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(anchor=tk.W, pady=(0, 5))
        
        self.content_text = tk.Text(right_frame,
                                    wrap=tk.WORD,
                                    font=('微软雅黑', 10),
                                    bg='#f8f9fa',
                                    relief='flat',
                                    highlightthickness=1)
        content_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=content_scroll.set)
        
        self.content_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_topics_tab(self, parent):
        """创建话题分析选项卡"""
        # 左侧话题列表
        left_frame = tk.Frame(parent, bg=self.colors['white'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(left_frame, text="🎯 检测到的话题",
                font=('微软雅黑', 10, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(anchor=tk.W, pady=(0, 5))
        
        self.topics_listbox = tk.Listbox(left_frame,
                                         font=('微软雅黑', 10),
                                         bg='#f8f9fa',
                                         fg='#2c3e50',
                                         selectbackground='#3498db',
                                         selectforeground='white',
                                         height=12)
        topics_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.topics_listbox.yview)
        self.topics_listbox.configure(yscrollcommand=topics_scroll.set)
        
        self.topics_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        topics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.topics_listbox.bind('<<ListboxSelect>>', self.on_topic_select)
        
        # 右侧话题详情
        right_frame = tk.Frame(parent, bg=self.colors['white'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(right_frame, text="📋 话题详情",
                font=('微软雅黑', 10, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(anchor=tk.W, pady=(0, 5))
        
        self.topic_details = tk.Text(right_frame,
                                     wrap=tk.WORD,
                                     font=('Consolas', 10),
                                     bg='#f8f9fa',
                                     relief='flat',
                                     height=12)
        details_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.topic_details.yview)
        self.topic_details.configure(yscrollcommand=details_scroll.set)
        
        self.topic_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 文档-话题分布表格
        dist_frame = ttk.LabelFrame(parent, text="📊 文档-话题分布", style='Card.TLabelframe')
        dist_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))
        
        dist_container = tk.Frame(dist_frame, bg=self.colors['white'])
        dist_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('文档', '主要话题', '置信度')
        self.doc_tree = ttk.Treeview(dist_container, columns=columns, show='headings', height=6)
        
        self.doc_tree.heading('文档', text='文档')
        self.doc_tree.heading('主要话题', text='主要话题')
        self.doc_tree.heading('置信度', text='置信度')
        
        self.doc_tree.column('文档', width=250)
        self.doc_tree.column('主要话题', width=100)
        self.doc_tree.column('置信度', width=80)
        
        tree_scroll = ttk.Scrollbar(dist_container, orient=tk.VERTICAL, command=self.doc_tree.yview)
        self.doc_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.doc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_viz_tab(self, parent):
        """创建可视化选项卡"""
        # matplotlib图形
        self.figure = Figure(figsize=(10, 7), dpi=80, facecolor='#f5f5f5')
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制按钮
        control_frame = tk.Frame(parent, bg=self.colors['white'])
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        buttons = [
            ("📊 话题分布图", self.plot_topic_distribution),
            ("📈 文档-话题热力图", self.plot_heatmap),
            ("🎯 关键词权重图", self.plot_keywords)
        ]
        
        for text, cmd in buttons:
            tk.Button(control_frame, text=text,
                      command=cmd,
                      font=('微软雅黑', 9),
                      bg=self.colors['secondary'],
                      fg='white',
                      padx=12, pady=4,
                      relief='flat',
                      cursor='hand2').pack(side=tk.LEFT, padx=5)
    
    def filter_documents(self):
        """防抖过滤文档列表"""
        if self._filter_timer:
            self.root.after_cancel(self._filter_timer)
        self._filter_timer = self.root.after(300, self._do_filter)
    
    def _do_filter(self):
        """执行文档过滤"""
        search_term = self.search_var.get().lower().strip()
        if not search_term:
            self.update_document_list()
            return
        
        filtered = [name for name in self.all_doc_names if search_term in name.lower()]
        self.doc_listbox.delete(0, tk.END)
        for name in filtered[:100]:  # 限制显示数量
            self.doc_listbox.insert(tk.END, name)
    
    def update_status(self, message, is_error=False):
        """更新状态栏"""
        icon = "❌" if is_error else "✅" if message == "分析完成" else "🔄"
        self.status_bar.config(text=f"{icon} {message}")
        self.root.update_idletasks()
    
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
                        if content.strip():
                            self.documents.append(content)
                            self.doc_names.append(os.path.basename(file_path))
                except Exception as e:
                    messagebox.showerror("错误", f"无法加载文件 {file_path}\n{str(e)}")
            
            self.all_doc_names = self.doc_names.copy()
            self.update_document_list()
            self.stats_label.config(text=f"📄 已加载 {len(self.documents)} 个文档")
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
                            if content.strip():
                                self.documents.append(content)
                                self.doc_names.append(file)
                                count += 1
                    except Exception as e:
                        print(f"无法加载 {file}: {str(e)}")
            
            self.all_doc_names = self.doc_names.copy()
            self.update_document_list()
            self.stats_label.config(text=f"📄 已加载 {len(self.documents)} 个文档")
            self.update_status(f"已从文件夹加载 {count} 个文档")
            messagebox.showinfo("成功", f"成功加载 {count} 个文档")
    
    def update_document_list(self):
        """更新文档列表显示"""
        self.doc_listbox.delete(0, tk.END)
        for name in self.doc_names[:200]:  # 限制显示数量
            self.doc_listbox.insert(tk.END, name)
    
    def on_document_select(self, event):
        """选择文档时显示内容"""
        selection = self.doc_listbox.curselection()
        if selection:
            index = selection[0]
            doc_name = self.doc_listbox.get(index)
            try:
                doc_index = self.doc_names.index(doc_name)
                self.content_text.delete(1.0, tk.END)
                self.content_text.insert(1.0, self.documents[doc_index])
            except ValueError:
                pass
    
    def on_topic_select(self, event):
        """选择话题时显示详情"""
        selection = self.topics_listbox.curselection()
        if selection and self.topics:
            idx = selection[0]
            if idx < len(self.topics):
                topic = self.topics[idx]
                self.topic_details.delete(1.0, tk.END)
                self.topic_details.insert(tk.END, f"话题 {idx + 1} 详情\n{'='*40}\n\n")
                
                for word, weight in zip(topic['words'], topic['weights']):
                    bar_len = int(weight * 25)
                    bar = "█" * bar_len + "░" * (25 - bar_len)
                    self.topic_details.insert(tk.END, f"{word:<15} [{bar}] {weight:.4f}\n")
    
    def start_analysis(self):
        """开始分析"""
        if len(self.documents) < 2:
            messagebox.showwarning("警告", "至少需要2个文档才能进行分析！")
            return
        
        self.progress.start()
        self.update_status("正在分析中...")
        self.btn_analyze.config(state='disabled')
        
        thread = threading.Thread(target=self._analyze_worker)
        thread.daemon = True
        thread.start()
    
    def _analyze_worker(self):
        """分析工作线程"""
        try:
            self.update_status("预处理文档...")
            self._preprocess_documents()
            
            self.update_status("向量化文档...")
            self._vectorize_documents()
            
            self.update_status("训练LDA模型...")
            self._train_lda()
            
            self.update_status("提取话题...")
            self._extract_topics()
            
            self.root.after(0, self._on_analysis_complete)
        except Exception as e:
            self.root.after(0, lambda: self._on_analysis_error(str(e)))
    
    def _on_analysis_complete(self):
        """分析完成"""
        self.display_results()
        self.progress.stop()
        self.btn_analyze.config(state='normal')
        self.update_status("分析完成")
    
    def _on_analysis_error(self, error_msg):
        """分析错误"""
        self.progress.stop()
        self.btn_analyze.config(state='normal')
        self.update_status("分析失败", True)
        messagebox.showerror("错误", f"分析失败：{error_msg}")
    
    def _preprocess_documents(self):
        """预处理文档"""
        self.processed_docs = []
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        try:
            lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet', quiet=True)
            lemmatizer = WordNetLemmatizer()
        
        try:
            word_tokenize("test")
        except:
            nltk.download('punkt', quiet=True)
        
        for doc in self.documents:
            doc = doc.lower()
            doc = re.sub(r'\d+', '', doc)
            doc = doc.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(doc)
            
            filtered = [lemmatizer.lemmatize(t) for t in tokens 
                       if len(t) > 2 and t not in stop_words]
            
            if filtered:
                self.processed_docs.append(' '.join(filtered))
    
    def _vectorize_documents(self):
        """向量化文档"""
        n_docs = len(self.processed_docs)
        min_df = 1 if n_docs < 3 else 2
        max_df = 1.0 if n_docs < 3 else 0.8
        
        self.vectorizer = CountVectorizer(
            max_df=max_df, min_df=min_df,
            stop_words='english', max_features=1000
        )
        self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
    
    def _train_lda(self):
        """训练LDA模型"""
        n_topics = min(int(self.topic_var.get()), len(self.processed_docs))
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics, max_iter=50,
            learning_method='online', random_state=42, n_jobs=-1
        )
        self.lda_model.fit(self.doc_term_matrix)
    
    def _extract_topics(self):
        """提取话题"""
        n_words = int(self.words_var.get())
        self.topics = []
        
        for idx, topic in enumerate(self.lda_model.components_):
            indices = topic.argsort()[:-n_words-1:-1]
            self.topics.append({
                'index': idx,
                'words': [self.feature_names[i] for i in indices],
                'weights': [topic[i] for i in indices]
            })
    
    def display_results(self):
        """显示结果"""
        # 更新话题列表
        self.topics_listbox.delete(0, tk.END)
        for topic in self.topics:
            self.topics_listbox.insert(tk.END, f"📌 话题 {topic['index'] + 1}")
        
        # 默认选中第一个话题
        if self.topics:
            self.topics_listbox.selection_set(0)
            self.on_topic_select(None)
        
        # 更新文档分布
        self._update_doc_distribution()
        
        # 绘制图表
        self.plot_topic_distribution()
    
    def _update_doc_distribution(self):
        """更新文档分布表格"""
        for item in self.doc_tree.get_children():
            self.doc_tree.delete(item)
        
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        
        for i, dist in enumerate(doc_topic_dist):
            if i < len(self.doc_names):
                main_topic = np.argmax(dist)
                confidence = dist[main_topic]
                
                self.doc_tree.insert('', tk.END, values=(
                    self.doc_names[i][:35],
                    f"话题 {main_topic + 1}",
                    f"{confidence:.1%}"
                ))
    
    def plot_topic_distribution(self):
        """绘制话题分布图"""
        if not self.topics:
            return
        
        self.figure.clear()
        
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        topic_avg = np.mean(doc_topic_dist, axis=0)
        
        ax = self.figure.add_subplot(111)
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_avg)))
        bars = ax.bar(range(len(topic_avg)), topic_avg, color=colors)
        
        ax.set_xlabel('话题编号', fontsize=10)
        ax.set_ylabel('平均概率', fontsize=10)
        ax.set_title('话题平均分布', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(topic_avg)))
        ax.set_xticklabels([f'话题{i+1}' for i in range(len(topic_avg))])
        
        for bar, val in zip(bars, topic_avg):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_heatmap(self):
        """绘制热力图"""
        if not self.topics:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        n_docs = min(15, len(doc_topic_dist))
        data = doc_topic_dist[:n_docs, :]
        
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('话题', fontsize=10)
        ax.set_ylabel('文档', fontsize=10)
        ax.set_title('文档-话题分布热力图', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(len(self.topics)))
        ax.set_xticklabels([f'T{i+1}' for i in range(len(self.topics))])
        ax.set_yticks(range(n_docs))
        ax.set_yticklabels([self.doc_names[i][:12] for i in range(n_docs)], fontsize=8)
        
        plt.colorbar(im, ax=ax, label='概率')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_keywords(self):
        """绘制关键词权重图"""
        if not self.topics:
            return
        
        self.figure.clear()
        n_topics = len(self.topics)
        n_cols = min(2, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        for i, topic in enumerate(self.topics):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)
            
            words = topic['words'][:8]
            weights = np.array(topic['weights'][:8])
            weights = weights / weights.max()
            
            colors = plt.cm.viridis(weights)
            y_pos = range(len(words))
            bars = ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.set_xlim(0, 1)
            ax.set_title(f'话题 {i+1}', fontsize=10, fontweight='bold')
            ax.set_xlabel('相对权重')
            
            for bar, w in zip(bars, weights):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{w:.2f}', va='center', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def refresh_charts(self):
        """刷新所有图表"""
        if self.topics:
            self.plot_topic_distribution()
            self.update_status("图表已刷新")
    
    def export_results(self):
        """导出结果"""
        if not self.topics:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("文本主题检测系统分析结果\n")
                    f.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write(f"文档数量：{len(self.documents)}\n")
                    f.write(f"话题数量：{len(self.topics)}\n\n")
                    
                    for topic in self.topics:
                        f.write(f"话题 {topic['index'] + 1}:\n")
                        f.write("-" * 40 + "\n")
                        for word, weight in zip(topic['words'], topic['weights']):
                            f.write(f"  {word}: {weight:.4f}\n")
                        f.write("\n")
                    
                    f.write("\n文档-话题分布:\n")
                    f.write("=" * 60 + "\n")
                    
                    doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
                    for i, dist in enumerate(doc_topic_dist):
                        if i < len(self.doc_names):
                            main_topic = np.argmax(dist)
                            f.write(f"{self.doc_names[i]}: 话题 {main_topic + 1} ({dist[main_topic]:.1%})\n")
                
                messagebox.showinfo("成功", f"结果已保存")
                self.update_status("结果已导出")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：{str(e)}")
    
    def clear_all(self):
        """清空所有"""
        if messagebox.askyesno("确认", "确定要清空所有文档吗？"):
            self.documents = []
            self.doc_names = []
            self.all_doc_names = []
            self.processed_docs = []
            self.topics = None
            self.lda_model = None
            
            self.doc_listbox.delete(0, tk.END)
            self.content_text.delete(1.0, tk.END)
            self.topics_listbox.delete(0, tk.END)
            self.topic_details.delete(1.0, tk.END)
            
            for item in self.doc_tree.get_children():
                self.doc_tree.delete(item)
            
            self.stats_label.config(text="📄 已加载 0 个文档")
            self.search_var.set("")
            self.figure.clear()
            self.canvas.draw()
            self.update_status("已清空")
    
    def show_help(self):
        """显示帮助"""
        help_text = """📚 使用说明

1. 加载文档：点击"加载文档"选择txt文件
2. 设置参数：调整话题数量和关键词数量
3. 开始分析：点击"开始分析"或按F5
4. 查看结果：在"话题分析"和"可视化"选项卡查看
5. 导出结果：菜单栏"文件"→"导出结果"

快捷键：
- F5：开始分析
- Ctrl+O：加载文档
- Ctrl+S：导出结果
- Ctrl+Delete：清空所有"""
        
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于"""
        about_text = """文本主题检测系统 v3.0

基于LDA算法的智能话题分析工具

技术栈：
- Python + Tkinter
- scikit-learn
- NLTK
- Matplotlib"""
        
        messagebox.showinfo("关于", about_text)
    
    def run(self):
        """运行程序"""
        self.root.mainloop()


if __name__ == "__main__":
    app = TopicModelingApp()
    app.run()