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

class ModernTopicModelingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("文本主题检测系统")
        self.root.geometry("1600x950")
        
        # 设置窗口背景色
        self.root.configure(bg='#f5f5f5')
        
        # 存储文档的列表
        self.documents = []
        self.doc_names = []
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
    
    def setup_styles(self):
        """设置现代样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 定义颜色方案
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
            'info': '#9b59b6',
            'teal': '#1abc9c'
        }
        
        # 配置标题样式
        style.configure('Title.TLabel', 
                       font=('微软雅黑', 22, 'bold'),
                       foreground=self.colors['primary'],
                       background='#f5f5f5')
        
        # 配置按钮样式
        style.configure('Primary.TButton',
                       font=('微软雅黑', 10),
                       padding=8,
                       background=self.colors['secondary'])
        style.map('Primary.TButton',
                 background=[('active', '#2980b9')])
        
        # 配置标签框样式
        style.configure('Card.TLabelframe',
                       relief='flat',
                       borderwidth=1,
                       background=self.colors['white'])
        style.configure('Card.TLabelframe.Label',
                       font=('微软雅黑', 12, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['white'])
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载文档", command=self.load_documents, accelerator="Ctrl+O")
        file_menu.add_command(label="加载文件夹", command=self.load_folder, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="导出结果", command=self.export_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit, accelerator="Ctrl+Q")
        
        # 分析菜单
        analyze_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="分析", menu=analyze_menu)
        analyze_menu.add_command(label="开始分析", command=self.start_analysis, accelerator="F5")
        analyze_menu.add_command(label="清空所有", command=self.clear_all, accelerator="Ctrl+Del")
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="刷新话题图", command=self.refresh_topic_chart)
        
        # 帮助菜单
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
        
        # ========== 顶部工具栏（包含开始分析按钮）==========
        top_toolbar = tk.Frame(self.root, bg=self.colors['primary'], height=70)
        top_toolbar.pack(fill=tk.X, side=tk.TOP)
        top_toolbar.pack_propagate(False)
        
        # 左侧Logo和标题
        logo_frame = tk.Frame(top_toolbar, bg=self.colors['primary'])
        logo_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        title_label = tk.Label(logo_frame, 
                               text="📊 文本主题检测系统——王子琛课设",
                               font=('微软雅黑', 18, 'bold'),
                               bg=self.colors['primary'],
                               fg='white')
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(logo_frame,
                                  text="基于LDA的智能话题分析",
                                  font=('微软雅黑', 9),
                                  bg=self.colors['primary'],
                                  fg='#bdc3c7')
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # 右侧按钮区域
        button_frame = tk.Frame(top_toolbar, bg=self.colors['primary'])
        button_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # 开始分析按钮（突出显示）
        self.btn_analyze = tk.Button(button_frame,
                                     text="🚀 开始分析",
                                     command=self.start_analysis,
                                     font=('微软雅黑', 12, 'bold'),
                                     bg=self.colors['teal'],
                                     fg='white',
                                     padx=25, pady=8,
                                     relief='flat',
                                     cursor='hand2',
                                     bd=0)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        # 加载文档按钮
        self.btn_load = tk.Button(button_frame,
                                  text="📂 加载文档",
                                  command=self.load_documents,
                                  font=('微软雅黑', 10),
                                  bg=self.colors['secondary'],
                                  fg='white',
                                  padx=15, pady=6,
                                  relief='flat',
                                  cursor='hand2',
                                  bd=0)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # 清空按钮
        self.btn_clear = tk.Button(button_frame,
                                   text="🗑️ 清空",
                                   command=self.clear_all,
                                   font=('微软雅黑', 10),
                                   bg=self.colors['warning'],
                                   fg='white',
                                   padx=15, pady=6,
                                   relief='flat',
                                   cursor='hand2',
                                   bd=0)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # ========== 参数设置栏 ==========
        param_frame = tk.Frame(self.root, bg=self.colors['white'], height=45)
        param_frame.pack(fill=tk.X, side=tk.TOP, padx=20, pady=(10, 0))
        param_frame.pack_propagate(False)
        
        param_inner = tk.Frame(param_frame, bg=self.colors['white'])
        param_inner.pack(fill=tk.BOTH, expand=True, padx=15)
        
        # 话题数量设置
        topic_frame = tk.Frame(param_inner, bg=self.colors['white'])
        topic_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(topic_frame, text="📈 话题数量：", 
                font=('微软雅黑', 10, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(side=tk.LEFT)
        
        self.topic_var = tk.StringVar(value="5")
        topic_spinbox = tk.Spinbox(topic_frame, from_=2, to=20, 
                                   textvariable=self.topic_var,
                                   width=6, font=('微软雅黑', 10),
                                   bg='white',
                                   relief='solid', bd=1)
        topic_spinbox.pack(side=tk.LEFT, padx=(8, 0))
        
        # 每个话题词数设置
        words_frame = tk.Frame(param_inner, bg=self.colors['white'])
        words_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(words_frame, text="📝 每个话题词数：", 
                font=('微软雅黑', 10, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['primary']).pack(side=tk.LEFT)
        
        self.words_var = tk.StringVar(value="10")
        words_spinbox = tk.Spinbox(words_frame, from_=5, to=20,
                                   textvariable=self.words_var,
                                   width=6, font=('微软雅黑', 10),
                                   bg='white',
                                   relief='solid', bd=1)
        words_spinbox.pack(side=tk.LEFT, padx=(8, 0))
        
        # 文档统计
        stats_label = tk.Label(param_inner, 
                               text="📄 已加载 0 个文档",
                               font=('微软雅黑', 10),
                               bg=self.colors['white'],
                               fg=self.colors['success'])
        stats_label.pack(side=tk.RIGHT)
        self.stats_label = stats_label
        
        # ========== 主内容区域 - 使用Notebook ==========
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 选项卡1：文档管理
        docs_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(docs_tab, text="📁 文档管理")
        
        # 选项卡2：话题分析结果
        topics_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(topics_tab, text="📌 话题分析结果")
        
        # 选项卡3：可视化图表
        viz_tab = tk.Frame(main_notebook, bg=self.colors['white'])
        main_notebook.add(viz_tab, text="📊 可视化图表")
        
        # ========== 文档管理选项卡 ==========
        # 左侧文档列表
        left_docs = tk.Frame(docs_tab, bg=self.colors['white'])
        left_docs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 搜索框
        search_frame = tk.Frame(left_docs, bg=self.colors['white'])
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(search_frame, text="🔍 搜索文档：", 
                font=('微软雅黑', 9),
                bg=self.colors['white']).pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_documents())
        
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                font=('微软雅黑', 9),
                                bg='#f8f9fa',
                                relief='solid', bd=1)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 文档列表
        doc_list_frame = tk.Frame(left_docs, bg=self.colors['white'])
        doc_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.doc_listbox = tk.Listbox(doc_list_frame,
                                      font=('微软雅黑', 10),
                                      bg='#f8f9fa',
                                      fg=self.colors['dark'],
                                      selectbackground=self.colors['secondary'],
                                      selectforeground='white',
                                      relief='flat',
                                      highlightthickness=1,
                                      bd=0)
        doc_scrollbar = ttk.Scrollbar(doc_list_frame, orient=tk.VERTICAL, 
                                      command=self.doc_listbox.yview)
        self.doc_listbox.configure(yscrollcommand=doc_scrollbar.set)
        
        self.doc_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        doc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.doc_listbox.bind('<<ListboxSelect>>', self.on_document_select)
        
        # 右侧文档内容预览
        right_preview = tk.Frame(docs_tab, bg=self.colors['white'])
        right_preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        preview_label = tk.Label(right_preview, text="📄 文档内容预览",
                                 font=('微软雅黑', 11, 'bold'),
                                 bg=self.colors['white'],
                                 fg=self.colors['primary'])
        preview_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.content_text = tk.Text(right_preview,
                                    wrap=tk.WORD,
                                    font=('微软雅黑', 10),
                                    bg='#f8f9fa',
                                    fg=self.colors['dark'],
                                    relief='flat',
                                    highlightthickness=1,
                                    bd=0)
        content_scroll = ttk.Scrollbar(right_preview, orient=tk.VERTICAL,
                                       command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=content_scroll.set)
        
        self.content_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== 话题分析结果选项卡 ==========
        # 左侧话题列表
        left_topics = tk.Frame(topics_tab, bg=self.colors['white'])
        left_topics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        topics_label = tk.Label(left_topics, text="🎯 检测到的话题",
                                font=('微软雅黑', 11, 'bold'),
                                bg=self.colors['white'],
                                fg=self.colors['primary'])
        topics_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.topics_listbox = tk.Listbox(left_topics,
                                         font=('微软雅黑', 10),
                                         bg='#f8f9fa',
                                         fg=self.colors['dark'],
                                         selectbackground=self.colors['secondary'],
                                         selectforeground='white',
                                         height=12)
        topics_scroll = ttk.Scrollbar(left_topics, orient=tk.VERTICAL,
                                      command=self.topics_listbox.yview)
        self.topics_listbox.configure(yscrollcommand=topics_scroll.set)
        
        self.topics_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        topics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.topics_listbox.bind('<<ListboxSelect>>', self.on_topic_select)
        
        # 右侧话题详情
        right_details = tk.Frame(topics_tab, bg=self.colors['white'])
        right_details.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        details_label = tk.Label(right_details, text="📋 话题详情",
                                 font=('微软雅黑', 11, 'bold'),
                                 bg=self.colors['white'],
                                 fg=self.colors['primary'])
        details_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.topic_details = tk.Text(right_details,
                                     wrap=tk.WORD,
                                     font=('Consolas', 10),
                                     bg='#f8f9fa',
                                     fg=self.colors['dark'],
                                     relief='flat',
                                     height=15)
        details_scroll = ttk.Scrollbar(right_details, orient=tk.VERTICAL,
                                       command=self.topic_details.yview)
        self.topic_details.configure(yscrollcommand=details_scroll.set)
        
        self.topic_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 文档-话题分布表格（放在话题选项卡下方）
        dist_frame = ttk.LabelFrame(topics_tab, text="📊 文档-话题分布", style='Card.TLabelframe')
        dist_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))
        
        dist_container = tk.Frame(dist_frame, bg=self.colors['white'])
        dist_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('文档', '主要话题', '置信度', '话题分布')
        self.doc_tree = ttk.Treeview(dist_container, columns=columns, show='headings', height=8)
        
        self.doc_tree.heading('文档', text='文档')
        self.doc_tree.heading('主要话题', text='主要话题')
        self.doc_tree.heading('置信度', text='置信度')
        self.doc_tree.heading('话题分布', text='话题分布')
        
        self.doc_tree.column('文档', width=180)
        self.doc_tree.column('主要话题', width=100)
        self.doc_tree.column('置信度', width=80)
        self.doc_tree.column('话题分布', width=300)
        
        tree_scroll = ttk.Scrollbar(dist_container, orient=tk.VERTICAL,
                                    command=self.doc_tree.yview)
        self.doc_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.doc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ========== 可视化选项卡 ==========
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 8), dpi=80, facecolor='#f5f5f5')
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制按钮
        viz_control = tk.Frame(viz_tab, bg=self.colors['white'])
        viz_control.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        tk.Button(viz_control, text="📊 话题分布图",
                  command=self.plot_topic_distribution,
                  font=('微软雅黑', 9),
                  bg=self.colors['secondary'],
                  fg='white',
                  padx=10, pady=4,
                  relief='flat',
                  cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        tk.Button(viz_control, text="📈 文档-话题热力图",
                  command=self.plot_doc_topic_heatmap,
                  font=('微软雅黑', 9),
                  bg=self.colors['teal'],
                  fg='white',
                  padx=10, pady=4,
                  relief='flat',
                  cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        tk.Button(viz_control, text="🎯 话题词云图",
                  command=self.plot_topic_wordclouds,
                  font=('微软雅黑', 9),
                  bg=self.colors['info'],
                  fg='white',
                  padx=10, pady=4,
                  relief='flat',
                  cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        # ========== 底部状态栏 ==========
        status_frame = tk.Frame(self.root, bg=self.colors['gray'], height=30)
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
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=150)
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 初始化
        self.all_doc_names = []
    
    def filter_documents(self):
        """过滤文档列表"""
        search_term = self.search_var.get().lower()
        if not search_term:
            self.update_document_list()
            return
        
        filtered = [name for name in self.all_doc_names if search_term in name.lower()]
        self.doc_listbox.delete(0, tk.END)
        for name in filtered:
            self.doc_listbox.insert(tk.END, name)
    
    def update_status(self, message, is_error=False):
        """更新状态栏"""
        if is_error:
            self.status_bar.config(text=f"❌ {message}")
        else:
            self.status_bar.config(text=f"🔄 {message}")
        self.root.update()
    
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
                            self.documents.append(content)
                            self.doc_names.append(file)
                            count += 1
                    except Exception as e:
                        print(f"无法加载 {file}: {str(e)}")
            
            self.all_doc_names = self.doc_names.copy()
            self.update_document_list()
            self.stats_label.config(text=f"📄 已加载 {len(self.documents)} 个文档")
            self.update_status(f"已从文件夹加载 {count} 个文档")
            if count > 0:
                messagebox.showinfo("成功", f"成功加载 {count} 个文档")
            else:
                messagebox.showwarning("警告", "文件夹中没有找到txt文件")
    
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
            doc_name = self.doc_listbox.get(index)
            try:
                doc_index = self.doc_names.index(doc_name)
                self.content_text.delete(1.0, tk.END)
                self.content_text.insert(1.0, self.documents[doc_index])
            except ValueError:
                pass
    
    def on_topic_select(self, event):
        """当选择话题时显示详情"""
        selection = self.topics_listbox.curselection()
        if selection and self.topics:
            index = selection[0]
            if index < len(self.topics):
                topic = self.topics[index]
                self.topic_details.delete(1.0, tk.END)
                self.topic_details.insert(tk.END, f"话题 {topic['index'] + 1} 详细分析\n")
                self.topic_details.insert(tk.END, "=" * 50 + "\n\n")
                
                self.topic_details.insert(tk.END, "关键词权重分布:\n")
                self.topic_details.insert(tk.END, "-" * 30 + "\n")
                
                for word, weight in zip(topic['words'], topic['weights']):
                    bar_length = int(weight * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    self.topic_details.insert(tk.END, f"  {word:<15} [{bar}] {weight:.4f}\n")
                
                # 显示该话题的文档分布
                self.topic_details.insert(tk.END, "\n\n包含此话题的文档:\n")
                self.topic_details.insert(tk.END, "-" * 30 + "\n")
                
                doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
                docs_with_topic = []
                for i, dist in enumerate(doc_topic_dist):
                    if dist[topic['index']] > 0.2:
                        docs_with_topic.append((self.doc_names[i], dist[topic['index']]))
                
                docs_with_topic.sort(key=lambda x: x[1], reverse=True)
                for doc_name, prob in docs_with_topic[:5]:
                    self.topic_details.insert(tk.END, f"  📄 {doc_name} (概率: {prob:.1%})\n")
    
    def start_analysis(self):
        """开始分析"""
        if len(self.documents) < 2:
            messagebox.showwarning("警告", "至少需要2个文档才能进行分析！")
            return
        
        self.progress.start()
        self.update_status("正在分析中，请稍候...")
        
        thread = threading.Thread(target=self.analyze_documents)
        thread.daemon = True
        thread.start()
    
    def analyze_documents(self):
        """分析文档"""
        try:
            self.update_status("正在预处理文档...")
            self.preprocess_documents()
            
            self.update_status("正在向量化文档...")
            self.vectorize_documents()
            
            self.update_status("正在训练LDA模型...")
            self.train_lda()
            
            self.update_status("正在提取话题...")
            self.extract_topics()
            
            self.root.after(0, self.display_results)
            self.root.after(0, lambda: self.update_status("✅ 分析完成"))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("错误", f"分析失败：{msg}"))
            self.root.after(0, lambda: self.update_status("❌ 分析失败", True))
        finally:
            self.root.after(0, self.progress.stop)
    
    def preprocess_documents(self):
        """预处理文档"""
        self.processed_docs = []
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        try:
            lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
        
        try:
            word_tokenize("test")
        except:
            nltk.download('punkt')
        
        for doc in self.documents:
            doc = doc.lower()
            doc = re.sub(r'\d+', '', doc)
            doc = doc.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(doc)
            
            filtered_tokens = []
            for token in tokens:
                if len(token) > 2 and token not in stop_words:
                    token = lemmatizer.lemmatize(token)
                    filtered_tokens.append(token)
            
            self.processed_docs.append(' '.join(filtered_tokens))
    
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
    
    def train_lda(self):
        """训练LDA模型"""
        num_topics = int(self.topic_var.get())
        
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=50,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.lda_model.fit(self.doc_term_matrix)
    
    def extract_topics(self):
        """提取话题"""
        num_words = int(self.words_var.get())
        self.topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
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
        # 更新话题列表
        self.topics_listbox.delete(0, tk.END)
        for topic in self.topics:
            self.topics_listbox.insert(tk.END, f"📌 话题 {topic['index'] + 1}")
        
        # 更新话题详情（默认显示第一个话题）
        if self.topics:
            self.on_topic_select(None)
        
        # 更新文档-话题分布
        self.update_doc_topic_distribution()
        
        # 绘制可视化图表
        self.plot_topic_distribution()
    
    def update_doc_topic_distribution(self):
        """更新文档-话题分布表格"""
        for item in self.doc_tree.get_children():
            self.doc_tree.delete(item)
        
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        
        for i, dist in enumerate(doc_topic_dist):
            if i < len(self.doc_names):
                main_topic = np.argmax(dist)
                confidence = dist[main_topic]
                
                dist_str = " | ".join([f"T{t}:{p:.1%}" for t, p in enumerate(dist)])
                
                self.doc_tree.insert('', tk.END, values=(
                    self.doc_names[i][:35] + "..." if len(self.doc_names[i]) > 35 else self.doc_names[i],
                    f"话题 {main_topic + 1}",
                    f"{confidence:.1%}",
                    dist_str[:250] + "..." if len(dist_str) > 250 else dist_str
                ))
    
    def plot_topic_distribution(self):
        """绘制话题分布图"""
        if not self.topics:
            return
        
        self.figure.clear()
        
        # 计算话题分布
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        topic_avg = np.mean(doc_topic_dist, axis=0)
        
        # 创建子图
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # 条形图
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_avg)))
        bars = ax1.bar(range(len(topic_avg)), topic_avg, color=colors)
        ax1.set_xlabel('话题编号', fontsize=10)
        ax1.set_ylabel('平均概率', fontsize=10)
        ax1.set_title('话题平均分布', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(topic_avg)))
        ax1.set_xticklabels([f'话题{i+1}' for i in range(len(topic_avg))])
        
        # 添加数值标签
        for bar, val in zip(bars, topic_avg):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9)
        
        # 饼图
        labels = [f'话题{i+1}' for i in range(len(topic_avg))]
        ax2.pie(topic_avg, labels=labels, autopct='%1.1f%%', colors=colors)
        ax2.set_title('话题分布占比', fontsize=12, fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_doc_topic_heatmap(self):
        """绘制文档-话题热力图"""
        if not self.topics:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
        
        # 限制显示数量（最多显示20个文档）
        n_docs = min(20, len(doc_topic_dist))
        n_topics = len(self.topics)
        
        data = doc_topic_dist[:n_docs, :]
        
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
        
        ax.set_xlabel('话题编号', fontsize=10)
        ax.set_ylabel('文档', fontsize=10)
        ax.set_title('文档-话题分布热力图', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(n_topics))
        ax.set_xticklabels([f'T{i+1}' for i in range(n_topics)])
        ax.set_yticks(range(n_docs))
        ax.set_yticklabels([self.doc_names[i][:15] for i in range(n_docs)], fontsize=8)
        
        plt.colorbar(im, ax=ax, label='概率')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_topic_wordclouds(self):
        """绘制话题词云图"""
        if not self.topics:
            return
        
        self.figure.clear()
        
        n_topics = len(self.topics)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        for i, topic in enumerate(self.topics):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)
            
            words = topic['words'][:10]
            weights = topic['weights'][:10]
            
            # 归一化权重
            weights = np.array(weights)
            weights = weights / weights.max()
            
            # 创建颜色映射
            colors = plt.cm.viridis(weights)
            
            # 绘制水平条形图
            y_pos = np.arange(len(words))
            bars = ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.set_xlim(0, 1)
            ax.set_title(f'话题 {topic["index"] + 1}', fontsize=10, fontweight='bold')
            ax.set_xlabel('相对权重')
            
            # 添加数值标签
            for bar, w in zip(bars, weights):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{w:.2f}', va='center', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def refresh_topic_chart(self):
        """刷新话题图表"""
        if self.topics:
            self.plot_topic_distribution()
    
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
                    f.write("=" * 70 + "\n")
                    f.write("文本主题检测系统分析结果\n")
                    f.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write(f"文档数量：{len(self.documents)}\n")
                    f.write(f"话题数量：{len(self.topics)}\n\n")
                    
                    for topic in self.topics:
                        f.write(f"话题 {topic['index'] + 1}:\n")
                        f.write("-" * 50 + "\n")
                        for word, weight in zip(topic['words'], topic['weights']):
                            f.write(f"  {word}: {weight:.4f}\n")
                        f.write("\n")
                    
                    f.write("\n" + "=" * 70 + "\n")
                    f.write("文档-话题分布\n")
                    f.write("=" * 70 + "\n\n")
                    
                    doc_topic_dist = self.lda_model.transform(self.doc_term_matrix)
                    for i, dist in enumerate(doc_topic_dist):
                        if i < len(self.doc_names):
                            main_topic = np.argmax(dist)
                            f.write(f"{self.doc_names[i]} -> 主要话题: {main_topic + 1}\n")
                            f.write(f"  话题分布: ")
                            for t, p in enumerate(dist):
                                f.write(f"T{t+1}:{p:.1%} ")
                            f.write("\n\n")
                
                messagebox.showinfo("成功", f"结果已保存到 {file_path}")
                self.update_status(f"✅ 结果已导出")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：{str(e)}")
    
    def clear_all(self):
        """清空所有文档"""
        if messagebox.askyesno("确认", "确定要清空所有文档吗？"):
            self.documents = []
            self.doc_names = []
            self.all_doc_names = []
            self.processed_docs = []
            self.vectorizer = None
            self.doc_term_matrix = None
            self.lda_model = None
            self.feature_names = None
            self.topics = None
            
            self.doc_listbox.delete(0, tk.END)
            self.content_text.delete(1.0, tk.END)
            self.topics_listbox.delete(0, tk.END)
            self.topic_details.delete(1.0, tk.END)
            
            for item in self.doc_tree.get_children():
                self.doc_tree.delete(item)
            
            self.stats_label.config(text="📄 已加载 0 个文档")
            self.search_var.set("")
            self.update_status("已清空")
            
            # 清空图形
            self.figure.clear()
            self.canvas.draw()
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        📚 文本主题检测系统 - 使用说明
        
        ============================================================
        
        📁 1. 加载文档
           • 点击顶部"📂 加载文档"按钮
           • 选择单个或多个txt文件
           • 或使用"加载文件夹"批量导入
        
        ⚙️ 2. 设置参数
           • 话题数量：建议2-10个，根据文档内容调整
           • 每个话题词数：每个话题显示的关键词数量
        
        🚀 3. 开始分析
           • 点击顶部"🚀 开始分析"按钮
           • 或按F5快捷键
           • 等待分析完成（右侧会显示结果）
        
        📊 4. 查看结果
           • "话题分析结果"选项卡：查看话题详情
           • "可视化图表"选项卡：查看分布图和热力图
           • 点击左侧话题列表可查看详细关键词
        
        💾 5. 导出结果
           • 菜单栏"文件" → "导出结果"
           • 可保存为txt或csv格式
        
        ⌨️ 快捷键
           • Ctrl+O    : 加载文档
           • F5        : 开始分析
           • Ctrl+S    : 导出结果
           • Ctrl+Del  : 清空所有
        
        💡 提示
           • 建议使用3个以上文档以获得更好的分析效果
           • 文档内容为英文时效果最佳
           • 分析结果可以在"可视化图表"中查看图形化展示
        """
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
        📊 文本主题检测系统 v3.0
        
        ====================================
        
        基于LDA（潜在狄利克雷分配）算法
        自动识别文档集合中的主题
        
        🛠️ 技术栈：
        • Python 3.8+
        • Tkinter（现代化界面）
        • scikit-learn（LDA模型）
        • NLTK（文本预处理）
        • Matplotlib（可视化）
        
        ✨ 功能特性：
        • 批量文档加载
        • 智能话题检测
        • 关键词权重分析
        • 文档-话题分布
        • 可视化图表展示
        • 结果导出功能
        
        📧 版本：3.0
        📅 更新：2024
        
        如有问题或建议，欢迎反馈！
        """
        messagebox.showinfo("关于", about_text)
    
    def run(self):
        """运行程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ModernTopicModelingApp()
    app.run()