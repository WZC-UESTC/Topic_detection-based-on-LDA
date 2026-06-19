# 📊 文本主题检测系统

基于 LDA（潜在狄利克雷分配）算法的智能文本主题分析工具，可自动识别文档集合中的潜在主题。

## ✨ 功能特点

- 📁 **文档管理**：支持单个/批量上传 txt 文档，带搜索功能
- 🎯 **主题检测**：基于 LDA 算法自动识别文档主题
- 📊 **可视化展示**：话题分布图、热力图、关键词权重图
- 💾 **结果导出**：支持导出为 txt 或 csv 格式
- 🚀 **高效处理**：多线程处理，支持大批量文档

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.8+ | 开发语言 |
| Tkinter | GUI 界面 |
| scikit-learn | LDA 主题建模 |
| NLTK | 文本预处理（分词、去停用词、词形还原）|
| Matplotlib | 数据可视化 |
| NumPy | 数值计算 |

## 📦 安装与运行

### 环境要求
- Python 3.8 或更高版本
- Windows / macOS / Linux
### 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
### 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

## 🚀 快速开始
1. 加载文档
点击 "📂 加载文档" 选择单个或多个 txt 文件

或点击 "加载文件夹" 批量导入
<img width="2826" height="1642" alt="3" src="https://github.com/user-attachments/assets/fd0cb529-3fdf-4e7d-ae4b-037e947290fd" />

2. 设置参数
话题数量：期望检测的话题个数（建议 3-10）

每个话题词数：每个话题显示的关键词数量

3. 开始分析
点击 "🚀 开始分析" 或按 F5
等待分析完成，结果自动显示

4. 查看结果
话题分析 选项卡：查看检测到的话题和关键词
可视化图表 选项卡：查看分布图和热力图
文档-话题分布 表格：查看每个文档的话题归属

###  ⌨️ 快捷键
快捷键	功能
F5	开始分析
Ctrl+O	加载文档
Ctrl+S	导出结果
Ctrl+Delete	清空所有
<img width="2860" height="1700" alt="2" src="https://github.com/user-attachments/assets/d496f502-a0d4-4403-8089-0e93e4dde8d7" />


6. 导出结果
菜单栏 文件 → 导出结果 保存分析结果

## 📁 项目结构
topic_detection/
├── main.py                    # 主程序（GUI界面）
├── topic_model_core.py        # 核心算法模块
├── download_bbc_dataset.py    # BBC数据集下载器
├── test_with_bbc.py           # BBC数据集测试脚本
├── requirements.txt           # 项目依赖
├── README.md                  # 项目说明
├── documents/                 # 存放待分析文档
└── test/                      # 测试相关
    ├── datasets/              # 测试数据集
    └── results/               # 测试报告

## 📊 算法原理
本系统采用 LDA（Latent Dirichlet Allocation，潜在狄利克雷分配） 算法：

文本预处理：分词、去停用词、词形还原

向量化：将文档转换为词频矩阵

模型训练：使用 LDA 算法提取主题

结果输出：输出每个主题的关键词及权重

## 📈 性能表现
基于 BBC News 数据集的测试结果：
<img width="1247" height="789" alt="屏幕截图 2026-05-03 194212" src="https://github.com/user-attachments/assets/7a2566d1-252c-48c7-989b-d5971927cd87" />

配置	文档数	准确率	耗时
小样本 + 3主题	50	~65%	2.1s
中样本 + 5主题	100	~72%	3.5s
大样本 + 5主题	250	~75%	8.2s

<img width="1095" height="267" alt="屏幕截图 2026-05-03 194233" src="https://github.com/user-attachments/assets/907aaf9b-f03a-405d-8fba-e8cfab8f5390" />

## 云端部署
26.5.30 目前已将该项目打包上传到云主机,形成网页端。并启用pm2自动托管，可以实时访问
由于云主机按量计费，访问截至2026/6/30
地址：http://8.160.166.149:80
<img width="2216" height="835" alt="屏幕截图 2026-05-30 183435" src="https://github.com/user-attachments/assets/8e25a9ba-2140-4d80-8b43-959d31e6c39e" />
<img width="2488" height="333" alt="屏幕截图 2026-05-30 183055" src="https://github.com/user-attachments/assets/592687a0-aebf-4b3f-8ffc-b0cf9f793eff" />



## 🤝 贡献
Wangzichen

## 📄 许可证
MIT License

# 📧 联系方式
GitHub: 13113
E-mail：2024080903021@std.uestc.edu.cn

# ⭐ 如果这个项目对你有帮助，欢迎给个 Star！
