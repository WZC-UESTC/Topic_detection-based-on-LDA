# 云服务器部署说明

当前桌面版入口是 `main.py`，它使用 Tkinter，不能直接通过浏览器访问。云端浏览器访问请运行新增的 `web_app.py`，核心算法仍复用 `topic_model_core.py`。

## 服务器环境

- Ubuntu 22.04/24.04 或其他 Linux 发行版
- Python 3.10 到 3.12 推荐
- 开放 80/443，或临时开放 5000

## 上传代码

如果服务器能访问 Git 仓库：

```bash
git clone <你的仓库地址> topic_detection
cd topic_detection
git checkout master
```

如果没有仓库权限，可以用 `scp` 上传项目目录，但建议排除 `.venv`、`build`、`dist`、`__pycache__` 和 `.exe` 文件。

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行如果 NLTK 资源缺失，程序会自动下载。若服务器不能联网，需要提前准备 `stopwords`、`wordnet`、`punkt`、`punkt_tab` 数据。

## 临时运行

```bash
source .venv/bin/activate
python web_app.py
```

浏览器访问：

```text
http://服务器IP:5000
```

## 生产运行

```bash
source .venv/bin/activate
gunicorn -w 2 -b 127.0.0.1:5000 web_app:app
```

再用 Nginx 反向代理到 `127.0.0.1:5000`，即可通过域名或服务器公网 IP 访问。
