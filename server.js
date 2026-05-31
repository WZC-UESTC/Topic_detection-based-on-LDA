const http = require("http");
const { spawn } = require("child_process");
const querystring = require("querystring");

const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT || 80);
const MAX_BODY_SIZE = 10 * 1024 * 1024;
const ANALYSIS_TIMEOUT_MS = Number(process.env.ANALYSIS_TIMEOUT_MS || 60000);
const PYTHON = process.env.PYTHON || process.env.PYTHON_BIN || "python3";

const pythonAnalyzer = String.raw`
import json
import sys
import numpy as np
from topic_model_core import TopicModelCore

payload = json.load(sys.stdin)
docs = payload.get("documents", [])
names = payload.get("names", [])
num_topics = int(payload.get("num_topics", 5))
num_words = int(payload.get("num_words", 10))
max_iter = int(payload.get("max_iter", 20))
max_features = int(payload.get("max_features", 600))
min_df = int(payload.get("min_df", 1))
max_df = float(payload.get("max_df", 0.85))

if len(docs) < 2:
    raise ValueError("请至少提供 2 篇文本。")

model = TopicModelCore()
model.load_documents(docs, names)
model.preprocess_documents()
model.vectorize_documents(min_df=min_df, max_df=max_df, max_features=max_features)
model.train_lda(num_topics=num_topics, max_iter=max_iter)
topics = model.extract_topics(num_words=num_words)
dist = model.get_doc_topic_distribution()

documents = []
for index, row in enumerate(dist):
    topic_index = int(np.argmax(row))
    documents.append({
        "name": names[index] if index < len(names) else f"doc_{index + 1}.txt",
        "topic": topic_index + 1,
        "confidence": float(row[topic_index]),
        "scores": [float(value) for value in row],
    })

print(json.dumps({
    "topics": [
        {
            "index": int(topic["index"]),
            "words": list(topic["words"]),
            "weights": [float(value) for value in topic["weights"]],
        }
        for topic in topics
    ],
    "documents": documents,
    "document_count": len(docs),
    "params": {
        "num_topics": num_topics,
        "num_words": num_words,
        "max_iter": max_iter,
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
    },
}, ensure_ascii=False))
`;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeJson(value) {
  return JSON.stringify(value || {}).replace(/</g, "\\u003c");
}

function page({ error = "", result = null, form = {} } = {}) {
  const numTopics = escapeHtml(form.num_topics || "5");
  const numWords = escapeHtml(form.num_words || "10");
  const maxIter = escapeHtml(form.max_iter || "20");
  const maxFeatures = escapeHtml(form.max_features || "600");
  const minDf = escapeHtml(form.min_df || "1");
  const maxDf = escapeHtml(form.max_df || "0.85");
  const pastedText = escapeHtml(form.pasted_text || "");
  const loadedCount = result ? result.document_count : 0;

  const docListHtml = result
    ? result.sources
        .map(
          (doc, index) => `
            <button class="list-item ${index === 0 ? "active" : ""}" type="button" data-doc-index="${index}">
              <span>${escapeHtml(doc.name)}</span>
              <small>${doc.text.length} 字符</small>
            </button>`
        )
        .join("")
    : `<div class="empty-line">尚未加载文档</div>`;

  const topicListHtml = result
    ? result.topics
        .map(
          (topic, index) => `
            <button class="list-item ${index === 0 ? "active" : ""}" type="button" data-topic-index="${index}">
              <span>话题 ${topic.index + 1}</span>
              <small>${escapeHtml(topic.words.slice(0, 3).join(" / "))}</small>
            </button>`
        )
        .join("")
    : `<div class="empty-line">分析后显示话题</div>`;

  const firstTopic = result ? result.topics[0] : null;
  const topicDetailHtml = firstTopic ? topicDetail(firstTopic) : "请先加载文档并开始分析。";

  const distributionRows = result
    ? result.documents
        .map(
          (doc) => `
            <tr>
              <td>${escapeHtml(doc.name)}</td>
              <td>话题 ${doc.topic}</td>
              <td>${(doc.confidence * 100).toFixed(1)}%</td>
            </tr>`
        )
        .join("")
    : "";

  const topicAverages = result ? result.topics.map((topic, index) => {
    const total = result.documents.reduce((sum, doc) => sum + Number(doc.scores[index] || 0), 0);
    return result.documents.length ? total / result.documents.length : 0;
  }) : [];

  const barChartHtml = result
    ? result.topics
        .map((topic, index) => {
          const value = topicAverages[index] || 0;
          return `
            <div class="bar-column">
              <div class="bar-track"><i style="height:${Math.max(2, Math.round(value * 100))}%"></i></div>
              <strong>话题 ${topic.index + 1}</strong>
              <span>${(value * 100).toFixed(1)}%</span>
            </div>`;
        })
        .join("")
    : `<div class="empty-line">分析后显示条形图</div>`;

  const heatmapHtml = result
    ? `<table class="heatmap-table">
        <thead>
          <tr>
            <th>文档</th>
            ${result.topics.map((topic) => `<th>话题 ${topic.index + 1}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${result.documents.slice(0, 15).map((doc) => `
            <tr>
              <td title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</td>
              ${doc.scores.map((score) => {
                const pct = Math.max(0, Math.min(1, Number(score || 0)));
                const bg = `rgba(52, 152, 219, ${0.12 + pct * 0.78})`;
                const color = pct > 0.55 ? "#fff" : "#1e2933";
                return `<td style="background:${bg};color:${color}">${(pct * 100).toFixed(0)}%</td>`;
              }).join("")}
            </tr>`).join("")}
        </tbody>
      </table>`
    : `<div class="empty-line">分析后显示热力图</div>`;

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文本主题检测系统</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      background: #f5f5f5;
      color: #2c3e50;
    }
    .topbar {
      height: 60px;
      background: #2c3e50;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
    }
    .brand { font-size: 20px; font-weight: 700; }
    .top-actions { display: flex; gap: 10px; }
    button, .file-button {
      border: 0;
      border-radius: 6px;
      padding: 9px 14px;
      font: inherit;
      cursor: pointer;
      color: #fff;
      background: #3498db;
    }
    .primary { background: #1abc9c; }
    .warning { background: #e67e22; }
    .danger { background: #e74c3c; }
    .file-button input { display: none; }
    .params {
      margin: 14px 20px 0;
      min-height: 46px;
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, auto));
      align-items: center;
      gap: 18px;
      padding: 8px 16px;
    }
    .params label { display: flex; align-items: center; gap: 8px; }
    .params input {
      width: 72px;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      padding: 7px 8px;
      font: inherit;
    }
    .stats { color: #27ae60; font-weight: 700; justify-self: end; }
    .alert {
      margin: 14px 20px 0;
      padding: 12px 14px;
      border: 1px solid #f2c6b7;
      border-radius: 6px;
      background: #fff0eb;
      color: #9b3f1e;
    }
    .workspace {
      margin: 14px 20px 20px;
      background: #fff;
      border: 1px solid #d9e1ea;
      border-radius: 8px;
      overflow: hidden;
    }
    .tabs {
      display: flex;
      gap: 0;
      border-bottom: 1px solid #d9e1ea;
      background: #f8fafc;
    }
    .tab {
      color: #2c3e50;
      background: transparent;
      border-radius: 0;
      border-right: 1px solid #d9e1ea;
      padding: 12px 18px;
    }
    .tab.active {
      background: #fff;
      color: #1abc9c;
      box-shadow: inset 0 3px 0 #1abc9c;
    }
    .tab-panel { display: none; padding: 18px; min-height: 620px; }
    .tab-panel.active { display: block; }
    .split {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
      min-height: 560px;
    }
    .side, .main-panel {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .side h2, .main-panel h2 {
      margin: 0;
      padding: 13px 15px;
      font-size: 16px;
      border-bottom: 1px solid #e2e8f0;
      background: #fbfdff;
    }
    .search {
      margin: 12px;
      width: calc(100% - 24px);
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }
    .list { max-height: 490px; overflow: auto; padding: 0 10px 10px; }
    .list-item {
      width: 100%;
      color: #2c3e50;
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      margin-bottom: 8px;
      display: grid;
      gap: 4px;
      text-align: left;
    }
    .list-item.active { border-color: #1abc9c; background: #ecfdf8; }
    .list-item small { color: #64748b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .content-box {
      padding: 14px;
      white-space: pre-wrap;
      line-height: 1.65;
      min-height: 500px;
      max-height: 560px;
      overflow: auto;
    }
    .paste-box {
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid #cbd5e1;
      border-radius: 7px;
      padding: 12px;
      font: inherit;
      line-height: 1.6;
    }
    .topic-detail table, .dist table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    td, th {
      border-bottom: 1px solid #edf2f7;
      padding: 10px 8px;
      text-align: left;
    }
    th { background: #fbfdff; color: #64748b; }
    .bar {
      display: block;
      height: 9px;
      border-radius: 999px;
      background: #e7edf3;
      overflow: hidden;
    }
    .bar i {
      display: block;
      height: 100%;
      background: #3498db;
      border-radius: inherit;
    }
    .viz-grid { display: grid; gap: 18px; }
    .bar-chart {
      min-height: 300px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(72px, 1fr));
      gap: 16px;
      align-items: end;
      padding: 18px;
    }
    .bar-column {
      height: 250px;
      display: grid;
      grid-template-rows: 1fr auto auto;
      gap: 8px;
      justify-items: center;
      color: #2c3e50;
      font-size: 13px;
    }
    .bar-track {
      width: 42px;
      height: 190px;
      border-left: 1px solid #d9e1ea;
      border-bottom: 1px solid #d9e1ea;
      background: #f8fafc;
      display: flex;
      align-items: flex-end;
      justify-content: center;
    }
    .bar-track i {
      display: block;
      width: 28px;
      background: #3498db;
      border-radius: 4px 4px 0 0;
    }
    .bar-column span { color: #64748b; }
    .heatmap-wrap { padding: 14px; overflow: auto; }
    .heatmap-table { min-width: 680px; table-layout: fixed; }
    .heatmap-table th,
    .heatmap-table td { text-align: center; }
    .heatmap-table th:first-child,
    .heatmap-table td:first-child {
      width: 180px;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      background: #fff;
      color: #2c3e50;
    }
    .empty-line { color: #94a3b8; padding: 16px; }
    @media (max-width: 900px) {
      .topbar { height: auto; align-items: flex-start; flex-direction: column; gap: 12px; padding: 16px; }
      .split { grid-template-columns: 1fr; }
      .stats { margin-left: 0; }
    }
  </style>
</head>
<body>
  <form id="analysis-form" method="post" enctype="multipart/form-data">
    <header class="topbar">
      <div class="brand">文本主题检测系统</div>
      <div class="top-actions">
        <label class="file-button">加载文档
          <input id="file-input" type="file" name="documents" multiple accept=".txt,text/plain">
        </label>
        <label class="file-button">加载文件夹
          <input id="folder-input" type="file" name="documents" multiple webkitdirectory directory accept=".txt,text/plain">
        </label>
        <button class="primary" type="submit">开始分析</button>
        <button class="warning" type="button" id="export-btn" ${result ? "" : "disabled"}>导出结果</button>
      </div>
    </header>

    <section class="params">
      <label>话题数量 <input type="number" name="num_topics" min="2" max="20" value="${numTopics}"></label>
      <label>每个话题词数 <input type="number" name="num_words" min="5" max="20" value="${numWords}"></label>
      <label>最大迭代次数 <input type="number" name="max_iter" min="5" max="100" value="${maxIter}"></label>
      <label>最大特征词数 <input type="number" name="max_features" min="100" max="3000" step="50" value="${maxFeatures}"></label>
      <label>最小文档频率 <input type="number" name="min_df" min="1" max="10" value="${minDf}"></label>
      <label>最大文档频率 <input type="number" name="max_df" min="0.1" max="1" step="0.05" value="${maxDf}"></label>
      <span class="stats" id="stats-label">已加载 ${loadedCount} 个文档</span>
    </section>
    <input type="hidden" name="client_documents" id="client-documents">

    ${error ? `<div class="alert">${escapeHtml(error)}</div>` : ""}

    <section class="workspace">
      <nav class="tabs">
        <button class="tab active" type="button" data-tab="docs">文档管理</button>
        <button class="tab" type="button" data-tab="topics">话题分析</button>
        <button class="tab" type="button" data-tab="dist">文档分布</button>
        <button class="tab" type="button" data-tab="viz">可视化</button>
      </nav>

      <section class="tab-panel active" id="tab-docs">
        <div class="split">
          <aside class="side">
            <h2>文档列表</h2>
            <input class="search" id="doc-search" type="search" placeholder="搜索文档">
            <div class="list" id="doc-list">${docListHtml}</div>
          </aside>
          <section class="main-panel">
            <h2>文档内容</h2>
            <div class="content-box" id="doc-content">${
              result ? escapeHtml(result.sources[0]?.text || "") : "上传 TXT 或粘贴文本后开始分析。"
            }</div>
          </section>
        </div>
        <h2 style="margin:18px 0 10px;">粘贴文本</h2>
        <textarea class="paste-box" name="pasted_text" placeholder="多篇文本之间用单独一行 --- 分隔">${pastedText}</textarea>
      </section>

      <section class="tab-panel" id="tab-topics">
        <div class="split">
          <aside class="side">
            <h2>检测到的话题</h2>
            <div class="list" id="topic-list">${topicListHtml}</div>
          </aside>
          <section class="main-panel">
            <h2>话题详情</h2>
            <div class="content-box topic-detail" id="topic-detail">${topicDetailHtml}</div>
          </section>
        </div>
      </section>

      <section class="tab-panel dist" id="tab-dist">
        <table>
          <thead><tr><th>文档</th><th>主要话题</th><th>置信度</th></tr></thead>
          <tbody>${distributionRows || `<tr><td colspan="3">分析后显示文档主题分布。</td></tr>`}</tbody>
        </table>
      </section>

      <section class="tab-panel" id="tab-viz">
        <div class="viz-grid">
          <div class="main-panel">
            <h2>话题平均分布条形图</h2>
            <div class="bar-chart">${barChartHtml}</div>
          </div>
          <div class="main-panel">
            <h2>文档-话题分布热力图</h2>
            <div class="heatmap-wrap">${heatmapHtml}</div>
          </div>
        </div>
      </section>
    </section>
  </form>

  <script id="result-data" type="application/json">${safeJson(result)}</script>
  <script>
    const result = JSON.parse(document.getElementById("result-data").textContent || "null");
    let pendingSources = result && result.sources ? result.sources.slice() : [];
    const tabs = document.querySelectorAll(".tab");
    const panels = document.querySelectorAll(".tab-panel");
    const docList = document.getElementById("doc-list");
    const docContent = document.getElementById("doc-content");
    const statsLabel = document.getElementById("stats-label");
    const clientDocuments = document.getElementById("client-documents");
    const fileInput = document.getElementById("file-input");
    const folderInput = document.getElementById("folder-input");
    const form = document.getElementById("analysis-form");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((item) => item.classList.remove("active"));
        panels.forEach((panel) => panel.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
      });
    });

    document.querySelectorAll("[data-doc-index]").forEach((item) => {
      item.addEventListener("click", () => {
        document.querySelectorAll("[data-doc-index]").forEach((node) => node.classList.remove("active"));
        item.classList.add("active");
        const doc = pendingSources[Number(item.dataset.docIndex)];
        docContent.textContent = doc ? doc.text : "";
      });
    });

    document.querySelectorAll("[data-topic-index]").forEach((item) => {
      item.addEventListener("click", () => {
        document.querySelectorAll("[data-topic-index]").forEach((node) => node.classList.remove("active"));
        item.classList.add("active");
        const topic = result.topics[Number(item.dataset.topicIndex)];
        document.getElementById("topic-detail").innerHTML = renderTopic(topic);
      });
    });

    const search = document.getElementById("doc-search");
    if (search) {
      search.addEventListener("input", () => {
        const keyword = search.value.trim().toLowerCase();
        document.querySelectorAll("[data-doc-index]").forEach((item) => {
          item.style.display = item.textContent.toLowerCase().includes(keyword) ? "" : "none";
        });
      });
    }

    if (fileInput) fileInput.addEventListener("change", () => loadFiles(fileInput.files));
    if (folderInput) folderInput.addEventListener("change", () => loadFiles(folderInput.files));
    if (form) {
      form.addEventListener("submit", () => {
        if (clientDocuments) clientDocuments.value = JSON.stringify(pendingSources);
      });
    }

    const exportBtn = document.getElementById("export-btn");
    if (exportBtn && result) {
      exportBtn.addEventListener("click", () => {
        const lines = ["文本主题检测结果", "", "文档数量：" + result.document_count, "话题数量：" + result.topics.length];
        if (result.params) {
          lines.push(
            "最大迭代次数：" + result.params.max_iter,
            "最大特征词数：" + result.params.max_features,
            "最小文档频率：" + result.params.min_df,
            "最大文档频率：" + result.params.max_df
          );
        }
        lines.push("");
        result.topics.forEach((topic) => {
          lines.push("话题 " + (topic.index + 1) + ":");
          topic.words.forEach((word, index) => lines.push("  " + word + "  " + topic.weights[index].toFixed(4)));
          lines.push("");
        });
        lines.push("文档主题分布:");
        result.documents.forEach((doc) => lines.push(doc.name + ": 话题 " + doc.topic + " (" + (doc.confidence * 100).toFixed(1) + "%)"));
        const blob = new Blob([lines.join("\\n")], { type: "text/plain;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "topic_results.txt";
        link.click();
        URL.revokeObjectURL(link.href);
      });
    }

    function renderTopic(topic) {
      if (!topic) return "暂无话题。";
      const max = Math.max(...topic.weights, 1);
      const rows = topic.words.map((word, index) => {
        const weight = topic.weights[index] || 0;
        const width = Math.max(4, Math.round(weight / max * 100));
        return "<tr><td>" + escapeText(word) + "</td><td><span class='bar'><i style='width:" + width + "%'></i></span></td><td>" + weight.toFixed(4) + "</td></tr>";
      }).join("");
      return "<table><tbody>" + rows + "</tbody></table>";
    }

    async function loadFiles(fileList) {
      const files = Array.from(fileList || []).filter((file) => {
        const name = file.webkitRelativePath || file.name || "";
        return name.toLowerCase().endsWith(".txt");
      });
      if (!files.length) {
        alert("请选择 .txt 文档。");
        return;
      }

      const loaded = [];
      for (const file of files) {
        const text = await readTextFile(file);
        if (text.trim()) {
          loaded.push({
            name: file.webkitRelativePath || file.name,
            text
          });
        }
      }

      pendingSources = loaded;
      renderDocumentList();
      if (clientDocuments) clientDocuments.value = JSON.stringify(pendingSources);
    }

    function readTextFile(file) {
      return file.arrayBuffer().then((buffer) => {
        const bytes = new Uint8Array(buffer);
        for (const encoding of ["utf-8", "gb18030", "gbk"]) {
          try {
            return new TextDecoder(encoding, { fatal: true }).decode(bytes);
          } catch (error) {
            // Try the next common text encoding.
          }
        }
        return new TextDecoder("utf-8").decode(bytes);
      });
    }

    function renderDocumentList() {
      if (!docList || !docContent || !statsLabel) return;
      statsLabel.textContent = "已加载 " + pendingSources.length + " 个文档";
      if (!pendingSources.length) {
        docList.innerHTML = '<div class="empty-line">尚未加载文档</div>';
        docContent.textContent = "上传 TXT 或粘贴文本后开始分析。";
        return;
      }

      docList.innerHTML = pendingSources.map((doc, index) => (
        '<button class="list-item ' + (index === 0 ? 'active' : '') + '" type="button" data-doc-index="' + index + '">' +
        '<span>' + escapeText(doc.name) + '</span>' +
        '<small>' + doc.text.length + ' 字符</small>' +
        '</button>'
      )).join("");
      docContent.textContent = pendingSources[0].text;

      document.querySelectorAll("[data-doc-index]").forEach((item) => {
        item.addEventListener("click", () => {
          document.querySelectorAll("[data-doc-index]").forEach((node) => node.classList.remove("active"));
          item.classList.add("active");
          const doc = pendingSources[Number(item.dataset.docIndex)];
          docContent.textContent = doc ? doc.text : "";
        });
      });
    }

    function escapeText(value) {
      return String(value ?? "").replace(/[&<>\"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;"
      }[char]));
    }
  </script>
</body>
</html>`;
}

function topicDetail(topic) {
  const maxWeight = Math.max(...topic.weights, 1);
  const rows = topic.words
    .map((word, index) => {
      const weight = topic.weights[index] || 0;
      const width = Math.max(4, Math.round((weight / maxWeight) * 100));
      return `<tr><td>${escapeHtml(word)}</td><td><span class="bar"><i style="width:${width}%"></i></span></td><td>${weight.toFixed(4)}</td></tr>`;
    })
    .join("");
  return `<table><tbody>${rows}</tbody></table>`;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;

    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_SIZE) {
        req.destroy();
        reject(new Error("上传内容超过 10MB，请减少文件数量或大小。"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function parseMultipart(body, contentType) {
  const match = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/i);
  if (!match) return { fields: {}, files: [] };

  const boundary = `--${match[1] || match[2]}`;
  const raw = body.toString("latin1");
  const fields = {};
  const files = [];

  for (const section of raw.split(boundary)) {
    if (!section || section === "--\r\n" || section === "--") continue;

    const cleaned = section.replace(/^\r\n/, "").replace(/\r\n--$/, "");
    const divider = cleaned.indexOf("\r\n\r\n");
    if (divider === -1) continue;

    const headerText = cleaned.slice(0, divider);
    let content = cleaned.slice(divider + 4);
    if (content.endsWith("\r\n")) content = content.slice(0, -2);

    const nameMatch = headerText.match(/name="([^"]+)"/);
    if (!nameMatch) continue;
    const name = nameMatch[1];
    const filenameMatch = headerText.match(/filename="([^"]*)"/);

    if (filenameMatch) {
      const filename = filenameMatch[1];
      if (!filename) continue;
      const data = Buffer.from(content, "latin1");
      files.push({ name, filename, text: data.toString("utf8") });
    } else {
      fields[name] = Buffer.from(content, "latin1").toString("utf8");
    }
  }

  return { fields, files };
}

function parseUrlEncoded(body) {
  return { fields: querystring.parse(body.toString("utf8")), files: [] };
}

function buildDocuments(fields, files) {
  const documents = [];
  const names = [];

  if (fields.client_documents) {
    try {
      const clientDocs = JSON.parse(String(fields.client_documents));
      if (Array.isArray(clientDocs)) {
        for (const doc of clientDocs) {
          const text = String(doc && doc.text ? doc.text : "").trim();
          if (!text) continue;
          documents.push(text);
          names.push(String(doc && doc.name ? doc.name : `doc_${documents.length}.txt`));
        }
      }
    } catch (error) {
      // Fall back to multipart file parsing below.
    }
  }

  for (const file of files) {
    if (file.name !== "documents") continue;
    if (!file.filename.toLowerCase().endsWith(".txt")) continue;
    const text = file.text.trim();
    if (!text) continue;
    documents.push(text);
    names.push(file.filename.split(/[\\/]/).pop());
  }

  const pasted = String(fields.pasted_text || "").trim();
  if (pasted) {
    const blocks = pasted
      .split(/\r?\n---\r?\n/g)
      .map((value) => value.trim())
      .filter(Boolean);
    for (const [index, block] of blocks.entries()) {
      documents.push(block);
      names.push(`pasted_${index + 1}.txt`);
    }
  }

  return { documents, names };
}

function analyze(payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ["-c", pythonAnalyzer], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, ANALYSIS_TIMEOUT_MS);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new Error("分析超时，请减少文档数量或文本长度后重试。"));
        return;
      }
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Python 分析进程退出码：${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Python 返回内容无法解析：${error.message}`));
      }
    });

    child.stdin.end(JSON.stringify(payload));
  });
}

async function handleRequest(req, res) {
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { status: "ok" });
    return;
  }

  if (req.method === "GET") {
    sendHtml(res, 200, page());
    return;
  }

  if (req.method !== "POST") {
    sendHtml(res, 405, page({ error: "不支持的请求方法。" }));
    return;
  }

  const form = {};

  try {
    const body = await readBody(req);
    const contentType = req.headers["content-type"] || "";
    const parsed = contentType.includes("multipart/form-data")
      ? parseMultipart(body, contentType)
      : parseUrlEncoded(body);

    form.num_topics = parsed.fields.num_topics || "5";
    form.num_words = parsed.fields.num_words || "10";
    form.max_iter = parsed.fields.max_iter || "20";
    form.max_features = parsed.fields.max_features || "600";
    form.min_df = parsed.fields.min_df || "1";
    form.max_df = parsed.fields.max_df || "0.85";
    form.pasted_text = parsed.fields.pasted_text || "";

    const numTopics = Math.max(2, Math.min(20, Number.parseInt(form.num_topics, 10) || 5));
    const numWords = Math.max(5, Math.min(20, Number.parseInt(form.num_words, 10) || 10));
    const maxIter = Math.max(5, Math.min(100, Number.parseInt(form.max_iter, 10) || 20));
    const maxFeatures = Math.max(100, Math.min(3000, Number.parseInt(form.max_features, 10) || 600));
    const minDf = Math.max(1, Math.min(10, Number.parseInt(form.min_df, 10) || 1));
    const maxDf = Math.max(0.1, Math.min(1, Number.parseFloat(form.max_df) || 0.85));
    const { documents, names } = buildDocuments(parsed.fields, parsed.files);

    if (documents.length < 2) {
      sendHtml(res, 200, page({ error: "请至少上传或粘贴 2 篇文本。", form }));
      return;
    }

    const result = await analyze({
      documents,
      names,
      num_topics: numTopics,
      num_words: numWords,
      max_iter: maxIter,
      max_features: maxFeatures,
      min_df: minDf,
      max_df: maxDf,
    });

    result.sources = documents.map((text, index) => ({
      name: names[index] || `doc_${index + 1}.txt`,
      text,
    }));

    sendHtml(res, 200, page({ result, form }));
  } catch (error) {
    sendHtml(res, 500, page({ error: `分析失败：${error.message}`, form }));
  }
}

function sendHtml(res, status, html) {
  res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
  res.end(html);
}

function sendJson(res, status, data) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(data));
}

http.createServer(handleRequest).listen(PORT, HOST, () => {
  console.log(`文本主题检测系统已启动：http://${HOST}:${PORT}`);
  console.log(`Python 命令：${PYTHON}`);
});
