# Demo — 跨 session 記憶生效

本資料夾展示記憶系統在真實 coding agent（Pi + 本地 gpt-oss:20b）上的運作:
agent 在一個 session 記下專案慣例 → 關閉 → 全新 session 仍能透過注入的記憶「記得」,
不需要使用者重新說明。

## 影片 / 截圖

影片連結（GitHub 不適合放大影片,請貼外部連結）:
- 

或截圖（放在本資料夾）:
- 

---

## Demo 流程與各步驟的意義

| 步驟 | 操作 | 畫面 | 證明什麼 |
|---|---|---|---|
| 1 | `cat ~/.pi-memory.json` | 顯示「沒有此一檔案或目錄」 | 起點乾淨,硬碟上沒有任何記憶 |
| 2 | Session A:告訴 agent 四項專案慣例 | agent 呼叫 4 次 remember,逐筆回 `Remembered: ...` | agent 主動辨識專案慣例並完成記憶捕捉(capture) |
| 3 | 退出 agent 後再 `cat ~/.pi-memory.json` | 顯示完整 JSON（四筆記憶） | 記憶以 JSON 持久化寫入硬碟,非暫存於對話 |
| 4 | Session B（全新 session）問「What package manager should I use?」 | 注入 `[記憶 - 來自過去的 session]`,agent 答出 pnpm | 新 session 開工前自動檢索並注入硬碟記憶,跨 session 生效 |

步驟 1→3→4 形成完整證據鏈:**無檔案 → 寫入硬碟 → 全新 session 從硬碟讀回並注入**。
Session A 與 Session B 是兩次獨立啟動的 Pi 程序,中間記憶完全靠硬碟檔案傳遞,
證明記得 pnpm 並非來自對話上下文,而是來自持久化的跨 session 記憶。

> 任務二的 hybrid 跨語言檢索能力（中文查詢命中英文記憶,純 BM25 接不到、hybrid 接到）
> 以 benchmark 統計與 CLI 對比作為客觀證據,詳見 `REPORT.md` 第 2 題。
> 該證明不經過 LLM、可重跑復現,比在 agent 對話中展示更嚴謹,故不納入本 demo 影片。

---

## 重現方式

環境（在 repo 根目錄）:

```bash
cd AIASE2026-HW4
source ~/hw4-venv/bin/activate
export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
export PYTHON=$(which python)
export RETRIEVAL_MODE=weighted
export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=""
```

本地模型（先確認 Ollama 已載入 gpt-oss:20b）:

```bash
ollama ps        # 應看到 gpt-oss:20b
```

啟動 agent 並掛上記憶系統:

```bash
pi -e ./pi-bridge/extension.ts
```

Demo 步驟:

```bash
# 1. 看硬碟，記憶是空的
cat ~/.pi-memory.json

# 2. 啟動 Pi（Session A），對 agent 說：
#    Remember that this project uses pnpm, not npm.
#    （agent 呼叫 remember）後 Ctrl+C 退出

# 3. 看硬碟，記憶已寫入
cat ~/.pi-memory.json

# 4. 再啟動 Pi（Session B），問：
#    What package manager should I use for this project?
#    （agent 透過注入的記憶答出 pnpm）
```

---

## 環境

- OS:Linux（實驗室 GPU 機器）
- 本地 LLM:gpt-oss:20b（Ollama,MXFP4 量化,20.9B 參數,GPU 推論）
- LLM context size:8192
- 後端:Ollama（OpenAI 相容 endpoint localhost:11434/v1）
- Agent:Pi（@earendil-works/pi-coding-agent）v0.79.0,Node v22.22.3
- 檢索 embedding:paraphrase-multilingual-MiniLM-L12-v2（CPU 推論）
- 檢索模式:weighted（任務二 hybrid;另支援 bm25 / rrf,見 REPORT）
