# AIASE2026 HW4 — Pi Memory（Python）

> 為本地 coding agent 打造跨 session 的持久記憶:capture → store → retrieve → inject。
> 核心、benchmark、評分皆為 Python;Pi 端只用已提供的輕量 TS bridge(無須修改)。

---

## 1. 設計概述

記憶系統把 agent 互動中的觀察捕捉並寫入硬碟,於下次 session 開工前依當前任務檢索最相關片段、在 token 預算內注入回 context,藉此跨越 session 保留專案慣例。

四個環節:
- **Capture / Store**（`memory/store.py`）:以 `summary` 的 SHA-256 為 id 去重,持久化為 JSON。
- **Retrieve**（`memory/bm25.py`）:標準 BM25 排序,確定性、不呼叫模型,供公開與隱藏測試驗證。
- **Inject**（`memory/core.py` 的 `build_injection`,已提供）:依分數高低在 token 預算內塞入。
- **任務二 Hybrid**（`memory/hybrid.py`）:BM25 + 本地多語言 embedding,改善純 BM25 接不到的語義/跨語言題。

設計遵循「確定性外殼包住機率性核心」:核心 BM25 與 store 完全確定性、可重跑驗證;embedding 屬準確定性(鎖模型/版本),作為加值層掛在外殼之上。

---

## 2. 環境

| 項目 | 值 |
|---|---|
| Python | 3.11（需 >= 3.10） |
| 本地 LLM（demo 用） | gpt-oss:20b（MXFP4 量化,20.9B 參數） |
| context size | 8192 |
| 後端 | Ollama（OpenAI 相容 endpoint `localhost:11434/v1`） |
| VRAM / GPU | NVIDIA A100 80GB PCIe（demo 機;LLM 走 GPU 推論） |
| Agent | Pi `@earendil-works/pi-coding-agent` v0.79.0,Node v22.22.3 |
| 檢索 embedding（任務二） | `paraphrase-multilingual-MiniLM-L12-v2`（多語言;CPU 推論即可） |

> 核心與 benchmark 與模型強弱無關,模型只在錄 demo 時用到。embedding 首次執行會從 Hugging Face 下載至本地快取(`~/.cache/huggingface`),之後推論完全在本地進行,不呼叫雲端 API。

---

## 3. 安裝與測試

```bash
pip install -r requirements.txt
pytest -q                 # 核心測試應全綠（13 passed）
```

核心(`bm25.py` / `store.py` / `core.py` / `cli.py`)只用標準函式庫;`sentence-transformers`、`numpy` 僅任務二 hybrid 需要。未安裝時,純 BM25 路徑與核心測試不受影響。

---

## 4. 跑 benchmark

純 BM25(老師原版,study tool):
```bash
python benchmark/run_benchmark.py --k 5 --per-query
python benchmark/run_benchmark.py --corpus corpus_large.jsonl --queries queries_large.jsonl --k 5 --per-query
```

任務二三種策略對照(新增的 `run_benchmark_hybrid.py`,以 `--mode` 一行切換):
```bash
python benchmark/run_benchmark_hybrid.py --mode bm25     --corpus corpus_large.jsonl --queries queries_large.jsonl --k 5
python benchmark/run_benchmark_hybrid.py --mode weighted --corpus corpus_large.jsonl --queries queries_large.jsonl --k 5
python benchmark/run_benchmark_hybrid.py --mode rrf      --corpus corpus_large.jsonl --queries queries_large.jsonl --k 5
```

`--mode` 會強制設定 `RETRIEVAL_MODE`,即使 shell 殘留舊值也以 `--mode` 為準,結果可重現。詳細數字與分析見 `REPORT.md`。

---

## 5. 掛 bridge 到 Pi 並重現 demo

### 5.1 本地模型(任選一種 OpenAI 相容 server)

本專案 demo 用 Ollama:
```bash
ollama serve &
ollama pull gpt-oss:20b
ollama ps                 # 確認模型已載入
```

複製並編輯模型設定(指向你的 server 與模型 id):
```bash
mkdir -p ~/.pi/agent
cp models.json.example ~/.pi/agent/models.json
# 編輯 ~/.pi/agent/models.json：baseUrl 指向 localhost:11434/v1，model id 設為 gpt-oss:20b
# gpt-oss 為 reasoning 模型，透過 Ollama 需設 compat.supportsDeveloperRole / supportsReasoningEffort 為 false
```

### 5.2 啟動 Pi（重要:cwd 與 PYTHONPATH）

bridge 透過 `python -m memory.cli` 呼叫 Python 記憶系統,**必須在 repo 根目錄執行並設定 `PYTHONPATH=.`**,Python 才找得到 `memory/` 套件:

```bash
cd AIASE2026-HW4
export PYTHONPATH=.
export RETRIEVAL_MODE=weighted        # 啟用任務二 hybrid；不設則預設純 BM25
export PYTHON=$(which python)          # 若用 venv/conda，指定該環境的 python，使 bridge 呼叫到含 embedding 的環境
pi -e ./pi-bridge/extension.ts
```

> bridge（`extension.ts`）預設以其所在 repo 根目錄為工作目錄,並用 `process.env.PYTHON`(預設 `python3`)呼叫 Python。若 embedding 裝在 venv/conda,務必如上設定 `PYTHON` 指向該環境的 python,否則 hybrid 會因找不到 `sentence-transformers` 而失效。

### 5.3 重現 demo（跨 session 記憶）

```bash
# 1) 看硬碟，記憶為空
cat ~/.pi-memory.json

# 2) Session A：啟動 Pi，對 agent 說「Remember that this project uses pnpm, not npm.」
#    agent 呼叫 remember 後 Ctrl+C 退出

# 3) 看硬碟，記憶已寫入 JSON
cat ~/.pi-memory.json

# 4) Session B：重新啟動 Pi，問「What package manager should I use?」
#    agent 透過注入的記憶答出 pnpm —— 跨 session 記憶生效
```

詳見 `demo/README.md`。

### 5.4 RETRIEVAL_MODE 說明（避免混淆）

- **未設 / 設為 `bm25`**:走確定性純 BM25(核心測試與老師原版 benchmark 的預設,行為與未改動前完全一致)。
- **設為 `weighted` 或 `rrf`**:走任務二 hybrid。
- ⚠️ 若在同一 shell 連續操作,跑純 BM25 前請先 `unset RETRIEVAL_MODE`(或開新 shell),避免殘留值影響結果。`run_benchmark_hybrid.py --mode` 已強制覆蓋此變數,不受殘留影響。

---

## 6. 任務二做了哪一項

**Hybrid retrieval**,並實作兩種融合策略對照:

- **Weighted**:分數正規化後加權相加(`α·BM25 + (1-α)·embedding`,α=0.5)。保留分數大小,Recall/nDCG 最高。
- **RRF（Reciprocal Rank Fusion）**:以排名融合(`Σ 1/(60+rank)`)。免正規化、免調權重,MRR 最高。

兩者皆透過 `core.retrieve()` 讀環境變數 `RETRIEVAL_MODE` 啟用,benchmark 與 Pi demo 共用同一條檢索路徑。純 BM25 路徑(`bm25_search`)維持確定性、不被污染,供測試驗證。完整數字、trade-off 與錯誤分析見 `REPORT.md`。

新增檔案:`memory/hybrid.py`、`benchmark/run_benchmark_hybrid.py`;`memory/core.py` 僅在 `retrieve()` 加入環境變數開關(預設行為不變)。
