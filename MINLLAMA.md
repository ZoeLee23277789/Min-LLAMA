
# 我的 Min-LLaMA 作業實作說明（作者：李柔儀）

本檔案紀錄我於 NLP244 課程中實作 Min-LLaMA 模型的各個模組之過程與方法，涵蓋原理、實作細節與應用目的。

---

## 📄 我的實作說明（作者：李柔儀）

### 1️⃣ llama.py — Llama 模型核心結構

#### 🧩 RMSNorm

**原理說明：**  
RMSNorm 是一種不需要減去均值的 Layer Normalization 變體，公式如下：  
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$  

**我的實作：**  
- 使用 `torch.mean(x.pow(2), dim=-1)` 計算平方平均。
- 保留資料型別 `.type_as(x)`。
- 使用 `nn.Parameter` 作為學習參數。

---

#### 🧠 Attention 模組（Multi-Head + Grouped Query Attention）

**原理說明：**  
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$  
Llama 使用 GQA（Grouped Query Attention）共享 Key/Value，以降低成本。

**實作細節：**
- 使用 `repeat_interleave` 讓 Key/Value 與 Query head 對齊。
- 將注意力計算封裝在 `compute_query_key_value_scores`。
- 使用 `Dropout` 並還原維度回 `(B, T, D)`。

---

#### 🔁 LlamaLayer

**結構：**
1. RMSNorm
2. Attention + residual
3. RMSNorm
4. FeedForward（SwiGLU）+ residual

**SwiGLU 公式：**
$$
\text{SwiGLU}(x) = \text{SiLU}(W_1 x) \cdot (W_3 x)
$$

---

#### 📝 forward & generate

**forward：**
- 若有 target，則計算整句的 logits。
- 若無 target，僅輸出最後一 token 的 logits。

**generate：**
- greedy（temperature=0）或抽樣（temperature>0）。
- 每次重新前傳（無 KV-cache，效率較低但簡化程式）。

---

### 2️⃣ classifier.py — 分類器設計

#### 📌 LlamaZeroShotClassifier

**方法：**
- 將每個 label 字串 tokenize 後，計算其出現在預測中的機率。
- 適用於無標訓練資料的場景。

#### 🏷️ LlamaEmbeddingClassifier

**步驟：**
1. 取最後一個 hidden state。
2. 套用 Dropout。
3. 使用 Linear 層分類後做 log_softmax。

支援 `pretrain`（freeze）與 `finetune`（微調）兩種模式。

---

### 3️⃣ optimizer.py — 自製 AdamW 優化器

**原理：**
AdamW 結合一階與二階動量與權重衰減：
$$
\theta = \theta - \eta \cdot \left( \frac{m}{\sqrt{v} + \epsilon} + \lambda \cdot \theta \right)
$$

**我的做法：**
- 用 `alpha_t` 做 bias 修正，簡化 `m_hat` 和 `v_hat`。
- 每次 step 都更新狀態字典中的 `m`, `v`, `step`。

---

### 4️⃣ rope.py — Rotary Position Embeddings

**原理：**
RoPE 使用複數旋轉方式保留相對位置資訊：
$$
x_{\text{rot}} = x \cdot \cos(\theta) + x_\perp \cdot \sin(\theta)
$$

**實作說明：**
- query/key 拆為實部/虛部，reshape 成 `(…, d/2, 2)`。
- 使用 `torch.outer` 生成頻率矩陣。
- 用 `cos`/`sin` 做複數旋轉後再組合回原來形狀。

---

這些模組共同組成了我手寫的簡化版 Llama2 模型，可支援生成、Zero-shot 分類與微調分類。
