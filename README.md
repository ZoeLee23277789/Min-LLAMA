
# 實作 Min-LLaMA 

本文件說明專案實作 LLaMA 模型各模組的細節。


## 🧠 作業簡介（Assignment Overview）

這份專案來自 UC Santa Cruz NLP244 課程，目的是透過手動實作 LLaMA 模型的核心模組，了解大型語言模型的運作方式。內容包含：

- 自行實作 Transformer 架構（Attention、Norm、FeedForward 等）。
- 加入 Rotary Positional Embeddings (RoPE) 處理相對位置資訊。
- 建構分類器處理情感分析（SST-5 與 CFIMDB）。
- 訓練與推論模型的流程控制皆經由 `run_llama.py` 指令觸發。



## 📁 主要程式結構與我的實作內容

### ✅ `llama.py` — 模型主架構

#### RMSNorm 實作細節

RMSNorm 是一種僅考慮張量平方平均值的正規化方法。相較於 LayerNorm，其計算更簡單也更穩定。其公式如下：
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$
- `self.eps` 為數值穩定性加入的小常數。
- `self.weight` 是 learnable scale。
- 實作上我使用 `torch.mean(x.pow(2), dim=-1, keepdim=True)` 搭配 `.type_as(x)` 保留資料型別一致性。

---

#### Attention 模組與 GQA 機制

這個模組實作了 Multi-Head Attention 機制，核心公式為：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

LLaMA 使用 [Grouped Query Attention]，即部分 query heads 共用 key/value head，以減少參數與記憶體。
我的實作包含：
- 預先以 `nn.Linear` 分別投影出 query、key、value。
- 使用 `repeat_interleave` 讓 key/value 的 head 數對應 query。
- 將 RoPE 相對位置編碼套用於 query、key。
- 使用 `Dropout` 處理 attention weights，並還原維度至 `(batch, seq_len, dim)`。

---

#### LlamaLayer 層級設計

每層 transformer block 包含：
1. RMSNorm
2. Self-Attention + 殘差連接
3. RMSNorm
4. SwiGLU 前饋神經網路 + 殘差連接

其中 SwiGLU 實作為：
$$
\text{SwiGLU}(x) = \text{SiLU}(W_1 x) \cdot (W_3 x)
$$

#### forward 與 generate 方法

- `forward()`：將輸入 token embedding 經過多層 transformer，並最後輸出 logits 或 hidden state。
- `generate()`：以 temperature 決定採用 argmax 或 multinomial sampling，每次都重新前傳（無使用 KV cache）。

---

### ✅ `classifier.py` — 分類器模組

#### ZeroShotClassifier

直接使用模型語言建模能力進行分類：
- 將每個 label 的字串經 tokenizer 轉為 token ids。
- 在 logits 上使用 `log_softmax`，取對應 token 的機率總和作為分數。

---

#### LlamaEmbeddingClassifier

此類別進行微調訓練並預測：
1. 前向傳遞取得最後一個 token 的 hidden state。
2. 套用 dropout，避免 overfitting。
3. 用 linear 層轉為 logit，再用 `log_softmax` 輸出類別分數。

支援模式切換（`pretrain` 固定 LLaMA、`finetune` 微調整體參數）。

---

### ✅ `optimizer.py` — 自製 AdamW 優化器

- 使用 `state['m']`、`state['v']` 記錄一階與二階動量。
- 使用簡化的 bias 修正公式計算 `alpha_t`：
$$
\alpha_t = lr \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$
- 權重更新包含 weight decay：
$$
\theta = \theta - \alpha_t \cdot \left( \frac{m}{\sqrt{v} + \epsilon} + \lambda \cdot \theta \right)
$$

---

### ✅ `rope.py` — Rotary Position Embedding

RoPE 將位置資訊編碼為複數旋轉，用來改善 transformer 處理長距序列的能力：
$$
\text{RoPE}(x) = x \cdot \cos(\theta) + x_{\perp} \cdot \sin(\theta)
$$

我依據 token 順序與頻率產生旋轉矩陣，將 query/key 分解為實數與虛數後進行旋轉再重組。使用 `torch.outer` 計算 position × frequency 表達方式。

---

## 🚀 三種評估模式與指令說明

### 1️⃣ 文字生成（generate）
指令：  
```bash
python run_llama.py --option generate
```
Temperature = 0：選取最有可能字元；=1 則增加隨機性。

---

### 2️⃣ Zero-Shot 分類（prompt）

```bash
python run_llama.py --option prompt --batch_size 10 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-prompting-output.txt \
  --test_out cfimdb-test-prompting-output.txt
```
Epoch | Train Acc | Dev Acc
    0 | 0.261 | 0.262
    1 | 0.273 | 0.253
    2 | 0.401 | 0.361
    3 | 0.517 | 0.392
    4 | 0.688 | 0.414 

Dev Accuracy：0.414
Test Accuracy：0.418

---

### 3️⃣ 微調分類（finetune）

```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-finetuning-output.txt \
  --test_out sst-test-finetuning-output.txt
```
Epoch | Train Loss | Train Acc | Dev Acc
    1 | 1.074 | 0.688 | 0.414
    2 | 1.002 | 0.742 | 0.480
    3 | 0.958 | 0.770 | 0.526
    4 | 0.912 | 0.796 | 0.558
    5 | 0.872 | 0.818 | 0.574 

精度最高可達 SST: 41.8%、CFIMDB: 80%。

---

## 🎯 作業提交與評分規範

- 使用 `prepare_submit.py` 進行驗證。



# Min-Llama Assignment
This is an assignment for NLP 244 Advanced Machine Learning for Natural Language Processing course in the UC Santa Cruz [NLP MS Program](https://nlp.ucsc.edu/).

Forked from CMU's [minllama-assignment](https://github.com/neubig/minllama-assignment) for [CS11-711 Advanced NLP](http://phontron.com/class/anlp2024/).

Originally created by Vijay Viswanathan (based on the previous [minbert-assignment](https://github.com/neubig/minbert-assignment))

## Introduction

This is an exercise in developing a minimalist version of Llama2.

In this assignment, you will implement some important components of the Llama2 model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with this model.

## Assignment Details

### Your task
The code to implement can be found in `llama.py`, `classifier.py` and `optimizer.py`. You are reponsible for writing _core components_ of Llama2 (one of the leading open source language models). In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from `stories42M.pt`; an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

Once you have implemented these components, you will test our your model in 3 settings:
1) Generate a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Perform task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.
4) If you've done #1-3 well, you will get an A! However, since you've come this far, try implementing something new on top of your hand-written language modeling system! If your method provides strong empirical improvements or demonstrates exceptional creativity, you'll get an A+ on this assignment.

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* The `data/cfimdb-test.txt` file provided to you does **not** contain gold-labels, and contains a placeholder negative (-1) label. Evaluating your code against this set will show lower accuracies so do not worry if the numbers don't make sense.
* We will run your code with commands below (under "Reference outputs/accuracies"), so make sure that whatever your best results are reproducible using these commands.
    * Do not change any of the existing command options (including defaults) or add any new required parameters

## Reference outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

### Submission Instructions

**Code:**
You will submit a full code package, with output files, on **Canvas**.

**Report (optional):** Your zip file can include a pdf file, named CRUZID-report.pdf, if (1) you've implemented something else on top of the requirements and further improved accuracy for possible extra points (see "Grading" below), and/or (2) if your best results are with some hyperparameters other than the default, and you want to specify how we should run your code. If you're doing (1), we expect your report should be 1-2 pages, but no more than 3 pages. If you're doing (2), the report can be very brief.

#### Canvas Submission

For submission via [Canvas](https://canvas.cmu.edu/),
the submission file should be a zip file with the following structure (assuming the
lowercase CruzID is ``CRUZID``):
```
CRUZID/
├── run_llama.py
├── base_llama.py
├── llama.py
├── rope.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── [OPTIONAL] sst-dev-advanced-output.txt
├── [OPTIONAL] sst-test-advanced-output.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── sst-dev-finetuning-output.txt
├── sst-test-finetuning-output.txt
├── [OPTIONAL] cfimdb-dev-advanced-output.txt
├── [OPTIONAL] cfimdb-test-advanced-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── cfimdb-dev-finetuning-output.txt
├── cfimdb-test-finetuning-output.txt
└── setup.sh
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It
will throw assertion errors if the format is not expected, and *submissions that fail
this check will be graded down*.

Usage:
1. To create and check a zip file with your outputs, run
   `python3 prepare_submit.py path/to/your/output/dir CRUZID`
2. To check your zip file, run
   `python3 prepare_submit.py path/to/your/submit/zip/file.zip CRUZID`

Please double check this before you submit to Canvas; most recently we had about 10/100
students lose a 1/3 letter grade because of an improper submission format.


### Grading
* A+: (Advanced implementation) You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements or demonstrate exceptional creativity. This improvement can be in either the zero-shot setting (no task-specific finetuning required) or in the finetuning setting (improving over our current finetuning implementation). Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the language modeling objective to do domain adaptation
    * enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.
    * perform [prompt-based finetuning](https://arxiv.org/abs/2109.01247)
    * add [regularization](https://arxiv.org/abs/1909.11299) to our finetuning process
    * try parameter-efficient finetuning (see Section 2.2 [here](https://arxiv.org/abs/2110.04366) for an overview)
    * try alternative fine-tuning algorithms e.g. [SMART](https://www.aclweb.org/anthology/2020.acl-main.197) or [WiSE-FT](https://arxiv.org/abs/2109.01903)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code such that coherent text (i.e. mostly grammatically well-formed) can be generated and the model achieves comparable accuracy (within 0.05 accuracy for SST or 0.15 accuracy for CFIMDB) to our reference implementation.
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option prompt`` and ``--option finetune`` code but coherent text is not generated (i.e. generated text is not well-formed English) or accuracy is not comparable to the reference (accuracy is more than 0.05 accuracy or 0.15 accuracy from our reference scores, for for SST and CFIMDB, respectively).
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (llama implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

If your results can be confirmed through the submitted files, but there are problems with your
code submitted through Canvas, such as not being properly formatted, not executing in
the appropriate amount of time, etc., you will be graded down 1/3 grade (e.g. A+ -> A or A- -> B+).

All assignments must be done individually and we will be running plagiarism detection
on your code. If we confirm that any code was plagiarized from that of other students
in the class, you will be subject to strict measure according to CMUs academic integrity
policy. That being said, *you are free to use publicly available resources* (e.g. papers or open-source
code), but you ***must provide proper attribution***.

### Acknowledgement
This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).


