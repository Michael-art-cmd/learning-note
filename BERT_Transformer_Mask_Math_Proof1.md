# BERT 掩码与 Transformer 掩码：数学推导与实现

---

## 第一部分：BERT Masked Language Model（MLM）

### 1.1 目标

给定一个句子 $x = (x_1, x_2, ..., x_n)$，随机掩码掉某些 token，让模型根据**双向上下文**预测被掩码的词。

### 1.2 掩码过程（数学描述）

设原始序列为 $w = [w_1, w_2, w_3, w_4, w_5]$

**步骤 1：选择掩码位置**

随机选择 15% 的位置集合 $\mathcal{M} \subset \{1, ..., n\}$

设 $n=5$，选择位置 $\{2, 4\}$ 作为掩码位置

**步骤 2：掩码策略（3 种情况）**

设 $p = 0.15$，
$i \in M$ 
为选中掩码的位置

$$
\text{maskedToken}_i = 
\begin{cases}
\text{[MASK]} & \text{概率 } 0.8 \\
\text{randomToken} & \text{概率 } 0.1 \\
\text{originalToken}_i & \text{概率 } 0.1 \\
\end{cases}
$$

**举例**：

原始: $[w_1, w_2, w_3, w_4, w_5] =$ [今天, 天气, 很好, 吧, ？]

| 位置 | 原始 | 随机数 r | 策略 | 结果 |
|------|------|----------|------|------|
| 2 | 天气 | r=0.3 → 0.8 | [MASK] | 天气→[MASK] |
| 4 | 吧 | r=0.85 → 0.1 | 随机 | 吧→很 |

最终输入: [今天, [MASK], 很好, 很, ？]

### 1.3 前向传播（数学推导）

**输入：token IDs**

设词表大小为 $|V|$，每个 token 转为 one-hot 向量 $e_{token} \in \mathbb{R}^{|V|}$

---

#### 步骤 1：Token Embedding

$$
\mathbf{x}^{(0)}_i = E \cdot e_{w_i} + \text{PosEmbedding}_i + \text{SegmentEmbedding}_i
$$

- $E \in \mathbb{R}^{d \times |V|}$：Token 嵌入矩阵
- $\text{PosEmbedding}_i$：位置编码
- $\text{SegmentEmbedding}_i$：句子 A/B 嵌入

**举例**（简化）：

假设 $d=4$，词表大小 $|V|=10000$

$w_2 = \text{[MASK]}$ → one-hot $e_{\text{[MASK]}} \in \mathbb{R}^{10000}$

$$x^{(0)}_2 = E \cdot e_{\text{[MASK]}} + \text{PosEmbedding}_2$$

设 $E \cdot e_{\text{[MASK]}} = [0.1, 0.3, -0.2, 0.5]^T$

---

#### 步骤 2：Transformer Encoder 层（多层）

每一层 $l = 1, ..., L$：

**Multi-Head Self-Attention：**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q = x^{(l-1)}W^Q$
- $K = x^{(l-1)}W^K$
- $V = x^{(l-1)}W^V$
- $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$

**举例**（单头，简化）：

设 $d=4$，$x^{(0)}_2 = [0.1, 0.3, -0.2, 0.5]^T$

随机初始化（训练前）：$W^Q = W^K = W^V = I_4$

则 $Q = K = V = x^{(0)}_2$

计算注意力分数：

$$
QK^T = 
\begin{bmatrix}
0.1 \\ 0.3 \\ -0.2 \\ 0.5
\end{bmatrix}
\begin{bmatrix}
0.1 & 0.3 & -0.2 & 0.5
\end{bmatrix}
=
\begin{bmatrix}
0.01 & 0.03 & -0.02 & 0.05 \\
0.03 & 0.09 & -0.06 & 0.15 \\
-0.02 & -0.06 & 0.04 & -0.10 \\
0.05 & 0.15 & -0.10 & 0.25
\end{bmatrix}
$$

缩放：$d_k = 4$，$\sqrt{d_k} = 2$

$$
\frac{QK^T}{\sqrt{d_k}} = 
\begin{bmatrix}
0.005 & 0.015 & -0.01 & 0.025 \\
0.015 & 0.045 & -0.03 & 0.075 \\
-0.01 & -0.03 & 0.02 & -0.05 \\
0.025 & 0.075 & -0.05 & 0.125
\end{bmatrix}
$$

softmax（按行）：

$\text{Attention}_2 = \text{softmax}(QK^T / \sqrt{d_k}) \cdot V$

计算第一行（位置 2 对所有位置的注意力）：

$\text{scores}_2 = [0.005, 0.015, -0.01, 0.025]$

softmax 结果：$\text{attn}_2 \approx [0.25, 0.27, 0.23, 0.25]$（所有位置几乎均匀）

---

**残差连接 + LayerNorm：**

$$
x^{(l)}_{\text{attn}} = \text{LayerNorm}(x^{(l-1)} + \text{Attention}(x^{(l-1)}))
$$

**FFN 前馈网络：**

$$
x^{(l)} = \text{LayerNorm}(x^{(l)}_{\text{attn}} + \text{FFN}(x^{(l)}_{\text{attn}}))
$$

其中 $\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2$，$\sigma$ 是 ReLU

---

#### 步骤 3：输出层

设最后一层输出 $h_i^{(L)}$，投影到词表：

$$
\text{logits}_i = h_i^{(L)} \cdot W^T + b
$$

其中 $W \in \mathbb{R}^{d \times |V|}$（与 token 嵌入共享）

**举例**：

假设 $h_2^{(L)} = [0.2, 0.1, -0.3, 0.4]^T$

$W$ 的某些行（假设）：

$W_{\text{天气}} = [1.0, 2.0, 0.5, 1.5]$  
$W_{\text{气候}} = [0.8, 1.9, 0.4, 1.3]$  
$W_{\text{情况}} = [0.5, 1.0, 0.2, 0.8]$

计算 logits：

$\text{logits}_2[\text{天气}] = h_2^{(L)} \cdot W_{\text{天气}}^T = 0.85$

$\text{logits}_2[\text{气候}] = h_2^{(L)} \cdot W_{\text{气候}}^T = 0.75$

### 1.4 损失函数（交叉熵）

**步骤：Softmax → Cross-Entropy**

对位置 $i \in M$（掩码位置）：

$$
p_i(\text{token}) = \text{softmax}(\text{logits}_i) = \frac{\exp(\text{logits}_i[\text{token}])}{\sum_{v \in V} \exp(\text{logits}_i[v])}
$$

**举例**（简化，假设词表只有 3 个词）：

| 词 | logits | exp(logits) | softmax |
|----|--------|-------------|---------|
| 天气 | 0.85 | 2.34 | 0.35 |
| 气候 | 0.75 | 2.12 | 0.32 |
| 情况 | 0.50 | 1.65 | 0.25 |

正确答案是"天气"，所以目标分布 $y$：$y_{\text{天气}} = 1, \ y_{\text{其他}} = 0$

---

**交叉熵损失**：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \sum_{v \in V} y_{i,v} \log p_{i}(v)
$$

简化为：

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p_{i}(\text{trueToken})
$$

**举例**：

$p_2(\text{天气}) = 0.35$

$\mathcal{L}_2 = -\log(0.35) = 1.05$

梯度计算（$p_i - y_i$）：

| 词 | softmax | 目标 y | 梯度 (p-y) |
|----|---------|--------|------------|
| 天气 | 0.35 | 1 | **-0.65** |
| 气候 | 0.32 | 0 | 0.32 |
| 情况 | 0.25 | 0 | 0.25 |

### 1.5 完整训练流程总结

```
输入句子 → [今天, 天气, 很好, 吧, ？]
            ↓
随机掩码 → [今天, [MASK], 很好, 很, ？]
            ↓
Token Embedding → x^{(0)}_1, ..., x^{(0)}_5
            ↓
12 层 Transformer Encoder
            ↓
输出每个位置的 logits
            ↓
计算掩码位置的交叉熵损失
            ↓
反向传播更新参数
```

---

## 第二部分：Transformer Attention Mask

### 2.1 Padding Mask（填充掩码）

**问题**：不同句子长度不同，需要 padding 到统一长度

**举例**：

| 句子 | Token IDs |
|------|-----------|
| "今天 天气 好" | [101, 202, 301, 202, 102] |
| "我 爱" | [501, 702, 102, 0, 0] |

其中 0 是 [PAD]，102 是 [SEP]

---

#### 掩码矩阵构建（数学）

设 batch size = 2，max_len = 5

**Padding Mask 矩阵** $M_{\text{pad}} \in \{0, 1\}^{B \times L}$：

$$
M_{\text{pad}}[b, i] = 
\begin{cases}
1 & \text{如果位置 } i \text{ 是有效 token} \\
0 & \text{如果位置 } i \text{ 是 [PAD]}
\end{cases}
$$

**举例**：

$$
M_{\text{pad}} = 
\begin{bmatrix}
1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 0 & 0
\end{bmatrix}
$$

---

#### 应用到注意力计算

标准注意力公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

其中 $M$ 是掩码矩阵，需要转换为 $- \infty$ 形式：

$$
M_{\text{soft}} = M_{\text{padMask}} \cdot 0 - (1 - M_{\text{padMask}}) \cdot \infty
$$

**举例**（batch 2，位置 i=0）：

原始注意力分数：

$$
\frac{QK^T}{\sqrt{d_k}} = 
\begin{bmatrix}
0.5 & 0.3 & 0.1 & 0.2 & 0.1 \\
0.2 & 0.6 & 0.4 & 0.1 & 0.3 \\
0.1 & 0.2 & 0.8 & 0.3 & 0.2 \\
0.3 & 0.1 & 0.2 & 0.7 & 0.4 \\
0.2 & 0.1 & 0.3 & 0.2 & 0.5
\end{bmatrix}
$$

应用掩码（位置 3, 4 设为 -inf）：

$$
\frac{QK^T}{\sqrt{d_k}} + M_{\text{soft}} = 
\begin{bmatrix}
0.5 & 0.3 & 0.1 & -\infty & -\infty \\
0.2 & 0.6 & 0.4 & -\infty & -\infty \\
0.1 & 0.2 & 0.8 & -\infty & -\infty \\
0.3 & 0.1 & 0.2 & -\infty & -\infty \\
0.2 & 0.1 & 0.3 & -\infty & -\infty
\end{bmatrix}
$$

softmax（按行）：

第一行：$\exp(0.5) = 1.65$，$\exp(0.3) = 1.35$，$\exp(0.1) = 1.11$

和 = $1.65 + 1.35 + 1.11 = 4.11$

$\text{softmax}_1 = [0.40, 0.33, 0.27, 0, 0]$

**结论**：padding 位置的概率为 0，不参与注意力计算

### 2.2 Causal Mask（因果掩码/序列掩码）

**问题**：自回归模型生成时，不能"偷看"未来 token

**举例**：

生成句子 "今天天气很好"

- 预测第 3 个词 "很" 时，只能看 [今天, 天气]
- 不能看到第 4 个词 "好"

---

#### 掩码矩阵构建（数学）

设序列长度 $L$，定义下三角矩阵：

$$
M_{\text{causal}}[i, j] = 
\begin{cases}
0 & \text{如果 } j \leq i \text{（可以看）} \\
-\infty & \text{如果 } j > i \text{（不能看未来）}
\end{cases}
$$

**举例**（L=4）：

$$
M_{\text{causal}} = 
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

含义：
- 第 0 行：只能看位置 0
- 第 1 行：只能看位置 0, 1
- 第 2 行：只能看位置 0, 1, 2
- 第 3 行：可以看所有位置

---

#### 应用举例

假设 $QK^T$ 计算结果（简化）：

$$
\frac{QK^T}{\sqrt{d_k}} = 
\begin{bmatrix}
0.5 & 0.3 & 0.1 & 0.2 \\
0.2 & 0.6 & 0.4 & 0.1 \\
0.1 & 0.2 & 0.8 & 0.3 \\
0.3 & 0.1 & 0.2 & 0.7
\end{bmatrix}
$$

加上 Causal Mask：

$$
\frac{QK^T}{\sqrt{d_k}} + M_{\text{causal}} = 
\begin{bmatrix}
0.5 & -\infty & -\infty & -\infty \\
0.2 & 0.6 & -\infty & -\infty \\
0.1 & 0.2 & 0.8 & -\infty \\
0.3 & 0.1 & 0.2 & 0.7
\end{bmatrix}
$$

按行 softmax：

| 位置 | 有效值 | softmax 结果 |
|------|--------|--------------|
| 0 | 0.5 | [1.0, 0, 0, 0] |
| 1 | 0.2, 0.6 | [0.40, 0.60, 0, 0] |
| 2 | 0.1, 0.2, 0.8 | [0.24, 0.27, 0.49, 0] |
| 3 | 0.3, 0.1, 0.2, 0.7 | [0.24, 0.20, 0.21, 0.35] |

### 2.3 PyTorch 代码实现对照

```python
import torch

# ========== Padding Mask ==========
def create_padding_mask(seq, pad_idx=0):
    """
    seq: [batch_size, seq_len]
    返回: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()  # 1=有效, 0=pad

# 测试
seq = torch.tensor([[1, 2, 3, 0, 0]])  # batch=1, len=5
mask = create_padding_mask(seq, pad_idx=0)
print(mask)
# tensor([[[[1., 1., 1., 0., 0.]]]])

# ========== Causal Mask ==========
def create_causal_mask(size):
    """
    返回: [1, 1, size, size]
    True = 需要 mask 掉
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)

# 测试
mask = create_causal_mask(4)
print(mask)
# tensor([[[[False, True,  True,  True],
#           [False, False, True,  True],
#           [False, False, False, True],
#           [False, False, False, False]]]])

# 实际使用时：attention_scores.masked_fill(mask, -1e9)
```

---

## 总结对比

| 掩码类型 | BERT MLM | Transformer Attention Mask |
|----------|----------|-----------------------------|
| **目的** | 预训练自监督 | 控制信息流 |
| **位置** | 输入层 | 注意力计算层 |
| **类型** | 随机掩码 token | Padding / Causal |
| **训练时** | 预测被掩码词 | 决定注意力范围 |
| **推理时** | 不使用 [MASK] | 决定是否能看未来 |

---

## 附加：学习到的语义规则是"黑盒"吗？

**部分是黑盒，部分可以解释。**

### 可解释的部分

- **Attention 权重**：可以看到模型在处理"银行"时，关注了"存款"还是"河岸"（消歧义）
- **Probing Probes**：设计探测任务可以发现"主语位置"、"时态信息"被编码在特定层
- **特征可视化**：CNN 第一层学到了边缘、纹理

### 难以解释的部分

- 深层抽象概念（如"因果关系"）分布式编码在整个网络中
- 数百维的向量空间无法直接对应人类可理解的概念
- 涌现能力（Emergent Abilities）不可预测

### 当前研究趋势

| 方法 | 作用 |
|------|------|
| Attention 可视化 | 展示 token 之间的注意力连接 |
| Probing | 探测特定层是否编码语法/语义信息 |
| KAN | 新架构试图让网络更可解释 |
| 概念瓶颈模型 | 强制中间层对应人类可解释概念 |

**结论**：Transformer 是"半可解释"的——可以观察其行为（输入→输出），但中间过程的具体语义规则难以完全显式化。
