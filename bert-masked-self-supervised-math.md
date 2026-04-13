# BERT掩码与Transformer掩码的自监督学习：严格数学推导

> 本文档从Transformer基础开始，一步一步推导BERT掩码自监督的形成过程，并分析语义规则的可解释性。

---

## 目录

- [一、Transformer基础：注意力机制的数学原理](#一transformer基础注意力机制的数学原理)
- [二、BERT掩码机制详解](#二bert掩码机制详解)
- [三、自监督形成过程：从数据到任务的完整推导](#三自监督形成过程从数据到任务的完整推导)
- [四、数学推导：从输入到损失函数的完整流程](#四数学推导从输入到损失函数的完整流程)
- [五、语义规则是黑盒吗？可解释性分析](#五语义规则是黑盒吗可解释性分析)
- [六、BERT vs Transformer原始掩码的区别](#六bert-vs-transformer原始掩码的区别)
- [七、论文推荐](#七论文推荐)

---

## 一、Transformer基础：注意力机制的数学原理

### 1.1 自注意力机制的数学定义

给定输入序列 $X = \{x_1, x_2, ..., x_T\}$，首先将每个token映射为向量：

$$X \in \mathbb{R}^{T \times d_{model}}$$

其中 $T$ 是序列长度，
$d_{model}$ 
是模型维度。

### 1.2 Query、Key、Value的计算

通过三个线性变换得到Query、Key、Value：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中：
- $W_Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_V \in \mathbb{R}^{d_{model} \times d_v}$

通常 $d_k = d_v = d_{model}$。

### 1.3 注意力分数的计算

注意力分数表示每个位置对其他位置的"关注度"：

$$A = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**逐步展开：**

**Step 1：计算Query与Key的点积**

$$S = QK^T \in \mathbb{R}^{T \times T}$$

矩阵 $S$ 的每个元素 $S_{ij}$ 表示位置 $i$ 对位置 $j$ 的原始注意力分数：

$$S_{ij} = q_i \cdot k_j = \sum_{m=1}^{d_k} Q_{im} K_{jm}$$

**Step 2：缩放（防止点积过大）**

$$S' = \frac{S}{\sqrt{d_k}}$$

缩放的原因：当 $d_k$ 很大时，点积的方差会变大，导致softmax的梯度趋近于0。

**数学证明：**

假设 $q$ 和 $k$ 的元素是独立的随机变量，均值为0，方差为1。

则点积 $q \cdot k$ 的方差为：

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

因此需要除以 $\sqrt{d_k}$ 来归一化方差。

**Step 3：Softmax归一化**

$$A_{\text{weights}} = \text{softmax}(S') \in \mathbb{R}^{T \times T}$$

对每一行独立进行softmax：

$$A_{ij} = \frac{\exp(S'_{ij})}{\sum_{j'=1}^{T} \exp(S'_{ij'})}$$

**性质：**
- $\sum_{j=1}^{T} A_{ij} = 1$（每行的注意力权重之和为1）
- $A_{ij} \geq 0$（所有权重非负）

**Step 4：加权求和Value**

$$O = A_{\text{weights}} V \in \mathbb{R}^{T \times d_v}$$

每个位置的输出是所有位置的Value的加权平均：

$$o_i = \sum_{j=1}^{T} A_{ij} v_j$$

### 1.4 多头注意力

为了让模型关注不同类型的关系，使用多头注意力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

其中每个head：

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

---

## 二、BERT掩码机制详解

### 2.1 BERT的两种掩码

BERT使用两种不同的掩码机制：

| 掩码类型 | 作用 | 实现方式 |
|----------|------|----------|
| **Attention Mask** | 控制注意力计算范围 | 在softmax前加上掩码矩阵 |
| **Token Mask (MLM)** | 随机遮蔽token用于预测 | 将token替换为[MASK]或随机词 |

### 2.2 Attention Mask的数学实现

Attention Mask用于：
- 区分真实token和padding token
- 在BERT中实现双向注意力（区别于GPT的单向）

**数学定义：**

定义掩码矩阵 $M \in \mathbb{R}^{T \times T}$：

$$M_{ij} = \begin{cases} 0 & \text{如果位置 } i \text{ 可以关注位置 } j \\ -\infty & \text{如果位置 } i \text{ 不能关注位置 } j \end{cases}$$

**修改后的注意力计算：**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**为什么这样实现？**

当 $M_{ij} = -\infty$ 时：

$$\exp(S'_{ij} + M_{ij}) = \exp(-\infty) = 0$$

因此该位置的注意力权重为0。

### 2.3 Token Mask (MLM)的实现

BERT的Masked Language Model随机遮蔽15%的token：

**遮蔽策略：**

对于每个被选中遮蔽的token：
- 80% 替换为 `[MASK]`
- 10% 替换为随机token
- 10% 保持不变

**为什么要这样设计？**

| 策略 | 原因 |
|------|------|
| 80% `[MASK]` | 主要的自监督任务 |
| 10% 随机token | 防止模型只学会预测`[MASK]`位置 |
| 10% 保持不变 | 让模型学习所有位置的表示，不只是被遮蔽的 |

---

## 三、自监督形成过程：从数据到任务的完整推导

### 3.1 Step 1：原始数据

假设我们有原始文本数据：

$$\text{原始句子} = \{w_1, w_2, w_3, ..., w_T\}$$

例如："科学家 发现 了 新物种"

### 3.2 Step 2：Token化

将文本转换为token ID：

$$X_{\text{raw}} = \{\text{id}_1, \text{id}_2, ..., \text{id}_T\}$$

同时生成词嵌入矩阵 $E \in \mathbb{R}^{|\mathcal{V}| \times d}$，其中 $\mathcal{V}$ 是词表。

每个token的嵌入：

$$e_i = E[\text{id}_i] \in \mathbb{R}^d$$

### 3.3 Step 3：构造遮蔽任务（自监督的关键）

**随机选择遮蔽位置：**

定义遮蔽指示器 $m \in \{0, 1\}^T$：

$$m_i = \begin{cases} 1 & \text{如果位置 } i \text{ 被遮蔽} \\ 0 & \text{如果位置 } i \text{ 未被遮蔽} \end{cases}$$

遮蔽比例约为15%：

$$\frac{\sum_{i=1}^{T} m_i}{T} \approx 0.15$$

**构造遮蔽后的输入：**

$$X_{\text{masked}} = \{\tilde{w}_1, \tilde{w}_2, ..., \tilde{w}_T\}$$

其中：

$$\tilde{w}_i = \begin{cases} [\text{MASK}] & \text{如果 } m_i=1 \text{ 且在80%情况} \\ w_{\text{random}} & \text{如果 } m_i=1 \text{ 且在10%情况} \\ w_i & \text{如果 } m_i=0 \text{ 或在10%情况} \end{cases}$$

**例子：**

```
原始：科学家 发现 了 新物种
遮蔽：科学家 [MASK] 了 新物种（遮蔽"发现"）
```

### 3.4 Step 4：输入编码

遮蔽后的序列经过嵌入层：

$$H^{(0)} = \{h_1^{(0)}, h_2^{(0)}, ..., h_T^{(0)}\}$$

每个位置的初始表示：

$$h_i^{(0)} = e_{\tilde{w}_i} + p_i$$

其中：
- $e_{\tilde{w}_i}$：token embedding
- $p_i$：position embedding（位置编码）

**位置编码的数学形式（BERT使用可学习的位置编码）：**

$$p_i \in \mathbb{R}^d \text{ 是可学习的参数向量}$$

### 3.5 Step 5：Transformer编码器处理

经过 $L$ 层Transformer编码器：

$$H^{(l)} = \text{TransformerLayer}(H^{(l-1)})$$

每一层包含：

**子层1：多头自注意力**

$$H' = \text{MultiHeadAttention}(H^{(l-1)})$$

$$H'' = \text{LayerNorm}(H^{(l-1)} + H')$$

**子层2：前馈网络**

$$H_{\text{FFN}} = \text{FFN}(H'') = \max(0, H''W_1 + b_1)W_2 + b_2$$

$$H^{(l)} = \text{LayerNorm}(H'' + H_{\text{FFN}})$$

**最终输出：**

$$H^{(L)} = \{h_1^{(L)}, h_2^{(L)}, ..., h_T^{(L)}\}$$

### 3.6 Step 6：预测被遮蔽的token

对于每个被遮蔽的位置 $i$（即 $m_i = 1$），计算预测概率：

$$P(w | \tilde{X}) = \text{softmax}(h_i^{(L)} W_{\text{pred}} + b_{\text{pred}})$$

其中 $W_{\text{pred}} \in \mathbb{R}^{d \times |\mathcal{V}|}$。

**具体概率：**

$$P(w_j | \tilde{X}) = \frac{\exp(h_i^{(L)} \cdot w_j^{\text{pred}})}{\sum_{k \in \mathcal{V}} \exp(h_i^{(L)} \cdot w_k^{\text{pred}})}$$

### 3.7 Step 7：计算损失函数

**Masked Language Model损失：**

$$\mathcal{L}_{\text{MLM}} = -\sum_{i: m_i=1} \log P(w_i | \tilde{X})$$

即对所有被遮蔽位置，计算预测正确token的负对数概率。

---

## 四、数学推导：从输入到损失函数的完整流程

### 4.1 完整推导示例

假设一个具体例子：

**原始句子：** $X = \{\text{科学家}, \text{发现}, \text{了}, \text{新物种}\}$

**词表：** $\mathcal{V} = \{\text{科学家}, \text{发现}, \text{了}, \text{新物种}, \text{研究}, \text{吃}, \text{睡觉}\}$

**词表大小：** $|\mathcal{V}| = 7$

**模型维度：** $d = 4$（简化）

**Step 1：Token ID**

假设token ID：
- 科学家 → id=0
- 发现 → id=1
- 了 → id=2
- 新物种 → id=3

**Step 2：嵌入矩阵**

$$E = \begin{bmatrix} 
e_{\text{科学家}} \\
e_{\text{发现}} \\
e_{\text{了}} \\
e_{\text{新物种}} \\
e_{\text{研究}} \\
e_{\text{吃}} \\
e_{\text{睡觉}}
\end{bmatrix} = \begin{bmatrix} 
0.1 & 0.2 & 0.3 & 0.4 \\
0.5 & 0.6 & 0.7 & 0.8 \\
0.2 & 0.3 & 0.1 & 0.4 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.4 & 0.5 & 0.6 & 0.7 \\
0.1 & 0.1 & 0.1 & 0.1 \\
0.0 & 0.0 & 0.0 & 0.0
\end{bmatrix}$$

**Step 3：遮蔽**

假设遮蔽位置2（"发现"），替换为`[MASK]`：

$$\tilde{X} = \{\text{科学家}, [\text{MASK}], \text{了}, \text{新物种}\}$$

遮蔽指示器：$m = \{0, 1, 0, 0\}$

**Step 4：初始嵌入**

假设`[MASK]`的嵌入：$e_{\text{MASK}} = [0.0, 0.0, 0.0, 0.0]$

位置编码（可学习）：

$$P = \begin{bmatrix} p_1 \\ p_2 \\ p_3 \\ p_4 \end{bmatrix} = \begin{bmatrix} 0.01 & 0.02 & 0.01 & 0.02 \\ 0.02 & 0.01 & 0.02 & 0.01 \\ 0.01 & 0.01 & 0.01 & 0.01 \\ 0.02 & 0.02 & 0.02 & 0.02 \end{bmatrix}$$

初始表示：

$$H^{(0)} = E[\tilde{X}] + P$$

$$= \begin{bmatrix} 
0.1+0.01 & 0.2+0.02 & 0.3+0.01 & 0.4+0.02 \\
0.0+0.02 & 0.0+0.01 & 0.0+0.02 & 0.0+0.01 \\
0.2+0.01 & 0.3+0.01 & 0.1+0.01 & 0.4+0.01 \\
0.3+0.02 & 0.4+0.02 & 0.5+0.02 & 0.6+0.02
\end{bmatrix}$$

$$= \begin{bmatrix} 
0.11 & 0.22 & 0.31 & 0.42 \\
0.02 & 0.01 & 0.02 & 0.01 \\
0.21 & 0.31 & 0.11 & 0.41 \\
0.32 & 0.42 & 0.52 & 0.62
\end{bmatrix}$$

**Step 5：自注意力计算（单层简化）**

假设权重矩阵：

$$W_Q = W_K = W_V = I_{4 \times 4}$$（简化为单位矩阵）

计算 $Q, K, V$：

$$Q = K = V = H^{(0)}$$

计算注意力分数：

$$S = QK^T = H^{(0)} H^{(0)T}$$

$$= \begin{bmatrix} 
0.11 & 0.22 & 0.31 & 0.42 \\
0.02 & 0.01 & 0.02 & 0.01 \\
0.21 & 0.31 & 0.11 & 0.41 \\
0.32 & 0.42 & 0.52 & 0.62
\end{bmatrix} \begin{bmatrix} 
0.11 & 0.02 & 0.21 & 0.32 \\
0.22 & 0.01 & 0.31 & 0.42 \\
0.31 & 0.02 & 0.11 & 0.52 \\
0.42 & 0.01 & 0.41 & 0.62
\end{bmatrix}$$

计算第一个元素（位置1对位置1的注意力）：

$$S_{11} = 0.11 \times 0.11 + 0.22 \times 0.22 + 0.31 \times 0.31 + 0.42 \times 0.42$$

$$= 0.0121 + 0.0484 + 0.0961 + 0.1764 = 0.333$$

类似计算其他元素（简化展示）：

$$S = \begin{bmatrix} 
0.333 & 0.032 & 0.297 & 0.447 \\
0.032 & 0.007 & 0.032 & 0.047 \\
0.297 & 0.032 & 0.267 & 0.397 \\
0.447 & 0.047 & 0.397 & 0.614
\end{bmatrix}$$

缩放（$\sqrt{d_k} = \sqrt{4} = 2$）：

$$S' = \frac{S}{2}$$

Softmax（每行独立）：

对于位置2（被遮蔽的位置）：

$$A_{21} = \frac{\exp(0.016)}{\exp(0.016) + \exp(0.0035) + \exp(0.016) + \exp(0.0235)}$$

$$= \frac{1.016}{1.016 + 1.0035 + 1.016 + 1.0235} = \frac{1.016}{4.059} \approx 0.250$$

完整注意力权重矩阵（近似）：

$$A_{\text{weights}} \approx \begin{bmatrix} 
0.25 & 0.24 & 0.25 & 0.26 \\
0.25 & 0.24 & 0.25 & 0.26 \\
0.24 & 0.24 & 0.25 & 0.27 \\
0.23 & 0.23 & 0.24 & 0.30
\end{bmatrix}$$

**关键观察：位置2（被遮蔽）的注意力分布**

被遮蔽位置`[MASK]`会从其他位置（科学家、了、新物种）聚合信息。

加权求和：

$$h_2^{(L)} = \sum_{j=1}^{4} A_{2j} v_j$$

$$= 0.25 \times h_1^{(0)} + 0.24 \times h_2^{(0)} + 0.25 \times h_3^{(0)} + 0.26 \times h_4^{(0)}$$

$$= 0.25[0.11, 0.22, 0.31, 0.42] + 0.24[0.02, 0.01, 0.02, 0.01] + ...$$

**Step 6：预测**

假设预测权重矩阵：

$$W_{\text{pred}} = E^T$$（使用词嵌入作为预测权重）

计算被遮蔽位置的预测分数：

$$s = h_2^{(L)} W_{\text{pred}}$$

对于词"发现"（id=1）：

$$s_1 = h_2^{(L)} \cdot e_{\text{发现}}$$

Softmax得到概率：

$$P(\text{发现} | \tilde{X}) = \frac{\exp(s_1)}{\sum_{k=0}^{6} \exp(s_k)}$$

**Step 7：损失函数**

$$\mathcal{L}_{\text{MLM}} = -\log P(\text{发现} | \tilde{X})$$

假设模型预测"发现"的概率为 $p = 0.6$：

$$\mathcal{L} = -\log(0.6) = 0.51$$

### 4.2 数学推导总结

完整流程的数学表达式：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i: m_i=1} \log \frac{\exp(h_i^{(L)} \cdot e_{w_i})}{\sum_{k \in \mathcal{V}} \exp(h_i^{(L)} \cdot e_k)}$$

其中：

$$h_i^{(L)} = \text{Transformer}^{(L)}(e_{\tilde{w}_i} + p_i)$$

---

## 五、语义规则是黑盒吗？可解释性分析

### 5.1 什么是"黑盒"？

**黑盒的定义：**
- 输入和输出之间的关系无法用人类可理解的规则解释
- 内部工作机制完全不可知

**BERT是否是黑盒？**

这是一个复杂的问题，需要从多个角度分析。

### 5.2 BERT学到的语义规则的类型

研究表明，BERT确实学到了多种可解释的语义规则：

#### 类型1：语法知识

| 语法现象 | BERT表现 | 证据 |
|----------|----------|------|
| 主谓一致 | 能区分单复数 | "The dog runs" vs "The dogs run" |
| 词序敏感 | 知道形容词在名词前 | "red car" vs "car red" |
| 句法树结构 | 注意力头对应句法关系 | 特定的head关注主谓关系 |

**数学验证：**

设probe task预测语法关系：

$$P(\text{relation} | h_i) = \text{softmax}(h_i W_{\text{probe}})$$

如果probe准确率高，说明 $h_i$ 蕴含语法信息。

#### 类型2：语义角色

| 语义角色 | BERT表现 |
|----------|----------|
| 施事者 | 能识别动作执行者 |
| 受事者 | 能识别动作承受者 |
| 时间/地点 | 能识别修饰成分 |

#### 类型3：世界知识

| 知识类型 | BERT表现 |
|----------|----------|
| 实体关系 | "巴黎是法国首都"能预测 |
| 常识推理 | 能做简单的常识判断 |

### 5.3 可解释性研究方法

#### 方法1：Probe任务

**定义：** 用简单的分类器探测BERT表示中是否包含特定信息。

$$\text{Probe}(h) = \arg\max_c P(c | h) = \arg\max_c \text{softmax}(hW_{\text{probe}})$$

**例子：探测词性标注**

```python
# 假设BERT的隐藏状态
h = BERT("The cat sat")  # h.shape = (3, 768)

# Probe分类器
probe = LinearProbe(768, num_pos_tags)

# 预测词性
pos_predictions = probe(h)
# 如果准确率高，说明h包含词性信息
```

**研究结果：**
- BERT的隐藏状态包含丰富的语法信息
- 不同层包含不同类型的信息
  - 低层：词法信息（词性、形态）
  - 中层：句法信息（句法树、依存关系）
  - 高层：语义信息（语义角色、共指消解）

#### 方法2：注意力可视化

**分析注意力权重矩阵：**

$$A_{ij} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)_{ij}$$

观察哪些位置对哪些位置的注意力高。

**研究发现：**

| 现象 | 注意力模式 |
|------|----------|
| 主谓关系 | 特定的head让主语关注谓语 |
| 修饰关系 | 形容词head关注名词 |
| 指代消解 | 代词关注其指代的实体 |

#### 方法3：Layer-wise分析

**不同层学到的信息：**

```
Layer 1-4:   词法信息（词性、词形变化）
Layer 5-8:   句法信息（句法树、依存关系）
Layer 9-12:  语义信息（语义角色、世界知识）
```

**数学表示：**

定义信息含量指标：

$$I(\text{feature}; h^{(l)}) = H(\text{feature}) - H(\text{feature} | h^{(l)})$$

其中 $H$ 是熵函数。

### 5.4 黑盒性分析

#### 部分可解释的部分

| 可解释的部分 | 证据 |
|--------------|------|
| 语法规则 | Probe实验高准确率 |
| 注意力模式 | 特定head对应句法关系 |
| 层次结构 | 不同层学不同信息 |

#### 黑盒的部分

| 黑盒的部分 | 原因 |
|------------|------|
| 组合语义 | 多个规则如何组合仍是黑盒 |
| 推理过程 | 从输入到输出的决策路径不清晰 |
| 高层语义 | 复杂语义关系难以完全解释 |

### 5.5 数学角度的黑盒分析

**可解释性的数学定义：**

给定模型 $f: X \rightarrow Y$，如果存在人类可理解的规则集 $\mathcal{R}$ 使得：

$$f(x) \approx \bigoplus_{r \in \mathcal{R}} r(x)$$

则模型是可解释的。

**BERT的情况：**

BERT的预测可以用部分规则解释：

$$P(w_i | \tilde{X}) = \text{softmax}(h_i^{(L)} \cdot E)$$

其中 $h_i^{(L)}$ 包含：

$$h_i^{(L)} = \sum_{\text{rule } r} \alpha_r \cdot r(\tilde{X})$$

但 $\alpha_r$ 的值和 $r$ 的具体形式不完全可解释。

### 5.6 结论：半黑盒

**BERT是"半黑盒"：**

1. **部分可解释：**
   - 学到的语法规则可以探测和验证
   - 注意力模式有一定的语言学意义
   - 层次结构对应语言学层次

2. **部分黑盒：**
   - 规则组合的方式不完全清楚
   - 高层语义决策难以完全解释
   - 存在"涌现"行为（大模型才有）

---

## 六、BERT vs Transformer原始掩码的区别

### 6.1 Transformer原始掩码（用于解码器）

**目的：** 防止解码器看到未来的token（单向注意力）

**数学实现：**

$$M_{ij} = \begin{cases} 0 & \text{如果 } j \leq i \\ -\infty & \text{如果 } j > i \end{cases}$$

**因果掩码矩阵：**

$$M = \begin{bmatrix} 
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}$$

**效果：**

$$\text{softmax}(S + M)_{ij} = \begin{cases} \frac{\exp(S_{ij})}{\sum_{k \leq i} \exp(S_{ik})} & \text{如果 } j \leq i \\ 0 & \text{如果 } j > i \end{cases}$$

### 6.2 BERT掩码（用于自监督学习）

**目的：** 构造预测任务，让模型学习双向理解

**数学实现：**

$$\tilde{w}_i = \begin{cases} [\text{MASK}] & \text{随机选择的15%位置} \\ w_i & \text{其他位置} \end{cases}$$

**损失函数：**

$$\mathcal{L} = -\sum_{i \in \mathcal{M}} \log P(w_i | \tilde{X})$$

### 6.3 关键区别总结

| 方面 | Transformer原始掩码 | BERT掩码 |
|------|---------------------|----------|
| **位置** | 解码器 | 编码器 |
| **目的** | 防止看到未来 | 构造自监督任务 |
| **类型** | Attention Mask | Token Mask |
| **注意力方向** | 单向 | 双向 |
| **自监督** | 无（用于生成任务） | 有（MLM任务） |

### 6.4 数学对比

**Transformer解码器：**

$$P(x_t | x_{<t}) = \text{softmax}(h_t \cdot E)$$

其中 $h_t$ 只能看到 $x_1, ..., x_{t-1}$。

**BERT编码器：**

$$P(x_t | \tilde{X}) = \text{softmax}(h_t \cdot E)$$

其中 $h_t$ 可以看到所有未被遮蔽的token。

---

## 七、论文推荐

### 7.1 自监督与BERT核心论文

| 论文 | 内容 | 链接 |
|------|------|------|
| **BERT: Pre-training of Deep Bidirectional Transformers** | BERT原始论文，MLM任务设计 | https://arxiv.org/abs/1810.04805 |
| **Attention Is All You Need** | Transformer原始论文 | https://arxiv.org/abs/1706.03762 |

### 7.2 可解释性研究论文

| 论文 | 内容 | 链接 |
|------|------|------|
| **"What Does BERT Look At?"** | 分析BERT注意力头 | https://arxiv.org/abs/1906.04341 |
| **"Probing Linguistic Features in BERT"** | Probe任务探测语法信息 | https://arxiv.org/abs/1905.05950 |
| **"BERT Rediscovers the Classical NLP Pipeline"** | 分析BERT层次结构 | https://aclanthology.org/P19-1452/ |
| **"Are Sixteen Heads Really Better than One?"** | 分析注意力头的作用 | https://arxiv.org/abs/1905.10650 |

### 7.3 阅读顺序建议

1. **Attention Is All You Need** → 理解Transformer基础
2. **BERT原始论文** → 理解MLM自监督任务
3. **"What Does BERT Look At?"** → 理解注意力可解释性
4. **"BERT Rediscovers the Classical NLP Pipeline"** → 理解层次结构

---

## 八、总结：自监督形成的关键数学表达式

### 8.1 完整的自监督过程

$$\boxed{\mathcal{L}_{\text{MLM}} = -\sum_{i: m_i=1} \log \frac{\exp(h_i^{(L)} \cdot e_{w_i})}{\sum_{k \in \mathcal{V}} \exp(h_i^{(L)} \cdot e_k)}}$$

其中各组件：

$$\begin{aligned}
h_i^{(L)} &= \text{Transformer}^{(L)}(e_{\tilde{w}_i} + p_i) \\
&= \text{LayerNorm}(\text{Attention}(H^{(L-1)}) + H^{(L-1)}) \\
\text{Attention}(H) &= \text{softmax}\left(\frac{HW_Q \cdot (HW_K)^T}{\sqrt{d_k}} + M\right)HW_V \\
m_i &= \mathbb{1}[\text{位置 } i \text{ 被遮蔽}] \\
\tilde{w}_i &= \begin{cases} [\text{MASK}] & m_i=1 \\ w_i & m_i=0 \end{cases}
\end{aligned}$$

### 8.2 关键洞察

**为什么自监督有效？**

$$\frac{\partial \mathcal{L}}{\partial W_Q} = -\sum_{i: m_i=1} \frac{\partial \log P(w_i)}{\partial h_i^{(L)}} \cdot \frac{\partial h_i^{(L)}}{\partial \text{Attention}} \cdot \frac{\partial \text{Attention}}{\partial W_Q}$$

梯度回传迫使：
- $W_Q, W_K$ 学习让相关位置的注意力权重更高
- $W_V$ 学习聚合有用的信息
- 整个模型学习理解上下文语义

---

## 附录：代码示例

### 简化的BERT MLM实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedBERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) 
            for _ in range(n_layers)
        ])
        
        # MLM prediction head
        self.mlm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask_positions):
        """
        x: [batch, seq_len] - masked input tokens
        mask_positions: [batch, seq_len] - binary mask indicating masked positions
        """
        # Step 1: Token + Position embedding
        seq_len = x.size(1)
        positions = torch.arange(seq_len).unsqueeze(0)
        h = self.embedding(x) + self.position_embedding(positions)
        
        # Step 2: Transformer encoding
        for layer in self.layers:
            h = layer(h)
        
        # Step 3: MLM prediction (only for masked positions)
        logits = self.mlm_head(h)
        
        return logits
    
    def compute_loss(self, logits, target_tokens, mask_positions):
        """
        logits: [batch, seq_len, vocab_size]
        target_tokens: [batch, seq_len] - original tokens before masking
        mask_positions: [batch, seq_len] - binary mask
        """
        # Only compute loss for masked positions
        masked_logits = logits[mask_positions.bool()]
        masked_targets = target_tokens[mask_positions.bool()]
        
        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention with residual
        x = self.norm1(x + self.attention(x, x, x))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        return output
```

---

## 参考文献完整列表

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
2. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
3. Clark, K., et al. "What Does BERT Look At? An Analysis of BERT's Attention." ACL 2019.
4. Tenney, I., et al. "BERT Rediscovers the Classical NLP Pipeline." ACL 2019.
5. Hewitt, J., & Manning, C. "A Structural Probe for Finding Syntax in Word Representations." NAACL 2019.
