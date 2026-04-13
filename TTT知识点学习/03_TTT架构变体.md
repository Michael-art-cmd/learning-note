# 03_TTT架构变体

## 目录
1. [TTT-Linear详细架构](#1-ttt-linear详细架构)
2. [TTT-MLP详细架构](#2-ttt-mlp详细架构)
3. [Titans架构（MAC、MAG、MAL）](#3-titans架构macmagmal)
4. [各架构的实现细节](#4-各架构的实现细节)
5. [架构对比分析](#5-架构对比分析)

---

## 1. TTT-Linear详细架构

### 1.1 架构概述

TTT-Linear是TTT的最简单实例，将隐藏状态实现为一个**线性模型**：

$$h_t(x) = W_t x + b_t$$

其中 $W_t \in \mathbb{R}^{d \times d}$，$b_t \in \mathbb{R}^d$。

### 1.2 前向传播

**输入**：token嵌入 $x_t \in \mathbb{R}^d$

**预测输出**：

$$\hat{y}_t = h_{t-1}(x_t) = W_{t-1} x_t + b_{t-1}$$

**损失计算**：

$$\mathcal{L}_t = \|x_t - \hat{y}_t\|^2$$

### 1.3 梯度计算

**对 $W_{t-1}$ 的梯度**：

$$\frac{\partial \mathcal{L}_t}{\partial W_{t-1}} = 2 (\hat{y}_t - x_t) x_t^\top$$

**对 $b_{t-1}$ 的梯度**：

$$\frac{\partial \mathcal{L}_t}{\partial b_{t-1}} = 2 (\hat{y}_t - x_t)$$

### 1.4 参数更新

$$W_t = W_{t-1} - \eta \cdot 2 (\hat{y}_t - x_t) x_t^\top$$

$$b_t = b_{t-1} - \eta \cdot 2 (\hat{y}_t - x_t)$$

### 1.5 内存占用

- $W_t$：$d \times d$ 矩阵，$O(d^2)$ 空间
- $b_t$：$d$ 维向量，$O(d)$ 空间
- **总计**：$O(d^2)$

### 1.6 对偶形式（Dual Form）

TTT-Linear的对偶形式可以直接计算闭式解：

设 batch 内输入为 $X \in \mathbb{R}^{b \times d}$，目标为 $Y \in \mathbb{R}^{b \times d}$：

$$W^* = (X^\top X + \lambda I)^{-1} X^\top Y$$

$$b^* = \frac{1}{b}(Y - XW^*)^\top \mathbf{1}$$

**优势**：
- 避免迭代梯度下降
- 可以利用矩阵乘法优化

### 1.7 层归一化融合

TTT论文提出了将层归一化与L2损失融合的技巧：

$$\mathcal{L}_t = \|\text{LN}(x_t) - \text{LN}(\hat{y}_t)\|^2$$

这可以稳定训练，防止梯度爆炸。

---

## 2. TTT-MLP详细架构

### 2.1 架构概述

TTT-MLP将隐藏状态实现为一个**两层的MLP**（多层感知机），具有更强的表达能力：

$$h_t(x) = W_2^{(t)} \sigma(W_1^{(t)} x + b_1^{(t)}) + b_2^{(t)}$$

其中：
- $W_1^{(t)} \in \mathbb{R}^{m \times d}$：第一层权重
- $W_2^{(t)} \in \mathbb{R}^{d \times m}$：第二层权重
- $m$：中间层维度（通常 $m > d$，如 $m = 4d$）
- $\sigma(\cdot)$：激活函数（如GELU）

### 2.2 前向传播

**第一层**：

$$h^{(1)} = \sigma(W_1^{(t-1)} x_t + b_1^{(t-1)})$$

**第二层**：

$$\hat{y}_t = W_2^{(t-1)} h^{(1)} + b_2^{(t-1)}$$

### 2.3 梯度计算

TTT-MLP的梯度计算比TTT-Linear更复杂，需要反向传播：

**输出层梯度**：

$$\delta^{(out)} = 2(\hat{y}_t - x_t)$$

**第二层梯度**：

$$\nabla_{W_2} \mathcal{L} = \delta^{(out)} \cdot h^{(1)\top}$$

$$\nabla_{b_2} \mathcal{L} = \delta^{(out)}$$

**第一层梯度**：

$$\delta^{(1)} = (W_2^{(t-1)\top} \delta^{(out)}) \odot \sigma'(W_1^{(t-1)} x_t + b_1^{(t-1)})$$

$$\nabla_{W_1} \mathcal{L} = \delta^{(1)} \cdot x_t^\top$$

$$\nabla_{b_1} \mathcal{L} = \delta^{(1)}$$

### 2.4 参数更新

$$\begin{aligned}
W_1^{(t)} &= W_1^{(t-1)} - \eta \cdot \nabla_{W_1} \mathcal{L} \\
b_1^{(t)} &= b_1^{(t-1)} - \eta \cdot \nabla_{b_1} \mathcal{L} \\
W_2^{(t)} &= W_2^{(t-1)} - \eta \cdot \nabla_{W_2} \mathcal{L} \\
b_2^{(t)} &= b_2^{(t-1)} - \eta \cdot \nabla_{b_2} \mathcal{L}
\end{aligned}$$

### 2.5 内存占用

- $W_1^{(t)}$：$m \times d$ 矩阵
- $W_2^{(t)}$：$d \times m$ 矩阵
- $b_1^{(t)}, b_2^{(t)}$：$O(m + d)$ 向量
- **总计**：$O(d \cdot m) = O(d^2)$（当 $m = O(d)$ 时）

### 2.6 与TTT-Linear的对比

| 特性 | TTT-Linear | TTT-MLP |
|------|------------|---------|
| 表达能力 | 线性函数 | 非线性函数 |
| 参数量 | $O(d^2)$ | $O(d \cdot m)$ |
| 计算量 | $O(d^2)$ | $O(d \cdot m)$ |
| 训练稳定性 | 较高 | 较低（需调参） |
| 内循环学习率 | 1.0 | 0.1 |

---

## 3. Titans架构（MAC、MAG、MAL）

### 3.1 Titans概述

**Titans: Learning to Memorize at Test Time** 是Google提出的新一代架构，将TTT的思想扩展为完整的神经网络架构。

**核心创新**：引入**神经长期记忆（Neural Long-term Memory, NLM）**模块，在推理时通过梯度下降自适应。

### 3.2 架构组件

Titans包含三个核心组件：

1. **持久记忆（Persistent Memory, PM）**：长期存储的先验知识
2. **神经长期记忆（Neural Long-term Memory, NLM）**：推理时学习的动态记忆
3. **短时记忆（Short-term Memory）**：传统注意力机制

### 3.3 MAC（Memory as Context）

**Memory as Context** 将长期记忆作为额外上下文：

**架构**：

$$y = \text{Attention}(q, [\text{PM}, \text{NLM}_t, c_t])$$

其中：
- $\text{PM}$：持久记忆（可学习的固定参数）
- $\text{NLM}_t$：当前长期记忆状态
- $c_t$：当前上下文（传统KV cache）

**长期记忆更新**：

$$\text{NLM}_t = \text{NLM}_{t-1} - \eta \nabla_{\text{NLM}} \mathcal{L}_t$$

### 3.4 MAG（Memory as Gating）

**Memory as Gating** 使用记忆控制信息流动：

**门控机制**：

$$g_t = \sigma(W_g [\text{NLM}_t; q_t] + b_g)$$

**输出**：

$$y_t = g_t \cdot \text{Attention}(q_t, K_t, V_t) + (1-g_t) \cdot \text{NLM}_t$$

**优势**：
- 动态决定使用短期还是长期记忆
- 更细粒度的记忆控制

### 3.5 MAL（Memory as Layer）

**Memory as a Layer** 将记忆作为独立层：

**多层架构**：
```
Input -> TTT Layer (NLM) -> Attention -> Output
```

每层都包含：
- **TTT层**：更新神经长期记忆
- **注意力层**：处理当前上下文

**输出**：

$$h_t, \text{NLM}_t = \text{TTT-Layer}(x_t, \text{NLM}_{t-1})$$

$$y_t = \text{Attention}(h_t, K_t, V_t)$$

### 3.6 变体对比

| 变体 | 记忆整合方式 | 适用场景 |
|------|-------------|---------|
| MAC | 作为额外上下文 | 需要全局信息 |
| MAG | 门控机制 | 动态权重分配 |
| MAL | 作为网络层 | 深度记忆整合 |

---

## 4. 各架构的实现细节

### 4.1 TTT层与Transformer的集成

TTT层可以集成到现有Transformer网络中：

```python
class TTTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ttt_type='linear'):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        if ttt_type == 'linear':
            self.ttt = TTTLinear(d_model)
        else:
            self.ttt = TTTMLP(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x, kv_cache=None):
        # 多头注意力
        attn_out = self.attn(x, kv_cache)
        x = self.norm1(x + attn_out)
        
        # TTT层
        ttt_out, hidden_state = self.ttt(x)
        x = self.norm2(x + ttt_out)
        
        # FFN
        x = self.ffn(x)
        
        return x, hidden_state
```

### 4.2 推理时的KV Cache

TTT需要维护两个缓存：
1. **传统KV Cache**：$c_t = (K_{1:t}, V_{1:t})$
2. **TTT隐藏状态**：$h_t$

```python
def generate_with_ttt(model, prompt, max_length):
    # 初始化
    hidden_state = model.ttt.init_hidden_state()
    cache = None
    
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    
    # 处理输入序列
    for x in input_ids:
        y, hidden_state = model.ttt_layer(x, hidden_state)
        cache = update_cache(cache, x)
    
    # 自回归生成
    for _ in range(max_length):
        x = last_token(y)
        y, hidden_state = model.ttt_layer(x, hidden_state)
        cache = update_cache(cache, x)
        
        # 采样 next token
        next_token = sample(y)
        output_ids.append(next_token)
    
    return output_ids
```

### 4.3 梯度检查点（Gradient Checkpointing）

由于TTT需要在每个token计算梯度，内存占用较大。使用梯度检查点可以节省内存：

```python
# 训练时使用梯度检查点
class TTTLinearWithCheckpoint(nn.Module):
    def forward(self, x, hidden_state):
        # 不保存中间激活值
        return torch.utils.checkpoint.checkpoint(
            self._forward, x, hidden_state
        )
```

---

## 5. 架构对比分析

### 5.1 TTT-Linear vs TTT-MLP

| 维度 | TTT-Linear | TTT-MLP |
|------|-------------|---------|
| **表达能力** | 线性 | 非线性 |
| **参数量** | $d^2$ | $2dm + m + 2d$ |
| **计算复杂度** | $O(d^2)$ | $O(dm)$ |
| **内存占用** | 较低 | 较高 |
| **训练稳定性** | 稳定 | 需要较小学习率 |
| **理论性质** | 等价于线性注意力 | 更灵活的表达 |

**选择建议**：
- 长上下文、高吞吐量：TTT-Linear
- 复杂模式、需要非线性：TTT-MLP

### 5.2 TTT vs Transformer vs Mamba

| 特性 | Transformer | Mamba | TTT |
|------|-------------|-------|-----|
| **架构** | 注意力机制 | 状态空间模型 | 可学习隐藏状态 |
| **复杂度** | $O(n^2 \cdot d)$ | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ |
| **推理延迟** | 线性增长 | 恒定 | 恒定 |
| **长上下文** | 受限于 $n^2$ | 优秀 | 优秀 |
| **表达能力** | $O(n^2)$ | $O(d)$ | $O(d^2)$ |

### 5.3 各架构适用场景

**Transformer**：
- 短序列（< 8K）
- 需要精确的逐位置注意力
- 充足计算资源

**Mamba**：
- 超长序列（> 100K）
- 需要高效的状态传递
- 硬件友好

**TTT**：
- 长序列（8K - 128K）
- 需要自适应学习能力
- 需要平衡效率和表达

### 5.4 性能总结

根据论文实验数据：
- **Pile 8k**：TTT-Linear(M) 困惑度优于 Mamba
- **Books 32k**：TTT-MLP 表现最佳
- **推理速度**：TTT延迟恒定，Transformer线性增长

---

## 本章小结

本章详细介绍了TTT的主要架构变体：

1. **TTT-Linear**：线性模型的隐藏状态，简单高效
2. **TTT-MLP**：两层的MLP，更强表达能力
3. **Titans**：Google的TTT扩展架构，MAC/MAG/MAL三种变体
4. **实现细节**：KV cache、梯度检查点、与Transformer集成
5. **架构对比**：各方案的优劣势和适用场景

TTT-Linear和TTT-MLP是TTT的两种基本实现，各有权衡。Titans则将TTT思想进一步发展，引入完整的神经长期记忆模块。