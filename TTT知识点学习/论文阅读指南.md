# 论文阅读指南

## 核心必读论文

### 1. TTT原始论文（必读）

**Paper 1: Learning to (Learn at Test Time): RNNs with Expressive Hidden States**
- arXiv: 2407.04620
- Authors: Sun et al.
- 发表时间：2024年7月

**阅读重点**：
1. 第1-2节：背景和动机，理解为什么要"在测试时学习"
2. 第3节：TTT-Linear和TTT-MLP的数学形式
3. 第4节：端到端训练（E2E）框架
4. 第5节：实验结果分析

**关键公式**：
- 隐藏状态更新：
$h_t = h_{t-1} - \eta \nabla \mathcal{L}_t$
- 自监督目标： $\mathcal{L}_{NTP} = \|x_t - h_{t-1}(x_t)\|^2$

---

### 2. Titans论文（推荐）

**Paper 2: Titans: Learning to Memorize at Test Time**
- arXiv: 2501.00663
- Authors: Behrouz et al. (Google)
- 发表时间：2025年1月

**阅读重点**：
1. 神经长期记忆（NLM）的概念
2. MAC、MAG、MAL三种变体
3. 与传统注意力机制的对比

---

## 参考文献

### 元学习基础

**Paper 3: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**
- arXiv: 1703.03400
- Authors: Finn et al. (2017)
- 重要性：理解MAML是理解TTT外循环的基础

**Paper 4: Meta-Learning in Neural Networks: A Survey**
- arXiv: 2112.09153
- 全面了解元学习领域

### RNN与状态空间模型

**Paper 5: Mamba: Linear-time Sequence Modeling with Selective State Spaces**
- arXiv: 2312.00752
- 了解与TTT竞争的Mamba架构

### 自监督学习

**Paper 6: A Theory of Gradient Sparsification for Distributed Learning**
- 理解梯度下降的理论基础

---

## 阅读顺序建议

**入门阶段**（1-2周）：
1. TTT原始论文第1-4节
2. MAML论文摘要和introduction

**进阶阶段**（2-3周）：
1. TTT原始论文第5-7节（实验和理论证明）
2. Titans论文

**深入阶段**（3-4周）：
1. 相关工作章节的引用论文
2. 实现细节

---

## 代码实现参考

**官方代码**：
- https://github.com/test-time-training/ttt-lm-pytorch
- https://github.com/mahdi-shafaee/titans-flax

**关键实现**：
```python
# TTT-Linear核心实现
class TTTLinear(nn.Module):
    def forward(self, x, hidden_state):
        # 1. 预测
        pred = hidden_state(x)
        
        # 2. 计算损失
        loss = (pred - x).pow(2).mean()
        
        # 3. 计算梯度并更新
        grad = torch.autograd.grad(loss, hidden_state)
        hidden_state = hidden_state - self.lr * grad
        
        return pred, hidden_state
```

---

## 常见问题

**Q1: TTT和MAML有什么区别？**
A: MAML在多个训练任务上学习一个好的初始化；TTT在每个测试序列上动态学习。

**Q2: TTT的推理速度为什么快？**
A: TTT层的计算复杂度是O(d²)，与序列长度无关，所以延迟恒定。

**Q3: TTT需要额外的训练数据吗？**
A: 不需要，TTT使用自监督的next-token prediction目标。
