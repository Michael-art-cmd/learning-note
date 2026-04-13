# 04_训练与推理流程

## 目录
1. [预训练阶段设计](#1-预训练阶段设计)
2. [测试时训练流程](#2-测试时训练流程)
3. [损失函数选择](#3-损失函数选择)
4. [超参数设置](#4-超参数设置)
5. [实际实现考虑](#5-实际实现考虑)

---

## 1. 预训练阶段设计

### 1.1 预训练目标

TTT-E2E的预训练同时优化两个目标：
1. **元学习目标**：使模型学会"如何在测试时学习"
2. **自监督目标**：在测试时能够进行有效的自我训练

### 1.2 两阶段训练框架

**第一阶段：内循环学习（自监督）**

在每个token位置上，用当前隐藏状态预测下一个token：

$$\mathcal{L}_{inner} = \sum_{t=1}^{n} \|x_t - h_{t-1}(x_t)\|^2$$

**第二阶段：外循环优化（元学习）**

根据更新后的隐藏状态计算最终损失：

$$\mathcal{L}_{outer} = \sum_{t=1}^{n} \|x_t - h_t(x_t)\|^2$$

**总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{outer} + \lambda \mathcal{L}_{inner}$$

### 1.3 预训练数据集

常用预训练数据：
- **The Pile**：多领域文本数据集
- **BooksCorpus**：书籍文本
- **Common Crawl**：网页文本
- **GitHub**：代码数据

**数据处理**：
1. 分词（Tokenization）
2. 数据清洗
3. 质量过滤
4. 去重

### 1.4 模型架构配置

TTT论文使用的模型规模：

| 参数规模 | 层数 | 隐藏维度 | 注意力头数 |
|---------|------|---------|-----------|
| 125M | 12 | 768 | 12 |
| 350M | 24 | 1024 | 16 |
| 760M | 24 | 1536 | 24 |
| 1.3B | 24 | 2048 | 32 |

---

## 2. 测试时训练流程

### 2.1 推理初始化

**隐藏状态初始化**：
```python
def init_hidden_state(d_model, ttt_type='linear'):
    if ttt_type == 'linear':
        # 初始化为零矩阵和零向量
        W = torch.zeros(d_model, d_model)
        b = torch.zeros(d_model)
        return (W, b)
    elif ttt_type == 'mlp':
        # 初始化两层MLP参数
        m = 4 * d_model  # 中间层维度
        W1 = torch.randn(m, d_model) * 0.01
        b1 = torch.zeros(m)
        W2 = torch.randn(d_model, m) * 0.01
        b2 = torch.zeros(d_model)
        return (W1, b1, W2, b2)
```

### 2.2 自回归生成流程

```python
def ttt_forward(x_t, hidden_state, ttt_layer):
    """
    TTT层前向传播
    
    参数:
        x_t: 当前输入token嵌入 [batch, d_model]
        hidden_state: 上一个隐藏状态
        ttt_layer: TTT层网络
    
    返回:
        y_t: 输出预测
        hidden_state: 更新后的隐藏状态
    """
    # 步骤1：用当前隐藏状态预测
    y_pred = hidden_state(x_t)  # 将x_t作为输入
    
    # 步骤2：计算损失
    loss = MSE(y_pred, x_t)
    
    # 步骤3：计算梯度
    grad = torch.autograd.grad(loss, hidden_state.parameters())
    
    # 步骤4：更新隐藏状态
    hidden_state = hidden_state - learning_rate * grad
    
    return y_pred, hidden_state
```

### 2.3 完整生成流程

```python
def generate_with_ttt(model, input_ids, max_length):
    # 初始化
    hidden_state = model.ttt.init_hidden_state()
    output_ids = []
    
    # 处理输入序列
    for x in input_ids:
        _, hidden_state = model.ttt_layer(x, hidden_state)
    
    # 自回归生成
    for _ in range(max_length):
        # TTT前向传播
        logits, hidden_state = model.ttt_layer(last_token, hidden_state)
        
        # 采样
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        
        if next_token == EOS:
            break
    
    return output_ids
```

### 2.4 上下文累积

TTT在处理长序列时自动累积信息：

```python
def process_long_sequence(model, tokens, chunk_size=512):
    """
    分块处理长序列
    
    参数:
        tokens: 输入token序列
        chunk_size: 块大小
    """
    hidden_state = model.ttt.init_hidden_state()
    
    # 分块处理
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        
        for token in chunk:
            # 每个token都进行TTT更新
            _, hidden_state = model.ttt_layer(token, hidden_state)
    
    return hidden_state
```

---

## 3. 损失函数选择

### 3.1 自监督损失

**MSE损失（均方误差）**：

$$\mathcal{L}_{MSE} = \|x_t - \hat{x}_t\|^2 = \|(I - W_{t-1})x_t - b_{t-1}\|^2$$

**交叉熵损失**：

$$\mathcal{L}_{CE} = -\sum_{v} x_{t,v} \log \hat{x}_{t,v}$$

### 3.2 层归一化融合

TTT论文推荐使用层归一化与损失融合：

```python
def ttt_loss_with_ln(x_t, h):
    """
    融合层归一化的TTT损失
    """
    # 预测
    pred = h(x_t)
    
    # 对输入和预测都应用层归一化
    pred_ln = F.layer_norm(pred, pred.shape)
    target_ln = F.layer_norm(x_t, x_t.shape)
    
    # 计算L2损失
    loss = (pred_ln - target_ln).pow(2).mean()
    
    return loss
```

### 3.3 对比损失

对于更复杂场景，可以使用对比学习损失：

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(h_t, x_t) / \tau)}{\sum_{j} \exp(\text{sim}(h_t, x_j) / \tau)}$$

### 3.4 损失函数选择建议

| 场景 | 推荐损失函数 | 原因 |
|------|-------------|------|
| 简单重建 | MSE | 稳定、高效 |
| 分类任务 | 交叉熵 | 适用于离散输出 |
| 长序列 | LN融合MSE | 稳定训练 |
| 表征学习 | 对比损失 | 学习更好表示 |

---

## 4. 超参数设置

### 4.1 训练超参数

**优化器配置**（TTT论文推荐）：
```python
optimizer_config = {
    'optimizer': 'AdamW',
    'beta': (0.9, 0.95),      # 动量参数
    'weight_decay': 0.1,     # 权重衰减
    'eps': 1e-8,              # 数值稳定性
}
```

**学习率调度**：
```python
scheduler_config = {
    'type': 'cosine',
    'warmup_ratio': 0.1,      # 10% warmup
    'min_lr': 1e-5,           # 最小学习率
}
```

### 4.2 TTT特有超参数

**内循环学习率**（最关键）：
```python
ttt_hyperparams = {
    'linear': {
        'eta_base': 1.0,       # TTT-Linear基础学习率
    },
    'mlp': {
        'eta_base': 0.1,      # TTT-MLP基础学习率（更小以稳定训练）
    },
}
```

**Mini-batch配置**：
```python
batch_config = {
    'batch_size': 16,          # micro batch size
    'gradient_accumulation': 4, # 累积步数
    'effective_batch_size': 64, # 有效batch size
}
```

### 4.3 不同规模的推荐配置

| 模型规模 | 学习率 | Batch Size | 序列长度 |
|---------|--------|------------|---------|
| 125M | 1e-4 | 32 | 2K |
| 350M | 8e-5 | 64 | 4K |
| 760M | 6e-5 | 32 | 8K |
| 1.3B | 4e-5 | 16 | 8K |

### 4.4 推理超参数

**生成参数**：
```python
generation_config = {
    'max_length': 1024,
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 0.9,
    'do_sample': True,
}
```

---

## 5. 实际实现考虑

### 5.1 内存优化

**梯度累积**：
```python
# 避免OOM
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    
    # 每4步更新一次
    if (step + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**混合精度训练**：
```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5.2 计算优化

**对偶形式加速**：
```python
def ttt_linear_dual_form(X, Y, lambda_reg=0.1):
    """
    对偶形式直接计算闭式解
    
    参数:
        X: 输入矩阵 [batch, d]
        Y: 目标矩阵 [batch, d]
        lambda_reg: 正则化系数
    
    返回:
        W: 最优权重矩阵
    """
    # W* = (X^T X + λI)^(-1) X^T Y
    XtX = X.T @ X + lambda_reg * torch.eye(X.shape[1])
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)
    
    return W.T
```

### 5.3 数值稳定性

**梯度裁剪**：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**学习率预热**：
```python
def warmup_cosine_schedule(step, warmup_steps, total_steps, min_lr=1e-6):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (1 - min_lr) * (1 + np.cos(np.pi * progress))
```

### 5.4 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 梯度爆炸 | 学习率过大 | 降低η、使用梯度裁剪 |
| 训练不稳定 | TTT-MLP非凸 | 使用TT-Linear或降低η |
| 内存不足 | 隐藏状态过大 | 使用梯度检查点、混合精度 |
| 推理慢 | 计算效率低 | 使用对偶形式、batch处理 |

---

## 本章小结

本章详细介绍了TTT的训练与推理流程：

1. **预训练设计**：两阶段训练框架，同时优化内循环和外循环
2. **推理流程**：自回归生成、隐藏状态更新
3. **损失函数**：MSE、交叉熵、层归一化融合
4. **超参数**：内循环学习率是关键参数
5. **实现考虑**：内存优化、数值稳定性

理解这些流程对于实现和部署TTT模型至关重要。