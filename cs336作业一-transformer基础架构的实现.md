# Transformer基础架构实现

## 一、基础知识点

### 1. 语言模型

> 语言模型是一种自回归的预测模型

+ 输入：

> 语言模型处理数字。每个词、标点符号或者更小的语言单位都会赋予一个唯一的整数ID，token
>
> 利用并行计算，模型可以按照batch同时处理多段文本

+ 输出：

> 对输入序列每一个词，模型都会预测下一个词是什么，给出概率分布，和为1

+ 训练

> 现在语言模型，大规模文本语料库的自监督学习，最大化给定上文序列的条件下，预测下一个token的条件概率
>
> 自监督，不需要人工标注，标签从数据自动获取
>
> 优化过程，最小化损失函数。当模型做出预测后，通过一个名为交叉熵损失的函数，量化其预测的概率分布与真实标签之间的差距。然后反向传播算法来调整内部数以亿计的参数，让损失值尽可能小

### 2. einsum /einops 

+ 这是干什么的？ 可以用一串“字母标签”描述张量运算 - 哪个维度要保留、哪个维度要相乘后求和

矩阵乘法最原始定义：

```
C[i][j] = Σ_k  A[i][k] × B[k][j]
k是内部下标，相乘时被消掉求和的维度，太麻烦

# 普通：Σ_i a[i]*b[i]
dot = (a * b).sum()

# einsum："i,i->"  两个一维数组，i 重复 → 乘起来求和（-> 后面空 = 标量）
dot = einsum(a, b, "i,i->")
# 逗号分隔的每个张量的 "维度名字"，重复出现的名字 = 相乘后求和
```

```
einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
```

+ D是三维数组，batch、sequence、d_in
+ A是二维数组，d_in、d_out
+ d_in同时出现在多个输入里=这个维度相乘后求和会消掉
+ ->后面就是输出保留哪些轴，输出什么顺序

### 3. 参数初始化

+ 初始参数设置不当，可能会出现无法收敛、收敛到局部最优解、梯度爆炸、梯度消失等问题

Xavier初始化：

> 保持每一层激活值的方差和反向传播时梯度的方差在前向和反向传播中保持不变

![image-20261005153255574](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005153255574.png)

He初始化：

> Xavier初始化在ReLU激活函数不佳，ReLU负输入都为0

文档给了三种参数的初始化方案，都用截断正态分布torch.nn.init.trunc_normal_:

<img src="C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261006194051793.png" alt="image-20261006194051793" style="zoom:100%;" />

## 二、 具体实现

### 1. 两大基础架构：

#### 1.线性模块

线性模块就是一个线性变换，将输入的向量映射到另一个向量

+ y = Wx + b

```
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 对权重进行Xacier初始化
        sigma = (2/(in_features + out_features))**0.5
        W = torch.empty((out_features, in_features), dtype=dtype, device=device)
        self.weight = nn.Parameter(W)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        # 输入...通配任意批量维
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

```

+ torch,empty(…)先分配一块未初始化的内存，再用trunc_normal_原地填入截断正态分布采样值（函数名结尾的下划线是pytorch的命名惯例，表示原地修改）
+ nn.Parameter(weight):把普通张量包装成可训练参数

#### 2.嵌入模块

将代表文本的整数token ID转换为高维的、模型能够理解的向量

```
# 将tokenID转换为更高维，模型能理解的向量表示
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings是词表大小（行数）
        # embedding_dim是每个token对应向量的维度（列数，d_model）
        super().__init__()
        table = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        self.weight = nn.Parameter(table)
        # 为什么这里不是Xavier的sqrt(2/(in + out))
        # embedding是查表，没有求和，直接标准正态N(0,1)截断[-3,3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

+ self.weight[token_ids]，weight 是一张 "词典表"，weight[token_ids] 就是按 token_ids 里的每个数字去查表，查到的行放到对应位置。
+ pytorch支持用整张张量直接索引另一个张量的第0维，token_ids形状是（batch_size, sequence_length）,自动变成(batch_size, sequence_length, embedding_dim)不需要手写循环

### 2. Pre-Norm Transformer Block

#### 1. RMS norm

+ 是简化版的层归一化，比LN节省了求均值的步骤

> 优点：计算效率更高、内存占用更少、简化模型

> LN和RMS norm总是沿着特征维度进行归一化（通常是最后一个维度，即hidden_size 或 embedding_dim）进行归一化，这种归一化**独立地**应用于每个样本的每个序列位置，不涉及批次维度或者序列长度维度上的统计计算

> 确保进入Transformer子层的每个token的向量表示

```
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # torch.one创建一个全1的向量，初始全1，不改变前向，训练中学习
        # nn.Parameter()标记可学习-训练时自动更新值
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        # 题目要求对于不同的精度要先转换为float32再进行归一化，最后在转换为原来的
        # 输入x：(batch_size, sequence_size, d_model)
        origin_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        norm_x = x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_norm = norm_x.to(origin_dtype)
        return x_norm * self.weight
        # 再次转为原来的精度
```

+ x.pow(2).mean(dim=-1, keepdim=True)：对最后一维度（d_model）求平方的均值，对应公式里面的mean(a_i^2);+self.eps再开方，避免分母算出0

#### 2. 位置级前馈网络（FFN）

原始Transformer的FFN（基于ReLU）

> + 包含两个线性变换层，中间夹着一个ReLU激活函数
>
> + FFN(x) = Linear(ReLU(Linear(x)))
> + 中间隐藏层(Linear_1的输出)的维度通常是输入维度d_model的四倍

SWiGLU

+ FFN 的作用是把 token 向量先 "变宽再变窄"（d_model → d_ff → d_model），中间必须插入非线性（不然堆多少层都等于一层）。SwiGLU 就是这个非线性，LLaMA 用的。

SiLU/Swish激活函数

+ 类似于ReLU，但在零点附近平滑，可缓解梯度消失
+ x * sigmod(x)

![image-20261005211621352](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005211621352.png)

门控线性单元（GLUs）

+ 将一个线性变换的结果通过sigmoid函数，再与另一个线性变换的结果进行逐元素相乘
+ GLU的直觉是为梯度提供提条线性通路，同时保留非线性能力，缓解深层网络的梯度消失问题

![image-20261005211634261](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005211634261.png)

![image-20261005202810460](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005202810460.png)

![image-20261005211646820](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005211646820.png)

```
def silu(x):
    return x * torch.sigmoid(x)
```

```
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # 建设参数矩阵w1，w2，w3
        self.weight1 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))
        self.weight2 = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype, device=device))
        self.weight3 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))

        # 参数初始化
        sigma = (2 / (d_model + d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight1, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        torch.nn.init.trunc_normal_(self.weight2, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        torch.nn.init.trunc_normal_(self.weight3, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x):
        gate = silu(x @ self.weight1.T) # W1门
        up = x @ self.weight3.T         # W3直通
        hidden = gate * up
        return hidden @ self.weight2.T  # W2最后投影
```

+ gate * up:两个形状相同的张量逐元素相乘，就是门控的核心–gate（过了SiLU的分支）决定value里面每个位置该保留多少信息

![image-20261005213532762](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005213532762.png)

#### 3. 相对位置嵌入（RoPE旋转位置编码）

Rope旋转位置编码属于相对位置编码

+ 为什么需要？
+ 因为需要位置信息，“我喜欢你”输入之后先BPE初步分词，然后转换为token数学向量形式，然后token经过input embedding，输出再进入到自注意力机制。对于一句话来说，如果不关注每一个字的位置信息，只关注字词内容本身，是不符合实际情况的
+ 旋转矩阵：乘以一个向量的时候改变了方向，但不改变大小，保持了手性
+ 旋转的作用到底在哪？
+ 作用发生在下游的q * k点积里；rope只管转，位置 3 的 q 被转了 3θ，位置 8 的 k 被转了 8θ。点积的本质是看 "夹角"。两个向量都转了以后，夹角 = 各自转角的**差** = 8θ − 3θ = 5θ。位置关系是点积自动算出来的

rope：将位置编码成向量的旋转角度

> 位置0的向量、位置1的向量、位置2的向量……一次多转一个角度
>
> 两个token的相对位置,旋转角差
>
> 模块没有学习参数，纯数学计算

![image-20261005213250347](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261005213250347.png)

> 旋转只在“每对元素内部”发生，对与对之间互不干扰，实现时候不用建立矩阵

```
d_k = 64 维 → 相邻两两配对 → 32 对（每对是 2D 向量）
第 0 对：转得最快（高频）
第 1 对：慢一点
……
第 31 对：几乎不动（低频）
```

频率公式 `f_k = Θ^(-2k/d_k)` 就是干这个的 ——**k 越大频率越小**。这就是为什么实现里要 `arange(0, d_k, 2)`（只看偶数下标 0, 2, 4... 代表第 0、1、2... 对）而不是 `arange(d_k)`。

```
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device:torch.device | None = None,
    ) -> None:
        super().__init__()

        # 1.频率数组f: d_k//2
        # d_k维向量，有d_k/2对，每对一个频率,f_k是计算之后的频率
        # 每一对维度k（0到d_k/2-1）对应一个频率
        index = torch.arange(0, d_k, 2)
        f_k = theta ** (-(index / d_k))

        # 2.位置数组pos,加上.float是为了和后面负电的f_k做乘法
        '''
        位置编号0..max_seq_len-1,跟频率做外积，得到每个（位置，维度对）
        '''
        pos = torch.arange(max_seq_len).float()
        # 3.角度矩阵
        angles = pos[:, None] * f_k[None, :]

        # 预先缓存cos/sin，forward时直接按位置切片查表，不用每次都重新算
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        #（persistent=False = 不进 state_dict，因为没参数、不学习）

    def forward(self, x, token_positions):
        '''
        x:(..., seq_len, d_k) --要旋转的Q或K向量
        token_positions:(..., seq_len) -- 每个位置对应的位置编号
        返回：(..., seq_len, d_k) -- 形状不变，只是旋转
        '''
        # 1.取角度
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # 2. 拆对
        # 把x的最后一位拆成两半：x1是偶数位，x2是奇数位，分别对应每一对旋转维度里的两个分量
        x1, x2 = x[..., 0::2], x[..., 1::2] # 各自(..., seq_len, d_k/2)
        # = x[:, :, 0::2]：取最后一维偶数位 0,2,4,...,62 → (4, 12, 32)

        # 二位旋转矩阵作用在(x1,x2)上
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * sin + x1 * cos

        # 把旋转后的两半按照原来交错的顺序拼回去，形状恢复成(..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_x1
        out[..., 1::2] = rotated_x2
        return out
```

#### 4. softmax

+ 把一组数字变成概率分布-每个数变成0-1之间、加起来等于1

```
softmax(v)_i = exp(v_i) / Σ_j exp(v_j)
```

> 直接计算容易在v_i很大时让exp(v_i)变成Inf（进而inf/inf变成NaN）。因为softmax对“给所有输入加一个常数”具有不变性（分子分母的exp(c)会同时约掉），先减去当前维度的最大值c，让新的最大值变成0，避免exp溢出

+ keepdim = True:让max/sum的结果保留被压缩的那一维（大小变成1），这样才能直接跟原始形状的 in_features 做元素减法/除法。如果不加，压缩掉的维度会消失，形状对不上，广播会出错或者算出错误结果

```
def softmax(x,dim):
    # 1.取最大值
    m = x.max(dim=dim, keepdim=True).values # keepdim 保留维度，方便广播
    # 2.每个数减最大值后取exp
    e = torch.exp(x - m)
    # 3.归一化
    result = e / e.sum(dim=dim, keepdim=True)
    return result
```

#### 5. dot_attention

| 角色                | 是什么            | 比喻                           |
| ------------------- | ----------------- | ------------------------------ |
| **Q**（Query 查询） | 每个 "问问题的人" | 每个 token 想知道 "我该关注谁" |
| **K**（Key 键）     | 每个位置的 "标签" | 每个 token 回答 "我是什么"     |
| **V**（Value 值）   | 每个位置的 "内容" | 每个 token 真正要提供的信息    |

三步走：

1. **打分**：Q 和所有 K 算相似度（点积大 = 像 = 该关注）
2. **归一化**：softmax 变成权重（每个 Q 的注意力权重和 = 1）
3. **取内容**：按权重把 V 加权求和 → 每个 Q 得到 "关注过的信息"

```
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

+ 为什么除以d_k的根

点积会随着维度的扩大而扩大，softmax会饱和，可能会出现梯度消失，除以这个是为了把点积拉回温和范围

> mask：True = 可以关注（保留分数），False = 禁止（-inf）

![image-20261006164824440](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261006164824440.png)

```
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # 1.打分，Q*K^T
    # K.transpose(-2, -1)：K 是 (…, n_k, d_k)，转置最后两维变 (…, d_k, n_k) 才能和 Q (…, n_q, d_k) 对齐相乘。用 einsum 就不用想转置
    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k**0.5)

    # 2.mask
    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))

    # 3.softmax,对最后一个维度（每个query的所有key）
    probs = softmax(scores, dim=-1)

    # 4.加权求和
    out = probs @ V
    return out
```

#### 6. 多头注意力

一个注意力头只能学**一种关注模式**。多个头 = 多个 "视角" 并行：

- 头 1 可能专关注 "代词指代谁"
- 头 2 可能专关注 "词和它左边紧挨的词"
- 头 3 可能专关注 "句法角色"

每个头有自己独立的Q/K/V投影，算完attention后拼回去，再过一个输出投影W_O把各头的视角混合成最终结果

```
MultiHeadSelfAttention(x) = W_O · Concat(head₁, ..., headₕ)
headᵢ = Attention(W_Qᵢx, W_Kᵢx, W_Vᵢx)
```

+ 作业提及，不要每个头单独乘一次 —— 把 h 个头合成一次矩阵乘法（W_Q 是 (d_model, d_model)，投影完再 "切开" 成 h 段）

```
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads      # （64//4 = 16）
        # 4 个 Linear（无 bias）：q_proj, k_proj, v_proj, o_proj，都是 (d_model, d_model)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # 1 个 RoPE：RotaryPositionalEmbedding(theta, d_k, max_seq_len)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def forward(self, x, token_positions=None):
        # x: (batch, seq, d_model)

        # 1. 投影（3 次 matmul，所有头合在一起）：
        # Q = x @ W_Qᵀ, K = x @ W_Kᵀ, V = x @ W_Vᵀ   → (batch, seq, d_model)
        # 投影：Q,K,V 各一次矩阵乘法（内部就是x @ weight.T）
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 切头：view(batch, seq, num_heads, d_k)
        #    → transpose(1, 2) → (batch, num_heads, seq, d_k)
        #    顺序：先 view 成 (batch, seq, h, d_k)，再转置让 h 提前
        b,s, _ = x.shape # 一次取出三个维度：batch=4, seq=12, d_model:_=64
        Q = Q.view(b, s, self.num_heads, self.d_k).transpose(1, 2) # (b, h, s, d_k)
        K = K.view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        # 3. RoPE（只转 Q、K，不转 V！）：
        #    直接调你的 RoPE：它接受 (..., seq, d_k)，(batch, num_heads) 自动当 ... 前缀
        #    token_positions 用 arange(seq) 或测试传来的
        # 没有token_postions就不用旋转
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4. causal mask（下三角）：
        #    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        #    True 在下三角（j ≤ i 能看），False 在上三角（未来被禁）
        mask = torch.tril(torch.ones(s, s, dtype=torch.bool))

        # 5. 调你写的 SDPA（自带 mask 支持）：
        #    out = scaled_dot_product_attention(Q, K, V, mask)  → (batch, num_heads, seq, d_k)
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 6. 拼回头：transpose(1, 2) → view(batch, seq, d_model)
        out = out.transpose(1, 2)  # (b, s, h, d_k)
        out = out.contiguous().view(b, s, -1)  # (b, s, h*d_k) = (b, s, d_model)
        # 7. 输出投影：out @ W_Oᵀ
        return self.o_proj(out)
```

+ 与其只用一组Q/K/V算一次attention，不如把d_model切成num_heads份、每份d_k = d_model / num_heads维，让每个head各自独立地取关注序列里不同类型的模型，最后把head结果拼回d_model维，再过一次投影W^O把信息重新混合
+ 作业明确要求：不能写个for循环跑num_heads次小attention，太慢。先用一次大的矩阵乘法把Q/K/V投影到完整的d_model维，再把最后一维reshape成（num_heads, d_k）,让多头变成一个额外的批量维度，一次性为给已经写好的dot_attention计算里面
+ rope要作用在每个head的d_k维上，而不是完整的d_model维，因为RoPE是逐位置旋转，跟head无关，所以旋转要在split成多头之后，算attention之前

#### 7. Pre-Norm Transformer Block

两个归一化子层+残差

整个Block是两个子层的堆叠，每个子层同一套模式

```
子层 1（注意力）：x → RMSNorm → 多头注意力 → 加回 x（残差）
子层 2（前馈）  ：x → RMSNorm → SwiGLU    → 加回 x（残差）
```

跟原始Transformer论文的post-norm不同，这里的pre-norm：先归一化，再送子层、最后把子层的输出加回没有被归一化的原始输入。好处：残差路径上是一条干净的加法链，不会被归一化层打断，梯度能更好地传回浅层

两个残差连接分别包住attention子层和FFN子层，各自用独立的一份RMSNorm，因为两个子层要归一化的分布不一样，不能共用一个参数

```
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)

        self.ln2 = RMSNorm(d_model)
        self.ffm = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions) # 子层1： norm + attn
        x = x + self.ffm(self.ln2(x))
        return x
```

### 3. 拼接模型，模型实现

语言模型接受（batch_size, sequence_length）的整数token id序列，输出(batch_size, sequence_length, vocab_size)的归一化概率分布（每个位置预测下一个token）

<img src="C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261006200440288.png" alt="image-20261006200440288" style="zoom:66%;" />

```
class Transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device, dtype)
            # 创建num_layers个TransformerBlock，装进一个列表
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.shape[-1], device=in_indices.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
```

![image-20261006200745460](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20261006200745460.png)