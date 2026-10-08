# CS336作业一 -train_BPE 的训练函数

## 1.函数总览

+ 词表初始化，256个字节+特殊 token
+ 预分词 + 计数
  + 预分词跟encode差不多，首先读语料，然后切分，不过训练的时候不需要特殊token
  + 语料 转化为counts字典：{key = （整词拆成单字节元组），值 = 词出现的频率}
    + count主要是词的次数{(b'l',b'o',b'w'): 5},low出现了5次
+ 合并循环（反复，直到词表满vocab_size）,一直合并就会一直产生新的token放到词表末尾
  + a. 从 counts 数所有相邻对，按词频加权 ->pair_counts     
  + b. max 挑最高频对，平局取字典序大的 ->max_pair      
  + c. 每个词里把 max_pair 切片替换合并，更新 counts      
  + d. max_pair 记进 merges，新 token 加进词表
+ return (vocab_train, merges)

![image-20260921231046636](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20260921231046636.png)

## 2.实现

### 2.1词表初始化

+ 256 个字节 + 特殊 token
  + 刚开始256字节可以采用循环赋值

```
# 1.首先词表初始化，初始词表 = 256个字节 + 特殊token
    vocab_train = {}
    for i in range(256):
        vocab_train[i] = bytes([i]) # 循环赋值
    for i in range(len(special_tokens)):
        vocab_train[i+256] = special_tokens[i].encode("utf-8", errors='ignore') # special是字符串
```

### 2.2 预分词＋计数

+ 首先读取语料，最后要加read方法，这样返回的才是字符串内容

```
# 2.预分词+计数
    # 读语料
    text = open(input_path, encoding="utf-8").read() # 如果不加read，返回的是文件对象，不是字符串内容
```

+ 与 encode 相同的特殊 token 切分法（排序 + 转义 + 捕获组 split）

  + > 先排序再转义避免转移之后数量产生变化
    >
    > 特殊 token 直接丢弃（不参与训练）

```
# 跟encode一样的切分方法
reverse_special_tokens = sorted(special_tokens, key=len, reverse=True)
# 用temp存储转义后的
temp_special_tokens = []
for token in reverse_special_tokens:
    # re.escape传入字符串，然后对其进行转义
    temp_special_tokens.append(re.escape(token))
# 拼pattern，把排好序，转移好的字符串用"|"拼成正则字符串
pattern = re.compile('(' + '|'.join(temp_special_tokens) + ')')
# 切分，用pattern把text切成两种文本，特殊的在这里面不需要处理，普通的走普通的
pieces = re.split(pattern, text)
```

+ GPT-2 正则预分词 ，每个词转字节串

```
# pre_token是去除了所有special_token之后的分词结果
pre_token = []
for piece in pieces:
    # 这里的special_tokens可以直接丢弃，只需要正常文本即可，特殊token不需要训练
    if piece not in special_tokens:
        # 按GPT-2的规则预分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        piece_pre_token = regex.findall(PAT, piece)
        # 转换成列表，存每一个token
        for i in range(len(piece_pre_token)):
            # pre_token里面是[b'the', b'cat', b'dog']
            pre_token.append(piece_pre_token[i].encode("utf-8", errors='ignore'))
```

+ 计数

  > 计数的时候就是记录每一个单词，记录这个单词出现的顺序，但是记录到counts里面的时候要把这个单词拆成一个一个字母的形式
  >
  > 因为key是元组，写的时候可以先写到列表里面，然后再转成元组赋值到counts的key里
  >
  > 对单词切成字母的时候用到python的切片

```
# 训练时要用的形式是counts: dict{tuple[bytes, ...], int}
# count主要是词的次数{(b'l',b'o',b'w'): 5},low出现了5次
# 为什么count里面的键要用元组，因为字典的键必须可以哈希，列表是可变的
counts = {}
for token in pre_token:
    # 和encode不同的是，这里不再是数字母了，就是将一个词拆开
    temp_lst = []
    for j in range(len(token)):
        temp_lst.append(token[j:j+1])  # word[0:1]=b't'，word[1:2]=b'h'，word[2:3]=b'e'（左闭右开，正好切1个字节）
        # get,查key现在的次数，没有就返回默认的0，然后+1存在value里面
    temp_tuple = tuple(temp_lst)
    counts[temp_tuple] = counts.get(temp_tuple, 0) + 1
```

### 3.合并循环

+ while判断，当词表数量没达到要求的时候，其实就是还没循环完，每次循环都会产生新的合并后的token放到词表里面，词表都会变大

```
# 3.合并循环
merges = []
# while里面为什么是这个判断
while(len(vocab_train) < vocab_size):
```

+ pair_counts负责存的是这个“词对”出现了几次，pair就是词对
  + 首先找count的key和value，key就是单词，value就是单词出现的次数
  + 然后遍历单词里面的字母，相邻的字母就会组成词对

```
# 1.首先，从counts里面推导相邻的计数对
    pair_counts = {}
    for key,value in counts.items(): #遍历counts.items(每个词元组+它的次数)
        # 词元组里面，相邻的两个元素可以组成对
        for i in range(len(key) - 1):
            pair = (key[i], key[i+1]) # 组成了相邻的元素对
            pair_counts[pair] = pair_counts.get(pair, 0) + value  # 这个字典里面存的就是字母对的个数多少
```

+ 挑最高频的对，平局自动取字典序大的

  + >第一项 `pair_counts[p]`：**先比次数**，谁多选谁；
    >
    >第二项 `p`：**次数相同比谁大**（字典序），这样结果确定、可复现。

    # 2.挑最高频的对，平局自动取字典序大的
    max_pair = max(pair_counts, key = lambda p : (pair_counts[p], p)) # 比较p的大小，也就是字母对个数的多少

+ 切片替换
  + 切片替换后列表短了 1
  + 替换的时候可以先把元组转换成列表，这样就可以替换操作了，还是切片，就是将两个最高频挨一块的合成一个
  + "containing" = [c,o,n,t,a,i,n,i,n,g]，合并 (i,n) 。结果：[c,o,n,t,a,in,in,g]

```
# 3.合并counts：把每个词里面相邻的max_pair替换成一个新字节
new_counts = {}
for key,value in counts.items():
    word = list(key) # 转换成列表，切片替换，再换成元组,因为元组不可变 [b'l',b'o',b'w']
    i = 0
    while i < len(word)-1:
        if word[i] == max_pair[0] and word[i+1] == max_pair[1]:
            word[i:i+2] = [word[i] + word[i+1]] # 合并完之后列表变短了
            i += 1
        else:
            i += 1
    new_counts[tuple(word)] = new_counts.get(tuple(word), 0) + value
counts = new_counts
```

+ 上文举例的in就可以进词表了
  + merges = (b'i', b'n')
  + vocab_train进入 b'in'

```
# 4.新字节进词表
merges.append(max_pair)
vocab_train[len(vocab_train)] = max_pair[0] + max_pair[1]
```

![image-20260921224939908](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20260921224939908.png)

