# CS336作业一 –BPE 构建

## 一、知识点总结

### 1.分词器

+ 分词器本身不直接提升算力，合适的分词策略，可以让模型参数、显存、计算资源被更好利用；坏的分词会浪费算力

#### 以字节为分词的分词器

+ 会面临长序列问题
  + 英文字母，UTF-8单字节
  + 中文字母，UTF-8编码三个字节
+ token序列更长
+ 用长token为代价，换取没有未知token的鲁棒性

#### 字符分词器

+ 最基础单元是Unicode字符（汉字、字母、标点各算1个基础单元）
+ 词表容量有限，如果遇到词表没有收录的生僻字、罕见符号将无法拆成字节继续表达，之恩那个输出<unk>,丧失原本文本信息

`"文本".encode("utf-8")` → bytes；`bytes.decode("utf-8")` → 文本。

#### BPE

+ 反复合并语料里面出现最多的两个相邻的token，直到词表达到目标大小
+ 现代 LLM 大多用 Byte-level BPE（字节 BPE，GPT2），无未知<unk>
  1. **字符 BPE（原始 BPE）**：最小原子单元是 Unicode 字符。初始词表是所有字符，有`<unk>`未知 token。
  2. **Byte-level BPE（字节 BPE，GPT2）**：最小原子单元是 UTF-8 字节。初始词表固定 256 个字节，**永远没有<unk>**，任何文本都能拆成字节表示，这是它最大亮点。

### 2.词表的构成

vocab_size由三部分构成：

| 256个初始字节token + 特殊token （如<\|endoftext\|>）+ 合并产生的新token |
| ------------------------------------------------------------ |

### 3.预分词（pre-tokenization）

直接对整篇文本合并代价很大（会跨词合并出奇怪的东西，比如空格和词粘在一起），GPT-2的做法是先用一个正则把文本切成“预分词单元”（大致是：单词、数字、标点、空白），合并只在每个预分词单元内部进行，不跨单元

| GPT2_PRETOKENIZE_PATTERN = r”‘“ ‘ (?:[sdmt] \| ll \| ve \| re ) \| ?\p{L}+\| ?\p{N}+ \| ?\[^s\p{L}\p{N}] + \| \s+(?!\S) \| \s+“”‘’ |
| ------------------------------------------------------------ |

+ ‘ (?:[sdmt] | ll | ve | re ) ：英文缩写，如’s  ’ll  ‘ve
+ ?\p{L}+ ：可选前导空格 + 连续字母（单词）
+  ?\p{N}+： 可选前导空格+ 连续数字
+  ?\[^s\p{L}\p{N}] +： 可选前导空格+连续标点符号
+ \s+(?!\S) | \s+:处理连续空白

### 4.特殊token不参与合并

特殊token（如<|endoftext|>）在训练前就先从文本中切分出去（re.split）保证：

​	1.预分词和合并都不会看到特殊token的内部结构

​	2.合并产生的新token不会跨越特殊token的边界，泄露出<|这种字节片段

### 5.合并规则与打破平局（tie-break）

每一轮：

​	1.统计当前所有相邻pair出现的次数

​	2.选出出现最多的pair

​	3.如果并列最多，选取字节字典序更大的pair（为了和GPT2保持一致，和参考答案对上）

### 6.增量更新—训练效率的关键

朴素实现：每合并一次就重新扫描全部文本统计一边pair频率—太慢了

更快： 维护一个“pair-包含该pair的预分词单元下标集合”的倒排索引，每次合并只需要：

	+ 找到受影响的预分词单元（倒排排序，不用扫全表）
	+ 只在这些单元内部减掉旧pair计数、做合并、加上新pair计数

## 代码实现

### 1.主体骨架

在 `cs336_basics/tokenizer.py` 里建一个 `Tokenizer` 类，**方法签名照抄作业 §2.6 给的接口**（`__init__`、`from_files`、`encode`、`encode_iterable`、`decode`），方法体先空着（`pass`）。这一步的目的是 "先把形状立起来"—— 就像写作文先列标题。

### 2.decode的实现

逻辑就三步：查词表（ID → bytes）→ 拼接 → `.decode("utf-8", errors="replace")`。

输入的vocab是一个字典，vocab = {0:b'a', 1:b'b', 2:b'ab'}，ids是一个数组[0,2,1],结果应该是aab\b

```
    def decode(self, ids):
        # res不能直接命名为空字符串，要命名为字节串
        res = b""
        for i in ids:
            # 字符串字节串没有append方法，列表有这个方法
            res += self.vocab[i]
        # 循环完之后整体解码，词表里面的一个词条未必是完整的UTF-8字符，可能把汉字拆成两个词条,res.decode会返回一个新的字符串
        res = res.decode("utf-8", errors="replace")
        return res
```

### 3.\_init_的实现

+ 这里需要实现刚开始vocab的反转

>  decode要是的ID->字节串，然后拼接转换成字符串

> encode要的是字节串->id,拿着字节串问，这个字节属于几号，比如想找b’ab’的ID，直接查反向表就可以，encode是给字符串，先转换成字节，再bpe分词，再转换成id

+ 为什么放在\_init_里面？

+ > 因为词表创建时就固定了，放在init里面只需要反转一次

merge是什么，用在encode函数里面，merge也是一个列表，里面每一项是一个规则，也就是一个    元组（A,B），反复合并相邻次数最多的A,B

```
    def __init__(self, vocab, merges, special_tokens=None):
        token_to_id = {}
        # 把vocab存成字典形式转换成数字对应，字节对应数字序列
        # 遍历字典的值，转换成新字典的键
        # i = 0
        # for key in vocab.values():
        #     token_to_id[key] = i
        #     i += 1

        # vocab.items()的每一项是(ID, 字节串)
        for key, value in vocab.items():
            token_to_id[value] = key
        self.token_to_id = token_to_id

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
```

### 4.decode的实现

+ decode的实现很简单，根据id，将字节一个一个拼起来，最后转换成字符

```
    def decode(self, ids):
        # res不能直接命名为空字符串，要命名为字节串
        res = b""
        for i in ids:
            # 字符串字节串没有append方法，列表有这个方法
            res += self.vocab[i]
        # 循环完之后整体解码，词表里面的一个词条未必是完整的UTF-8字符，可能把汉字拆成两个词条,res.decode会返回一个新的字符串
        res = res.decode("utf-8", errors="replace")
        return res
```

### 5.encode的实现

#### 5.1 普通encode

+ 1.预分词,先将字符串文本按GPT-2标准分词转换成字节串

```
def encode_normol(self, text):
        # 1.预分词,先将字符串文本按GPT-2标准分词转换成字节串
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 正则化之后的pre_token对象变成了列表
        pre_token = regex.findall(PAT, text)
        for i in range(len(pre_token)):
            pre_token[i] = pre_token[i].encode("utf-8", errors='ignore')
```

+ 2.将字节串，按照每个字母每个字母的分，拆开

```
 # 2.将字节串，按照每个字母每个字母的分，拆开
        separate_token = []
        # 先遍历外层再遍历内层，取出每一个单词再取出每一个字母
        for i in range(len(pre_token)):
            temp_token = pre_token[i]
            # son_sepatate_token建立在循环里面，就可以指向不同的对象，从而有不同的列表元素
            son_separate_token = []
            for j in range(len(pre_token[i])):
                son_separate_token.append(temp_token[j:j + 1])
            # 创建完之后的列表[[b's', b'o', b'm', b'e'], [b' ', b't', b'e', b'x', b't']]
            separate_token.append(son_separate_token)
```

+ 3.合并（对每个子列表，反复找第一个能套用的合并并应用，知道没有合并能套用为止）
  + 合并过程中，可以分别对每个小字节串处理，调用函数

```
# 3.合并（对每个子列表，反复找第一个能套用的合并并应用，知道没有合并能套用为止）
        for i in range(len(separate_token)):
            separate_token[i] = self.merge_one(separate_token[i])
```

```
    # 合并所有的子序列一起是非常复杂的，这个merge_one函数是根据merge的元组，对一个子序列sub进行操作，sub只是一个子序列
    def merge_one(self,sub):
        while True:
            # flag判断这一轮有没有命中过，如果所有规则都没有命中任何一个子序列，flag = false，结束
            flag = False
            # merges表是训练好的，这里可以直接用
            # 将子表里面的元素检查的元素与merges每一个元组都判断一遍
            for merge in self.merges:
                for i in range(len(sub) - 1):
                    if (sub[i] == merge[0]) and (sub[i+1] == merge[1]):
                        # 找到可以合并的，就将子列表的这两项合并到一起
                        # 合并的时候用切片语法，先拼，再用切片替换sub[j:j+2] = [sub[j] + sub[j+1]]
                        # 这里开始是j+2，因为引号后面的事不包含的
                        sub[i:i + 2] = [sub[i] + sub[i + 1]]
                        flag = True
                        break  # 跳出这个子序列，根据规则重新扫描
                if (flag == True): break # 合并过的话就跳出循环，不然会导致一个规则（元组）没判断完就到另一个规则
            if (flag == False): break  #全部扫完也没有合并的，说明这个子序列完成了
        return sub
```

+ 4.字节串->ID

```
        # 4.字节串->ID
        ids = []
        for i in range(len(separate_token)):
            for token in separate_token[i]:
                # 遍历每个字节串，根据函数查出id，放进ids
                ids.append(self.token_to_id[token])
        # 传进来的字符串最后转为了id的形式返回
        return ids
```

#### 5.2 特殊token的处理

+ 因为|在正则表达式里面是或的意思，会出现问题，可以用re.escape(token)对每一个特殊token处理之后，就可以转义
+ 转义之后，要进行排序，长的在前面，作业要求了，两个一样的出现时要识别成一个token

> 在转移之前先对特殊token排序，排序完再转义，可以避免转移之后长度发生变化

| 拼法                | 结果 |           |        |           |      | 含义                                       |
| ------------------- | ---- | --------- | ------ | --------- | ---- | ------------------------------------------ |
| 短的在前（`单|双`） | `['< | endoftext | >', '< | endoftext | >']` | 正则先试短的、，双连被**拆成两个 token** ✗ |
| 长的在前（`双|单`） | `['< | endoftext | ><     | endoftext | >']` | 正则先试长的、整体匹配，**一个 token** ✓   |

+ 特殊toen和普通token互不影响
  + 特殊 token → 走 "查 ID" 捷径（不预分词、不合并）；
  + 普通文本 → 走完整管线（预分词 → 拆字节 → 合并 → 查 ID）

+ 拼 pattern：把排好序、转义好的字符串，用 `"|"` 拼成一个正则字符串
  + 可能有**多个**特殊 token（测试里有 1 个的，也有 2 个的：`<|endoftext|>` 和双连）。你需要**一个**正则，能在**一次从左到右的扫描**里把**任意一个**特殊 token 都找出来。

```
# 如果没有特殊文本，直接走普通，不然sorted(None)会崩
        if self.special_tokens is not None:
            # 首先，对特殊字符按长度降序，防止escape之后长度变化，转义后的排序跟前面的不一样
            reverse_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 用temp存储转义后的
            temp_special_tokens = []
            for token in reverse_special_tokens:
                # re.escape传入字符串，然后对其进行转义
                temp_special_tokens.append(re.escape(token))
            # 拼pattern，把排好序，转移好的字符串用"|"拼成正则字符串
            pattern = re.compile('(' + '|'.join(temp_special_tokens) + ')')
```

+ 切分：把文本切成两个片段，特殊token走特殊token的路，普通文本片段走普通文本片段的路

切分之后普通段走普通的，特殊的直接查id，append

```
# 切分，用pattern把text切成两种文本，特殊的走特殊的，普通的走普通的
            pieces = re.split(pattern, text)

            # 存储答案的列表
            res = []
            # 判断，特殊文本直接查id，普通文本调用函数
            for piece in pieces:
                if piece in self.special_tokens:
                    res.append(self.token_to_id[piece.encode("utf-8")]) # piece原本是字符串，转成字节串去查id
                else:
                    # 这里如果用append方法会产生多个对象
                    res.extend(self.encode_normol(piece))
            return res
        else:
            return self.encode_normol(text)
```

### 6.encode_iterable的实现

+ 这段代码存在的意义是懒加载-大文件一行一行喂过来，不能全文读完再解码，需要“取一段、算一段、交一段”

  > 普通函数，return直接结束
  >
  > yield函数（生成器），每次迭代到yield就“暂停”，交值出去，下次for继续从暂停出走

```
        for piece in iterable:
            yield from self.encode(piece)
```

