bilstm+attention：

双向lstm，会利用两边的隐向量，最终句子表示可以通过self-attention

![1781148582389](image/report/1781148582389.png)

$$
out=(B,L,2*H_{out})\\
W=(2*H_{out},1)\\
\alpha=\text{softmax}(tanh{(out)}W)=(B,L,1)\\
\sum_l\alpha*out=(B,2*H_{out})
$$



### 数据预处理

原数据为jsonl格式（即每行一个json对象，可以独立解析）

```json
{ label, 	// 为标签数字
label_desc,  	// 标签含义
sentence,    	// 句子
keywords}	// 关键词
```


空样本或纯空白样本，非法标签，重复样本（句子和标签完全一致），类别图片见 `image\data_analysis\train_label_distribution.png`

![1781151066296](image/report/1781151066296.png)

没有空文本和非法标签，有多个重复样本

news_stock 类比的数据较少


`unicodedata.category` 通过这个可以判断unicode的一些不可见字符

全角转半角 `str` 提供 `translate` 对表格进行转换，没出现的字符保留

通过正则表达式 `\s+` 表示连续空白字符，可以通过 `re.sub` 进行合并空白

对于数字和英文，他们是有特定含义的，因此不能去除，文本分类中标点符号不太重要，可以去除，数字由于直接保留会导致词汇表太大，因此可以使用 `<NUM> 表示`

中文区间：\u4e00-\u9fff (基本汉字) + \u3400-\u4dbf (扩展A)

清洗的文本存在 `cleaned` 字段


jieba分词：使用精确模式，不存在冗余单词（搜索引擎模式，会将长词再次切分，有冗余，可以增加搜索准确率），对于文本分类任务精确模式就足够了

缺点：OOV现象

![1781182440426](image/report/1781182440426.png)

数据多为短文本，因此平均token数在10.9左右，出现了较多OOV现象

分词样例

```
  样例(train[0]):
    原文  : 新骗局！一分不投两年赚2400万 至少已有数百人入群
    清洗后: 新骗局一分不投两年赚<NUM>万 至少已有数百人入群
    分词  : ['新', '骗局', '一分', '不', '投', '两年', '赚', '<NUM>', '万', '至少', '已有', '数百人', '入群']
```


lr_scheduler：step方式每轮epoch对lr进行下降

精确率：所有预测为正的有多少是正的；召回率：所有真实正样本有多少被找出来

$$
Precision = \frac{TP}{TP + FP}\\
Recall = \frac{TP}{TP + FN}
$$

$$
F1_i = \frac{2 \cdot Precision_i \cdot Recall_i}{Precision_i + Recall_i}
$$

多分类F1，对每个类别做二分类f1计算，然后取平均


对于CE loss：为15分类，随机为 $-\log(1/15) =2.7$


adamW增加了weight_decay：用于减小权重，越大时模型参数会越小（类似L2正则化）

![1781249250689](image/report/1781249250689.png)

lr调度器：step（每隔几个epoch，将lr乘上系数gamma）；cosine（随着epoch缓慢降低学习率）


测试小样本（10000个训练数据，1000个测试和验证集）

出现了一个非常奇怪的现象，就是验证集的准确率平稳慢慢上升，但是loss竟然也在大幅度上升

![1781248607781](image/report/1781248607781.png)

原因：根据CE公式，当模型对错误样本过于自信会导致loss极端上升
