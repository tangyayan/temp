# **中文短新闻主题分类**

---

唐雅妍23320143

本项目基于 PyTorch 实现 TNEWS 新闻短文本分类，支持 BiLSTM、BiLSTM + Attention 和 Transformer 三种模型，并提供 jieba 分词和字符级两种数据处理方式。

## 项目结构

```text
.
├── main.py                  # 训练与测试主入口
├── config.py                # 模型、训练和数据路径配置
├── dataset.py               # TNEWS 数据集封装与 DataLoader collate 函数
├── train.py                 # 训练、验证、测试逻辑
├── plot.py                  # 训练曲线和混淆矩阵绘制
├── produce.py               # 汇总实验结果
├── model/
│   ├── bilstm.py            # BiLSTM 模型
│   ├── bilstm_att.py        # BiLSTM + Attention 模型
│   └── transformer.py       # Transformer 模型
├── mydatasets/
│   ├── tnews_public/        # 原始 TNEWS 数据
│   ├── tnews_cleared/       # 清洗后的数据
│   ├── jieba/               # jieba 分词后的数据、词表和词向量
│   ├── char/                # 字符级切分后的数据、词表和词向量
│   ├── clear_data.py        # 原始数据清洗脚本
│   ├── jieba_data.py        # jieba 分词与词表构建脚本
│   ├── char_data.py         # 字符级切分与词表构建脚本
│   └── wordvector.py        # 预训练词向量转换脚本
├── image/                   # 报告图片与数据分析图
├── result/                  # 训练结果输出目录
└── report.pdf               # 实验报告
```

## 数据集说明

使用 CLUE benchmark 中的 TNEWS 中文新闻分类数据集。每条样本主要包含：

- `sentence`：新闻标题或短文本
- `label`：类别编号
- `label_desc`：类别名称
- `keywords`：关键词，原始数据中提供

数据集共 15 个类别，包括故事、文化、娱乐、体育、财经、房产、汽车、教育、科技、军事、旅游、国际、证券、农业、游戏等。原始数据位于 `mydatasets/tnews_public/`，处理后的训练数据位于 `mydatasets/jieba/` 或 `mydatasets/char/`。

当前已处理完的数据规模为：

- `train.jsonl`：45248 条
- `dev.jsonl`：5028 条
- `test.jsonl`：9793 条

## 数据预处理

如果需要从原始数据重新生成处理后的数据，可以依次运行：

```bash
python mydatasets/clear_data.py
python mydatasets/jieba_data.py
```

如果使用字符级切分，则运行：

```bash
python mydatasets/clear_data.py
python mydatasets/char_data.py
```

预训练词向量文件已转换为 `sogou.pt`，训练时可通过配置选择 `sogou` 或 `random`。

## 训练方法

训练入口为 `main.py`。运行前先在 `main.py` 中修改以下配置：

```python
dataset = "./mydatasets/jieba"      # 可选 "./mydatasets/jieba" 或 "./mydatasets/char"
embedding = "sogou"                 # 可选 "sogou" 或 "random"
datamethod = "jieba"                # 可选 "jieba" 或 "char"
model_name = "Transformer"          # 可选 "BiLSTM_att"、"BiLSTM" 或 "Transformer"
```

然后执行：

```bash
python main.py
```

训练完成后，结果会保存在：

```text
result/{model_name}_{dataset_method}_{embedding}/
```

Transformer 使用 `cls` pooling 时，目录名会额外带上 `_cls`。目录中通常包含：

- `best_model.pt`：验证集 macro-F1 最优模型
- `history.json`：训练过程记录
- `test_results.json`：测试集评估结果
- `hyperparameters.json`：本次训练配置

目前 `resulst/` 为使用 Transformer+CLS+Jieba分词的结果。

### 自定义超参数

超参数定义位于 `config.py`，可以修改配置，目前显示为经实验训练测试得到的各模型最佳配置。

## 绘图

训练完成后，可以绘制训练曲线和混淆矩阵：

```bash
python plot.py --model_name Transformer_jieba_sogou_cls --result_dir ./result
```
