
运行其中模块：
`python -m model.bilstm`

1. 在 `main.py` 中 `Config(dataset="./mydatasets/jieba", embedding="random", dataset_method="jieba")`
修改数据集地址，以及数据集方法
2. `config.py`中 修改 `model_name`，所有结果文件会存储在 `result/model_name`
3. `plot.py` 绘制训练曲线图和混淆矩阵，修改里面的 parser
