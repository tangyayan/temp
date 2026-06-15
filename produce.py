import json
file_dir = './result/'
# file_names = ['Transformer_cls_char_sogou','Transformer_jieba_sogou_mean', 'Transformer_jieba_sogou_attention', 
#               'Transformer_jieba_sogou_cls', 'Transformer_jieba_sogou_cls_w']
# output_names = ['CLS','Mean Pooling', 'Attention Pooling', 
#                 'CLS', 'CLS + CE Weight']
# word = ['Char','Jieba', 'Jieba', 'Jieba', 'Jieba']

file_names = ['BiLSTM_att_char_sogou','BiLSTM_char_sogou', 'BiLSTM_att_jieba_sogou', 
              'BiLSTM_jieba_sogou', 'BiLSTM_att_jieba_sogou_w']
output_names = ['Attention','-', 'Attention', '-', 'Attention + CE Weight']
word = ['Char','Char', 'Jieba', 'Jieba', 'Jieba']
for i,file_name in enumerate(file_names):
    with open(file_dir+file_name+'/test_results.json', 'r') as f:
        result = json.load(f)
        acc = result['accuracy'] *100
        f1 = result['macro_f1'] *100
    with open(file_dir+file_name+'/hyperparameters.json', 'r') as f:    
        result = json.load(f)
        time = result['training_time_seconds']
    # print(f'& {output_names[i]} & {time:.2f} & {acc:.2f} & {f1:.2f} \\\\')
    print(f'& {word[i]} & {output_names[i]} & {time:.2f} & {acc:.2f} & {f1:.2f} \\\\')

"""
& Char  & —                  & xx.xx & xx.xx & xx.xx \\
& Jieba & —                  & xx.xx & xx.xx & xx.xx \\
& Char  & Attention          & xx.xx & xx.xx & xx.xx \\
& Jieba & Attention          & xx.xx & xx.xx & xx.xx \\
& Jieba & Attention + CE Weight & xx.xx & xx.xx & xx.xx \\
\midrule
\multirow{4}{*}{Transformer}
& Jieba & Mean Pooling          & 140.06 & 54.96 & 52.95 \\
& Jieba & Attention Pooling     & 151.89 & 54.78 & 52.35 \\
& Jieba & CLS                   & 179.78 & 55.14 & 52.75 \\
& Jieba & CLS + CE Weight       &  92.27 & 52.98 & 51.58 \\
"""