# -*- coding: utf-8 -*-
import os
from transformers import BertTokenizer
import torch

import bert_get_data
from bert_get_data import BertClassifier,label_encoder

bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = input("请输入要测试的模型名称（best_{num}.pt）:")

save_path = './bert_checkpoint'
model = BertClassifier(classifier_num=2520)
model.load_state_dict(torch.load(os.path.join(save_path, model_name), weights_only=True))
model = model.to(device)
model.eval()

while True:
    text = input('请输入分析语句：')
    bert_input = tokenizer(text, padding='max_length',
                           max_length=50,
                           truncation=True,
                           return_tensors="pt")
    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].unsqueeze(1).to(device)
    output = model(input_ids, masks)
    pred = output.argmax(dim=1)
    # 获取两个最高概率的类别
    _, top_2 = torch.topk(output, 2)
    top_2_labels = [x.item() for x in top_2[0]]

    pred_str = label_encoder.inverse_transform([top_2_labels[0].item()])
    label_str = label_encoder.inverse_transform([top_2_labels[1].item()])
    print('第一个答案:', pred_str)
    print('第二个答案:', label_str)

    # print(real_labels[pred])

# 输出相关的类别
# 做分词转化(提取信息),动作行为（怎么生成数据集呢？）