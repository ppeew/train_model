# -*- coding: utf-8 -*-
import os

from transformers import BertTokenizer
import torch

from bert_get_data import InitLabelEncoder

bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'best_2.pt'
save_path = './bert_checkpoint'

# 加载训练好的模型
model = torch.load(os.path.join(save_path, model_name), map_location=device,weights_only=False)
model = model.to(device)
model.eval()

# 初始化 LabelEncoder
label_encoder = InitLabelEncoder()
# 获取LabelEncoder中已经见过的类别数
known_labels = label_encoder.classes_

while True:
    text = input('请输入分析语句：')

    # 处理输入文本
    bert_input = tokenizer(text, padding='max_length',
                           max_length=500,
                           truncation=True,
                           return_tensors="pt")

    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].to(device)

    # 模型预测
    with torch.no_grad():
        output = model(input_ids, masks)

    # 获取输出并去除梯度信息
    output = output.cpu().numpy()

    # 获取两个最高概率的类别
    top_2 = output[0].argsort()[-2:][::-1]

    # 检查预测的类别是否在训练时的标签范围内
    valid_top_2 = [label for label in top_2 if label in range(len(known_labels))]

    if not valid_top_2:
        print("预测的类别超出了已知标签范围。")
    else:
        pred_str = label_encoder.inverse_transform([valid_top_2[0]])[0]
        print('第一个答案:', pred_str)

        if len(valid_top_2) > 1:
            label_str = label_encoder.inverse_transform([valid_top_2[1]])[0]
            print('第二个答案:', label_str)
        else:
            print('没有找到第二个有效标签')
