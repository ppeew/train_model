# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader

import bert_get_data
from bert_get_data import BertClassifier, MyDataset, GenerateData, tokenizer
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(save_path, save_name))

# 训练超参数
epoch = 2
batch_size = 32
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20240725
save_path = './bert_checkpoint'
setup_seed(random_seed)

label_encoder = bert_get_data.InitLabelEncoder()

# 构建数据集
train_dataset = GenerateData(mode='train',label_encoder=label_encoder)
dev_dataset = GenerateData(mode='test',label_encoder=label_encoder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)


def find_max_num_file(directory):
    max_num = -1
    max_file = None
    for filename in os.listdir(directory):
        if filename.startswith('best_') and filename.endswith('.pt'):
            num_str = filename.split('_')[1].split('.')[0]
            num = int(num_str)
            if num > max_num:
                max_num = num
                max_file = filename
    return max_file

# 定义模型
model = BertClassifier(classifier_num=2520)
file=find_max_num_file("bert_checkpoint")
print("load model from:",file)

# model.load_state_dict(torch.load(os.path.join(save_path, file),weights_only=True))
# model = model.to(device)
model = torch.load(os.path.join(save_path, file), map_location=device,weights_only=False)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader):
        input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
        masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
        labels = labels.to(device)
        output = model(input_ids, masks)

        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    
    with torch.no_grad():
        for inputs, labels in dev_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()

            for i,(input,label) in enumerate(zip(inputs['input_ids'],labels)):
                pred = output.argmax(dim=1)[i]

                # 解码预测结果，复原为原始的中文文本
                input_text = tokenizer.decode(input.squeeze().cpu().numpy(), skip_special_tokens=True)

                pred_str = label_encoder.inverse_transform([pred.item()])
                label_str = label_encoder.inverse_transform([label.item()])
                print(
                    f'''input_text:{input_text.replace(" ", "")} | pred: {pred_str} | label: {label_str} 
                                   ------> 预测{['正确' if pred.item() == label.item() else '错误']}''')

        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')
        
        # 保存最优的模型
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model("best_{}.pt".format(int(best_dev_acc*100)))
        
    model.train()

# 保存最后的模型
save_model('last.pt')

