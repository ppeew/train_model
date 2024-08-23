import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder

bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)


class MyDataset(Dataset):
    def __init__(self, df,label_encoder):
        # 初始化标签编码器
        self.label_encoder = label_encoder
        # 将文本转为BERT输入格式
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=500,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]

        # 将中文标签转换为数值化标签
        self.labels = torch.tensor(self.label_encoder.transform(df['label']), dtype=torch.long)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def getLabelEncoder(self):
        return self.label_encoder


class BertClassifier(nn.Module):
    def __init__(self, classifier_num):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, classifier_num)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def GenerateData(mode,label_encoder):
    train_data_path = 'DataWords/data/train_data.csv'
    dev_data_path = 'DataWords/data/test_data.csv'
    test_data_path = 'DataWords/data/test_data.csv'

    train_df = pd.read_csv(train_data_path, sep='|', header=None)
    dev_df = pd.read_csv(dev_data_path, sep='|', header=None)
    test_df = pd.read_csv(test_data_path, sep='|', header=None)

    new_columns = ['text', 'label']
    train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
    dev_df = dev_df.rename(columns=dict(zip(dev_df.columns, new_columns)))
    test_df = test_df.rename(columns=dict(zip(test_df.columns, new_columns)))

    # 对训练集和测试集的标签进行编码
    train_dataset = MyDataset(train_df,label_encoder)
    dev_dataset = MyDataset(dev_df,label_encoder)
    test_dataset = MyDataset(test_df,label_encoder)

    if mode == 'train':
        return train_dataset
    elif mode == 'val':
        return dev_dataset
    elif mode == 'test':
        return test_dataset


def GetClassifierLen(train_dataset):
    label_set = set()
    for data in train_dataset.labels:
        label_set.add(data)
    return len(label_set)


def InitLabelEncoder():
    label_encoder = LabelEncoder()
    train_data_path = 'DataWords/data/train_data.csv'

    train_df = pd.read_csv(train_data_path, sep='|', header=None)

    new_columns = ['text', 'label']
    train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))

    # 使用整个数据集的标签来拟合 LabelEncoder
    label_encoder.fit(train_df['label'])
    return label_encoder
