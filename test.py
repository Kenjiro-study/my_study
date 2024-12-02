# model_best.ptの中身を確認する
"""
import argparse
import random
import os
import time as tm
from itertools import chain
import torch
import torch.nn as nn
from torch import cuda
# 中身のテンソルの形とか確認してみよう
model1 = torch.load('checkpoint/lf2lf/model_best.pt', torch.device('cpu'))
model2 = torch.load('test/model_best.pt', torch.device('cpu'))


a = model1["model"]
for k, v in a.items():
    print(k)
    print("-----------------")
    print(len(v))
    print("-----------------")
    #break
print("-1-11-1-1-1-1-1-1--1-1-1--1--1")
b = model2["model"]
for k, v in b.items():
    print(k)
    print("-----------------")
    print(len(v))
    print("-----------------")
    #break

for k, v in model1.items():
    print(k)
    print("-----------------")
    print(v)
    print("-----------------")
    #break

print("-1-11-1-1-1-1-1-1--1-1-1--1--1")
for k, v in model1.items():
    print(k)
    print("-----------------")
    print(v)
    print("-----------------")
"""
# train-parsed.jsonの中身を確認する
"""
import json
import pandas as pd

json_open = open('craigslistbargain/data/train-parsed.json', 'r')
json_load = json.load(json_open)

data_list = []
meta_list = []

for i in range(len(json_load)):
    events = json_load[i]['events']
    for j in range(len(events)):
        data_list.append(events[j]['data'])
        meta_list.append(events[j]['metadata'])

dict1 = dict(data=data_list, meta_data=meta_list)
dia_df = pd.DataFrame(data=dict1)

print(dia_df)

dia_df.to_csv('train.csv')
"""

"""
# 各ダイアログアクトがどのような文についているか確認する
import pandas as pd
import math

data_list = []
meta_list = []
#count = 0

df = pd.read_csv("train.csv")
#print(df)

for i in range(0, 47986):
    data = (df.loc[i,"data"])
    meta = (df.loc[i,"meta_data"])
    
    if isinstance(meta, float):
        continue
    elif 'offer' in meta:
        #count += 1
        #print(data)
        #print(meta)
        data_list.append(data)
        meta_list.append(meta)

#print(count)

dict1 = dict(data=data_list, meta_data=meta_list)
dia_df = pd.DataFrame(data=dict1)

print(dia_df)

dia_df.to_csv('offer.csv')

#import torch
#print(torch.cuda.is_available())
import string
print(string.printable)
"""
"""
# pklファイルの中身を覗く
import pickle
# pklファイルのパスを指定
pkl_file_path = 'test/deep/mappings/lf2lf/vocab.pkl'

# pklファイルを読み込む
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 内容を表示
print(data["utterance_vocab"].finished)

#data.to_csv('rule-train-template.csv')
"""
import json

def extract_data(input_file, output_file):
    # 入力JSONファイルを読み込む
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # データがリストの場合に備えて、適切なデータ構造にアクセス
    if isinstance(data, list):
        data = data[0]  # リストの最初の要素を使用
    elif not isinstance(data, dict):
        raise ValueError("Invalid JSON structure: expected a dictionary or a list of dictionaries.")
    
    # 新しいデータを格納するリストを初期化
    extracted_data = []
    
    # eventsのリストからagent, action, data, intentを抽出
    for event in data.get("events", []):
        # intentの値を取得し、存在しない場合はNoneを設定
        intent = (event.get("metadata", {})
                      .get("sent", {})
                      .get("logical_form", {})
                      .get("intent", None) if event.get("metadata") else None)
        
        # 必要な情報を追加
        extracted_data.append({
            "agent": event.get("agent"),
            "action": event.get("action"),
            "data": event.get("data"),
            "intent": intent
        })
    
    # 抽出したデータを新しいJSONファイルに書き込む
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# 使用例
extract_data("scripts/transcripts_m.json", "detail.json")