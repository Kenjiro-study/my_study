import pandas as pd
import linecache
import json
from glob import glob
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.nn.functional import softmax

def classify_intent_neural(texts, path):
    # GPUの指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 作成したモデルの読み込み
    checkpoint = path # 好きなモデルを選択
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
    model = model.to(device) # GPUにモデルを送る

    inputs = tokenizer(texts['pre_text'], texts['text'], max_length=512, truncation=True, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    outputs = model(**inputs)
    
    # 推論結果の取得
    logits = outputs.logits # ロジットの取得
    probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
    predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
    #print("a: ", predicted_class)
    predicted_class = model.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換
    #print("b: ", predicted_class)
        
    return predicted_class