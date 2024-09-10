from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.nn.functional import softmax

def oneshot_classify_intent(model, tokenizer, pre_text, text):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(pre_text, text, max_length=512, truncation=True, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    outputs = model(**inputs)

    # 推論結果の取得
    logits = outputs.logits # ロジットの取得
    probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
    predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
    predicted_class = model.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換

    return predicted_class

def classify_intent_neural(examples, path):
    # GPUの指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 作成したモデルの読み込み
    checkpoint = path # 使用したいモデルのパスを持ってくる
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
    model = model.to(device) # GPUにモデルを送る

    intent_dic = {} # 各対話のintentが格納された配列をまとめるための辞書
    for i in range(len(examples)):
        print(f"Dialogue {i + 1}") # 何対話目まで処理したかの確認
        events = examples[i].events # テキストが入っているeventsを取り出す
        intent_list = [] # 対話ごとのintentを格納するリスト
        for j in range(len(events)):
            text = events[j].data # テキストを取り出す
            if j == 0:
                pre_text = "[PAD]"
            else:
                if type(events[j-1].data) is str:
                    pre_text = events[j-1].data # 一つ前の発話を取得
                else:
                    if type(events[j-2].data) is str:
                        pre_text = events[j-2].data
                    else:
                        pre_text = "[PAD]"

            # テキストのみにダイアログアクトをつける(accept, reject, offer, quitはすでにつけてある)
            if type(text) is str:
                inputs = tokenizer(pre_text, text, max_length=512, truncation=True, return_tensors="pt")
                inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
                outputs = model(**inputs)

                # 推論結果の取得
                logits = outputs.logits # ロジットの取得
                probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
                predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
                predicted_class = model.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換

                intent_list.append(predicted_class)
            else:
                intent_list.append(None) # テキスト以外のところはNoneを入れて数合わせ
        
        intent_dic[f"dialogue{i+1}"] = intent_list
        
    return intent_dic