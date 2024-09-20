"""
実験で得たsurveys.jsonファイルを結合するプログラム
"""
import json

# ファイルを読み込む
#with open('rulechat_output/transcripts/surveys.json', 'r') as f1, open('deepchat_output/transcripts/surveys2.json', 'r') as f2: # 実験用
with open('rulechat_output/transcripts/surveys.json', 'r') as f1, open('deepchat_output/transcripts/surveys2.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# 結合の処理
merged_data = []

# マージして新しいリストに追加
for dict1, dict2 in zip(data1, data2):
    merged_entry = {}
    
    # 両方の辞書に対してキーをループしてマージ
    for key in dict1.keys() | dict2.keys():
        if key in dict1 and key in dict2:
            # サブ辞書もマージする
            merged_entry[key] = {**dict1[key], **dict2[key]}
        elif key in dict1:
            merged_entry[key] = dict1[key]
        elif key in dict2:
            merged_entry[key] = dict2[key]
    
    merged_data.append(merged_entry)

# マージした結果を新しいファイルに保存する
with open('scripts/surveys_merge.json', 'w') as f_out:
    json.dump(merged_data, f_out, indent=4, ensure_ascii=False)

print("ファイルの結合が完了しました。")