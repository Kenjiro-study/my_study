"""
実験で得たtranscripts.jsonファイルを結合するプログラム
"""
import json

# ファイルを読み込む
#with open('transcripts1.json', 'r') as f1, open('transcripts2.json', 'r') as f2: # 実験用
with open('rulechat_output/transcripts/transcripts.json', 'r') as f1, open('deepchat_output/transcripts/transcripts.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# マージ処理
merged_data = data1 + data2  # リスト同士を結合する

# 同じscenario_uuidがあった場合、イベントを結合する
uuid_to_entry = {}
for entry in merged_data:
    uuid = entry["scenario_uuid"]
    
    if uuid not in uuid_to_entry:
        uuid_to_entry[uuid] = entry
    else:
        # 既に存在する場合、イベントをマージ
        existing_entry = uuid_to_entry[uuid]
        existing_entry["events"].extend(entry["events"])
        
        # もし追加したい他の要素があれば、ここで処理を追加
        # 例: outcome, scenario など
        if "outcome" in entry and "outcome" not in existing_entry:
            existing_entry["outcome"] = entry["outcome"]
        if "scenario" in entry and "scenario" not in existing_entry:
            existing_entry["scenario"] = entry["scenario"]

# 辞書からリストに戻す
final_merged_data = list(uuid_to_entry.values())

# 結果を新しいファイルに保存
with open('scripts/transcripts_merge.json', 'w') as f_out:
    json.dump(final_merged_data, f_out, indent=4, ensure_ascii=False)

print("ファイルの結合が完了しました。")