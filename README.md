# Let's Negotiate! (Craigslistbargainデータセットにおける交渉対話実験)

このプロジェクトは商品の価格交渉に関するデータセットであるCRAIGSLISTBARGAINを使用した, 交渉対話システムとの対話実験を行うプロジェクトです.
先行研究である[Decoupling Strategy and Generation in Negotiation Dialogues (He, 2018)](https://aclanthology.org/D18-1256/)にて提案された, 
パーサー, マネージャー, ジェネレーターの3つのモジュールからなる交渉対話システムを基に, 新たに深層学習を使用したパーサーを提案し, 交渉対話システムを作成しました.

このプロジェクトで対話をすることができる交渉対話システムは以下の8種類です.
**先行研究のBot(Ruleと呼びます)**:
- **SL-rule**: 強化学習なしのルールベースパーサー使用のBot
- **RL-rule-margin**: 強化学習の報酬に目的効用を使用した, ルールベースパーサー使用のBot
- **RL-rule-length**: 強化学習の報酬に対話の長さを使用した, ルールベースパーサー使用のBot
- **RL-rule-fair**: 強化学習の報酬に公平性を使用した, ルールベースパーサー使用のBot

**本研究の提案手法を用いたBot(Deepと呼びます)**:
- **SL-deep**: 強化学習なしのDLベースパーサー使用のBotです
- **RL-deep-margin**: 強化学習の報酬に目的効用を使用した, DLベースパーサー使用のBot
- **RL-deep-length**: 強化学習の報酬に対話の長さを使用した, DLベースパーサー使用のBot
- **RL-deep-fair**: 強化学習の報酬に公平性を使用した, DLベースパーサー使用のBot

掲載論文は以下の通りです.
- [交渉対話システムにおける深層学習に基づくパーサーによるダイアログアクトの推定](https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_1G3GS605/_article/-char/ja).
森本賢次郎, 藤田桂英.
人工知能学会全国大会(第38回), 2024.

## 環境構築
1. まず, 作業ディレクトリ(ここでは例としてworkディレクトリとします)に本プロジェクトのディレクトリ及びファイルをインストールします.
2. 次にworkディレクトリに移動し, 以下の順序でdockerのコンテナを作成します.

**None**: ポート番号を指定する「-p 5000:5000」は使用可能なポート番号に適宜変更してください

```
docker build --shm-size=4gb --force-rm=true -t <任意のdockerイメージ名> .
docker run --gpus 1 -tid --ipc=host --name <任意のdockerコンテナ名> -p 5000:5000 -v <workディレクトリへの絶対パス>:/workspace <先ほど作成したdockerイメージ名>:latest
```

3. コンテナを作成したら, 以下の順序でコンテナを起動し, セットアップを行います.

```
docker exec -i -t 〈コンテナID〉 bash
python setup.py develop
python stopword_set.py
pip install gevent
```
----------
## 交渉対話システムとのチャット
このプロジェクトにはすでに上記のBot8種類が用意されています.
以下のコマンドを実行することで, Webアプリケーション上で交渉対話システムと商品の価格交渉を行うことができます.

**先行研究のBotと交渉を行う場合**:
```
PYTHONPATH=craigslistbargain python craigslistbargain/web/chat_app.py --port 5000 --config craigslistbargain/web/app_params_rulesys.json --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/dev-scenarios.json --price-tracker craigslistbargain/data/price_tracker.pkl --templates src/rule/train-templates.pkl --policy src/rule/train-model.pkl --output rulechat_output --sample
```

**本研究のBotと交渉を行う場合**:
```
PYTHONPATH=craigslistbargain python craigslistbargain/web/chat_app.py --port 5000 --config craigslistbargain/web/app_params_deepsys.json --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/dev-scenarios.json --price-tracker craigslistbargain/data/price_tracker.pkl --templates src/deep/train-templates.pkl --policy src/deep/train-model.pkl --parserpath transformers/model/roberta_fold_1/checkpoint-82304 --output deepchat_output --sample --neuralflag
```

上記のうち, どちらかのコマンドを実行すると, ターミナル上に **App setup complete** の文字が表示されたら準備完了です.
URL: 「http://<プログラムを実行しているPCまたはサーバーのIPアドレス>:5000」にアクセスすると交渉が開始されます.
交渉の詳しい説明に関しては, 本プロジェクト内の「nogotiation_instruction.pdf」をご確認ください.

なお, 何らかのエラーが原因でWebアプリケーション上でチャットができない場合は, 下記の**チャットBotの作成**における**5. コマンドラインインタフェース上でのBotとのチャット**に従ってコマンドを実行することにより, コマンドラインインタフェース上で簡易的に交渉対話システムとチャットを行うことができます.
お試しください.

**交渉結果の可視化**

Webアプリケーション上で行った交渉は結果が記録されています.
以下, 結果の確認方法について記述します.

まず, 結果が保存されたSQLデータベースから, JSONファイルに交渉結果情報を移動します.
```
(Rule)
PYTHONPATH=craigslistbargain python scripts/web/dump_db.py --db rulechat_output/chat_state.db --output rulechat_output/transcripts/transcripts.json --surveys rulechat_output/transcripts/surveys.json --schema craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/dev-scenarios.json
```
```
(Deep)
PYTHONPATH=craigslistbargain python scripts/web/dump_db.py --db deepchat_output/chat_state.db --output deepchat_output/transcripts/transcripts.json --surveys deepchat_output/transcripts/surveys.json --schema craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/dev-scenarios.json
```

次に作成したruleとdeepのjsonファイルを結合します.

```
python scripts/surveys_merge.py 
python scripts/transcripts_merge.py 
```

最後にJSONファイルに保存された情報をコマンドライン上に表示します.
```
PYTHONPATH=craigslistbargain python scripts/visualize_transcripts.py --dialogue-transcripts scripts/transcripts_merge.json --survey-transcripts scripts/surveys_merge.json --summary --survey-only
```


## DLベースパーサーの作成

今後追記します


## チャットBotの作成
上記の「交渉対話システムとのチャット」で使用したチャットbotの作成方法について説明します.

先行研究(Rule)のBotを作成するコマンドには(Rule), 本研究の提案手法である深層学習ベースのパーサーを使用したBotを作成するコマンドには(Deep)の印を記載しています.

**1. Price Trackerの作成**

発話から価格の言及が行われているか否かを判別する Price Trackerを作成します.

```
PYTHONPATH=craigslistbargain python craigslistbargain/core/price_tracker.py --train-examples-path craigslistbargain/data/train.json --output craigslistbargain/data/price_tracker.pkl
```

出力として**price_tracker.pkl**が得られます.

**2. パーサーによる学習・検証データの解析**

学習データ**train.json**と検証データ**dev.json**の両方をパーサーによって解析します.

```
(Rule)
PYTHONPATH=craigslistbargain python craigslistbargain/parse_dialogue.py --transcripts craigslistbargain/data/train.json --price-tracker craigslistbargain/data/price_tracker.pkl --max-examples -1 --templates-output src/rule/train-templates.pkl --model-output src/rule/train-model.pkl --transcripts-output src/rule/train-parsed.json

PYTHONPATH=craigslistbargain python craigslistbargain/parse_dialogue.py --transcripts craigslistbargain/data/dev.json --price-tracker craigslistbargain/data/price_tracker.pkl --max-examples -1 --templates-output src/rule/dev-templates.pkl --model-output src/rule/dev-model.pkl --transcripts-output src/rule/dev-parsed.json
```
```
(Deep)
PYTHONPATH=craigslistbargain python craigslistbargain/parse_dialogue.py --transcripts craigslistbargain/data/train.json --price-tracker craigslistbargain/data/price_tracker.pkl --max-examples -1 --templates-output src/deep/train-templates.pkl --model-output src/deep/train-model.pkl --transcripts-output src/deep/train-parsed.json --parserpath transformers/model/roberta_fold_1/checkpoint-82304 --neuralflag

PYTHONPATH=craigslistbargain python craigslistbargain/parse_dialogue.py --transcripts craigslistbargain/data/dev.json --price-tracker craigslistbargain/data/price_tracker.pkl --max-examples -1 --templates-output src/deep/dev-templates.pkl --model-output src/deep/dev-model.pkl --transcripts-output src/deep/dev-parsed.json --parserpath transformers/model/roberta_fold_1/checkpoint-82304 --neuralflag
```

出力として次の3つのファイルが得られます
- **train(dev)-parsed.json**: パーサーを用いて, データセット内の発話をダイアログアクト(発話の意図)に分類したデータ.
- **train(dev)-model.pkl**: Botのマネージャーで使用するダイアログアクトについてn-gramモデルを学習したもの.
- **train(dev)-templates.pkl**: 検索ベースのジェネレーターで使用する発話のテンプレート.

**3. マネージャーの学習**

パーサーによって解析したデータ**train(dev)-parsed.json**を使用し, ダイアログアクトを用いてseq2seqモデルをトレーニングします.

```
(Rule)
PYTHONPATH=craigslistbargain python craigslistbargain/main.py --schema-path craigslistbargain/data/craigslist-schema.json --train-examples-paths src/rule/train-parsed.json --test-examples-paths src/rule/dev-parsed.json --price-tracker craigslistbargain/data/price_tracker.pkl --model lf2lf --model-path src/rule/checkpoint/lf2lf --mappings src/rule/mappings/lf2lf --word-vec-size 300 --pretrained-wordvec '' '' --rnn-size 300 --rnn-type LSTM --global-attention multibank_general --num-context 2 --stateful --batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 --epochs 20 --report-every 500 --cache src/cache/lf2lf --ignore-cache --verbose --best-only
```
```
(Deep)
PYTHONPATH=craigslistbargain python craigslistbargain/main.py --schema-path craigslistbargain/data/craigslist-schema.json --train-examples-paths src/deep/train-parsed.json --test-examples-paths src/deep/dev-parsed.json --price-tracker craigslistbargain/data/price_tracker.pkl --model lf2lf --model-path src/deep/checkpoint/lf2lf --mappings src/deep/mappings/lf2lf --word-vec-size 300 --pretrained-wordvec '' '' --rnn-size 300 --rnn-type LSTM --global-attention multibank_general --num-context 2 --stateful --batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01 --epochs 20 --report-every 500 --cache src/deep/cache/lf2lf --ignore-cache --verbose --best-only
```

出力として得られるcheckpoint/lf2lf/model_best.ptが強化学習なしのBot(SL-rule または SL-deep)となります.

**4. 強化学習によるマネージャーのファインチューニング**

3の「マネージャーの学習」で取得したmodel_best.ptを特定の報酬関数を用いたREINFORCEで強化学習します.
まず, 学習・検証用のシナリオを生成します

```
PYTHONPATH=craigslistbargain python scripts/chat_to_scenarios.py --chats craigslistbargain/data/train.json --scenarios craigslistbargain/data/train-scenarios.json
PYTHONPATH=craigslistbargain python scripts/chat_to_scenarios.py --chats craigslistbargain/data/dev.json --scenarios craigslistbargain/data/dev-scenarios.json
```
出力としてシナリオtrain-scenarios.jsonとdev-scenarios.jsonが得られます.

その後, 3種類の報酬関数のもと, REINFORCEを実行します.
使用する報酬関数は次の通りです.

1. margin : **目的効用値**. 自分の目標価格を1, 相手の目標価格を-1, それらの中間価格を0とした線形関数で表現される.
2. length : **対話の長さ**. 一つの対話中に何回発話がなされたかで表現される.
3. fair : **取引の公平性**. 自分と相手の目的効用値の差で表現される.

```
PYTHONPATH=craigslistbargain python craigslistbargain/reinforce.py --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/train-scenarios.json --valid-scenarios-path craigslistbargain/data/dev-scenarios.json --price-tracker craigslistbargain/data/price_tracker.pkl --agent-checkpoints src/rule/checkpoint/lf2lf/model_best.pt src/rule/checkpoint/lf2lf/model_best.pt --model-path src/rule/checkpoint/lf2lf-margin --optim adagrad --learning-rate 0.001 --agents pt-neural pt-neural --report-every 500 --max-turns 20 --num-dialogues 5000 --sample --temperature 0.5 --max-length 20 --reward margin --best-only
```

```
PYTHONPATH=craigslistbargain python craigslistbargain/reinforce.py --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/train-scenarios.json --valid-scenarios-path craigslistbargain/data/dev-scenarios.json --price-tracker craigslistbargain/data/price_tracker.pkl --agent-checkpoints src/deep/checkpoint/lf2lf/model_best.pt src/deep/checkpoint/lf2lf/model_best.pt --model-path src/deep/checkpoint/lf2lf-margin --optim adagrad --learning-rate 0.001 --agents pt-neural pt-neural --report-every 500 --max-turns 20 --num-dialogues 5000 --sample --temperature 0.5 --max-length 20 --reward margin --best-only
```

実行する際には, 上記のコマンドの「--model-path src/rule/checkpoint/lf2lf-margin」と「--reward margin」の**margin**の部分を任意の報酬関数名(margin, length, fair)に変更してください.

**5. コマンドラインインタフェース上でのBotとのチャット**

次のコマンドを実行することで, コマンドラインインタフェース上でBotとチャットを行うことができます.

```
(Rule)
PYTHONPATH=craigslistbargain python scripts/generate_dataset.py --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/train-scenarios.json --results-path src/rule/checkpoint/bot-chat-transcripts-r.json --max-examples 100 --agents hybrid cmd --price-tracker craigslistbargain/data/price_tracker.pkl --agent-checkpoints src/rule/checkpoint/lf2lf-margin/model_best.pt "" --templates src/rule/train-templates.pkl --policy src/rule/train-model.pkl --max-turns 20 --random-seed 11 --sample --temperature 0.2
```

```
(deep)
PYTHONPATH=craigslistbargain python scripts/generate_dataset.py --schema-path craigslistbargain/data/craigslist-schema.json --scenarios-path craigslistbargain/data/train-scenarios.json --results-path src/deep/checkpoint/bot-chat-transcripts-d.json --max-examples 100 --agents hybrid cmd --price-tracker craigslistbargain/data/price_tracker.pkl --agent-checkpoints src/deep/checkpoint/lf2lf-margin/model_best.pt "" --templates src/deep/train-templates.pkl --policy src/deep/train-model.pkl --max-turns 20 --random-seed 11 --sample --temperature 0.2 --parserpath transformers/model/roberta_fold_1/checkpoint-82304 --neuralflag
```
実行する際には, 上記のコマンドの「--agent-checkpoints src/deep/checkpoint/lf2lf-margin/model_best.pt ""」の**lf2lf-margin**のところを任意のモデル(lf2lf, lf2lf-margin, lf2lf-length, lf2lf-fair)に変更してください.

コマンドライン上で交渉を行う場合には, 以下の方法でチャットを行ってください.

- **交渉メッセージを送る** : メッセージを英語で入力してEnterキーを押す.
- **交渉に合意する** : 「&lt;accept&gt;」と入力してEnterキーを押す.
- **交渉を断る** : 「&lt;reject&gt;」と入力してEnterキーを押す.
- **交渉を止める** : 「&lt;quit&gt;」と入力してEnterキーを押す.
- **価格オファーをする** : 「&lt;offer&gt; オファー価格」と入力してEnterキーを押す(例: &lt;offer&gt; 120).