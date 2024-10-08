# Let's Negotiate! (Craigslistbargainデータセットにおける交渉対話実験)

このプロジェクトは商品の価格交渉に関するデータセットであるCRAIGSLISTBARGAINを使用した, 交渉対話システムとの対話実験を行うプロジェクトです。
先行研究である[Decoupling Strategy and Generation in Negotiation Dialogues (He, 2018)](https://aclanthology.org/D18-1256/)にて提案された, 
パーサー, マネージャー, ジェネレーターの3つのモジュールからなる交渉対話システムを基に, 新たに深層学習を使用したパーサーを提案し, 交渉対話システムを作成しました。

このプロジェクトで対話をすることができる交渉対話システムは以下の8種類です。
**先行研究のBot**:
- **SL-rule**: 強化学習なしのルールベースパーサー使用のBot
- **RL-rule-margin**: 強化学習の報酬に目的効用を使用した, ルールベースパーサー使用のBot
- **RL-rule-length**: 強化学習の報酬に対話の長さを使用した, ルールベースパーサー使用のBot
- **RL-rule-fair**: 強化学習の報酬に公平性を使用した, ルールベースパーサー使用のBot

**本研究の提案手法を用いたBot**:
- **SL-deep**: 強化学習なしのDLベースパーサー使用のBotです
- **RL-deep-margin**: 強化学習の報酬に目的効用を使用した, DLベースパーサー使用のBot
- **RL-deep-length**: 強化学習の報酬に対話の長さを使用した, DLベースパーサー使用のBot
- **RL-deep-fair**: 強化学習の報酬に公平性を使用した, DLベースパーサー使用のBot

掲載論文は以下の通りです.
- [交渉対話システムにおける深層学習に基づくパーサーによるダイアログアクトの推定](https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_1G3GS605/_article/-char/ja).
森本賢次郎, 藤田桂英.
人工知能学会全国大会(第38回), 2024.

----------
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
交渉の詳しい説明に関しては, 本プロジェクト内の「nogotiation_instruction.pdf」をご確認ください

----------
## DLベースパーサーの作成

今後追記します

----------
## チャットBotの作成
上記の「交渉対話システムとのチャット」で使用したチャットbotの作成方法について説明します。



### Systems and sessions
A dialogue **agent** is instantiated in a **session** which receives and sends messages. A **system** is used to create multiple sessions (that may run in parallel) of a specific agent type. For example, ```system = NeuralSystem(model)``` loads a trained model and ```system.new_session()``` is called to create a new session whenever a human user is available to chat.

### Events and controllers
A dialogue **controller** takes two sessions and have them send/receive **events** until the task is finished or terminated. The most common event is ```message```, which sends some text data. There are also task-related events, such as ```select``` in MutualFriends, which sends the selected item. 

### Examples and datasets
A dialogue is represented as an **example** which has a scenario, a series of events, and some metadata (e.g. example id). Examples can be read from / write to a JSON file in the following structure:
```
examples.json
|--[i]
|  |--"uuid": "<uuid>"
|  |--"scenario_uuid": "<uuid>"
|  |--"scenario": "{scenario dict}"
|  |--"agents": {0: "agent type", 1: "agent type"}
|  |--"outcome": {"reward": R}
|  |--"events"
|     |--[j]
|        |--"action": "action"
|        |--"data": "event data"
|        |--"agent": agent_id
|        |--"time": "event sent time"
```
A **dataset** reads in training and testing examples from JSON files.

## Code organization
CoCoA is designed to be modular so that one can add their own task/modules easily.
All tasks depend on the `cocoa` pacakge.
See documentation in the task folder for task-specific details. 

### Data collection
We provide basic infrastructure (see `cocoa.web`) to set up a website that pairs two users or a user and a bot to chat in a given scenario.

#### Generate scenarios
The first step is to create a ```.json``` schema file and then (randomly) generate a set of scenarios that the dialogue will be situated in.

#### <a name=web>Setup the web server</a>
The website pairs a user with another user or a bot (if available). A dialogue scenario is displayed and the two agents can chat with each other.
Users are then directed to a survey to rate their partners (optional).
All dialogue events are logged in a SQL database.

Our server is built by [Flask](http://flask.pocoo.org/).
The backend (```cocoa/web/main/backend.py```) contains code for pairing, logging, dialogue quality check.
The frontend code is in ```task/web/templates```.

To deploy the web server, run
```
cd <name-of-your-task>;
PYTHONPATH=. python web/chat_app.py --port <port> --config web/app_params.json --schema-path <path-to-schema> --scenarios-path <path-to-scenarios> --output <output-dir>
```
- Data and log will be saved in `<output-dir>`. **Important**: note that this will delete everything in `<output-dir>` if it's not empty.
- `--num-scenarios`: total number of scenarios to sample from. Each scenario will have `num_HITs / num_scenarios` chats.
You can also specify ratios of number of chats for each system in the config file.
Note that the final result will be an approximation of these numbers due to concurrent database calls.

To collect data from Amazon Mechanical Turk (AMT), workers should be directed to the link ```http://your-url:<port>/?mturk=1```.
`?mturk=1` makes sure that workers will receive a Mturk code at the end of the task to submit the HIT.

#### <a name=visualize>Dump the database</a>
Dump data from the SQL database to a JSON file (see [Examples and datasets](#examples-and-datasets) for the JSON structure).
```
cd <name-of-your-task>;
PYTHONPATH=. python ../scripts/web/dump_db.py --db <output-dir>/chat_state.db --output <output-dir>/transcripts/transcripts.json --surveys <output-dir>/transcripts/surveys.json --schema <path-to-schema> --scenarios-path <path-to-scenarios> 
```
Render JSON transcript to HTML:
```
PYTHONPATH=. python ../scripts/visualize_transcripts.py --dialogue-transcripts <path-to-json-transcript> --html-output <path-to-output-html-file> --css-file ../chat_viewer/css/my.css
```
Other options for HTML visualization:
- `--survey-transcripts`: path to `survey.json` if survey is enabled during data collection.
- `--survey-only`: only visualize dialgoues with submitted surveys.
- `--summary`: statistics of the dialogues.

### Dialogue agents
To add an agent for a task, you need to implement a system ```<name-of-your-task>/systems/<agent-name>_system.py```
and a session ```<name-of-your-task>/sessions/<agent-name>_session.py```.

### Model training and testing
See documentation in the under each task (e.g., `./craigslistbargain`).

### Evaluation
To deploy bots to the web interface, add the `"models"` field in the website config file,
e.g.
```
"models": {
    "rulebased": {
        "active": true,
        "type": "rulebased",
    }
}
```
See also [set up the web server](#web).
