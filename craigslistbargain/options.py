import cocoa.options
from datetime import datetime

# parser.add_argument_group('⚪︎⚪︎') → コマンドライン引数のグループ化

# =============== core ===============
def add_price_tracker_arguments(parser):
    parser.add_argument('--price-tracker-model', help='Path to price tracker model') # price_trackerモデルへのパス


# =============== data ===============
def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the encoder') # entityフォームをエンコーダに入力
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder') # entityフォームをデコーダに入力
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder') # entityフォームをデコーダに出力
    parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches') # 前処理されたバッチのキャッシュへのパス
    parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache') # 既存のキャッシュを無視するかどうか
    parser.add_argument('--mappings', help='Path to vocab mappings') # 語彙マッピングへのパス

def add_data_generator_arguments(parser):
    cocoa.options.add_scenario_arguments(parser)
    cocoa.options.add_dataset_arguments(parser)
    add_preprocess_arguments(parser)
    add_price_tracker_arguments(parser)


# =============== model ===============
def add_model_arguments(parser):
    from onmt.modules.SRU import CheckSRU
    group = parser.add_argument_group('Model')

    # srcとtgtのための単語埋め込みサイズ
    group.add_argument('--word-vec-size', type=int, default=300,
                       help='Word embedding size for source and target.')
    
    # 入力に共有重み行列を使用し, デコーダに単語埋め込みを出力するかどうか
    group.add_argument('--share-decoder-embeddings', action='store_true',
                       help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    
    # 使用するエンコーダ層のタイプ. オプションはrnn, brnn, transformer, cnn
    group.add_argument('--encoder-type', type=str, default='rnn',
                       choices=['rnn', 'brnn', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    
    # 前のターンにコンテキスト(文脈)を埋め込むために使用するエンコーダ
    group.add_argument('--context-embedder-type', type=str, default='mean',
                       choices=['rnn', 'mean', 'brnn'],
                       help="Encoder to use for embedding prev turns context")
    
    # 使用するデコーダ層のタイプ. オプションはrnn, transformer, cnn
    group.add_argument('--decoder-type', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are [rnn|transformer|cnn].""")
    
    # コピーアテンション層を訓練するかどうか
    group.add_argument('-copy_attn', action="store_true",
                       help='Train copy attention layer.')
    
    # エンコーダ/デコーダのレイヤ(層)数
    group.add_argument('--layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    
    # エンコーダ内のレイヤ(層)数
    group.add_argument('--enc-layers', type=int, default=1,
                       help='Number of layers in the encoder')
    
    # デコーダ内のレイヤ(層)数
    group.add_argument('--dec-layers', type=int, default=1,
                       help='Number of layers in the decoder')
    
    # RNNの隠れ状態のサイズ
    group.add_argument('--rnn-size', type=int, default=500,
                       help='Size of rnn hidden states')
    
    # RNNで使用するゲートのタイプ
    group.add_argument('--rnn-type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'SRU'], action=CheckSRU,
                       help="""The gate type to use in the RNNs""")
    
    # 各時間ステップでコンテキストベクトルを追加入力として(単語埋め込みとの連結を介して)デコーダにフィードするかどうか
    group.add_argument('--input-feed', action='store_true',
                       help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    
    # 使用するアテンションのタイプ: dotprod, general(Luong), MLP (Bahdanau), コンテキストを追加するためにmultibankを先頭に追加する
    group.add_argument('--global-attention', type=str, default='multibank_general',
                       choices=['dot', 'general', 'mlp',
                       'multibank_dot', 'multibank_general', 'multibank_mlp'],
                       help="""The attention type to use: dotprod or general (Luong)
                       or MLP (Bahdanau), prepend multibank to add context""")
    # モデルのタイプ(今回はseq2seqを選んでいる)
    group.add_argument('--model', type=str, default='seq2seq',
                       choices=['seq2seq', 'lf2lf'],
                       help='Model type')
    
    # (エンコーダ入力に加えて)対話コンテキストとして考慮する文の数
    group.add_argument('--num-context', type=int, default=2,
                       help='Number of sentences to consider as dialogue context (in addition to the encoder input)')
    
    # 対話エンコーディング/デコーディングの過程を通して隠れ状態を伝えるかどうか
    group.add_argument('--stateful', action='store_true',
                       help='Whether to pass on the hidden state throughout the dialogue encoding/decoding process')
    
    # ソースとターゲットの語彙の埋め込みを共有するかどうか
    group.add_argument('--share-embeddings', action='store_true',
                       help='Share source and target vocab embeddings')

def add_trainer_arguments(parser):
    cocoa.options.add_trainer_arguments(parser)

def add_rl_arguments(parser):
    cocoa.options.add_rl_arguments(parser)
    parser.add_argument('--reward', choices=['margin', 'length', 'fair'],
            help='Which reward function to use') # どの報酬関数を使用するか


# =============== systems ===============
def add_neural_system_arguments(parser):
    cocoa.options.add_generator_arguments(parser)

def add_system_arguments(parser):
    cocoa.options.add_rulebased_arguments(parser)
    add_price_tracker_arguments(parser)
    add_neural_system_arguments(parser)
    # NOTE: hybrid system の引数は neural system と rulebased system でカバーされる

def add_hybrid_system_arguments(parser):
    cocoa.options.add_rulebased_arguments(parser)
    add_neural_system_arguments(parser)


# =============== website ===============
def add_website_arguments(parser):

    # サーバを起動するポート番号
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    
    # アプリを実行するホストIPアドレス. デフォルトはローカルホスト
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    
    # Webサイトの構成を含むJSONファイルへのパス
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    
    # Webサイトの出力(デバッグとエラーのログ, チャット, データベース)を保存するディレクトリの名前
    # デフォルトはweb_output/current_dateで, current_dateの部分は%%Y-%%m-%%dの形で表示される
    # 指定されたディレクトリが既に存在する場合は, --reuseパラメータの指定がない限り全てのデータが上書きされる
    parser.add_argument('--output', type=str,
                        default="web_output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help="""Name of directory for storing website output (debug and error logs, chats, and database). 
                             Defaults to a web_output/current_date, with the current date formatted as %%Y-%%m-%%d.
                             If the provided directory exists, all data in it is overwritten unless the --reuse parameter is provided.""")
    
    # 出力ディレクトリ内の既存のデータベースファイルを使用するかどうか
    parser.add_argument('--reuse', action='store_true',
                        help='If provided, reuses the existing database file in the output directory.')
