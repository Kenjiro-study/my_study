# =============== core ===============
def add_dataset_arguments(parser):
    
    # 入力する訓練examples
    parser.add_argument('--train-examples-paths', nargs='*', default=[],
        help='Input training examples')
    
    # 入力するテストexamples
    parser.add_argument('--test-examples-paths', nargs='*', default=[],
        help='Input test examples')
    
    # 訓練examplesの最大数
    parser.add_argument('--train-max-examples', type=int,
        help='Maximum number of training examples')
    
    # テストexamplesの最大数
    parser.add_argument('--test-max-examples', type=int,
        help='Maximum number of test examples')
    
    # 複数応答の評価ファイルへのパス
    parser.add_argument('--eval-examples-paths', nargs='*', default=[],
        help='Path to multi-response evaluation files')

def add_scenario_arguments(parser):
    #  ドメインのスキーマを説明する入力パス
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain')

    # 生成されたシナリオのための出力パス
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated')


# =============== model ===============
def add_logging_arguments(parser):
    group = parser.add_argument_group('Logging')
    # ここで指定したバッチ感覚で統計情報を出力する
    group.add_argument('--report-every', type=int, default=5,
                       help="Print stats at this many batch intervals")
    
    # モデルのファイル名
    # モデルは<filname>_acc_ppl_e.ptとして保存される, ACCはaccuracy(精度), PPLはperplexity, Eはepoch
    group.add_argument('--model-filename', default='model',
                       help="""Model filename (the model will be saved as
                       <filename>_acc_ppl_e.pt where ACC is accuracy, PPL is
                       the perplexity and E is the epoch""")
    
    # モデルのチェックポイントが保存されるファイル
    group.add_argument('--model-path', default='data/checkpoints',
                       help='Which file the model checkpoints will be saved')
    
    # ここで指定したエポックを含むそれ以降の全てのエポックでチェックポイントを作成する
    group.add_argument('--start-checkpoint-at', type=int, default=0,
                       help='Start checkpointing every epoch after and including this epoch')
    
    # 最適なチェックポイントのみを保存するかどうか
    group.add_argument('--best-only', action='store_true',
                       help="Only store the best checkpoint")

def add_trainer_arguments(parser):
    group = parser.add_argument_group('Training')
    """Initialization"""
    # 有効なパスが指定されている場合, 事前学習済み単語埋め込みを読み込む
    # リストに2つ埋め込みが含まれている場合は, 2番目のものはitemのタイトルと説明用
    group.add_argument('--pretrained-wordvec', nargs='+', default=['', ''],
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings, if list contains two embeddings,
                       then the second one is for item title and description""")
    
    # パラメータはサポート(-param_init, param_init)を使用して一様分布で初期化される
    # 初期化をしない場合は0を使用する
    group.add_argument('--param-init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    
    # 事前学習された単語埋め込みを修正する
    group.add_argument('--fix-pretrained-wordvec',
                       action='store_true',
                       help="Fix pretrained word embeddings.")

    """Optimization"""
    # トレーニングの最大バッチサイズ
    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    
    # データはジェネレータから取得され, 無制限であるため, 人為的な制限を設定する
    #group.add_argument('--batches_per_epoch', type=int, default=10,
                        #help='Data comes from a generator, which is unlimited, so we need to set some artificial limit.')
    
    # トレーニングエポック数
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')
    
    # 最適化手法(確率的勾配降下法(SGD), AgaGrad, AdaDelta, ADAMが使える)
    group.add_argument('--optim', default='sgd', help='Optimization method.',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    
    # 勾配ベクトルのノルムの最大値
    # これを超える場合は, ノルムがこの値に等しくなるよう再正規化する
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    
    # ドロップアウトの確率. LSTMスタックに適用される
    group.add_argument('--dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    
    # 学習率の初期値
    # 各最適化手法の推奨設定は次の通り → SGD=1, AdaGrad=0.1, AdaDelta=1, ADAM=0.001
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd=1, adagrad=0.1, adadelta=1, adam=0.001""")
    
    # ここで指定した番号のGPUをCUDAで使用する
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")
    
    # 実験の再現性を確保するために乱数のシードを設定する
    group.add_argument('-seed', type=int, default=-1,
                       help='Random seed used for the experiments reproducibility.')
    
    # ラベルのスムージング値εの値
    # 全てのnon-Trueのラベルの確率はε/(vocab_size - 1)で平滑化される
    # ラベルのスムージングをオフにするにはゼロを設定する
    # 詳細はhttps://arxiv.org/abs/1512.00567参照
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels will be smoothed
                       by epsilon / (vocab_size - 1). Set to zero to turn off
                       label smoothing. For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")

    """Logging"""
    add_logging_arguments(parser)

def add_rl_arguments(parser):
    group = parser.add_argument_group('Reinforce')
    # 最大ターン数
    group.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')

    # 生成/訓練する対話の数
    group.add_argument('--num-dialogues', default=10000, type=int,
            help='Number of dialogues to generate/train')
    
    # 値計算の際に各タイムステップの報酬を割り引く量(通常γと記載される)
    group.add_argument('--discount-factor', default=1.0, type=float,
            help="""Amount to discount the reward for each timestep when
            calculating the value, usually written as gamma""")
    
    # 詳細な出力を行うかどうか
    group.add_argument('--verbose', default=False, action='store_true',
            help='Whether or not to have verbose prints')

    group = parser.add_argument_group('Training')
    group.add_argument('--optim', default='sgd', help="""Optimization method.""",
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')
    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")

    """Logging"""
    add_logging_arguments(parser)


def add_generator_arguments(parser):
    """事前学習されたニューラルモデルからテキストを生成するための引数
    """
    # チェックポイントへのパス
    parser.add_argument('--checkpoint', help='Path to checkpoint')
    

    group = parser.add_argument_group('Beam')
    # Beamのサイズ
    group.add_argument('--beam-size',  type=int, default=5, help='Beam size')

    # 最小の予測長
    group.add_argument('--min-length', type=int, default=1, help='Minimum prediction length')

    # 最大の予測長
    group.add_argument('--max-length', type=int, default=50, help='Maximum prediction length.')
    
    # verboseがセットされている場合, n-bestにデコードされた文を出力する
    group.add_argument('--n-best', type=int, default=1, help='If verbose is set, will output the n_best decoded sentences')

    # 長さのペナルティパラメータ(higher = longer generation)
    group.add_argument('--alpha', type=float, default=0.5, help='length penalty parameter (higher = longer generation)')


    group = parser.add_argument_group('Sample')
    # beamサーチの代わりにサンプルを採取するかどうか
    group.add_argument('--sample', action="store_true", help='Sample instead of beam search')
    
    # サンプル温度
    group.add_argument('--temperature', type=float, default=1, help='Sample temperature')


    group = parser.add_argument_group('Efficiency')
    # バッチサイズ
    group.add_argument('--batch-size', type=int, default=30, help='Batch size')
    
    # ここで指定した番号のGPUをCUDAで使用する
    group.add_argument('--gpuid', default=[], nargs='+', type=int, help="Use CUDA on the listed devices.")

    
    group = parser.add_argument_group('Logging')
    # 各文のスコアと予測を出力するかどうか
    group.add_argument('--verbose', action="store_true", help='Print scores and predictions for each sentence')


# =============== system ===============
def add_rulebased_arguments(parser):
    # pklファイルのtemplatesへのパス
    parser.add_argument('--templates', help='Path to templates (.pkl)')
    
    # pklファイルのマネージャーmodelへのパス
    parser.add_argument('--policy', help='Path to manager model (.pkl)')

