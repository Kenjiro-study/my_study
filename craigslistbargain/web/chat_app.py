import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit
from gevent.pywsgi import WSGIServer

from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import read_json
from cocoa.systems.human_system import HumanSystem
from cocoa.web.main.logger import WebLogger
import cocoa.options

from core.scenario import Scenario
from systems import get_system
from main.db_reader import DatabaseReader
from main.backend import DatabaseManager
import options

__author__ = 'anushabala'

DB_FILE_NAME = 'chat_state.db'
LOG_FILE_NAME = 'log.out'
ERROR_LOG_FILE_NAME = 'error_log.out'
TRANSCRIPTS_DIR = 'transcripts'

from flask import g
from web.main.backend import Backend

###############
from flask import Flask, current_app

from flask_socketio import SocketIO
socketio = SocketIO()


def close_connection(exception):
    backend = getattr(g, '_backend', None)
    if backend is not None:
        backend.close()


def create_app(debug=False, templates_dir='templates'):
    """アプリケーションの作成"""

    app = Flask(__name__, template_folder=os.path.abspath(templates_dir))
    app.debug = debug
    app.config['SECRET_KEY'] = 'gjr39dkjn344_!67#'
    app.config['PROPAGATE_EXCEPTIONS'] = True

    from web.views.action import action
    from cocoa.web.views.chat import chat
    app.register_blueprint(chat) # cocoa.web.views.chatのchat変数に登録された複数のページ(Blueprint)を登録
    app.register_blueprint(action) # web.views.actionのaction変数に登録された複数のページ(Blueprint)を登録

    app.teardown_appcontext_funcs = [close_connection] # データベースやファイルの解放, ログ記録などを行う設定のリスト

    socketio.init_app(app) # FlaskアプリにSocket.IOを統合し, リアルタイムチャットを可能にする
    return app


def add_systems(args, config_dict, schema, debug=False):
    """
    Params:
    config_dict: bot名をbotの設定(config)を含む辞書にマッピングする辞書
        辞書にはbotのタイプ(key 'type')が含まれている必要があり、生成に基礎モデルを使用するbotの場合は, 
        モデルのパラメータ、語彙などを含むディレクトリへのパスが含まれる
    Returns:
    agents: bot名からそのbotのシステムオブジェクトへのマッピングを行う辞書
    pairing_probabilities: bot名からユーザーがそのbotとペアになる確率にマッピングする辞書
        そこには人間とペアになる確率文字含まれる (backend.Partner.Human)
    """

    total_probs = 0.0
    systems = {HumanSystem.name(): HumanSystem()}
    pairing_probabilities = {}
    timed = False if debug else True
    for (sys_name, info) in config_dict.items():
        if "active" not in info.keys():
            warnings.warn("active status not specified for bot %s - assuming that bot is inactive." % sys_name)
        if info["active"]:
            name = info["type"]
            try:
                model = get_system(name, args, schema, timed, info.get('checkpoint')) # システムの取得
            except ValueError:
                warnings.warn(
                    'Unrecognized model type in {} for configuration '
                    '{}. Ignoring configuration.'.format(info, sys_name))
                continue
            systems[sys_name] = model
            if 'prob' in info.keys():
                # jsonファイルにあらかじめ各システムが選ばれる確率を定義しておけるみたい(今は定義していないのでここはするスルー)
                prob = float(info['prob'])
                pairing_probabilities[sys_name] = prob
                total_probs += prob
    print('{} systems loaded'.format(len(systems)))
    for name in systems:
        print(name)

    # ペアリング確率の設定
    if total_probs > 1.0:
        raise ValueError("Probabilities for active bots can't exceed 1.0.")
    if len(pairing_probabilities.keys()) != 0 and len(pairing_probabilities.keys()) != len(systems.keys()):
        remaining_prob = (1.0-total_probs)/(len(systems.keys()) - len(pairing_probabilities.keys()))
    else:
        remaining_prob = 1.0 / len(systems.keys()) # 全てのシステムを等分の確率で選ばれるようにする
    inactive_bots = set()
    for system_name in systems.keys():
        if system_name not in pairing_probabilities.keys():
            if remaining_prob == 0.0:
                inactive_bots.add(system_name)
            else:
                pairing_probabilities[system_name] = remaining_prob

    for sys_name in inactive_bots:
        systems.pop(sys_name, None)

    return systems, pairing_probabilities


def cleanup(flask_app):
    db_path = flask_app.config['user_params']['db']['location']
    transcript_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'transcripts.json')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], transcript_path)
    if flask_app.config['user_params']['end_survey'] == 1:
        surveys_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'surveys.json')
        DatabaseReader.dump_surveys(cursor, surveys_path)
    conn.close()


def init(output_dir, reuse=False):
    db_file = os.path.join(output_dir, DB_FILE_NAME)
    log_file = os.path.join(output_dir, LOG_FILE_NAME)
    error_log_file = os.path.join(output_dir, ERROR_LOG_FILE_NAME)
    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_DIR)
    # TODO: 全て削除するな
    if not reuse:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        db = DatabaseManager.init_database(db_file)

        if os.path.exists(transcripts_dir):
            shutil.rmtree(transcripts_dir)
        os.makedirs(transcripts_dir)
    else:
        db = DatabaseManager(db_file)

    return db, log_file, error_log_file, transcripts_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scenarios', type=int)
    parser.add_argument('--parserpath', default=None, help='Path of the deep learning-based parser you created') ##### new!! ディープラーニングベースパーサーのパス #####
    parser.add_argument('--neuralflag', default=False, action='store_true', help='which parser do you use, neural-base or rule-base') ###### new!! ルールベース, ニューラルベースどちらのパーサーを使用するか #####
    options.add_website_arguments(parser)
    cocoa.options.add_scenario_arguments(parser)
    options.add_system_arguments(parser)
    args = parser.parse_args()

    params_file = args.config
    with open(params_file) as fin:
        params = json.load(fin) # app_param.jsonの内容をparamsに格納

    db, log_file, error_log_file, transcripts_dir = init(args.output, args.reuse)
    error_log_file = open(error_log_file, 'w')

    WebLogger.initialize(log_file)
    params['db'] = {}
    params['db']['location'] = db.db_file
    params['logging'] = {}
    params['logging']['app_log'] = log_file
    params['logging']['chat_dir'] = transcripts_dir

    # "task_title" は "Let's Negotiate!" で設定されている
    if 'task_title' not in params.keys():
        raise ValueError("Title of task should be specified in config file with the key 'task_title'")

    instructions = None
    # "instructions" は "craigslistbargain/web/templates/craigslist-instructions.html" で設定されている
    if 'instructions' in params.keys():
        instructions_file = open(params['instructions'], 'r')
        instructions = "".join(instructions_file.readlines())
        instructions_file.close()
    else:
        raise ValueError("Location of file containing instructions for task should be specified in config with the key "
                         "'instructions")

    templates_dir = None
    # "templates_dir" は "craigslistbargain/web/templates" で設定されている
    if 'templates_dir' in params.keys():
        templates_dir = params['templates_dir']
    else:
        raise ValueError("Location of HTML templates should be specified in config with the key templates_dir")
    if not os.path.exists(templates_dir):
            # "templates_dir" で指定したパスが存在するかどうかの確認
            raise ValueError("Specified HTML template location doesn't exist: %s" % templates_dir)

    app = create_app(debug=False, templates_dir=templates_dir) # FlaskにSocket.IOを統合したリアルタイムチャットアプリを作成

    schema_path = args.schema_path # スキーマはcraigslistbargain/data/craigslist-schema.jsonで設定
    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)
    schema = Schema(schema_path)

    scenarios = read_json(args.scenarios_path) # シナリオはcraigslistbargain/data/dev-scenarios.jsonで設定
    if args.num_scenarios is not None:
        scenarios = scenarios[:args.num_scenarios]
    scenario_db = ScenarioDB.from_dict(schema, scenarios, Scenario)
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    if 'quit_after' not in params.keys():
        params['quit_after'] = params['status_params']['chat']['num_seconds'] + 1

    if 'skip_chat_enabled' not in params.keys():
        params['skip_chat_enabled'] = False

    if 'end_survey' not in params.keys() :
        params['end_survey'] = 0

    if 'debug' not in params:
        params['debug'] = False

    # systems → システム名とシステムそのものが辞書型で入っている (例: {'human': <cocoa.systems.human_system.HumanSystem object at 0x7f196d8ec5b0>})
    # pairing_probabilities → システム名とそのシステムが選ばれる確率が辞書型で入っている (例: {'human': 0.16666666666666666})
    systems, pairing_probabilities = add_systems(args, params['models'], schema, debug=params['debug'])

    db.add_scenarios(scenario_db, systems, update=args.reuse) # cocoa/main/web/backend.pyのDatabaseManagerのadd_scenario

    app.config['systems'] = systems
    app.config['sessions'] = defaultdict(None)
    app.config['pairing_probabilities'] = pairing_probabilities
    app.config['num_chats_per_scenario'] = params.get('num_chats_per_scenario', {k: 1 for k in systems})
    for k in systems:
        assert k in app.config['num_chats_per_scenario']
    app.config['schema'] = schema
    app.config['user_params'] = params
    app.config['controller_map'] = defaultdict(None)
    app.config['instructions'] = instructions
    app.config['task_title'] = params['task_title']

    ##### new! DLベースパーサーを使うための処理
    if (args.parserpath is not None) and args.neuralflag:
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = args.parserpath 
        app.config["tokenizer"] = AutoTokenizer.from_pretrained(checkpoint) # configにトークナイザーを追加
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
        app.config['dlparser'] = model.to(device) # configにDLベースパーサーを追加
        app.config['flag'] = args.neuralflag # configにflagを追加
    else:
        app.config["tokenizer"] = None
        app.config["dlparser"] = None
        app.config["flag"] = False

    
    if 'icon' not in params.keys():
        app.config['task_icon'] = 'handshake.jpg'
    else:
        app.config['task_icon'] = params['icon']

    print("App setup complete")

    server = WSGIServer(('0.0.0.0', args.port), app, log=WebLogger.get_logger(), error_log=error_log_file)
    atexit.register(cleanup, flask_app=app) # 終了時のクリーンアップ
    server.serve_forever() # サーバーを無限ループで実行!
