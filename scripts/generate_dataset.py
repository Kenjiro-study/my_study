"""
二つの実装されたエージェントを取得し, ダイアログを生成する
"""

import argparse
import random
import json
import numpy as np

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
import cocoa.options

from core.scenario import Scenario
from core.controller import Controller
from systems import get_system
import options

def generate_examples(num_examples, scenario_db, examples_path, max_examples, remove_fail, max_turns):
    examples = []
    num_failed = 0
    scenarios = scenario_db.scenarios_list
    #scenarios = [scenario_db.scenarios_map['S_8COuPdjZZkYgrzhb']]
    #random.shuffle(scenarios)

    for i in range(max_examples): # max_exampleはデフォルトで20 この数だけ対話をシミュレート
        scenario = scenarios[num_examples % len(scenarios)]
        sessions = [agents[0].new_session(0, scenario.kbs[0]), agents[1].new_session(1, scenario.kbs[1])]       
        controller = Controller(scenario, sessions)
        ex = controller.simulate(max_turns, verbose=args.verbose)
        if not controller.complete():
            num_failed += 1
            if remove_fail:
                continue
        examples.append(ex)
        num_examples += 1 # num_examplesをインクリメント

    with open(examples_path, 'w') as out:
        print(json.dumps([e.to_dict() for e in examples]), file=out)
    if num_failed == 0:
        print('All {} dialogues succeeded!'.format(num_examples))
    else:
        print('Number of failed dialogues:', num_failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1) # 乱数のシード
    parser.add_argument('--agents', default=['rulebased', 'rulebased'], help='What kind of agent to use', nargs='*') # どのエージェントを使うか
    parser.add_argument('--agent-checkpoints', nargs='+', default=['', ''], help='Directory to learned models') # 学習済みモデルのディレクトリ
    parser.add_argument('--scenario-offset', default=0, type=int, help='Number of scenarios to skip at the beginning') # 開始時にスキップするシナリオ数
    parser.add_argument('--remove-fail', default=False, action='store_true', help='Remove failed dialogues') # 失敗した対話を削除するかどうか
    parser.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns') # 最大ターン数
    parser.add_argument('--results-path', default=None, help='json path to store the results of the chat examples') # チャットの例の結果を保存するためのjsonパス
    parser.add_argument('--max-examples', default=20, type=int, help='Number of test examples to predict') # 予測するテストexamplesの数
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='whether or not to have verbose prints') # 詳細を出力するかどうか
    cocoa.options.add_scenario_arguments(parser) # --schema-path, --scenarios-path
    cocoa.options.add_dataset_arguments(parser) # --train-examples-paths, --test-examples-paths, --train-max-examples, --test-max-examples, --eval-examples-paths
    options.add_system_arguments(parser) # --templates, --policy, --price-tracker-model, --checkpoint, --beam-size, --min-length, --max-length, --n-best, --alpha, --sample, --temperature, --batch-size, --gpuid, --verbose
    args = parser.parse_args()

    # 再現性のため乱数のシード固定
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path) # craigslistbargain/data/craigslist-schema.json
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario) # craigslistbargain/data/train-scenarios.json

    assert len(args.agent_checkpoints) == len(args.agents) # 指定したエージェント名の個数と, チェックポイントの個数が一致しているかの確認

    # craigslistbargain/systems/__init__.pyのget_system関数によりエージェントを定義する. 使えるエージェントはrulebased, hybrid, cmd, pt-neuralの4種(それ以外はエラー)
    agents = [get_system(name, args, schema, model_path=model_path) for name, model_path in zip(args.agents, args.agent_checkpoints)]
    
    num_examples = args.scenario_offset # 開始時にスキップするシナリオ数でデフォルトの0を使っている

    generate_examples(num_examples, scenario_db, args.results_path, args.max_examples, args.remove_fail, args.max_turns)
    # 引数の詳細
    # 1. num_example → 開始時にスキップするシナリオ数=0
    # 2. scenario_db → craigslist-schema.jsonとtrain-scenarios.jsonの内容を読み込んだシナリオのデータベース
    # 3. args.results_path → 結果を出力するパス=bot-chat-transcripts.json
    # 4. args.max_examples → 予測するテストexamplesの数=20
    # 5. args.remove_fail → 失敗した対話を削除するかどうか=False
    # 6. args.max_turns → 最大ターン数=20