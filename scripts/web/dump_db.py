import sqlite3
import json
import math
import os
from argparse import ArgumentParser

from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.util import read_json, write_json

import cocoa.options

# Task-specific modules
from web.main.db_reader import DatabaseReader
from core.scenario import Scenario

def read_results_csv(csv_file):
    '''
    mturk_code から worker_id に辞書を返す
    '''
    import csv
    reader = csv.reader(open(csv_file, 'r'))
    header = reader.next()
    worker_idx = header.index('WorkerId')
    code_idx = header.index('Answer.surveycode')
    d = {}
    for row in reader:
        workerid = row[worker_idx]
        code = row[code_idx]
        d[code] = workerid
    return d

def chat_to_worker_id(cursor, code_to_wid):
    '''
    chat_id: {'0': workder_id, '1': worker_id}
    workder_id が None の場合は botであることを意味する
    '''
    d = {}
    cursor.execute('SELECT chat_id, agent_ids FROM chat')
    for chat_id, agent_uids in cursor.fetchall():
        agent_wid = {}
        agent_uids = eval(agent_uids)
        for agent_id, agent_uid in agent_uids.iteritems():
            if not (isinstance(agent_uid, basestring)): #and agent_uid.startswith('U_')):
                agent_wid[agent_id] = None
            else:
                cursor.execute('''SELECT mturk_code FROM mturk_task WHERE name=?''', (agent_uid,))
                res = cursor.fetchall()
                if len(res) > 0:
                    mturk_code = res[0][0]
                    if mturk_code not in code_to_wid:
                        continue
                    else:
                        agent_wid[agent_id] = code_to_wid[mturk_code]
        d[chat_id] = agent_wid
    return d

def log_worker_id_to_json(db_path, batch_results):
    '''
    {chat_id: {'0': worker_id; '1': worker_id}}
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    code_to_wid = read_results_csv(batch_results)
    worker_ids = chat_to_worker_id(cursor, code_to_wid)
    output_dir = os.path.dirname(batch_results)
    write_json(worker_ids, output_dir + '/worker_ids.json')


if __name__ == "__main__":
    parser = ArgumentParser()
    cocoa.options.add_scenario_arguments(parser)
    parser.add_argument('--db', type=str, required=True, help='Path to database file containing logged events') # 記録されたイベントを含むDBファイルへのパス
    parser.add_argument('--output', type=str, required=True, help='File to write JSON examples to.') # 出力であるJSONファイルの書き出し先
    parser.add_argument('--uid', type=str, nargs='*', help='Only print chats from these uids') # 指定のuser IDによるチャットのみをプリントする
    parser.add_argument('--surveys', type=str, help='If provided, writes a file containing results from user surveys.') # ユーザーによる調査の結果をファイルに出力するかどうか
    parser.add_argument('--batch-results', type=str, help='If provided, write a mapping from chat_id to worker_id') # chat_idからworker_idへのマッピングを出力するかどうか
    args = parser.parse_args()
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, scenario_db, args.output, args.uid)
    if args.surveys:
        DatabaseReader.dump_surveys(cursor, args.surveys)
    # TODO: これを db_reader に移動する
    if args.batch_results:
        log_worker_id_to_json(args.db, args.batch_results)
