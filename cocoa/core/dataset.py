"""
events, examples, datasetsのためのデータ構造
"""

from cocoa.core.util import read_json
from cocoa.core.event import Event
from cocoa.core.kb import KB

class Example(object):
    """
    一例として, シナリオに基づいた対話に一連のイベントがあり, 最後に何らかの報酬が得られる
    ライブ会話を通して作成され, シリアル化し, その後トレーニングのために読み込まれる
    """
    def __init__(self, scenario, uuid, events, outcome, ex_id, agents, agents_info=None):
        self.scenario = scenario
        self.uuid = uuid
        self.events = events
        self.outcome = outcome
        self.ex_id = ex_id
        self.agents = agents
        self.agents_info = agents_info

    def add_event(self, event):
        self.events.append(event)

    @classmethod
    def from_dict(cls, raw, Scenario, scenario_db=None):
        if 'scenario' in raw:
            scenario = Scenario.from_dict(None, raw['scenario']) # シナリオの取得
        # Compatible with old data formats (to be removed)
        elif scenario_db:
            print('WARNING: scenario should be provided in the example')
            scenario = scenario_db.get(raw['scenario_uuid'])
        else:
            raise ValueError('No scenario')
        uuid = raw['scenario_uuid'] # シナリオuuidの設定
        events = [Event.from_dict(e) for e in raw['events']] # eventsの設定
        outcome = raw['outcome'] # outcomeの設定
        ex_id = raw['uuid'] # exampleのidとしてuuidを設定
        if 'agents' in raw:
            agents = {int(k): v for k, v in raw['agents'].items()} # agentsの設定
        else:
            agents = None
        agents_info = raw.get('agents_info', None) # agents_infoの取得
        return Example(scenario, uuid, events, outcome, ex_id, agents, agents_info=agents_info)

    @classmethod
    def test_dict(cls, raw):
        uuid = raw['scenario_uuid']
        events = [Event.from_dict(e) for e in raw['events']]
        outcome = raw['outcome']
        ex_id = raw['uuid']
        if 'agents' in raw:
            agents = {int(k): v for k, v in raw['agents'].items()}
        else:
            agents = None
        agents_info = raw.get('agents_info', None)
        return Example(None, uuid, events, outcome, ex_id, agents, agents_info=agents_info)


    def to_dict(self):
        return {
            'scenario_uuid': self.scenario.uuid,
            'events': [e.to_dict() for e in self.events],
            'outcome': self.outcome,
            'scenario': self.scenario.to_dict(),
            'uuid': self.ex_id,
            'agents': self.agents,
            'agents_info': self.agents_info,
        }

class Dataset(object):
    """
    データセットは, トレーニング例とテスト例のリストで構成される
    """
    def __init__(self, train_examples, test_examples):
        self.train_examples = train_examples
        self.test_examples = test_examples

class EvalExample(object):
    """
    turkesからスコアを持つContext-responseのペア
    """
    def __init__(self, uuid, kb, agent, role, prev_turns, prev_roles, target, candidates, scores):
        self.ex_id = uuid
        self.kb = kb
        self.agent = agent
        self.role = role
        self.prev_turns = prev_turns
        self.prev_roles = prev_roles
        self.target = target
        self.candidates = candidates
        self.scores = scores

    @staticmethod
    def from_dict(schema, raw):
        ex_id = raw['exid']
        kb = KB.from_dict(schema.attributes, raw['kb'])
        agent = raw['agent']
        role = raw['role']
        prev_turns = raw['prev_turns']
        prev_roles = raw['prev_roles']
        target = raw['target']
        candidates = raw['candidates']
        scores = raw['results']
        return EvalExample(ex_id, kb, agent, role, prev_turns, prev_roles, target, candidates, scores)

############################################################

def read_examples(paths, max_examples, Scenario):
    """
    最大の|max_examples|examplesを|paths|から読み取る
    """
    examples = []
    for path in paths:
        print('read_examples: %s' % path) # 使用する学習・検証データ名を表示
        for raw in read_json(path):
            #if max_examples >= 0 and len(examples) >= max_examples:
                #break
            examples.append(Example.from_dict(raw, Scenario))
    return examples

def read_dataset(args, Scenario):
    """
    与えられた引数によって指定されたデータセットを返す
    """
    train_examples = read_examples(args.train_examples_paths, args.train_max_examples, Scenario)
    test_examples = read_examples(args.test_examples_paths, args.test_max_examples, Scenario)
    print("We found {0} train examples and {1} test examples".format(len(train_examples), len(test_examples)))
    dataset = Dataset(train_examples, test_examples)
    return dataset

if __name__ == "__main__":
    lines = read_json("fb-negotiation/scr/data/transformed_test.json")
    for idx, raw in enumerate(lines):
        print(Example.from_dict(raw))
