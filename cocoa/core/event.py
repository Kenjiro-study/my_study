class Event(object):
    """
    対話の原子的なイベント. 誰かが話したり選択したりする

    Params:
    agent: イベントをトリガーにしたエージェントのインデックス
    time: イベントの発生時刻
    action: このイベントが対応するアクション('select', 'message', ..)
    data: イベントの一部である全てのデータ
    start_time: イベントアクションが開始された時刻(例: エージェントが送信するメッセージの入力を開始した時刻
    """

    decorative_events = ('join', 'leave', 'typing', 'eval')

    def __init__(self, agent, time, action, data, start_time=None, metadata=None):
        self.agent = agent
        self.time = time
        self.action = action
        self.data = data
        self.start_time = start_time
        self.metadata = metadata

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'], start_time=raw.get('start_time'), metadata=raw.get('metadata'))

    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data,
                'start_time': self.start_time, 'metadata': self.metadata}

    @classmethod
    def MessageEvent(cls, agent, data, time=None, start_time=None, metadata=None):
        return cls(agent, time, 'message', data, start_time=start_time, metadata=metadata)

    @classmethod
    def JoinEvent(cls, agent, userid=None, time=None):
        return cls(agent, time, 'join', userid)

    @classmethod
    def LeaveEvent(cls, agent, userid=None, time=None):
        return cls(agent, time, 'leave', userid)

    @classmethod
    def TypingEvent(cls, agent, data, time=None):
        return cls(agent, time, 'typing', data)

    @classmethod
    def EvalEvent(cls, agent, data, time):
        return cls(agent, time, 'eval', data)

    @staticmethod
    def gather_eval(events):
        event_dict = {e.time: e for e in events if e.action != 'eval'}
        for e in events:
            if e.action == 'eval':
                event_dict[e.time].tags = [k for k, v in e.data['labels'].items() if v != 0]
            else:
                event_dict[e.time].tags = []
        events_with_eval = [v for k, v in sorted(event_dict.items(), key=lambda x: x[0])]
        return events_with_eval
