from __future__ import print_function

import json
import random

from .util import generate_uuid
from .dataset import Example
from .event import Event
from threading import Lock

class Controller(object):
    """
    Interface of the controller: 2つのシステムを使用し, それらを実行してダイアログを生成する
    """
    def __init__(self, scenario, sessions, chat_id=None, allow_cross_talk=False, session_names=(None, None)):
        self.lock = Lock()
        self.scenario = scenario
        self.sessions = sessions
        self.session_names = session_names
        self.chat_id = chat_id
        assert len(self.sessions) == 2
        self.events = []
        self.max_turns = None
        self.allow_cross_talk = allow_cross_talk
        self.session_status = {agent: 'received' for agent, _ in enumerate(self.sessions)}

    def describe_scenario(self):
        print('='*50) # 見やすいように==を表示
        for session in self.sessions:
            print('\nAGENT={}'.format(session.agent)) # エージェントの名前(0か1)を表示
            session.kb.dump() # 各エージェントの詳細表示
        print('='*50) # 見やすいように==を表示
        return True

    def event_callback(self, event):
        raise NotImplementedError

    def get_outcome(self):
        raise NotImplementedError

    def get_result(self, agent_idx):
        return None

    def simulate(self, max_turns=None, verbose=False):
        '''
        対話をシミュレートする
        '''
        self.events = []
        self.max_turns = max_turns
        time = 0
        num_turns = 0
        game_over = False
        self.describe_scenario()
        # 乱数によって1/2で最初に話し始めるエージェントを決定
        if random.random() < 0.5: # seedが固定されているからいつも同じ値(変えたかったらseedを変える)
            first_speaker = 0
        else:
            first_speaker = 1
        while not game_over:
            # hybrid は craigslistbargain/sessions/hybrid_session.py の SellerHybridSessionクラス or BuyerHybridSessionクラス
            # cmd は craigslistbargain/sessions/cmd_session.py の CmdSessionクラス
            for agent, session in enumerate(self.sessions):
                print(session)
                if num_turns == 0 and agent != first_speaker: # first_speakerに選ばれたエージェントから対話開始
                    continue
                event = session.send() # ここでevent(発話やダイアログアクト等)が決定されている
                time += 1 # timeのインクリメント
                if not event:
                    continue

                event.time = time
                self.event_callback(event)
                self.events.append(event)
                
                if verbose:
                    # この時点でdataにintentが入り, metadataはNoneになってしまっている
                    print('agent=%s: session=%s, event=%s' % (agent, type(session).__name__, event.to_dict()))
                else:
                    action = event.action
                    data = event.data
                    event_output = data if action == 'message' else "Action: {0}, Data: {1}".format(action, data)
                    print('agent=%s, event=%s' % (agent, event_output))
                print("---------------------------------") #####
                num_turns += 1
                if self.game_over() or (max_turns and num_turns >= max_turns):
                    game_over = True
                    break

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

        uuid = generate_uuid('E')
        outcome = self.get_outcome()
        if verbose:
            print('outcome: %s' % outcome)
            print('----------------')
        # TODO: 構成可能な名前をシステムとセッションに追加するべき
        agent_names = {'0': self.session_names[0], '1': self.session_names[1]}
        return Example(self.scenario, uuid, self.events, outcome, uuid, agent_names)


    def step(self, backend=None):
        '''
        Webのバックエンドによって呼び出される
        '''
        with self.lock:
            # あるセッションから他のセッションにメッセージを送信しようとする
            for agent, session in enumerate(self.sessions):
                if session is None:
                    # 無言で失敗した場合、セッションがリセットされ、コントローラが事実上非アクティブになっていることを意味する
                    continue
                if (not self.allow_cross_talk) and self.session_status[agent] != 'received':
                    continue
                event = session.send()
                if event is None:
                    continue

                if not event.action in Event.decorative_events:
                    self.session_status[agent] = 'sent'
                self.event_callback(event)
                self.events.append(event)
                if backend is not None:
                    backend.add_event_to_db(self.get_chat_id(), event)

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)
                        if not event.action in Event.decorative_events:
                            self.session_status[partner] = 'received'

    def inactive(self):
        """
        このControllerが現在アクティブなアクティブなチャットセッションを制御しているかどうかを返す
        (両方のユーザーがまだアクティブ(交渉している)かどうかを確認することによって行う)
        
        :return: チャットがアクティブな場合はTrue(両方のセッションがNoneではない場合), それ以外の場合はFalse
        """
        for s in self.sessions:
            if s is None:
                return True
        return False

    def set_inactive(self, agents=[]):
        """
        Controller内の任意の数のセッションを[None]に設定すると, Controllerが非アクティブとしてマークされる.
        デフォルトの動作では全てのセッションが[None]に設定されますが(関数にパラメータが設定されていない場合), インデックスのリストを渡して
        それらのインデックスのSessionオブジェクトを[None]に設定することができる

        :param agents: 非アクティブとしてマークするSessionsのインデックスのリスト
                       これが[None]の場合, 関数は何も実行しない
                       何もリストが渡されない場合は, 関数は全てのSessionオブジェクトにNoneを設定する
        """
        with self.lock:
            if agents is None:
                return
            elif len(agents) == 0:
                self.sessions = [None] * len(self.sessions)
            else:
                for idx in agents:
                    self.sessions[idx] = None

    def get_chat_id(self):
        return self.chat_id

    def game_over(self):
        # game/sessionが終了状態に達したかどうか
        raise NotImplementedError

    def complete(self):
        # taskが正常に完了したかどうか
        raise NotImplementedError
