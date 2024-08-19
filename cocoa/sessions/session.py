import time
import string
from cocoa.core.event import Event


class Session(object):
    """
    エージェントをインスタンス化するための抽象クラス
    セッションは対話状態を維持し対話イベントを送受信する
    """
    def __init__(self, agent, config=None):
        """
        エージェントのセッションを構築する
        Args: agent(int): agent id (0 or 1).
        """
        self.agent = agent  # 0 or 1 (どちらのプレイヤーが自分か?)
        self.partner = 1 - agent
        self.config = config

    def receive(self, event):
        """
        受信したイベントを解析し, 対話状態を更新する
        Args: event(Event)
        """
        raise NotImplementedError

    def send(self):
        """
        イベントを送る
        Returns: event(Event)
        """
        raise NotImplementedError

    # 表示可能な文字のみの文字列に変更する(漢字かなや非表示文字を含まない)
    # 表示可能な(string.printableに定義されている)文字は次の通り
    # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    @staticmethod
    def remove_nonprintable(raw_tokens):
        tokens = []
        for token in raw_tokens:
            all_valid_characters = True
            for char in token:
                if not char in string.printable:
                    all_valid_characters = False
            if all_valid_characters:
                tokens.append(token)
        return tokens

    @staticmethod
    def timestamp():
        return str(time.time())

    def message(self, text, metadata=None):
        return Event.MessageEvent(self.agent, text, time=self.timestamp(), metadata=metadata)

    def wait(self):
        return None
