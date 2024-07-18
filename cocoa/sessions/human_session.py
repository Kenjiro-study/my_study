__author__ = 'anushabala'
from cocoa.sessions.session import Session


class HumanSession(Session):
    """
    HumanSessionは対話内の単一の人間エージェントを表す.
    このクラスはエージェントによって送信されたメッセージをキューに入れ, 対話内の他のエージェントからのメッセージを取得するために使用できる
    """
    def __init__(self, agent):
        super(HumanSession, self).__init__(agent)
        self.outbox = []
        self.inbox = []
        self.cached_messages = []
        # TODO: メッセージ履歴を保存するためのキャッシュを実装しないといけない
        
    def send(self):
        if len(self.outbox) > 0:
            return self.outbox.pop(0)
        return None

    def poll_inbox(self):
        if len(self.inbox) > 0:
            return self.inbox.pop(0)
        return None

    def receive(self, event):
        self.inbox.append(event)

    def enqueue(self, event):
        self.outbox.append(event)


