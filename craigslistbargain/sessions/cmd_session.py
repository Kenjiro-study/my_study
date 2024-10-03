from sessions.session import Session


class CmdSession(Session):
    def __init__(self, agent, kb):
        super().__init__(agent) # 3系Ver.
        self.kb = kb

    def send(self):
        message = input()
        event = self.parse_input(message)
        return event

    def parse_input(self, message):
        """
        コマンドラインからユーザーの入力を解析する
        Args: message(str)
        Returns: Event
        """
        raw_tokens = message.split()
        tokens = self.remove_nonprintable(raw_tokens) # 英数字+一部の表示可能文字のみの文章に変更する

        if len(tokens) >= 2 and tokens[0] == '<offer>':
            return self.offer({'price': int(tokens[1]), 'sides': ''})
        elif tokens[0] == '<accept>':
            return self.accept()
        elif tokens[0] == '<reject>':
            return self.reject()
        elif tokens[0] == '<quit>':
            return self.quit()
        else:
            return self.message(message)

    def receive(self, event):
        print('【intent: {}, utterance: {}】'.format(event.metadata['sent']['logical_form']['intent'], event.data))
        #print("intent: ", event.data)
