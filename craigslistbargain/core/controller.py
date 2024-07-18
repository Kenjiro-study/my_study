from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, session_names=(None, None)):
        super(Controller, self).__init__(scenario, sessions, chat_id, session_names=session_names)
        # self.prices = [None, None]
        self.offers = [None, None]
        # self.sides = [None, None]
        self.outcomes = [None, None]
        self.quit = False

    def event_callback(self, event):
        if event.action == 'offer':
            self.offers[event.agent] = event.data
        elif event.action == 'accept':
            self.outcomes[event.agent] = True
        elif event.action == 'reject':
            self.outcomes[event.agent] = False
        elif event.action == 'quit':
            self.quit = True
            self.outcomes[event.agent] = False

    def get_outcome(self):
        offer = None
        reward = 0
        if self.offers[0] is not None and self.outcomes[1] is True:
            reward = 1
            offer = self.offers[0]
        elif self.offers[1] is not None and self.outcomes[0] is True:
            reward = 1
            offer = self.offers[1]
        else:
            if (self.offers[0] is not None or self.offers[1] is not None) and False in self.outcomes:
                reward = 0
                offer = self.offers[0] if self.offers[1] is None else self.offers[1]

        # 結果として起きうるoutcomesの種類:
        # reward が 1 で offer は null ではない: complete dialogue(完全な対話)
        # reward が 0 で offer は null ではない: incomplete dialogue (disagreement): offer was made and not accepted(不完全な対話(非合意): オファーは行われたが受け入れられなかった)
        # reweard が 0 で offer は null: incomplete dialogue: no offer was made(不完全な対話: オファーが行われなかった)
        return {'reward': reward, 'offer': offer}

    def game_over(self):
        return not self.inactive() and \
               ((self.offers[0] is not None and self.outcomes[1] is not None) or
                (self.offers[1] is not None and self.outcomes[0] is not None) or
                 self.quit)

    def get_result(self, agent_idx):
        ##### 【超重要】分析結果を表示したい場合はこの関数を修正すること! ##### 
        # TODO: fix this if we ever want to display results in the survey
        return None

    def complete(self):
        return (self.offers[0] is not None and self.outcomes[1] is True) or (self.offers[1] is not None and self.outcomes[0] is True)

    def get_winner(self):
        # 交渉の勝者を表示したい場合はこの関数を修正すること!
        # TODO: fix this if we ever want to calculate who the winner is
        return -1
