from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, session_names=(None, None)):
        super().__init__(scenario, sessions, chat_id, session_names=session_names) # 3系Ver.
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

    # 交渉の結果を取得するメソッド
    def get_result(self, agent_idx, agent_data, partner_data):
        if self.offers[agent_idx] is not None:
            agent_dict = {'Target': agent_data['Target'], 'Bottomline': agent_data['Bottomline'], 'Offer': self.offers[agent_idx]['price']}
        else:
            agent_dict = {'Target': agent_data['Target'], 'Bottomline': agent_data['Bottomline'], 'Offer': None}

        if self.offers[1 - agent_idx] is not None:
            partner_dict = {'Target': partner_data['Target'], 'Bottomline': partner_data['Bottomline'], 'Offer': self.offers[1 - agent_idx]['price']}
        else:
            partner_dict = {'Target': partner_data['Target'], 'Bottomline': partner_data['Bottomline'], 'Offer': None}
            
        if agent_idx == 0:
            results = [agent_dict, partner_dict]
        else:
            results = [partner_dict, agent_dict]
        
        return results

    def get_statistics(self, agent_idx):
        outcome = self.get_outcome() # 交渉対話の結果を取り出す {'reward': reward, 'offer': offer}の辞書
        ag = outcome['reward'] # 合意か非合意か
        partner_idx = 1 - agent_idx # 相手のインデックスを作成
        
        agent_target = self.sessions[agent_idx].kb.target # 定量評価したいエージェントの目標価格を取得
        partner_target = self.sessions[partner_idx].kb.target # 相手の目標価格を取得

        if ag == 0:
            ut = 0 # 非合意であれば効用は計算しない
            leng = 0 # 非合意であれば対話の長さも計算しない
        else:
            slope = 2 / (agent_target - partner_target) # 2はagentの最大効用1から最小効用-1を引いたもの これで傾きを求める
            intercept = 1 - (slope * agent_target) # 切片を求める
            ut = (slope * outcome['offer']['price']) + intercept # 効用
            leng = self.events[-1].time # 対話の長さ
        
        fa = (-2) * abs(ut) # 対話の公平性
        return ag, ut, fa, leng


    def complete(self):
        return (self.offers[0] is not None and self.outcomes[1] is True) or (self.offers[1] is not None and self.outcomes[0] is True)

    def get_winner(self):
        # 交渉の勝者を表示したい場合はこの関数を修正すること!
        # TODO: fix this if we ever want to calculate who the winner is
        return -1
