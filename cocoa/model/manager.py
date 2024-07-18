from cocoa.core.util import read_pickle, write_pickle
from cocoa.model.counter import build_vocabulary, count_ngrams
from cocoa.model.ngram import MLENgramModel
from cocoa.model.util import entropy

class Manager(object):
    def __init__(self, model, actions):
        self.model = model
        self.actions = actions

    @classmethod
    def from_train(cls, sequences, n=3):
        vocab = build_vocabulary(1, *sequences) # 各ダイアログアクトが何回出てきているかをカウント
        counter = count_ngrams(n, vocab, sequences, pad_left=True, pad_right=False) #####
        model = MLENgramModel(counter)
        actions = vocab.keys() # intentの種類を格納
        # print(model.score('intro', ('<start>',))) # init-priceが<start>の次にくる確率
        # print(model.ngrams.most_common(10))
        return cls(model, actions)

    def available_actions(self, state):
        actions = [a for a in self.actions if a != 'unknown']
        return actions

    def most_likely_action(self, context, freqdist):
        best_action = max(freqdist, key=lambda x: x[1])[0]
        return best_action

    def min_entropy_action(self, context, freqdist):
        ent = []
        for a, _ in freqdist:
            c = (context[-1], a)
            f = self.model.freqdist(c)
            e = entropy([x[1] for x in f], normalized=False)
            ent.append((a, e))
        best_action = min(ent, key=lambda x: x[1])[0]
        return best_action

    def choose_action(self, state, context=None):
        if not context:
            context = (state.my_act, state.partner_act) # 自分の一つ前のダイアログアクトと相手のダイアログアクトを格納する
        freqdist = self.model.freqdist(context) # 次(応答)をどのintentにするのが最も良いかを計算
        actions = self.available_actions(state) # このタスクにおける使用できるintentの種類を格納
        freqdist = [x for x in freqdist if x[0] in actions]
        print('context:', context)
        print('dist:', freqdist)
        print('available actions:', actions)
        # TODO: backoff
        if len(freqdist) == 0:
            return None
        best_action = max(freqdist, key=lambda x: x[1])[0] # 計算した値から次のベストなintentを決定
        print('action:', best_action)
        return best_action

    def save(self, output):
        data = {'model': self.model, 'actions': list(self.actions)}
        write_pickle(data, output)

    @classmethod
    def from_pickle(cls, path):
        data = read_pickle(path)
        return cls(data['model'], data['actions'])
