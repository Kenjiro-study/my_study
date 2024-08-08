import re
import numpy as np
from nltk import ngrams
from nltk.corpus import stopwords
from collections import defaultdict

from cocoa.core.entity import is_entity
from cocoa.model.new_parser3 import Parser as BaseParser, LogicalForm as LF, Utterance ##### お試し用にparserをnew_parser3参照に変更
from cocoa.model.inference import classify_intent_neural

from core.tokenizer import tokenize

class Parser(BaseParser):
    stopwords = set(stopwords.words('english')) # 頻繁に現れるため検索処理から除外すべき単語(stopwords)の設定
    stopwords.update(['may', 'might', 'rent', 'new', 'brand', 'low', 'high', 'now', 'available']) # 交渉対話用にstopwordsを追加

    price_patterns = [
            r'come down',
            r'(highest|lowest)',
            r'go (lower|higher)',
            r'too (high|low)',
            ]

    greeting_patterns = [
            r'how are you',
            r'interested in',
            ]

    agreement_patterns = [
        r'^that works[.!]*$',
        r'^great[.!]*$',
        r'^(ok|okay)[.!]*$',
        r'^great, thanks[.!]*$',
        r'^deal[.!]*$',
        r'^[\w ]*have a deal[\w ]*$',
        r'^i can do that[.]*$',
    ]

    @classmethod
    def is_price_token(cls, token):
        return is_entity(token)

    @classmethod
    def is_greeting(cls, utterance):
        result = super(Parser, cls).is_greeting(utterance)
        if not result:
            for pattern in cls.greeting_patterns:
                if re.search(pattern, utterance.text, re.IGNORECASE):
                    return True
            return False
        return result

    @classmethod
    def is_agreement(cls, utterance):
        for pattern in cls.agreement_patterns:
            # 大文字と小文字の区別なしで(re.IGNORECASE), agreement_patternsとマッチング
            if re.match(pattern, utterance.text, re.IGNORECASE) is not None:
                return True
        return False

    @classmethod
    def is_price(cls, utterance):
        for pattern in cls.price_patterns:
            # price_patternsとマッチング
            if re.search(pattern, utterance.text):
                return True
        return False

    def compare(self, x, y, inc):
        """Compare two prices.

        Args:
            x (float)
            y (float)
            inc (float={1,-1}): 1 means higher is better and -1 mean lower is better.
        """
        if x == y:
            r = 0
        elif x < y:
            r = -1
        else:
            r = 1
        r *= inc
        return r

    def parse_offer(self, event):
        intent = 'offer' # intentをofferに決定
        try:
            price = float(event.data.get('price')) # 価格のみを取り出す(offerのdataは{"price": val1 ,"sides": val2}という構造)
        except TypeError:
            price = None
        return Utterance(logical_form=LF(intent, price=price), template=['<offer>'])

    def tag_utterance(self, utterance, tokens_with_parsed_price):
        """Tag the utterance with basic speech acts.
        """
        tags = super(Parser, self).tag_utterance(utterance)
        if self.is_price(utterance):
            tags.append('vague-price')
        if self.is_agreement(utterance):
            tags.append('agree')
        price_types = list(set([token.canonical.type for token in tokens_with_parsed_price if self.is_price_token(token)]))
        tags.extend(price_types)
        if price_types:
            tags.append('has_price')
        return tags

    def get_proposed_price(self, tokens, dialogue_state):
        # パートナーが複数の価格について言及しているときは, どちらが相手の求めている価格なのか判断する
        partner_prices = []
        partner_prev_price = dialogue_state.partner_price # パートナーの一つ目の提案価格を取得
        partner_role = 'seller' if self.kb.role == 'buyer' else 'buyer' # 自分のroleからパートナーが買い手, 売り手どちらなのかを決定
        inc = 1. if partner_role == 'seller' else -1. # パートナーが売り手ならincは1, 買い手ならincは-1

        # for文で取り出す価格は[発話内の表示価格|小数点第一位まで表示した価格]になる (例: How about $8 → [$8|8.0])
        for price in filter(lambda x: self.is_price_token(x), tokens):
            # priceは小数点第一位まで表示した価格, type_はlisting_price, partner_price, my_price, priceのいずれか
            price, type_ = price.canonical.value, price.canonical.type
            # 1) New price doest repeat current price
            # 2) One's latest price is worse than previous ones (compromise)
            # 3) Seller might propose the listing price but the buyer won't
            if type_ != 'curr_price' and \
                (partner_prev_price is None or self.compare(price, partner_prev_price, inc) <= 0) and \
                (partner_role == 'seller' or type_ != 'listing_price'):
                    partner_prices.append(price)
        # 複数の価格が考えられる場合はincを使って最適なものを選択する
        if partner_prices:
            i = np.argmax(inc * np.array(partner_prices))
            return partner_prices[i]
        return None

    def _parse_price(self, price_token, dialogue_state):
        price = price_token.canonical.value
        if price == dialogue_state.listing_price:
            type_ = 'listing_price'
        elif price == dialogue_state.my_price:
            type_ = 'my_price'
        elif price == dialogue_state.partner_price:
            type_ = 'partner_price'
        else:
            type_ = 'price'
        canonical = price_token.canonical._replace(type=type_)
        price_token = price_token._replace(canonical=canonical)
        return price_token

    def parse_prices(self, tokens, dialogue_state):
        tokens = [self._parse_price(token, dialogue_state) if self.is_price_token(token) else token for token in tokens]
        return tokens

    # 商品説明のタイトルを元に, 発話の一部をプレースホルダに置き換える
    def _parse_title(self, tokens, dialogue_state):
        title = dialogue_state.kb.title.lower().split() # データのscenario.kbs.item.title(要は交渉する商品のタイトル)の文を小文字化してトークナイズ
        new_tokens = []
        for token in tokens:
            if token in title and not token in self.stopwords: # 発話内のトークンがtitleに含まれているかつ, ストップワードに含まれていない場合
                if len(new_tokens) == 0 or new_tokens[-1] != '{title}': # 発話内の最初のトークンか, 一つ前のトークンがプレースホルダーに置き換えられていない場合
                    new_tokens.append('{title}') # {title}プレースホルダに置き換え
            else:
                new_tokens.append(token) # 条件を満たさない場合は置き換えずにそのまま
        return new_tokens

    def extract_template(self, tokens, dialogue_state):
        tokens = self.parse_prices(tokens, dialogue_state) # 価格を検出してトークナイズ
        tokens = ['{%s}' % token.canonical.type if self.is_price_token(token) else token for token in tokens] # 価格の部分を{price}のプレースホルダに置き換える
        tokens = self._parse_title(tokens, dialogue_state) # 発話内における商品説明のタイトルにも出てくる単語を{title}のプレースホルダに置き換える
        return tokens

    def classify_intent(self, utterance, tokens_with_parsed_price, dialogue_state):
        # タグに応じてルールベースでマッチングを行いダイアログアクトを決定
        tags = self.tag_utterance(utterance, tokens_with_parsed_price)
        if dialogue_state.time == 0 and not 'has_price' in tags:
            intent = 'intro'
        elif 'has_price' in tags and dialogue_state.curr_price is None:
            intent = 'init-price'
        elif (not 'has_price' in tags) and 'vague-price' in tags:
            intent = 'vague-price'
        elif 'price' in tags or 'listing_price' in tags:
            intent = 'counter-price'
        elif 'partner_price' in tags:
            intent = 'insist'
        elif tags == ['question']:
            intent = 'inquiry'
        elif 'agree' in tags or 'my_price' in tags:
            intent = 'agree'
        elif not tags and dialogue_state.my_act == 'inquiry':
            intent = 'inform'
        elif tags == ['negative']:
            intent = 'disagree'
        else:
            intent = 'unknown'
        return intent

    def parse_message(self, event, dialogue_state, intents=None):
        tokens = self.lexicon.link_entity(tokenize(event.data), kb=self.kb, scale=False) # craigslistbargain/core/price-tracker.pyで価格を検出してトークナイズ
        template = self.extract_template(tokens, dialogue_state) # タイトルやprice-trackerを使用して発話の一部をプレースホルダに置き換えてテンプレートを作成
        utterance = Utterance(raw_text=event.data, tokens=tokens) # 正しい形の発話に成形
        tokens_with_parsed_price = self.parse_prices(tokens, dialogue_state) # 価格の部分をlisting_price(定価),my_price(自分の提案価格),partner_price(相手の提案価格),price(それ以外)に分ける
        #######
        if self.flag == True: ##### DLベースパーサーの使用
            intent = intents
            #print(intent)
        else: ##### ルールベースパーサーの使用
            intent = self.classify_intent(utterance, tokens_with_parsed_price, dialogue_state) # ダイアログアクトの決定(ここを置き換えよう!)
        #######
        proposed_price = self.get_proposed_price(tokens_with_parsed_price, dialogue_state) # 各発話で提案されている価格を取得
        utterance.lf = LF(intent, price=proposed_price) # LogicalForm形式でダイアログアクトを保存(intent=⚪︎⚪︎ price=××)
        utterance.template = template # プレースホルダに置き換えたテンプレートを格納
        return utterance

    def parse(self, event, dialogue_state, intents=None):
        # パートナーの発話を解析する
        assert event.agent == 1 - self.agent # パートナーの発話を確認するため, agent番号が自分と同じ場合はエラーにする
        if event.action == 'offer':
            u = self.parse_offer(event) # offerの場合はintent=offerで価格を追加
        elif event.action == 'message':
            u = self.parse_message(event, dialogue_state, intents) # messageの場合はパーサーで解析
        elif event.action in ('reject', 'accept', 'quit'):
            u = self.parse_action(event) # この三つの場合はactionをそのままintentにする
        else:
            return False
        return u

def parse_dialogue(example, lexicon, counter):
    kbs = example.scenario.kbs
    parsers = [Parser(agent, kbs[agent], lexicon) for agent in (0, 1)]
    dialogue_state = {
            'time': 0,
            'act': [None, None],
            'price': [None, None],
            'curr_price': None,
            }
    lfs = []
    for i, event in enumerate(example.events):
        dialogue_state['time'] = i
        lf = parsers[1 - event.agent].parse(event, dialogue_state, update_state=True)
        if lf:
            lfs.append(lf)
    counter['total_lf'] += len(lfs)
    counter['unk_lf'] += len([lf for lf in lfs if lf.intent == 'unknown'])
    if len(lfs) > 2:
        for seq in ngrams(lfs, 3):
            seq = [s.intent for s in seq]
            counter['seqs'][seq[0]][seq[1]][seq[2]] += 1


if __name__ == '__main__':
    import argparse
    from collections import defaultdict
    from cocoa.core.dataset import read_examples
    from core.price_tracker import PriceTracker
    from core.scenario import Scenario
    from neural.preprocess import Preprocessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--price-tracker-model')
    parser.add_argument('--max-examples', default=-1, type=int)
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker_model)
    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    counter = {'total_lf': 0, 'unk_lf': 0, 'seqs': defaultdict(lambda : defaultdict(lambda : defaultdict(int)))}
    for example in examples:
        if Preprocessor.skip_example(example):
            continue
        parse_dialogue(example, price_tracker, counter)
    print('% unk:', counter['unk_lf'] / float(counter['total_lf']))
    print('intent seqs:')
    seqs = counter['seqs']
    for first in seqs:
        total_first = sum([sum(seqs[first][a].values()) for a in seqs[first]])
        unks = sum([sum(seqs[first][a].values()) for a in ('unknown',)])
        unks = float(unks) / total_first
        print(first, 'unk={:.3f}'.format(unks))
        for second in seqs[first]:
            if second == 'unknown':
                continue
            total = sum(seqs[first][second].values())
            if total < 50:
                continue
            unks = seqs[first][second]['unknown']
            unks = float(unks) / total
            print(first, second, '{:.3f}'.format(total/float(total_first)), 'unk={:.3f}'.format(unks))
        print('------------------------')
