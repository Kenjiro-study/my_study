"""
データセット内のexamplesを前処理し, モデルのためのデータを生成する
"""

import random
import re
import time
import os
import numpy as np

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.model.vocab import Vocabulary

from core.price_tracker import PriceTracker, PriceScaler
from core.tokenizer import tokenize
from .batcher import DialogueBatcherFactory, Batch
from .symbols import markers
from .vocab_builder import create_mappings
from neural import make_model_mappings

category_to_marker = {
        'car': markers.C_car,
        'phone': markers.C_phone,
        'housing': markers.C_housing,
        'bike': markers.C_bike,
        'furniture': markers.C_furniture,
        'electronics': markers.C_electronics,
        }


def price_filler(x):
    return x == '<price>'


class TextIntMap(object):
    """
    結果を視覚的に表すためにtextとintの間のマップを作成する
    """
    def __init__(self, vocab, preprocessor):
        self.vocab = vocab
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor

    def pred_to_input(self, preds, prices=None):
        """
        デコーダ出力をデコーダ入力に変換する
        """
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred) for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def text_to_int(self, utterance, stage=None):
        """
        エンコード, デコード, グラウンドトゥルースのいずれに使用されるかに基づいて, 発話内のエンティティを処理する
        """
        tokens = self.preprocessor.process_utterance(utterance, stage)
        return [self.vocab.to_ind(token) for token in tokens]

    def int_to_text(self, inds, stage=None, prices=None):
        """
        text_to_intの逆
        """
        toks = [self.vocab.to_word(ind) for ind in inds]
        if prices is not None:
            assert len(inds) == len(prices)
            toks = [CanonicalEntity(value=p, type='price') if price_filler(x) else x for x, p in zip(toks, prices)]
        return toks

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, agent, kb, uuid, model='seq2seq'):
        """
        モデルに必要な対話データ
        """
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.model = model
        self.agent_to_role = self.get_role_mapping(agent, kb)
        # KB context
        # NOTE: context_to_intはカテゴリ, タイトル, 説明を整数に変更する
        self.category_str = kb.category
        self.category = kb.category
        self.title = tokenize(re.sub(r'[^\w0-9]', ' ', kb.facts['item']['Title']))
        self.description = tokenize(re.sub(r'[^\w0-9]', ' ', ' '.join(kb.facts['item']['Description'])))
        # token_turns: トークンとエンティティ(エンティティのリンクの出力)
        self.token_turns = []
        # 解析済みのlogical forms
        self.lfs = []
        # turns: エンコーダ, デコーダの入力とターゲットの入力トークン. 後で整数に変換される
        self.turns = [[], [], []]
        # entities: turnsと同じ構造を持つ. non-entityトークンはNoneとする
        self.entities = []
        self.agents = []
        self.roles = []
        self.is_int = False  # 整数に変換したかどうかを示すフラグ
        self.num_context = None

    @property
    def num_turns(self):
        return len(self.turns[0])

    def join_turns(self):
        for i, utterances in enumerate(self.turns):
            self.turns[i] = [x for utterance in utterances for x in utterance]

    @staticmethod
    def get_role_mapping(agent, kb):
        my_id = agent
        my_role = kb.role

        partner_id = 1 - agent
        partner_role = 'buyer' if my_role == 'seller' else 'seller'

        return {my_id: my_role, partner_id: partner_role}

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_utterance(self, agent, utterance, lf=None):
        # 常にパートナーエージェントから開始する
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
        self._add_utterance(agent, utterance, lf=lf)

    @classmethod
    def scale_price(cls, kb, utterance):
        return [PriceScaler.scale_price(kb, x) if is_entity(x) else x for x in utterance]

    @classmethod
    def original_price(cls, kb, utterance):
        s = [PriceScaler.unscale_price(kb, x) if is_entity(x) else x for x in utterance]
        return s

    def lf_to_tokens(self, kb, lf):
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            tokens.append(PriceScaler.scale_price(kb, price).canonical)
        return tokens

    def _insert_markers(self, agent, utterance, new_turn):
        # 文の終わりをマークする
        utterance.append(markers.EOS)

        # GOを挿入する
        if new_turn:
            cat_symbol = category_to_marker[self.category_str]
            utterance.insert(0, cat_symbol)

            role = self.agent_to_role[agent]
            start_symbol = markers.GO_S if role == 'seller' else markers.GO_B
            utterance.insert(0, start_symbol)

        return utterance

    def _add_utterance(self, agent, utterance, lf=None):
        # 同じエージェントが話す
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]
        if lf:
            lf = self._insert_markers(agent, self.lf_to_tokens(self.kb, lf), new_turn)
        else:
            lf = []

        if new_turn:
            self.agents.append(agent)
            role = self.agent_to_role[agent]
            self.roles.append(role)

            self.token_turns.append(utterance)
            self.entities.append(entities)
            self.lfs.append(lf)
        else:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
            self.lfs[-1].extend(lf)

    def kb_context_to_int(self):
        self.category = self.mappings['cat_vocab'].to_ind(self.category)
        self.title = map(self.mappings['kb_vocab'].to_ind, self.title)
        self.description = map(self.mappings['kb_vocab'].to_ind, self.description)
        self.title = list(self.title) # 追加
        self.description = list(self.description) # 追加

    def lf_to_int(self):
        self.lf_token_turns = []
        for i, lf in enumerate(self.lfs):
            self.lf_token_turns.append(lf)
            self.lfs[i] = map(self.mappings['lf_vocab'].to_ind, lf)

    def convert_to_int(self):
        if self.is_int:
            return

        for turn in self.token_turns:
            # turnはエージェントが自分のターンに話したトークンのリスト
            # self.turnsは[[], [], []]で始まるため, 各部分はエンコード部分, デコード部分, ターゲット部分のいずれかのトークンを保持するリストになる
            for portion, stage in zip(self.turns, ('encoding', 'decoding', 'target')):
                portion.append(self.textint_map.text_to_int(turn, stage))

        self.kb_context_to_int()
        self.lf_to_int()

        self.is_int = True

    def _pad_list(self, l, size, pad):
        for i in range(len(l), size):
            l.append(pad)
        return l

    def pad_turns(self, num_turns):
        """
        Pad_turnsを長さnum_turnsにする
        """
        self.agents = self._pad_list(self.agents, num_turns, None)
        self.roles = self._pad_list(self.roles, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])
        self.lfs = self._pad_list(self.lfs, num_turns, [])

    def get_price_turns(self, pad):
        """
        フラット化されたエンティティターンを指定して, 各トークンの価格を返す
        pad: non-priceターゲットを満たす(記述する)ために使用する.
        """
        def to_float_price(entity):
            return float('{:.2f}'.format(PriceTracker.get_price(entity)))
        prices = [[to_float_price(entity) if entity else pad for entity in entities] for entities in self.entities]
        return prices

class Preprocessor(object):
    """
    生の発話を前処理する: トークン化, エンティティの紐付け(リンク)
    サンプルをDataGeneratorで使用されるDialogueデータ構造に変換する
    """
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form, model='seq2seq'):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}
        self.model = model

    @classmethod
    def get_entity_form(cls, entity, form):
        """
        entityは(surface_form, (canonical_form, type))として表される
        """
        assert len(entity) == 2
        if form == 'surface':
            return entity.surface
        elif form == 'type':
            return '<%s>' % entity.canonical.type
        elif form == 'canonical':
            return entity._replace(surface='')
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage=None):
        """
        Input: utteranceはトークンのリスト, stageはエンコード, デコード, ターゲットのいずれか
        Output: ほとんどの場合, stageが宣言される.
                model_typeとstageの組み合わせに基づいて, 発話を要約するかどうかを選択する
                "sum"を含むモデルは選択したキーワードのみを含むように要約する必要があり, "seq"を含むモデルは完全なシーケンスを保持する
        """
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            if stage == 'encoding':
                summary = self.summarize(utterance) if self.model in ["sum2sum", "sum2seq"] else utterance
            elif (stage == 'decoding') or (stage == 'target'):
                if self.model == "sum2sum":
                    summary = self.summarize(utterance)
                elif self.model == "sum2seq":
                    summary = self.summarize(utterance)
                    summary.append(markers.END_SUM)
                    summary.extend(utterance)
                else:
                    summary = utterance
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in summary]

    def lf_to_tokens(self, kb, lf):
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            tokens.append(PriceScaler.scale_price(kb, price))
        return tokens

    def _process_example(self, ex):
        """
        exampleを各エージェントの視点からターンベースの対話に変換する
        exampleごとにDialogueオブジェクトを生成する
        """
        kbs = ex.scenario.kbs
        for agent in (0, 1):
            dialogue = Dialogue(agent, kbs[agent], ex.ex_id, model=self.model)
            for e in ex.events:
                if self.model in ('lf2lf',):
                    lf = e.metadata
                    assert lf is not None
                    utterance = self.lf_to_tokens(dialogue.kb, lf)
                else:
                    utterance = self.process_event(e, dialogue.kb)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance, lf=e.metadata)
            yield dialogue

    @classmethod
    def price_to_entity(cls, price):
        return Entity(price, CanonicalEntity(price, 'price'))

    def process_event(self, e, kb):
        """
        トークン化, エンティティを紐づける(リンクする)
        """
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.link_entity(tokenize(e.data), kb=kb, scale=True, price_clip=4.)
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'offer':
            data = e.data['price']
            if data is None:
                return None
            price = PriceScaler._scale_price(kb, data)
            entity_tokens = [markers.OFFER, self.price_to_entity(price)]
            return entity_tokens
        elif e.action == 'quit':
            entity_tokens = [markers.QUIT]
            return entity_tokens
        elif e.action == 'accept':
            entity_tokens = [markers.ACCEPT]
            return entity_tokens
        elif e.action == 'reject':
            entity_tokens = [markers.REJECT]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.')

    @classmethod
    def skip_example(cls, example):
        tokens = {0: 0, 1: 0}
        turns = {0: 0, 1: 0}
        for event in example.events:
            if event.action == "message":
                msg_tokens = tokenize(event.data)
                tokens[event.agent] += len(msg_tokens) # 各エージェントの一対話における総単語(トークン)数を計算
                turns[event.agent] += 1 # 各エージェントの一対話における総ターン数を計算
        if tokens[0] < 40 and tokens[1] < 40:
            return True
        if turns[0] < 2 or turns[1] < 2:
            return True
        return False # どちらかのエージェントの総単語数が40未満, またはターン数が2未満の短い対話の場合はFalseを返してスキップ

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            if self.skip_example(ex):
                continue
            for d in self._process_example(ex):
                dialogues.append(d)
        return dialogues

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor,
            schema, mappings_path=None, cache='.cache',
            ignore_cache=False, num_context=1, batch_size=1,
            model='seq2seq'):
        examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.items()}
        self.num_context = num_context
        self.model = model

        self.cache = cache
        self.ignore_cache = ignore_cache
        if (not os.path.exists(cache)) or ignore_cache:
            # NOTE: 各対話は各エージェントの視点から2つのexamplesにまとめられる
            self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.items() if v}

            for fold, dialogues in self.dialogues.items():
                print('%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold]))
        else:
            self.dialogues = {k: None  for k, v in examples.items() if v}
            print('Using cached data from', cache)

        self.mappings = self.load_mappings(model, mappings_path, schema, preprocessor)
        self.textint_map = TextIntMap(self.mappings['utterance_vocab'], preprocessor)

        Dialogue.mappings = self.mappings
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = num_context

        self.dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model,
                        kb_pad=self.mappings['kb_vocab'].to_ind(markers.PAD),
                        mappings=self.mappings, num_context=num_context)

        self.batches = {k: self.create_batches(k, dialogues, batch_size) for k, dialogues in self.dialogues.items()}

    def load_mappings(self, model_type, mappings_path, schema, preprocessor):
        vocab_path = os.path.join(mappings_path, 'vocab.pkl')
        if not os.path.exists(vocab_path):
            print('Vocab not found at', vocab_path)
            mappings = create_mappings(self.dialogues['train'], schema,
                preprocessor.entity_forms.values())
            write_pickle(mappings, vocab_path)
            print('Wrote mappings to {}.'.format(vocab_path))
        else:
            print('Loading vocab from', vocab_path)
            mappings = read_pickle(vocab_path)

        for k, v in mappings.items():
            print(k, v.size)
        mappings = make_model_mappings(model_type, mappings)
        return mappings

    def convert_to_int(self):
        """
        トークンを整数に変換する
        """
        for fold, dialogues in self.dialogues.items():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def get_dialogue_batch(self, dialogues):
        return DialogueBatcher(dialogues).create_batch()

    def dialogue_sort_score(self, d):
        # 対話をターン数で並び替える
        return len(d.turns[0])

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogue_batches = []
        dialogues.sort(key=lambda d: self.dialogue_sort_score(d))
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: 十分なexamplesがない場合は最後のバッチサイズが小さくなる場合がある
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(self.dialogue_batcher.create_batch(dialogue_batch))
            start = end
        return dialogue_batches

    def get_all_responses(self, name):
        dialogues = self.dialogues[name]
        responses = {'seller': [], 'buyer': []}
        for dialogue in dialogues:
            for turn, role in zip(dialogue.token_turns, dialogue.roles):
                responses[role].extend(turn)
        return responses

    def create_batches(self, name, dialogues, batch_size):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            for dialogue in dialogues:
                dialogue.convert_to_int()

            dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
            print('Write %d batches to cache %s' % (len(dialogue_batches), cache_file))
            start_time = time.time()
            write_pickle(dialogue_batches, cache_file)
            print('[%d s]' % (time.time() - start_time))
        else:
            start_time = time.time()
            dialogue_batches = read_pickle(cache_file)
            print('Read %d batches from cache %s' % (len(dialogue_batches), cache_file))
            print('[%d s]' % (time.time() - start_time))
        return dialogue_batches

    def generator(self, name, shuffle=True, cuda=True):
        dialogue_batches = self.batches[name]
        yield sum([len(b) for b in dialogue_batches])
        inds = range(len(dialogue_batches))
        if shuffle:
            _inds = list(inds)
            random.shuffle(_inds)
        for ind in inds:
            for batch in dialogue_batches[ind]:
                yield Batch(batch['encoder_args'],
                            batch['decoder_args'],
                            batch['context_data'],
                            self.mappings['utterance_vocab'],
                            num_context=self.num_context, cuda=cuda)
            # 対話の終わり
            yield None

