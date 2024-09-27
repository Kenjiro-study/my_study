from __future__ import print_function

import random
import numpy as np

from cocoa.core.entity import is_entity

from neural.symbols import markers
from core.event import Event
from sessions.rulebased_session import CraigslistRulebasedSession
from cocoa.model.inference import oneshot_classify_intent

class HybridSession(object):
    @classmethod
    def get_session(cls, agent, kb, lexicon, generator, manager, config=None):
        if kb.role == 'buyer':
            return BuyerHybridSession(agent, kb, lexicon, config, generator, manager)
        elif kb.role == 'seller':
            return SellerHybridSession(agent, kb, lexicon, config, generator, manager)
        else:
            raise ValueError('Unknown role: %s', kb.role)

class BaseHybridSession(CraigslistRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super().__init__(agent, kb, lexicon, config, generator, manager) # 3系Ver.
        self.price_actions = ('init-price', 'counter-price', markers.OFFER)
        self.parser_model = None # パーサーに使うDLモデル
        self.tokenizer = None # DLモデルに対応するトークナイザ

    def receive(self, event, pre_text=None):
        if event.action in Event.decorative_events:
            return
        # パーサーの部分を処理する
        if self.parser.flag and (type(event.data) is str):
            # DLベースパーサー
            print("self.parser_model: ", self.parser_model)
            print("self.tokenizer: ", self.tokenizer)
            print("pre_text: ", pre_text)
            print("event.data: ", event.data)
            intent = oneshot_classify_intent(self.parser_model, self.tokenizer, pre_text, event.data)
            print("intent: ", intent)
            utterance = self.parser.parse(event, self.state, intent)
        else:
            # ルールベースパーサー
            utterance = self.parser.parse(event, self.state)

        self.state.update(self.partner, utterance)
        # マネージャーの部分を処理する
        if event.action == "message":
            logical_form = {"intent": utterance.lf.intent, "price": utterance.lf.price}
            entity_tokens = self.manager.env.preprocessor.lf_to_tokens(self.kb, logical_form)
        else:
            entity_tokens = self.manager.env.preprocessor.process_event(event, self.kb)
        if entity_tokens:
            self.manager.dialogue.add_utterance(event.agent, entity_tokens)
        print('【intent: {}, utterance: {}】'.format(utterance.lf.intent, utterance.text)) ######

    #ジェネレーターはアクションが有効であることを確認する
    def is_valid_action(self, action_tokens):
        if not action_tokens:
            return False
        if action_tokens[0] in self.price_actions and \
                not (len(action_tokens) > 1 and is_entity(action_tokens[1])):
            return False
        return True

    def send(self):
        action_tokens = self.manager.generate() # sessions.neural_session.PytorchNeuralSessionのgenerateメソッド
        if action_tokens is None:
            return None
        self.manager.dialogue.add_utterance(self.agent, list(action_tokens))

        price = None
        if not self.is_valid_action(action_tokens):
            action = 'unknown'
        else:
            action = action_tokens[0]
            if action in self.price_actions:
                price = self.manager.builder.get_price_number(action_tokens[1], self.kb)

        if action == markers.OFFER:
            return self.offer(price)
        elif action == markers.ACCEPT:
            return self.accept()
        elif action == markers.REJECT:
            return self.reject()
        elif action == markers.QUIT:
            return self.quit()
        
        return self.template_message(action, price=price)


class SellerHybridSession(BaseHybridSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super().__init__(agent, kb, lexicon, config, generator, manager) # 3系Ver.
        self.inc = 1.
        self.init_price()

    def init_price(self):
        # Seller: 目標価格, 定価が表示される
        self.state.my_price = self.target

class BuyerHybridSession(BaseHybridSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        super().__init__(agent, kb, lexicon, config, generator, manager) # 3系Ver.
        self.inc = -1.
        self.init_price()

    def init_price(self):
        self.state.my_price = self.round_price(self.target * (1 + self.inc * self.config.overshoot))
