from cocoa.neural.utterance import Utterance
from cocoa.neural.utterance import UtteranceBuilder as BaseUtteranceBuilder

from .symbols import markers, category_markers
from core.price_tracker import PriceScaler
from cocoa.core.entity import is_entity

class UtteranceBuilder(BaseUtteranceBuilder):
    """
    word-basedの発話をジェネレータのバッチ出力と基となる辞書から作成する
    """
    def build_target_tokens(self, predictions, kb=None):
        tokens = super(UtteranceBuilder, self).build_target_tokens(predictions, kb)
        tokens = [x for x in tokens if not x in category_markers] # カテゴリー以外を抽出する(インテントだけが出てくるが本当はここで自然言語応答も欲しい?)
        return tokens # インテントだけが返っている

    def _entity_to_str(self, entity_token, kb):
        raw_price = PriceScaler.unscale_price(kb, entity_token)
        human_readable_price = "${}".format(raw_price.canonical.value)
        return human_readable_price

    def get_price_number(self, entity, kb):
        raw_price = PriceScaler.unscale_price(kb, entity)
        return raw_price.canonical.value
