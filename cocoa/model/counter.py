# Natural Language Toolkit: Language Model Counters
#
# Copyright (C) 2001-2016 NLTK Project
# Author: Ilia Kurenkov <ilia.kurenkov@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import unicode_literals

from collections import Counter, defaultdict
from copy import copy
from itertools import chain

from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import compat


def build_vocabulary(cutoff, *texts):
    combined_texts = chain(*texts)
    return NgramModelVocabulary(cutoff, combined_texts)


def count_ngrams(order, vocabulary, training_sents, **counter_kwargs):
    counter = NgramCounter(order, vocabulary, **counter_kwargs)
    counter.train_counts(training_sents)
    return counter

class NgramModelVocabulary(Counter):
    """言語モデルの語彙を保存する

    語彙に関する二つの共通の言語モデリング要件を満たします:
    - メンバーシップを確認して, そのサイズを計算する時に, カウントをカットオフ値と比較することで項目をフィルター処理する
    - "unknown"トークンを考慮して, そのサイズに1を加える
    """

    def __init__(self, unknown_cutoff, *counter_args):
        Counter.__init__(self, *counter_args)
        self.cutoff = unknown_cutoff

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff):
        if isinstance(new_cutoff, int): # new_cutoffが辞書ではなくintの場合もあったのでif文追加 2024/4/16
            if new_cutoff < 1:
                msg_template = "Cutoff value cannot be less than 1. Got: {0}"
                raise ValueError(msg_template.format(new_cutoff))
            self._cutoff = new_cutoff
        else:
            for v in new_cutoff.values(): # 辞書のvalueを取り出すようfor文追加 2024/4/9
                if v < 1:
                    msg_template = "Cutoff value cannot be less than 1. Got: {0}"
                    raise ValueError(msg_template.format(new_cutoff))
            self._cutoff = new_cutoff

    def __contains__(self, item):
        """Only consider items with counts GE to cutoff as being in the vocabulary."""
        return self[item] >= self.cutoff

    def __len__(self):
        """これは a)カウントによるフィルタリング, b) "unknown"の考慮 を反映する必要がある

        一つ目はメンバーシップチェックの実装に依存することで実現される
        二つ目はvocabularyサイズに1を足すことで実現される
        """
        # the if-clause here looks a bit dumb, should we make it clearer?
        return sum(1 for item in self if item in self) + 1

    def __copy__(self):
        return self.__class__(self._cutoff, self)


class EmptyVocabularyError(Exception):
    pass


class NgramCounter(object):
    """ngramsをカウントするためのクラス"""

    def __init__(self, order, vocabulary, unk_cutoff=None, unk_label="<UNK>", **ngrams_kwargs):
        """
        :type training_text: List[List[str]]
        """

        if order < 1:
            message = "Order of NgramCounter cannot be less than 1. Got: {0}"
            raise ValueError(message.format(order))

        self.order = order # ngramのnの値(今回は3)
        self.unk_label = unk_label # 存在しないintentにつけるラベル<UNK>を格納する

        # いくつかの共通のdefaultsをプリセットする
        self.ngrams_kwargs = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }
        # While allowing whatever the user passes to override them
        self.ngrams_kwargs.update(ngrams_kwargs)
        # 語彙をセットアップ
        self._set_up_vocabulary(vocabulary, unk_cutoff)

        self.ngrams = defaultdict(ConditionalFreqDist) # ムズイ
        self.unigrams = FreqDist() # ムズイ

    def _set_up_vocabulary(self, vocabulary, unk_cutoff):
        self.vocabulary = copy(vocabulary)  # 状態の共有を防ぐためにコピーが必要
        if unk_cutoff is not None:
            # カットオフ値が指定されている場合は語彙のカットオフをオーバーライドする
            self.vocabulary.cutoff = unk_cutoff

        if self.ngrams_kwargs['pad_left']:
            lpad_sym = self.ngrams_kwargs.get("left_pad_symbol")
            self.vocabulary[lpad_sym] = self.vocabulary.cutoff

        if self.ngrams_kwargs['pad_right']:
            rpad_sym = self.ngrams_kwargs.get("right_pad_symbol")
            self.vocabulary[rpad_sym] = self.vocabulary.cutoff

    def _enumerate_ngram_orders(self):
        return enumerate(range(self.order, 1, -1))

    def train_counts(self, training_text):
        # ここでは "1" が空の語彙を示していることに注意!
        # 詳細は, "NgramModelVocabulary __len__" メソッド参照
        if len(self.vocabulary) <= 1:
            # 語彙が一個以下の場合はngramにならないのでエラー
            raise EmptyVocabularyError("Cannot start counting ngrams until "
                                       "vocabulary contains more than one item.")
        # training_textは対話ごとに分かれたダイアログアクトの並びのリスト
        for i, sent in enumerate(training_text):
            checked_sent = (self.check_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                # ngramには('init-price', 'unknown', 'insist')のようにtri-gramで分割したintentが入っている
                context, word = tuple(ngram[:-1]), ngram[-1] # contextにngramの前半2つ, wordにngramの最後の1つを格納
                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False

                for trunc_index, ngram_order in self._enumerate_ngram_orders():
                    trunc_context = context[trunc_index:]
                    # 上の行の内容は最初の反復ではコンテキストに影響を与えないことに注意
                    self.ngrams[ngram_order][trunc_context][word] += 1
                self.unigrams[word] += 1

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word # 語彙の中にあるintentの場合はそのまま返す
        return self.unk_label # 語彙の中にないintentの場合は<UNK>を返す

    def to_ngrams(self, sequence):
        """初期化中に保存される便利なオプションを含むutil.ngramsのラッパー

        :param sequence: nltk.util.ngramsと同じ
        :type sequence: 任意の反復が可能
        """
        return ngrams(sequence, self.order, **self.ngrams_kwargs)
