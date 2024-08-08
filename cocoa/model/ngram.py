# Natural Language Toolkit: Language Models
#
# Copyright (C) 2001-2016 NLTK Project
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import unicode_literals, division
from math import log

from nltk import compat
from cocoa.model.util import safe_div


NEG_INF = float("-inf")


class BaseNgramModel(object):
    """
    NgramCounterを使用して言語モデルを作成する方法の例

    このクラスは直接使用することはできない
    独自のngramモデルを作成するときに継承して使用する
    """

    def __init__(self, ngram_counter):

        self.ngram_counter = ngram_counter
        # for convenient access save top-most ngram order ConditionalFreqDist
        self.ngrams = ngram_counter.ngrams[ngram_counter.order]
        self._ngrams = ngram_counter.ngrams
        self._order = ngram_counter.order

    def _check_against_vocab(self, word):
        return self.ngram_counter.check_against_vocab(word)

    @property
    def order(self):
        return self._order

    def check_context(self, context):
        """コンテキストがモデルのngramよりも長くなく、タプルであることを確認"""
        if len(context) >= self._order:
            raise ValueError("Context is too long for this ngram order: {0}".format(context))
        # ensures the context argument is a tuple
        return tuple(context)

    def score(self, word, context):
        """
        これはダミー実装
        子クラスは独自の実装を定義する必要がある

        :param word: 確率を取得する単語
        :type word: str
        :param context: 単語が含まれるコンテキスト
        :type context: Tuple[str]
        """
        return 0.5

    def logscore(self, word, context):
        """
        このコンテキストでのこの単語のlog確率を評価する

        この実装はこのままで機能する
        子クラスで再定義する必要はない

        :param word: 確率を取得する単語
        :type word: str
        :param context: 単語が含まれるコンテキスト
        :type context: Tuple[str]
        """
        score = self.score(word, context)
        if score == 0.0:
            return NEG_INF
        return log(score, 2)

    def entropy(self, text, average=True):
        """
        与えられた評価テキストのn-gramモデルの近似クロスエントロピーを計算する
        これはテキスト内の各単語の平均対数確率である

        :param text: 評価に使用する単語
        :type text: Iterable[str]
        """

        normed_text = (self._check_against_vocab(word) for word in text)
        H = 0.0     # entropy is conventionally denoted by "H"
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            H += self.logscore(word, context)
            processed_ngrams += 1
        if processed_ngrams == 0:
            H = 0.
        if average:
            return -1. * safe_div(H, processed_ngrams)
        else:
            return -1. * H, processed_ngrams

    def perplexity(self, text):
        """
        指定されたテキストのperplexityを計算する
        これは単純にテキストの 2 ** cross-entropy である

        :param text: perplexityを計算する単語
        :type text: Iterable[str]
        """

        return pow(2.0, self.entropy(text))


class MLENgramModel(BaseNgramModel):
    """MLEngramModelのスコアを提供するクラス

    BaseNgramModelから初期化方法を継承している
    """

    def score(self, word, context):
        """Returns the MLE score for a word given a context.

        Args:
        - wordは文字列であることが期待される
        - contextはタプルに合理的に変換できるものであることが期待される
        """
        context = self.check_context(context)
        dist = self._ngrams[len(context)+1][context]
        # TODO: backoff
        return dist.freq(word)

    def freqdist(self, context):
        context = self.check_context(context)
        dist = self._ngrams[len(context)+1][context]
        return dist.items()

class LidstoneNgramModel(BaseNgramModel):
    """
    Lidstone-smoothedスコアを算出する

    BaseNgramModelから初期化引数に加えて、カウントを増やす数値、ガンマが必要
    """

    def __init__(self, gamma, *args):
        super(LidstoneNgramModel, self).__init__(*args)
        self.gamma = gamma
        # This gets added to the denominator to normalize the effect of gamma
        self.gamma_norm = len(self.ngram_counter.vocabulary) * gamma

    def score(self, word, context):
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        ctx_count = context_freqdist.N()
        return (word_count + self.gamma) / (ctx_count + self.gamma_norm)


#@compat.python_2_unicode_compatible
class LaplaceNgramModel(LidstoneNgramModel):
    """
    ラプラス(1を加算)平滑化を実装

    ガンマは常に1なので初期化はBaseNgramModelと同じ
    """

    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)



#####################################################
if __name__ == '__main__':
    from counter import build_vocabulary, count_ngrams
    sents = [['a', 'b', 'c'], ['a', 'c', 'c']]
    vocab = build_vocabulary(1, *sents)
    counter = count_ngrams(2, vocab, sents)
    model = MLENgramModel(counter)
    print(model.score('b', ('a',)))
    print(model.order)
