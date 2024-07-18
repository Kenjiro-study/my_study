"""
このファイルは, トレーニング中のロス関数の詳細を処理する

これには, LossComputeBase, 標準のNMTLossCompute,  分割化された損失関数計算機能が含まれる
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io


class LossComputeBase(nn.Module):
    """
    効率的な損失関数を管理するためのクラス
    次のステップ予測のshardingと複数の損失計算のaccumulatingを扱う

    ユーザーはこのクラスのサブクラスを作成することで, 独自の損失計算戦略を実装できる
    ユーザーは_compute_loss()メソッドとmake_shard_state()メソッドを実装する必要がある

    Args:
        generator (:obj:'nn.Module') :
             デコーダーの出力をターゲット語彙にわたる分布にマッピングするモジュール
        tgt_vocab (:obj:'Vocab') :
             ターゲット出力正規化を表すtorchtext語彙オブジェクト(str): "sents" または "tokens" によって正規化する
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        効率的な損失計算のために反復可能なshardsを返すように, shards()のshard状態の辞書を作成する
        サブクラスは独自に作成した_compute_loss()インタフェースと一致するようにこのメソッドを定義する必要がある
        Args:
            batch: 現在のバッチ
            output: モデルからの予測出力
            range_: 計算用のexamplesの範囲. バッチ全体か一部のどちらか
            attns: モデルから返されたアテンションの辞書
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        損失の計算. サブクラスはこのメソッドを定義する必要がある

        Args:
            batch: 現在のバッチ
            output: モデルからの予測出力
            target: 出力を比較する検証用ターゲット
            **kwargs(optional): 損失計算に関する追加情報
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        バッチの順方向の損失計算

        Args:
          batch (batch): ラベル付のexamplesのバッチ
          output (:obj:'FloatTensor'): デコーダーモデル'[tgt_len x batch x hidden]'の出力
          attns (dict of :obj:'FloatTensor') : アテンションの分布の辞書 '[tgt_len x batch x src_len]'

        Returns:
            :obj:'onmt.Statistics': 損失統計
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns, cur_trunc, trunc_size, shard_size, normalization):
        """
        順方向の損失計算をして逆伝播する
        計算はshardsを用いて実行され,  オプションでメモリ効率を高めるために切り捨てられる

        また, デコーダー出力シーケンスの範囲を取得して逆伝播することにより, 長いシーケンスの切り捨てられたBPTTもサポートする
        範囲は'(cur_trunc, cur_trunc + trunc_size)'からである

        shardingは生成バッファに必要なメモリを軽減する正確な効率化の仕組みであることに注意!
        TruncationはRNNバッファに必要なメモリを軽減するためのおおよその効率化の仕組み 

        Args:
          batch (batch) : ラベル付examplesのバッチ
          output (:obj:'FloatTensor') : デコーダーモデル'[tgt_len x batch x hidden]' の出力
          attns (dict) : アテンション分布の辞書'[tgt_len x batch x src_len]'
          cur_trunc (int) : truncation windowの開始位置
          trunc_size (int) : truncation windowの長さ
          shard_size (int) : shard内のexamplesの最大数
          normalization (int) : 損失をこの数値で割る

        Returns:
            :obj:`onmt.Statistics`: 検証用損失の統計
        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)

            loss.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:'FloatTensor'): loss criterionによって計算された損失
            scores (:obj:'FloatTensor'): 可能な各出力のスコア
            target (:obj:'FloatTensor'): 真のターゲット

        Returns:
            :obj:'Statistics' : このバッチの統計
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    標準的なNMT損失計算
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)

        if label_smoothing > 0:
            """
            ラベルの平滑化がonになっている場合, q_{平滑化されたグラウンドトゥルース確率}(w)とp_{モデルによって算出された確率}(w)の間のKL-divergenceは最小化される
            ラベルの平滑化の値がゼロに設定されている場合, 損失は NLLLoss か CrossEntropyLoss と同じになる
            全ての非真のラベルは一律に低信頼度に設定される
            """
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            loss_data = - likelihood.sum(0)
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: *LossCompute._make_shard_state()の出力に対応する辞書
               これらのキーの値はテンソルかNone
        shard_size: モデルによって生成されるshardsの最大サイズ
        eval: Trueの場合, 状態のみを生成し, 他は何も生成しない
              それ以外の場合はshardsが生成される

    Yields:
        生成された各shardは辞書型

    Side effect:
        最後のshardの後, この関数は逆伝播を実行する
    """
    if eval:
        yield state
    else:
        # non_none: 値がNondeではないstate辞書のsubdict
        non_none = dict(filter_shard_state(state))

        # さて, iteration: stateはテンソルのようなシーケンスの辞書だが, テンソルの辞書のシーケンスが必要
        # まず, 辞書のキーのシーケンスとテンソルのようなシーケンスのシーケンスに解凍する
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        """
        次に, 各shardの辞書を生成する
        キーは常に同じである
        値は長さのkeysのシーケンスであり, 各要素は長さのshardsのシーケンスである
        ここではキーではなく, shardsを反復処理したい: 
        従って, 値をshardごとに再圧縮してから, 各shardをキーとペアにする必要がある
        """
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # backprop'dを想定
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
