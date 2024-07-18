from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from .attention import MultibankGlobalAttention, GlobalAttention, MultibankConcatGlobalAttention

def rnn_factory(rnn_type, **kwargs):
    # 利用可能な場合はpytorch versionを使用する!
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRUはPackedSequenceをサポートしていない
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoderクラス
    様々なエンコーダタイプによって使用され, onmt.Models.NMTModelオブジェクトで必要なインターフェースを指定する

    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`): スパースインデックスの埋め込みシーケンス '[src_len x batch x nfeat]'
            lengths (:obj:`LongTensor`): 各シーケンスの長さ '[batch]'
            encoder_state (rnn-class specific): encoder_stateの初期状態

        Returns:
            (tuple of :obj:'FloatTensor', :obj:'FloatTensor'):
                * デコーダの初期化に使用する最終的なエンコーダの状態
                * attentionのためのmemory bank, '[src_len x batch x hidden]'
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """単純なnon-recurrentエンコーダ
       単純に平均プーリングを適用する

    Args:
       num_layers (int): 複製されたレイヤーの数
       embeddings (:obj:'onmt.modules.Embeddings'): 使用する埋め込みモジュール
       embed_type (:str:): ダイアログ, KB(タイトルと説明用), またはカテゴリのいずれか
    """
    def __init__(self, num_layers, embeddings, embed_type='utterance'):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.embed_type = embed_type

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:'EncoderBase.forward()'"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank

class StdRNNEncoder(EncoderBase):
    """ 一般的なリカレントニューラルネットワークエンコーダ

    Args:
       rnn_type (:obj:'str'): 使用するリカレントユニットのスタイル. 次のうちいずれか → [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : 双方向RNNの使用
       num_layers (int) : 積み重ねるレイヤー数
       hidden_size (int) : 各レイヤーの隠れサイズ
       dropout (float) : 'nn.Dropout'オブジェクトのドロップアウト値
       embeddings (:obj:'onmt.modules.Embeddings'): 使用する埋め込みモジュール
       embed_type (:str:): ダイアログ, KB(タイトルと説明用), またはカテゴリのいずれか
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout=0.0, embeddings=None, embed_type='utterance',
                 use_bridge=False):
        super(StdRNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.embed_type = embed_type

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # ブリッジ層の初期化
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengthsのデータは変数内にラッピングされる
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTMには隠れ状態とセル状態があり, 他は一つだけである
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # 状態の総数
        self.total_hidden_dim = hidden_size * num_layers

        # それぞれに線形レイヤーを構築する
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        ブリッジを介して隠れ状態を転送する
        """
        def bottle_hidden(linear, states):
            """
            3Dから2Dに変換し, 線形(変換)を適用して初期サイズを返す
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class RNNDecoderBase(nn.Module):
    """
    ベースとなるrecurrent attention-based decoderのクラス.
    様々なデコーダータイプで使用され, 'onmt.Models.NMTModel'で必要とされるインターフェースを指定する

    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`): 使用するリカレントユニットのスタイル. 次のうちいずれか → [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : 双方向エンコーダの使用
       num_layers (int) : 積み重ねるレイヤー数
       hidden_size (int) : 各レイヤーの隠れサイズ
       attn_type (str) : 'onmt.modules.GlobalAttention'参照
       coverage_attn (str): 'onmt.modules.GlobalAttention'参照
       context_gate (str): 'onmt.modules.ContextGate'参照
       copy_attn (bool): 別のcopy attentionのメカニズムをセットアップする
       dropout (float) : 'nn.Dropout'オブジェクトのドロップアウト値
       embeddings (:obj:'onmt.modules.Embeddings'): 使用する埋め込みモジュール
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False,
                 pad=None):
        super(RNNDecoderBase, self).__init__()

        # 基本となるattributes
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.pad = pad

        # RNNを構築
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # context gateのセットアップ
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # 標準的なアテンション機構の設定
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # 必要に応じて, 分離されたcopy attention layerを設定する
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def get_final_non_pad_output(self, tgt, outputs):
        # non-pad elementsは1
        mask = torch.eq(tgt, self.pad).long().eq(0).long()  # (seq_len, batch_size)
        last_ind = torch.sum(mask, dim=0, keepdim=True) - 1  # (1, batch_size)
        last_ind = torch.max(last_ind, torch.zeros_like(last_ind))  # (1, batch_size)
        # outputs: (seq_len, batch_size, rnn_size)
        gather_ind = last_ind.unsqueeze(2).expand(1, outputs.size(1), outputs.size(2))
        final_output = torch.gather(outputs, 0, gather_ind).squeeze(0)  # (batch_size, rnn_size)
        return final_output

    def forward(self, tgt, memory_banks, state, memory_lengths=None, lengths=None):
        """
        Args:
            tgt (`LongTensor`): パディングされたトークンのシーケンス '[tgt_len x batch]'
            memory_bank(s) (`FloatTensor`): エンコーダ '[src_len x batch x hidden]' からのベクトル. おそらくベクトルのリスト
            state (:obj:`onmt.Models.DecoderState`): デコーダを初期化するためのデコーダ状態オブジェクト
            memory_lengths (`LongTensor`): パディングされたソースの長さ '[batch]'
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: デコーダーからの出力 (アテンション後) '[tgt_len x batch x hidden]'
                * decoder_state: デコーダからの最終的な隠れ状態
                * attns: 各tgt'[tgt_len x batch x src_len]'でのsrc上の分散
        """
        # Check
        assert isinstance(state, RNNDecoderState)

        tgt_len, tgt_batch = tgt.size()
        if isinstance(memory_banks, list):
            _, memory_batch, _ = memory_banks[0].size()
        else:
            _, memory_batch, _ = memory_banks.size()
        aeq(tgt_batch, memory_batch)
        # END

        # RNNのフォワードパスを実行
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_banks, state, memory_lengths=memory_lengths, lengths=lengths)

        # 結果を使用して状態を更新
        #final_output = decoder_outputs[-1]
        final_output = self.get_final_non_pad_output(tgt, decoder_outputs)

        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # 新しい次元に沿ってテンソルのシーケンスを連結する
        #decoder_outputs = torch.stack(decoder_outputs)
        #for k in attns:
        #    attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # 隠れているエンコーダは'(layers*directions) x batch x dim'で表される
            # これを'layers x batch x (directions*dim)'に変更する必要がある
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
    """
    attentionを伴う, 標準的な完全にバッチ化されたRNNデコーダー
    より高速な実装にはCuDNNを使用する
    optionsについては'RNNDecoderBase'オブジェクト参照

    "Neural Machine Translation By Jointly Learning To Align and Translate"のアプローチに基づいている
    :cite:`Bahdanau2015`

    input_feedingなしで実装されており, 現在は'coverage_attn'と'copy_attn'はサポートしていない
    """
    def _run_forward_pass(self, tgt, memory_banks, state, memory_lengths=None, lengths=None):
        """
        特定のRNNフォワードパスを実行するためのプライベートヘルパー
        全てのサブクラスによってオーバーライドされる必要がある
        Args:
            tgt (LongTensor): 入力トークンのテンソル [len x batch] のシーケンス
            memory_banks (FloatTensor): エンコーダからの出力(テンソルのシーケンス)
                        サイズが(src_len x batch_size x hidden_size)のRNN
                        あるいはメモリーバンクのリストもある
            state (FloatTensor): デコーダを初期化するためのエンコーダRNN型の隠れ状態
            memory_lengths (LongTensor): ソースのメモリバンクの長さ
        Returns:
            decoder_final (Variable): デコーダからの最終的な隠れ状態
            decoder_outputs ([FloatTensor]): デコーダからの各タイムステップの出力の配列
            attns (dict of (str, [FloatTensor]): デコーダからの各タイムステップの異なるタイプのアテンションテンソル配列の辞書
        """
        assert not self._copy  # TODO, まだサポートしていない
        assert not self._coverage  # TODO, まだサポートしていない

        # ローカル変数を初期化して変数を返す
        attns = {}
        emb = self.embeddings(tgt)

        lengths = None
        # TODO: ソートとエンべディング
        packed_emb = emb  # (seq_len, batch_size, emb_size)
        #if lengths is not None:
        #    lengths, ind = torch.sort(lengths, 0, descending=True)  # (batch_size,)
        #    gather_inds = Variable(ind.view(1, -1, 1).expand(*emb.size()))
        #    sorted_emb = torch.gather(emb, 1, gather_inds)
        #    packed_emb = pack(sorted_emb, lengths.view(-1).tolist())

        # RNNのフォワードパスを実行する
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(packed_emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(packed_emb, state.hidden)

        ## TODO: 出力と最終出力の並べ替えを解除する
        #def unsort(t, ind, dim):
        #    gather_inds = Variable(ind.view(1, -1, 1).expand(*t.size()))
        #    t.scatter_(dim, gather_inds, t)
        #    return t

        #if lengths is not None:
        #    rnn_output = unpack(rnn_output)[0]
        #    rnn_output = unsort(rnn_output, ind, 1)
        #    h = unsort(decoder_final[0], ind, 1)
        #    c = unsort(decoder_final[1], ind, 1)
        #    decoder_final = (h, c)

        # Check
        tgt_len, tgt_batch = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # attentionを計算
        if isinstance(memory_banks, list):
            memory_banks = [bank.transpose(0,1) for bank in memory_banks]
            # encoder_memory_bankとprev_context_memory_bankは両方とも (seq_len, batch_size, hidden) → (batch_size, seq_len, hidden)
        else:
            memory_banks = memory_banks.transpose(0,1)

        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(), memory_banks,
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # コンテキストゲートの計算
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        予測されるfeaturesの数を返すプライベートヘルパー
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    入力フィーディングベースのデコーダ. optionsは'RNNDecoderBase'オブジェクト参照

    "Effective Approaches to Attention-based Neural Machine Translation"の入力フィードアプローチに基づく
    :cite:'Luong2015'

    .. mermaid::
       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        引数と戻り値の説明に関してはStdRNNDecoder._run_forward_pass()を参照
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # ローカル変数を初期化して変数を返す
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # 入力フィードは隠れ状態を連結する
        # タイムステップごとに入力する
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: 2つ目のRNNトランスフォーマーの代わりにコンテキストゲートを使用する必要がある
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # coverage attentionの更新
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # copy attention layerのフォワードパス(順方向のパス)を実行する
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # 結果を返す
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        入力をアテンションベクトルと連結することで入力フィードを使用する
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    OpenNMTのコアのトレーニング可能なオブジェクト
    シンプルで汎用的なエンコーダ+デコーダモデルのトレーニング可能なインタフェースを実装する

    Args:
      encoder (:obj:`EncoderBase`): エンコーダオブジェクト
      decoder (:obj:`RNNDecoderBase`): デコーダオブジェクトa decoder object
      multi<gpu (bool): マルチGPUサポートのセットアップ
    """
    def __init__(self, encoder, decoder, multigpu=False, stateful=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.stateful = stateful

    def forward(self, src, tgt, lengths, dec_state=None, enc_state=None, tgt_lengths=None):
        """
        トレーニング用に'src'と'tgt'のペアを順伝搬する
        デコーダの開始状態で初期化される可能性がある

        Args:
            src (:obj:`Tensor`):
                エンコーダに渡されるソースシーケンス
                通常, 入力の場合, これはサイズ'[len x batch x features]'のパディングされた'LongTensor'オブジェクトになる
                ただし, エンコーダによってはその他の一般的な入力になる場合がある
            tgt (:obj:`LongTensor`): サイズ'[tgt_len x batch]'のターゲットシーケンス
            lengths(:obj:`LongTensor`): '[batch]'を事前にパディングしたsrcの長さ
            dec_state (:obj:`DecoderState`, optional): デコーダの初期状態
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):
                 * デコーダ出力 '[tgt_len x batch x hidden]'
                 * '[tgt_len x batch x src_len]'の辞書アテンション
                 * 最終的なデコーダの状態
        """
        # tgt = tgt[:-1]  本来, これは最後のターゲット(a <EOS> token)を除外する
        # デコーダ入力からのものだが, 前処理ですでにこれを処理している

        enc_final, memory_bank = self.encoder(src, lengths, enc_state)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths,
                         lengths=tgt_lengths)
        if self.multigpu:
            # TODO: マルチGPUではまだサポートされていない
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class DecoderState(object):
    """
    リカレントデコーダの現在の状態をグループ化するためのインターフェース
    最も単純なケースでは, モデルの非表示状態を表すだけ
    ただし, 様々な形式のinput_feedモデルやnon-recurrentモデルの実装に使用できる

    デコーディングを利用するには, モジュールでこれを実装する必要がある
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): デコーダの隠れ層のサイズ
            rnnstate: エンコーダからの最終的な隠れ状態. 次の形に変換: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # 入力フィードを開始する
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ バッチ次元に沿って, beam_size回繰り返す """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

class MultiAttnDecoder(StdRNNDecoder):

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
            hidden_size, attn_type="general", coverage_attn=False,
            context_gate=None, copy_attn=False, dropout=0.0,
            embeddings=None, reuse_copy_attn=False, pad=None):
        attn_type = attn_type[10:]
        super(MultiAttnDecoder, self).__init__(rnn_type, bidirectional_encoder,
              num_layers, hidden_size, attn_type, coverage_attn,
              context_gate, copy_attn, dropout, embeddings, reuse_copy_attn, pad)

        self.attn = MultibankGlobalAttention(
            hidden_size, coverage=coverage_attn, attn_type=attn_type)

class MultiAttnConcatDecoder(StdRNNDecoder):

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
            hidden_size, memory_sizes, attn_type="general", coverage_attn=False,
            context_gate=None, copy_attn=False, dropout=0.0,
            embeddings=None, reuse_copy_attn=False, pad=None):
        attn_type = attn_type[10:]
        super(MultiAttnConcatDecoder, self).__init__(rnn_type, bidirectional_encoder,
              num_layers, hidden_size, attn_type, coverage_attn,
              context_gate, copy_attn, dropout, embeddings, reuse_copy_attn, pad)

        self.attn = MultibankConcatGlobalAttention(
            hidden_size, memory_sizes, coverage=coverage_attn, attn_type=attn_type)

