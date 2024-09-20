"""
このファイルはモデル作成用であり, optionsを参照する.
それに応じて各エンコーダとデコーダを作成する.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator
from onmt.Utils import use_gpu

from cocoa.neural.models import MeanEncoder, StdRNNEncoder, StdRNNDecoder, \
              MultiAttnDecoder, NMTModel
from .models import NegotiationModel

from cocoa.io.utils import read_pickle

from .symbols import markers
from neural import make_model_mappings


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def make_embeddings(opt, word_dict, for_encoder=True):
    """
    Embeddingsのインスタンスを生成する
    Args:
        opt: 現在の環境でのオプション
        word_dict(Vocabulary): 単語辞書
        for_encoder(bool): エンコーダかデコーダの埋め込みを作成するかどうか
    """
    embedding_dim = opt.word_vec_size

    word_padding_idx = word_dict.to_ind(markers.PAD)
    num_word_embeddings = len(word_dict)

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=False,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      word_vocab_size=num_word_embeddings)


def make_encoder(opt, embeddings):
    """
    各種エンコーダディスパッチャーの関数
    Args:
        opt: 現在の環境のoption
        embeddings (Embeddings): このエンコーダの語彙埋め込み
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        bidirectional = True if opt.encoder_type == 'brnn' else False
        return StdRNNEncoder(opt.rnn_type, bidirectional, opt.enc_layers,
                        opt.rnn_size, opt.dropout, embeddings)

def make_context_embedder(opt, embeddings, embed_type='utterance'):
    """
    各種コンテキストエンベッダーディスパッチャの関数
    optionsについてはmake_encoder参照
    embed_type (:str:): either dialogue, kb (for title and desc) or category
    """
    if opt.context_embedder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings, embed_type)
    else:
        # "rnn" or "brnn"
        bidirectional = True if opt.context_embedder_type == 'brnn' else False
        return StdRNNEncoder(opt.rnn_type, bidirectional, opt.enc_layers,
                        opt.rnn_size, opt.dropout, embeddings, embed_type, False)

def make_decoder(opt, embeddings, tgt_dict):
    """
    各種デコーダディスパッチャの関数
    Args:
        opt: 現在の環境のoption
        embeddings (Embeddings): このデコーダの語彙埋め込み
    """
    bidirectional = True if opt.encoder_type == 'brnn' else False
    pad = tgt_dict.to_ind(markers.PAD)
    if "multibank" in opt.global_attention:
        return MultiAttnDecoder(opt.rnn_type, bidirectional,
                             opt.dec_layers, opt.rnn_size,
                             attn_type=opt.global_attention,
                             dropout=opt.dropout,
                             embeddings=embeddings,
                             pad=pad)
    elif opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, bidirectional,
                                   opt.dec_layers, opt.rnn_size,
                                   attn_type=opt.global_attention,
                                   dropout=opt.dropout,
                                   embeddings=embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, bidirectional,
                             opt.dec_layers, opt.rnn_size,
                             attn_type=opt.global_attention,
                             dropout=opt.dropout,
                             embeddings=embeddings,
                             pad=pad)

def load_test_model(model_path, opt, dummy_opt):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    for attribute in ["share_embeddings", "stateful"]:
        if not hasattr(model_opt, attribute):
            model_opt.__dict__[attribute] = False

    # TODO: これは直す必要あり
    if model_opt.stateful and not opt.sample:
        raise ValueError('Beam search generator does not work with stateful models yet')

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))

    # mappings = read_pickle('{0}/{1}/vocab.pkl'.format(model_opt.mappings, model_opt.model))
    mappings = make_model_mappings(model_opt.model, mappings)

    model = make_base_model(model_opt, mappings, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return mappings, model, model_opt

def make_base_model(model_opt, mappings, gpu, checkpoint=None):
    """
    Args:
        model_opt: チェックポイントから読み込まれたオプション
        fields: モデルのための`Field`オブジェクト
        gpu(bool): GPUを使うかどうか
        checkpoint: trainフェーズによって生成されたモデル, 
                    または停止したtrainingから再開されたスナップショットモデル
    Returns:
        the NMTModel.
    """
    # エンコーダの作成
    src_dict = mappings['src_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict)
    encoder = make_encoder(model_opt, src_embeddings)

    # コンテキストエンベッダーの作成
    if model_opt.num_context > 0:
        context_dict = mappings['utterance_vocab']
        context_embeddings = make_embeddings(model_opt, context_dict)
        context_embedder = make_context_embedder(model_opt, context_embeddings)

    # kbエンベッダーの作成
    if "multibank" in model_opt.global_attention:
        if model_opt.model == 'lf2lf':
            kb_embedder = None
        else:
            kb_dict = mappings['kb_vocab']
            kb_embeddings = make_embeddings(model_opt, kb_dict)
            kb_embedder = make_context_embedder(model_opt, kb_embeddings, 'kb')

    # デコーダの作成
    tgt_dict = mappings['tgt_vocab']
    tgt_embeddings = make_embeddings(model_opt, tgt_dict)

    # 埋め込み行列を共有する - share_vocabによる前処理が必要
    if model_opt.share_embeddings:
        # `-share_vocab`が指定されている場合, src/tgt vocabは同じである必要がある
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings, tgt_dict)

    if "multibank" in model_opt.global_attention:
        model = NegotiationModel(encoder, decoder, context_embedder, kb_embedder, stateful=model_opt.stateful)
    else:
        model = NMTModel(encoder, decoder, stateful=model_opt.stateful)

    model.model_type = 'text'

    # ジェネレーターの作成
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(tgt_dict)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size, fields["tgt"].vocab)

    # モデルの状態をチェックポイントからロードするか, 初期化する
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

        wordvec = {'utterance': model_opt.pretrained_wordvec[0]}
        if len(model_opt.pretrained_wordvec) > 1:
            wordvec['kb'] = model_opt.pretrained_wordvec[1]

        def load_wordvec(embeddings, name):
            embeddings.load_pretrained_vectors(
                    wordvec[name], model_opt.fix_pretrained_wordvec)

        # LFsには事前学習された単語ベクトルは必要ない
        if not model_opt.model in ('lf2lf',):
            load_wordvec(model.encoder.embeddings, 'utterance')
            if hasattr(model, 'context_embedder'):
                load_wordvec(model.context_embedder.embeddings, 'utterance')
        if hasattr(model, 'kb_embedder') and model.kb_embedder is not None:
            load_wordvec(model.kb_embedder.embeddings, 'kb')

        if model_opt.model == 'seq2seq':
            load_wordvec(model.decoder.embeddings, 'utterance')

    # モデルにジェネレータの追加(これはモデルのパラメータとして登録される).
    model.generator = generator

    # GPUが指定されている場合はモデル全体でGPUを使用する
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
