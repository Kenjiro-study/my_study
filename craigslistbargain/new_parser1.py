import argparse
import copy

from cocoa.core.dataset import read_examples
from cocoa.model.manager import Manager
from cocoa.analysis.utils import intent_breakdown
from cocoa.io.utils import write_json

from core.event import Event
from core.scenario import Scenario
from core.price_tracker import PriceTracker
from neural.preprocess import Preprocessor
from model.new_parser2 import Parser ##### お試し用にparserをnew_parser2参照に変更
from model.dialogue_state import DialogueState
from model.generator import Templates, Generator

def parse_example(example, lexicon, templates, flag, path=None): # (example, price_tracker, templates, arg.neural-flag, arg.neural-parser)
    """exampleを解析し, templatesを収集する
    """
    kbs = example.scenario.kbs
    # エージェントごとのパーサー(解析器)と状態を定義
    parsers = [Parser(agent, kbs[agent], lexicon, flag, path) for agent in (0, 1)]
    states = [DialogueState(agent, kbs[agent]) for agent in (0, 1)]
    # 最初の発話としてintent及び文に<start>を追加する
    parsed_utterances = [states[0].utterance[0], states[1].utterance[1]]

    events = example.events

    # textとpre_textをリストで取得
    if flag == True:
        text_list = two_text_get(events)
    
    # 発話を一つずつ解析する
    for i in range(len(events)):
        writing_agent = events[i].agent  # 話し手
        reading_agent = 1 - writing_agent # 聞き手
        # print(event.agent)

        if flag == True:
            received_utterance = parsers[reading_agent].parse(events[i], states[reading_agent], text_list[i]) # DLベースの方はtextと　pre_textの辞書を持っていく
        else:
            received_utterance = parsers[reading_agent].parse(events[i], states[reading_agent]) # 発話文, ダイアログアクト, テンプレートの三つをまとめて作成
        
        if received_utterance:
            events[i].metadata = received_utterance.lf # parseによって生成されたlf属性のものをmetadata(ダイアログアクト)にする
            sent_utterance = copy.deepcopy(received_utterance) # received_utteranceのコピーをsent_utteranceに作成
            if sent_utterance.tokens:
                sent_utterance.template = parsers[writing_agent].extract_template(sent_utterance.tokens, states[writing_agent]) # partner_priceとmy_priceを入れ替えて送る側と受け取る側のテンプレートを作る

            templates.add_template(sent_utterance, states[writing_agent])
            parsed_utterances.append(received_utterance)
            #print('sent:', ' '.join(sent_utterance.template))
            #print('received:', ' '.join(received_utterance.template))

            # statesのアップデート
            states[reading_agent].update(writing_agent, received_utterance)
            states[writing_agent].update(writing_agent, sent_utterance)
            
    return parsed_utterances

def two_text_get(events):
    text_list = [] # textとpre_textの辞書を格納するリスト
    for i in range(len(events)):
        text = events[i].data # テキストを取り出す
        if i == 0:
            pre_text = "[PAD]"
        else:
            if type(events[i-1].data) is str:
                pre_text = events[i-1].data # 一つ前の発話を取得
            else:
                if type(events[i-2].data) is str:
                    pre_text = events[i-2].data
                else:
                    pre_text = "[PAD]"
            
        text_list.append({'text':text, 'pre_text':pre_text}) # リストに追加
    
    return text_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates') # 学習・検証データ
    parser.add_argument('--transcripts-output', help='JSON transcripts of parsed dialogues') # 解析済み学習・検証データ
    parser.add_argument('--price-tracker', help='The price tracker recognizes price mentions in an utterance') # プライストラッカー
    parser.add_argument('--max-examples', default=-1, type=int) # -1で初期化
    parser.add_argument('--templates', help='Path to load templates') # なぜかあらかじめtemplatesを読み込むこともできる(まだ作ってないのに...)
    parser.add_argument('--templates-output', help='Path to save templates') # テンプレートの出力先
    parser.add_argument('--model-output', help='Path to save the dialogue manager model') # モデルの出力先
    parser.add_argument('--parserpath', help='Path of the deep learning-based parser you created') ##### new!! ディープラーニングベースパーサーのパス #####
    parser.add_argument('--neuralflag', action='store_true', help='which parser do you use, neural-base or rule-base') ###### new!! ルールベース, ニューラルベースどちらのパーサーを使用するか #####
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker)
    examples = read_examples(args.transcripts, args.max_examples, Scenario) # 学習・検証データを読み込む
    parsed_dialogues = [] # 解析した対話を格納するための配列
    templates = Templates() # テンプレートのインスタンスを作成
    flag = args.neuralflag # ルールベース, DLベースどちらを使うかを判別するフラグ

    for example in examples:
        if Preprocessor.skip_example(example): # このスキップ文があるからNanがあったのか！
            continue

        # DLベースはモデル名が引数に必要なので一旦ここで分岐
        if flag == True:
            model_path = args.parserpath
            utterances = parse_example(example, price_tracker, templates, flag, model_path) ##### flagとモデル名を追加 #####
        else:
            utterances = parse_example(example, price_tracker, templates, flag) ##### flag追加 #####
        
        parsed_dialogues.append(utterances)

    #for d in parsed_dialogues[:2]: # parse_dialoguesの中身を2対話だけ確認
        #for u in d:
            #print(u)
    #import sys; sys.exit()

    if args.transcripts_output:
        write_json([e.to_dict() for e in examples], args.transcripts_output) # 解析したデータを出力

    # n-gram modelの学習
    sequences = []
    for d in parsed_dialogues:
        sequences.append([u.lf.intent for u in d]) # 一対話ごとにintentをsequences配列に格納
        # こんな感じ → [['<start>', '<start>', 'init-price', 'unknown', 'insist', 'counter-price', 'counter-price', 'counter-price', 'agree', 'offer', 'accept']]
    manager = Manager.from_train(sequences)
    manager.save(args.model_output) # model.pklに保存

    templates.finalize() # テンプレートに処理を施して完成させる
    templates.save(args.templates_output) # templates.pklに保存

    #templates.dump(n=10) # 指定された状況に合わせた文をテンプレートからn個作成する(ジェネレータの模擬使用?)
    #intent_breakdown(parsed_dialogues) # 各intentの出現回数とその全体に占める割合を計算し, 出力

    # modelとgeneratorのテストをする
    generator = Generator(templates)
    action = manager.choose_action(None, context=('intro', 'init-price'))
    #print(action)
    print(generator.retrieve('intro', context_tag='init-price', tag=action, category='car', role='seller').template)