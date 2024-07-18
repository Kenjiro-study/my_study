from argparse import ArgumentParser

from cocoa.core.util import write_json
from analysis.visualizer import Visualizer
from analysis.html_visualizer import HTMLVisualizer

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--survey-transcripts', nargs='+', help='Path to directory containing evaluation transcripts') # 評価transcriptsを含むディレクトリへのパス
    parser.add_argument('--dialogue-transcripts', nargs='+', help='Path to directory containing dialogue transcripts') # 対話transcriptsを含むディレクトリへのパス
    parser.add_argument('--summary', default=False, action='store_true', help='Summarize human ratings') # 人間の評価の要約
    parser.add_argument('--html-visualize', action='store_true', help='Output html files') # HTMLファイルの出力
    parser.add_argument('--outdir', default='.', help='Output dir') # 出力用ディレクトリ
    parser.add_argument('--stats', default='stats.json', help='Path to stats file') # 統計ファイルへのパス
    parser.add_argument('--partner', default=False, action='store_true', help='Whether this is from partner survey') # これがパートナーのsurveyによるものかどうか
    parser.add_argument('--task', default='cl-neg', choices=['cl-neg','fb-neg', 'mutual', 'movies'], help='which task you are trying run') # 実行しようとしているタスク
    parser.add_argument('--worker-ids', nargs='+', help='Path to json file containing chat_id to worker_id mappings') # chat_idからworker_idへのマッピングを含むjsonファイルへのパス
    parser.add_argument('--hist', default=False, action='store_true', help='Plot histgram of ratings') # 評価のヒストグラムをプロットする
    parser.add_argument('--survey-only', default=False, action='store_true', help='Only analyze dialogues with survey (completed)') # surveyによる対話のみを分析
    parser.add_argument('--base-agent', default='human', help='Agent to compare against') # 比較するエージェント

    HTMLVisualizer.add_html_visualizer_arguments(parser)
    args = parser.parse_args()

    visualizer = Visualizer(args.dialogue_transcripts, args.survey_transcripts)
    results = visualizer.compute_effectiveness(with_survey=args.survey_only, base_agent=args.base_agent)
    visualizer.print_results(results)

    if args.hist:
        visualizer.hist(question_scores, args.outdir, partner=args.partner)
    if args.worker_ids:
        visualizer.worker_stats()

    # TODO: 概要(summary)と履歴(hist)をアナライザーに移動
    if args.summary:
        summary = visualizer.summarize()
        write_json(summary, args.stats)
    if args.html_output:
        visualizer.html_visualize(args.viewer_mode, args.html_output,
            css_file=args.css_file, img_path=args.img_path,
            worker_ids=visualizer.worker_ids)
