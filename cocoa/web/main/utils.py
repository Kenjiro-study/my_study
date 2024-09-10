__author__ = 'anushabala'
import time
import datetime
#import src.config as config


class Status(object):
    Waiting = "waiting"
    Chat = "chat"
    Finished = "finished"
    Survey = "survey"
    Redirected = "redirected"
    Incomplete = "incomplete"
    Reporting = "reporting"


class UnexpectedStatusException(Exception):
    def __init__(self, found_status, expected_status):
        self.expected_status = expected_status
        self.found_status = found_status


class ConnectionTimeoutException(Exception):
    pass


class InvalidStatusException(Exception):
    pass


class StatusTimeoutException(Exception):
    pass


class NoSuchUserException(Exception):
    pass


class Messages(object):
    ChatExpired = '時間切れです!'
    PartnerConnectionTimeout = "パートナーの接続がタイムアウトしました。 新しいチャットに接続します..."
    ConnectionTimeout = "接続がタイムアウトしました。元のURLを使用してこのWebサイトに再度アクセスしてください。" \
                        "新しいチャットを開始できます。"
    YouLeftRoom = 'チャットをスキップしました!'
    PartnerLeftRoom = 'パートナーがチャットから離れました!'
    WaitingTimeExpired = "申し訳ありません。現在アクティブなユーザーが他にいないようです。しばらくしてから再度アクセスしてください。"
    ChatCompleted = "チャットが全て完了しました!"
    ChatIncomplete = ConnectionTimeout
    HITCompletionWarning = "HITクレジットは意欲的に交渉を行った場合のみ付与されます。ご了承ください。"
    Waiting = '新しいチャットが始まるまで今しばらくお待ちください...'


def current_timestamp_in_seconds():
    return int(time.mktime(datetime.datetime.now().timetuple()))


class User(object):
    def __init__(self, row):
        self.name = row[0]
        self.status = row[1]
        self.status_timestamp = row[2]
        self.connected_status = row[3]
        self.connected_timestamp = row[4]
        self.message = row[5]
        self.partner_type = row[6]
        self.partner_id = row[7]
        self.scenario_id = row[8]
        self.agent_index = row[9]
        self.selected_index = row[10]
        self.chat_id = row[11]
