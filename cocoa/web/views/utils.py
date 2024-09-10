import uuid
from datetime import datetime
from flask import request, g

def generate_userid(prefix="U_"):
    return prefix + uuid.uuid4().hex

def userid():
    return request.args.get('uid')

def format_message(message, status_message):
    """メッセージの文字列のフォーマット

    Args:
        message (str)
        status_message (bool): メッセージが action (例.select) なのか utterance なのか
    """
    timestamp = datetime.now().strftime(u'%x %X')
    left_delim = u"<" if status_message else u""
    right_delim = u">" if status_message else u""
    return u"[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)
