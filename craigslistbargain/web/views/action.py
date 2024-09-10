from flask import Blueprint, jsonify, request
from cocoa.web.views.utils import userid, format_message

from web.main.backend import Backend
get_backend = Backend.get_backend

action = Blueprint('action', __name__)

@action.route('/_offer/', methods=['GET'])
def offer():
    backend = get_backend()
    price = float(request.args.get('price'))
    sides = request.args.get('sides')

    offer = {'price': price,
             'sides': sides}

    if offer is None or price == -1:
        return jsonify(message=format_message("無効なオファーです! 再度入力してください。", True))
    backend.make_offer(userid(), offer)

    displayed_message = format_message("オファーを受け取りました!", True)
    return jsonify(message=displayed_message)


@action.route('/_accept_offer/', methods=['GET'])
def accept_offer():
    backend = get_backend()
    backend.accept_offer(userid())

    msg = format_message("オファーを受け入れました!", True)
    return jsonify(message=msg)


@action.route('/_reject_offer/', methods=['GET'])
def reject_offer():
    backend = get_backend()
    backend.reject_offer(userid())

    msg = format_message("オファーを拒否しました!", True)
    return jsonify(message=msg)


@action.route('/_quit/', methods=['GET'])
def quit():
    backend = get_backend()
    backend.quit(userid())
    displayed_message = format_message("このタスクの終了を選択しました。", True)
    return jsonify(message=displayed_message)
