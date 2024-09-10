import sqlite3
import json
import time

from cocoa.web.main.backend import Backend as BaseBackend
from cocoa.web.main.backend import DatabaseManager as BaseDatabaseManager
from cocoa.web.main.utils import Status, Messages
from cocoa.web.views.utils import format_message

from analysis.utils import reject_transcript
from .db_reader import DatabaseReader
from core.event import Event

class DatabaseManager(BaseDatabaseManager):
    @classmethod
    def add_survey_table(cls, cursor):
        cursor.execute(
            '''CREATE TABLE survey (name text, chat_id text, partner_type text, fluent integer,
            honest integer, persuasive integer, fair integer, negotiator integer, coherent integer, comments text)''')

    @classmethod
    def init_database(cls, db_file):
        super().init_database(db_file) # 3系ver.
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE bot (chat_id text, type text, config text)'''
        )
        cls.add_survey_table(c)
        conn.commit()
        conn.close()
        return cls(db_file)


class Backend(BaseBackend):
    def display_received_event(self, event):
        if event.action == 'offer':
            message = format_message("パートナーがオファーを出しました。右側でオファーを確認し, accept または reject を選択してください。", True)
            return {'message': message, 'status': False, 'price': event.data['price']}
        elif event.action == 'accept':
            message = format_message("パートナーがあなたのオファーを受け入れました!", True)
            return {'message': message, 'status': False}
        elif event.action == 'reject':
            message = format_message("残念ですが, パートナーがあなたのオファーを拒否しました.", True)
            return {'message': message, 'status': False}
        else:
            return super().display_received_event(event) # 3系ver.

    def should_reject_chat(self, userid, agent_idx, min_tokens):
        with self.conn:
            controller = self.controller_map[userid]
            cursor = self.conn.cursor()
            chat_id = controller.get_chat_id()
            ex = DatabaseReader.get_chat_example(cursor, chat_id, self.scenario_db).to_dict()
            return reject_transcript(ex, agent_idx, min_tokens=min_tokens)

    def check_game_over_and_transition(self, cursor, userid, partner_id):
        agent_idx = self.get_agent_idx(userid)
        game_over, game_complete = self.is_game_over(userid)
        controller = self.controller_map[userid]
        chat_id = controller.get_chat_id()

        def verify_chat(userid, agent_idx, is_partner, min_tokens=40):
            user_name = 'partner' if is_partner else 'user'
            if self.should_reject_chat(userid, agent_idx, min_tokens):
                self.logger.debug("Rejecting chat with ID {:s} for {:s} {:s} (agent ID {:d}), and "
                                  "redirecting".format(chat_id, user_name, userid, agent_idx))
                self.end_chat_and_redirect(cursor, userid,
                                           message=self.messages.Redirect + " " + self.messages.Waiting)
            else:
                msg, _ = self.get_completion_messages(userid)
                self.logger.debug("Accepted chat with ID {:s} for {:s} {:s} (agent ID {:d}), and redirecting to "
                                  "survey".format(chat_id, user_name, userid, agent_idx))
                self.end_chat_and_finish(cursor, userid, message=msg)

        if game_over:
            if not self.is_user_partner_bot(cursor, userid):
                min_tokens = 40
                verify_chat(partner_id, 1 - agent_idx, True, min_tokens=min_tokens)
            else:
                min_tokens = 30
            verify_chat(userid, agent_idx, False, min_tokens=min_tokens)
            return True

        return False

    def get_completion_messages(self, userid):
        """
        二つの完了メッセージを返す: 1つは現在のユーザー用, もう一つはユーザーのパートナー用
        この関数はユーザーのパートナーがbotかどうかをチェックしない
        交渉における勝者がどちらかを決定し, それに応じて完了のメッセージを割り当てることだけをする
        :param userid:
        :return:
        """

        _, game_complete = self.is_game_over(userid)
        if game_complete:
            msg = self.messages.ChatCompleted
            partner_msg = msg
        else:
            msg = self.messages.ChatIncomplete
            partner_msg = msg

        return msg, partner_msg

    def make_offer(self, userid, offer):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                self._update_user(cursor, userid, connected_status=1)
                self.send(userid, Event.OfferEvent(u.agent_index,
                                                   offer,
                                                   str(time.time())))
        except sqlite3.IntegrityError:
            print("注意!: Rolled back transaction")
            return None

    def accept_offer(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                self._update_user(cursor, userid, connected_status=1)
                self.send(userid, Event.AcceptEvent(u.agent_index,
                                                   str(time.time())))
        except sqlite3.IntegrityError:
            print("注意!: Rolled back transaction")
            return None

    def reject_offer(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                self._update_user(cursor, userid, connected_status=1)
                self.send(userid, Event.RejectEvent(u.agent_index,
                                                   str(time.time())))
        except sqlite3.IntegrityError:
            print("注意!: Rolled back transaction")
            return None

    def quit(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                self._update_user(cursor, userid, connected_status=1)
                self.send(userid, Event.QuitEvent(u.agent_index,
                                                  None,
                                                  str(time.time())))
        except sqlite3.IntegrityError:
            print("注意!: Rolled back transaction")
            return None

    def submit_survey(self, userid, data):
        def _user_finished(userid):
            self._update_user(cursor, userid, status=Status.Finished)

        def _update_scenario_db(chat_id, scenario_id, partner_type):
            # 両方のエージェントが人間の場合, シナリオの完了したダイアログの数が一度だけ更新されることを確認する
            cursor.execute('''SELECT complete FROM scenario WHERE scenario_id=? AND partner_type=?''',
                           (scenario_id, partner_type))
            complete_set = set(json.loads(cursor.fetchone()[0]))
            complete_set.add(chat_id)
            cursor.execute('''
                UPDATE scenario
                SET complete=?
                WHERE scenario_id=? AND partner_type=?
                AND (SELECT COUNT(survey.name)
                    FROM survey
                    WHERE survey.chat_id=?) = 0;
            ''', (json.dumps(list(complete_set)), scenario_id, partner_type, chat_id))

        try:
            with self.conn:
                cursor = self.conn.cursor()
                user_info = self._get_user_info_unchecked(cursor, userid)
                cursor.execute('''SELECT scenario_id FROM chat WHERE chat_id=?''', (user_info.chat_id,))
                scenario_id = cursor.fetchone()[0]
                _update_scenario_db(user_info.chat_id, scenario_id, user_info.partner_type)
                cursor.execute('INSERT INTO survey VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (userid, user_info.chat_id, user_info.partner_type,
                                data['fluent'], data['honest'], data['persuasive'],
                                data['fair'], data['negotiator'], data['coherent'], data['comments']))
                _user_finished(userid)
                self.logger.debug("User {:s} submitted survey for chat {:s}".format(userid, user_info.chat_id))

        except sqlite3.IntegrityError:
            print("注意!: Rolled back transaction")
