__author__ = 'anushabala'
from cocoa.systems.system import System
from cocoa.sessions.human_session import HumanSession


class HumanSystem(System):
    def __init__(self):
        super().__init__() # 3系ver.

    @classmethod
    def name(cls):
        return 'human'

    def new_session(self, agent, kb):
        return HumanSession(agent)
