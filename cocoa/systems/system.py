class System(object):
    """
    Sessionオブジェクトを構築するための抽象クラス
    """
    def new_session(self, agent, kb):
        raise NotImplementedError

    @classmethod
    def name(cls):
        return 'base'
