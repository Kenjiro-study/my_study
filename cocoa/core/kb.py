class KB(object):
    """
    エージェントの知識を表すオブジェクト
    knowledge baseの略
    様々な情報がここに入っている
    """
    def __init__(self, attributes):
        self.attributes = attributes

    def dump(self):
        raise NotImplementedError
