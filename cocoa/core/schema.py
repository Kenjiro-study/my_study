"""
スキーマはドメインに関する情報(types, entities, relations)を指定する
"""

import json


class Attribute(object):
    def __init__(self, name, value_type, unique=False, multivalued=False, entity=True):
        self.name = name
        self.value_type = value_type
        self.unique = unique
        self.multivalued = multivalued
        # この属性の値がentityであるかどうか
        self.entity = entity

    @staticmethod
    def from_json(raw):
        return Attribute(raw['name'], raw['value_type'], raw.get('unique', False), raw.get('multivalued', False), raw.get('entity', True))

    def to_json(self):
        return {'name': self.name, 'value_type': self.value_type, 'unique': self.unique, 'multivalued': self.multivalued, 'entity': self.entity}


class Schema(object):
    """
    スキーマには存在しうるentitiesとrelationsに関する情報が含まれる
    """
    def __init__(self, path, domain=None):
        raw = json.load(open(path)) # criagslist-schema.jsonをまるまる読み込み
        # type(例:hobby)から値のリスト(例:hiking)へのマッピング
        values = raw['values'] # valueの取り出し(最初は{})
        # 属性のリスト(例:place_of_birth)
        attributes = [Attribute.from_json(a) for a in raw['attributes']] # unique属性を足してAttributeオブジェクトに変換
        self.attr_names = [attr.name for attr in attributes] # 各属性の名前(name)を取得
        self.values = values # 値(value)の取得
        self.attributes = attributes # 属性値(attributes)の取得
        self.domain = domain # ドメインの取得(初期値がNoneなのでNone)

    def get_attributes(self):
        """
        全ての属性の辞書{name: value_type}を返す
        """
        return {attr.name: attr.value_type for attr in self.attributes}

    def get_ordered_attribute_subset(self, attribute_subset):
        """
        スキーマ内の属性の元々の順序を使用して, このスキーマの属性のサブセットを順序付けする
        attribute_subset: サブセット内に存在する属性の名前を含むリスト
        return: スキーマ内の属性の元々の順序を保持した同じリストを返す
        """
        subset_ordered = sorted([(attr, self.attributes.index(attr)) for attr in attribute_subset], key=lambda x: x[1])
        return [x[0] for x in subset_ordered]

    def get_ordered_item(self, item):
        """
        get_ordered_attribute_subsetに従ってitem内の属性を順序付けし, リストを返す
        """
        ordered_item = []
        for name in self.attr_names:
            try:
                ordered_item.append((name, item[name]))
            except KeyError:
                continue
        return ordered_item
