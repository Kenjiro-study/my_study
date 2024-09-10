from cocoa.model.generator import Templates as BaseTemplates, Generator as BaseGenerator

from core.tokenizer import detokenize

class Generator(BaseGenerator):
    def get_filter(self, used_templates=None, category=None, role=None, context_tag=None, tag=None, **kwargs):
        locs = [super().get_filter(used_templates)]
        assert category and role
        self._add_filter(locs, self.templates.role == role)
        self._add_filter(locs, self.templates.category == category)
        if tag:
            self._add_filter(locs, self.templates.tag == tag)
        if context_tag:
            self._add_filter(locs, self.templates.context_tag == context_tag)
        return self._select_filter(locs)

class Templates(BaseTemplates):
    def ambiguous_template(self, template):
        """曖昧さがあるかどうかを確認し, 曖昧さがある場合はテンプレートを破棄する
        """
        if len(template) == 0:
            return True
        num_prices = sum([1 if token == '{price}' else 0 for token in template])
        if num_prices > 1:
            return True
        num_titles = sum([1 if token == '{title}' else 0 for token in template])
        if num_titles > 1:
            return True
        return False

    def add_template(self, utterance, dialogue_state):
        if self.finalized:
            print('Cannot add templates.')
            return
        if not utterance.template or self.ambiguous_template(utterance.template):
            return
        row = {
                'category': dialogue_state.kb.category,
                'role': dialogue_state.kb.role,
                'tag': utterance.lf.intent,
                'template': detokenize(utterance.template), # 応答のテンプレート
                'context_tag': dialogue_state.partner_act,
                'context': detokenize(dialogue_state.partner_template), # 現在の発話
                'id': self.template_id,
                }
        #print('add template:')
        #print('context:', row['context'])
        #print('template:', row['template'])
        self.template_id += 1
        self.templates.append(row)

    # (おそらく)templatesに従って, 指定されたカテゴリ, 役割, 一つ前のintent, 応答したいintentの4つの情報に沿った文を生成する関数
    def dump(self, n=-1):
        df = self.templates.groupby(['category', 'role', 'context_tag', 'tag'])
        for group in df.groups:
            category, role, context_tag, response_tag = group
            if response_tag == 'offer':
                continue
            print('='*40)
            print('category={}, role={}, context={}, response={}'.format(category, role, context_tag, response_tag)) # category, role, context, responseを決定
            print('='*40)
            rows = [x[1] for x in df.get_group(group).iterrows()]
            #rows = sorted(rows, key=lambda r: r['count'], reverse=True)
            for i, row in enumerate(rows):
                if i == n:
                    break
                print(row['template'].encode('utf-8')) # nに指定した数だけcategory, role, context, responseに合う文を生成する
