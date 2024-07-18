import torch
from itertools import chain

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class ONMTDatasetBase(torch.utils.data.Dataset):
    def __init__(self, examples, fields, data_type):
        self.examples = examples
        self.fields = fields
        self.data_type = data_type

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        return super(ONMTDatasetBase, self).__reduce_ex__(proto)

    def load_fields(self, vocab_dict):
        from onmt.io.IO import load_fields_from_vocab

        fields = load_fields_from_vocab(vocab_dict.items(), self.data_type)
        self.fields = {k: f for k, f in fields.items() if k in self.examples[0].__dict__}

    @staticmethod
    def extract_text_features(tokens):
        if not tokens:
            return [], [], -1

        split_tokens = [token.split("￨") for token in tokens]
        split_tokens = [token for token in split_tokens if token[0]]
        token_size = len(split_tokens[0])

        assert all(len(token) == token_size for token in split_tokens), \
            "all words must have the same number of features"
        words_and_features = list(zip(*split_tokens))
        words = words_and_features[0]
        features = words_and_features[1:]

        return words, features, token_size - 1

    def _join_dicts(self, *args):
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
        ex = {}
        for (name, field), val in zip(fields, data):
            if field is not None:
                ex[name] = field.preprocess(val)
            else:
                ex[name] = val
        return ex

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]