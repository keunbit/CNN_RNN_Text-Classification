from konlpy.tag import Hannanum, Kkma, Komoran, Mecab, Okt


class BasicTokenizer:
    def __init__(self):
        self.type_ = "Basic"

    def tokenize(self, text):
        tokens = text.split()
        tokens_len = len(tokens)

        return tokens, tokens_len


class MecabTokenizer:
    def __init__(self):
        self.mecab = Mecab()
        self.type_ = "Mecab"

    def tokenize(self, text):
        tokens = self.mecab.morphs(text)
        tokens_len = len(tokens)

        return tokens, tokens_len


class HannanumTokenizer:
    def __init__(self):
        self.hannanum = Hannanum()
        self.type_ = "Hannanum"

    def tokenize(self, text):
        tokens = self.hannanum.morphs(text)
        tokens_len = len(tokens)

        return tokens, tokens_len


class KkmaTokenizer:
    def __init__(self):
        self.kkma = Kkma()
        self.type_ = "Kkma"

    def tokenize(self, text):
        tokens = self.kkma.morphs(text)
        tokens_len = len(tokens)

        return tokens, tokens_len


class KomoranTokenizer:
    def __init__(self):
        self.komoran = Komoran()
        self.type_ = "Komoran"

    def tokenize(self, text):
        tokens = self.komoran.morphs(text)
        tokens_len = len(tokens)

        return tokens, tokens_len


class OktTokenizer:
    def __init__(self):
        self.okt = Okt()
        self.type_ = "Okt"

    def tokenize(self, text):
        tokens = self.okt.morphs(text)
        tokens_len = len(tokens)

        return tokens, tokens_len
