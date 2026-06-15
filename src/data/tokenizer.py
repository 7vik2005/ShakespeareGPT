import pickle

from configs.config import (
    DATA_PATH,
    VOCAB_PATH,
    CHAR2IDX_PATH,
    IDX2CHAR_PATH,
    create_directories
)


class CharacterTokenizer:
    def __init__(self):
        self.vocab=None
        self.char2idx=None
        self.idx2char=None

    def load_text(self):
        with open(DATA_PATH,"r",encoding="utf-8") as file:
            return file.read()

    def build_vocabulary(self,text):
        self.vocab=sorted(list(set(text)))

        self.char2idx={
            char:index
            for index,char in enumerate(self.vocab)
        }

        self.idx2char={
            index:char
            for char,index in self.char2idx.items()
        }

    def encode(self,text):
        if self.char2idx is None:
            raise ValueError(
                "Vocabulary has not been initialized."
            )

        return [
            self.char2idx[char]
            for char in text
        ]

    def decode(self,indices):
        if self.idx2char is None:
            raise ValueError(
                "Vocabulary has not been initialized."
            )

        return "".join(
            self.idx2char[index]
            for index in indices
        )

    def save(self):
        create_directories()

        with open(VOCAB_PATH,"wb") as file:
            pickle.dump(self.vocab,file)

        with open(CHAR2IDX_PATH,"wb") as file:
            pickle.dump(self.char2idx,file)

        with open(IDX2CHAR_PATH,"wb") as file:
            pickle.dump(self.idx2char,file)

    def load(self):
        with open(VOCAB_PATH,"rb") as file:
            self.vocab=pickle.load(file)

        with open(CHAR2IDX_PATH,"rb") as file:
            self.char2idx=pickle.load(file)

        with open(IDX2CHAR_PATH,"rb") as file:
            self.idx2char=pickle.load(file)

    @property
    def vocab_size(self):
        if self.vocab is None:
            return 0

        return len(self.vocab)

    def fit(self):
        text=self.load_text()

        self.build_vocabulary(text)

        self.save()

        return text

    def fit_transform(self):
        text=self.fit()

        return self.encode(text)

    def transform(self,text):
        return self.encode(text)

    def inverse_transform(self,indices):
        return self.decode(indices)
