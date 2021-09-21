import re
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk


class NLP:
    def __init__(self):
        nltk.download("stopwords")
        self.stop_words = nltk.corpus.stopwords.words("portuguese")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def analize(self, sentences):
        sentences = self.strip_pontuation(sentences).lower()

        onehot_df = self.tolkenizator(sentences)
        print(onehot_df)

        bow_df = self.bow(sentences)
        print(bow_df.columns)
        print(bow_df)

        tokens = self.do_ngrams(
            self.casual_token(sentences, reduce_len=True, strip_handles=True)
        )
        print(tokens)

        bow_df_ngrams = self.bow(sentences, 2)

        print(f"{self.scalar_prod(bow_df_ngrams)=}")
        print(f"{self.scalar_prod(bow_df)=}")

    def tolkenizator(
        self, sentence: str, remove_stop_words: bool = True
    ) -> pd.DataFrame:
        token_sentence = str.split(sentence)
        if remove_stop_words:
            token_sentence = [
                word for word in token_sentence if word not in self.stop_words
            ]

        vocab = sorted(set(token_sentence))

        num_tokens = len(token_sentence)
        vocab_size = len(vocab)

        onehot_vectors = np.zeros((num_tokens, vocab_size), int)

        for i, word in enumerate(token_sentence):
            onehot_vectors[i, vocab.index(word)] = 1

        return pd.DataFrame(onehot_vectors, columns=vocab)

    def bow(self, sentence: str, ngrams: int = None) -> pd.DataFrame:
        """Bag of Words"""
        corpus = {}
        for i, sent in enumerate(sentence.split("\n")):
            if ngrams:
                corpus[f"sent{i}"] = dict(
                    (tok, 1)
                    for tok in self.do_ngrams(
                        self.casual_token(
                            self.stemming(sent), reduce_len=True, strip_handles=True
                        ),
                        ngrams,
                    )
                )
            else:
                corpus[f"sent{i}"] = dict(
                    (tok, 1) for tok in self.casual_token(self.stemming(sent))
                )

        return pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

    def casual_token(self, sentence: str, *args, **kwargs) -> List[str]:
        return casual_tokenize(sentence, *args, **kwargs)

    def strip_pontuation(
        self, text: str, pattern_to_strip: str = r"([-.,;!?])+"
    ) -> str:
        pattern = re.compile(pattern_to_strip)
        return "".join(
            [tok for tok in pattern.split(text) if tok not in pattern_to_strip]
        )

    def do_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        return [" ".join(pair) for pair in list(ngrams(tokens, n))]

    def scalar_prod(self, df: pd.DataFrame) -> Tuple[Any, int]:
        df = df.T
        ocorr = [(k, v) for (k, v) in (df.sent0 & df.sent1).items() if v]
        prod = df.sent0.dot(df.sent1)

        return ocorr, prod

    def stemming(self, text: str) -> str:
        return " ".join([self.stemmer.stem(w) for w in text.split()])


if __name__ == "__main__":
    sentences = """Bolsonaro ignora crise energética,
        infla números de manifestação bolsonarista,
        diz que desmatamento ilegal da Amazônia teve
        queda de mais de 30% e ataca governadores,
        prefeitos e imprensa brasileira em defesa de
        tratamento precoce contra Covid-19 na ONU:
        https://bit.ly/2XDrV0M"""

    nlp = NLP()
    nlp.analize(sentences)
