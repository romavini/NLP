from nlp.psql.database import read_db
import re
from typing import List, Tuple, Any
import pandas as pd  # type: ignore
import numpy as np
from nltk.tokenize.casual import casual_tokenize  # type: ignore
from nltk.util import ngrams  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
import spacy
import nltk  # type: ignore


class NLP:
    def __init__(self):
        nltk.download("stopwords")
        self.stop_words = nltk.corpus.stopwords.words("portuguese")

        self.stemmer = PorterStemmer()
        self.scy = spacy.load("pt_core_news_sm")

    def analize(self, sentences: str):
        sentences = self.strip_pontuation(sentences).lower()

        onehot_df = self.tolkenizator(sentences)
        print(f"{onehot_df = }")

        bow_df = self.bow(sentences)
        print(f"{bow_df.columns = }")
        print(f"{bow_df = }")

        tokens = self.do_ngrams(
            self.casual_token(sentences, reduce_len=True, strip_handles=True)
        )
        print(f"{tokens = }")

        bow_df_ngrams = self.bow(sentences, 2)

        print(f"{self.scalar_prod(bow_df_ngrams) = }")
        print(f"{self.scalar_prod(bow_df) = }")

        self.scalar_matrix(bow_df)

        tokens_scy, doc = self.grammatic(sentences)
        tokens_lemma = self.lemma(tokens_scy)
        ent = self.entities(doc)

        print(f"{tokens_lemma = }")
        print(f"{ent = }")

        return bow_df

    def tolkenizator(self, sentence: str, remove_stop_words: bool = True) -> pd.DataFrame:
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

    def grammatic(self, sentence: str) -> Tuple[Any, Any]:
        doc = self.scy(sentence)

        return [token for token in doc], doc

    def lemma(self, tokens: List[Any]) -> List[Any]:
        return [
            (token.lemma_, token.pos_)
            if token.pos_ == "VERB"
            else (token.orth_, token.pos_)
            for token in tokens
        ]

    def entities(self, doc: spacy.tokens.doc.Doc) -> List[Any]:
        print(f"{type(doc) = }")

        return [(entity, entity.label_) for entity in doc.ents]

    def casual_token(self, sentence: str, *args, **kwargs) -> List[str]:
        return casual_tokenize(sentence, *args, **kwargs)

    def strip_pontuation(self, text: str, pattern_to_strip: str = r"([-.,;!?])+") -> str:
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

    def scalar_matrix(self, sentences: pd.DataFrame) -> pd.DataFrame:
        # TODO
        pass

    def stemming(self, text: str) -> str:
        return " ".join([self.stemmer.stem(w) for w in text.split()])


if __name__ == "__main__":
    df = read_db("poems")

    sentence = "\n".join(df.loc[df["category"] == "Poesias › Amor"]["text"].to_list())

    nlp = NLP()
    bow_df = nlp.analize(sentence)
