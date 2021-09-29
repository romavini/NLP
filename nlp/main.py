from nltk import tokenize
from nlp.psql.database import read_db
import re
import copy
from typing import Dict, List, Tuple, Any
import pandas as pd  # type: ignore
import numpy as np
from nltk.tokenize.casual import casual_tokenize  # type: ignore
from nltk.tokenize import TreebankWordTokenizer  # type: ignore
from nltk.util import ngrams  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from collections import Counter
from collections import OrderedDict
import spacy
import nltk  # type: ignore


class NLP:
    def __init__(self):
        nltk.download("stopwords")
        self.stop_words = nltk.corpus.stopwords.words("portuguese")

        self.stemmer = PorterStemmer()
        self.spacy_mode_pt = spacy.load("pt_core_news_sm")

    def vectorize_many(self, docs):
        tokenizer = TreebankWordTokenizer()

        doc_tokens = []

        for doc in docs:
            tokens = sorted(tokenizer.tokenize(doc.lower()))
            doc_tokens.append([e for e in tokens if e not in self.stop_words])

        print(f"{doc_tokens = }")
        print(f"{len(doc_tokens) = }")

        all_tokens = sum(doc_tokens, [])
        print(f"{len(all_tokens) = }")

        lexicon = sorted(set(all_tokens))
        print(f"{lexicon = }")
        print(f"{len(lexicon) = }")

        # Vectorize
        zeros = OrderedDict((token, 0) for token in lexicon)
        doc_vec = []
        for token_list in doc_tokens:
            vec = copy.copy(zeros)
            token_count = Counter(token_list)
            for key, value in token_count.items():
                vec[key] = value / len(lexicon)
            doc_vec.append(vec)

        print(f"{len(doc_vec)}")
        print(f"{doc_vec[0]}")

    def analize(self, sentences: str):
        # sentences = self.strip_pontuation(sentences).lower()

        tokens = self.nltk_to1kenize(sentences)

        # removing stopwords
        tokens = [e for e in tokens if e not in self.stop_words]

        bag_of_words = Counter(tokens)  # type: ignore
        print(f"{bag_of_words.most_common(15) = }")

        print(f"{self.tf(bag_of_words, 'amor')}")

        # onehot_df = self.tolkenizator(sentences)
        # print(f"{onehot_df = }")

        # bow_df = self.bow(sentences)
        # print(f"{bow_df.columns = }")
        # print(f"{bow_df = }")

        # tokens = self.do_ngrams(
        #     self.casual_token(sentences, reduce_len=True, strip_handles=True)
        # )
        # print(f"{tokens = }")

        # bow_df_ngrams = self.bow(sentences, 2)

        # print(f"{self.scalar_prod(bow_df_ngrams) = }")
        # print(f"{self.scalar_prod(bow_df) = }")

        # self.scalar_matrix(bow_df)

        # tokens_scy, doc = self.grammatic(sentences)
        # tokens_lemma = self.lemma(tokens_scy)
        # ent = self.entities(doc)

        # print(f"{tokens_lemma = }")
        # print(f"{ent = }")

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

    def nltk_to1kenize(self, sentence: str) -> List[str]:
        tokenizer = TreebankWordTokenizer()

        return tokenizer.tokenize(sentence.lower())

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

    def tf(self, bag: Dict[str, int], word: str) -> float:
        """Return the term freaquency of a word in a bag_of_words."""
        return round(bag[word] / len(bag), 4)

    def grammatic(self, sentence: str) -> Tuple[Any, Any]:
        doc = self.spacy_mode_pt(sentence)

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
    # nlp.analize(sentence)

    sentences_list = df.loc[df["category"] == "Poesias › Amor"]["text"].to_list()
    nlp.vectorize_many(sentences_list)
