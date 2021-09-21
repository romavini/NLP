import re
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download("stopwords")
stop_words = nltk.corpus.stopwords.words("portuguese")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def tolkenizator(sentence: str, remove_stop_words: bool = True) -> pd.DataFrame:
    token_sentence = str.split(sentence)
    if remove_stop_words:
        token_sentence = [word for word in token_sentence if word not in stop_words]

    vocab = sorted(set(token_sentence))

    num_tokens = len(token_sentence)
    vocab_size = len(vocab)

    onehot_vectors = np.zeros((num_tokens, vocab_size), int)

    for i, word in enumerate(token_sentence):
        onehot_vectors[i, vocab.index(word)] = 1

    return pd.DataFrame(onehot_vectors, columns=vocab)


def bow(sentence: str, ngrams: int = None) -> pd.DataFrame:
    """Bag of Words"""
    corpus = {}
    for i, sent in enumerate(sentence.split("\n")):
        if ngrams:
            corpus[f"sent{i}"] = dict(
                (tok, 1)
                for tok in do_ngrams(
                    casual_token(stemming(sent), reduce_len=True, strip_handles=True),
                    ngrams,
                )
            )
        else:
            corpus[f"sent{i}"] = dict((tok, 1) for tok in casual_token(stemming(sent)))

    return pd.DataFrame.from_records(corpus).fillna(0).astype(int).T


def casual_token(sentence: str, *args, **kwargs) -> List[str]:
    return casual_tokenize(sentence, *args, **kwargs)


def strip_pontuation(text: str, pattern_to_strip: str = r"([-.,;!?])+") -> str:
    pattern = re.compile(pattern_to_strip)
    return "".join([tok for tok in pattern.split(text) if tok not in pattern_to_strip])


def do_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    return [" ".join(pair) for pair in list(ngrams(tokens, n))]


def scalar_prod(df: pd.DataFrame) -> Tuple[Any, int]:
    df = df.T
    ocorr = [(k, v) for (k, v) in (df.sent0 & df.sent1).items() if v]
    prod = df.sent0.dot(df.sent1)

    return ocorr, prod


def stemming(text: str) -> str:
    return " ".join([stemmer.stem(w) for w in text.split()])


if __name__ == "__main__":
    sentence = "Lorem ipsum dolor sit amet,"
    sentences = """Passei a vida ouvindo q precisava acordar mais cedo, q dormir tarde faz mal, q de manhã temos mais energia.
        Depois desses 2 meses ridículos [e somando o ensino médio e um emprego matutino q tive], posso dizer: NADA DE BOM acontece antes das 9h da manhã."""

    sentences = strip_pontuation(sentences).lower()
    # sentences = stemming(sentences)

    onehot_df = tolkenizator(sentences)
    print(onehot_df)

    bow_df = bow(sentences)
    print(bow_df.columns)
    print(bow_df)

    tokens = do_ngrams(casual_token(sentences, reduce_len=True, strip_handles=True))
    print(tokens)

    bow_df_ngrams = bow(sentences, 2)

    print(scalar_prod(bow_df))

    print(len(stop_words))
