# This code is a modified copy of the `NLTKWordTokenizer` class from `NLTK` library.

from underthesea import word_tokenize

class VietnameseTokenizer:
    @staticmethod
    def tokenize(text: str) -> list[str]:
        return word_tokenize(text, format="text").split()
