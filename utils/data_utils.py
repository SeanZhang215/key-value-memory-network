# utils/data_utils.py

import torch
import numpy as np
from typing import Dict, List, Tuple
import re
from unidecode import unidecode

class Vocab:
    """
    Vocabulary class for mapping between words and indices.
    
    Attributes:
        name (str): Name of the vocabulary
        word2index (Dict[str, int]): Mapping from words to indices
        word2count (Dict[str, int]): Word frequency counter
        index2word (Dict[int, str]): Mapping from indices to words
        n_words (int): Number of unique words
    """
    
    def __init__(self, name: str = 'vocab'):
        self.name = name
        self._word2index = {}
        self._word2count = {}
        self._index2word = {}
        self._n_words = 0
        self.add_word('<UNK>')
    
    def add_word(self, word: str) -> None:
        """Add a word to the vocabulary."""
        if word not in self._word2index:
            self._word2index[word] = self._n_words
            self._word2count[word] = 1
            self._index2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, removing punctuation and converting to lowercase.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    text = unidecode(text)  # Handle unicode characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text.lower().split()

def multihot(text: str, vocab: Vocab, preserve_counts: bool = False) -> np.ndarray:
    """
    Convert text to multi-hot encoding based on vocabulary.
    
    Args:
        text: Input text string
        vocab: Vocabulary object
        preserve_counts: Whether to preserve word counts or use binary encoding
        
    Returns:
        Multi-hot encoded vector
    """
    if not text or len(text.strip()) == 0:
        return np.zeros(vocab._n_words)
        
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(vocab._n_words)
        
    # Convert tokens to indices
    indices = [vocab.word2index(t) for t in tokens if t in vocab._word2index]
    if not indices:
        return np.zeros(vocab._n_words)
        
    # Create encoding
    encoding = np.zeros((len(indices), vocab._n_words))
    encoding[np.arange(len(indices)), indices] = 1
    
    if preserve_counts:
        return encoding.sum(0)
    return encoding.sum(0) >= 1