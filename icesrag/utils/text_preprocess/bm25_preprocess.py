import string
from typing import List
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingStrategy

logger = logging.getLogger(__name__)

# Prerequirement!
# nltk.download('punkt')  # For word tokenization
# nltk.download('punkt_tab')
# nltk.download('stopwords')  # For stopwords list
# nltk.download('wordnet')  # For lemmatization

class BM25PreProcess(TextPreprocessingStrategy):

    def preprocess(self, text: str, **kwargs) -> str:
        """
        Process a single text string.

        Args:
            text (str): The input text to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            str: The preprocessed version of the input text.
        """        
        logger.debug(f"BM25 preprocessing text: {text[:50]}...")
        
        # Step 1: Convert all words to lowercase
        logger.debug("Converting text to lowercase")
        text = text.lower()
        
        # Step 2: Remove all punctuation
        logger.debug("Removing punctuation")
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Step 3: Tokenize the text into words
        logger.debug("Tokenizing text")
        words = word_tokenize(text)
        logger.debug(f"Tokenized into {len(words)} words")
        
        # Step 4: Remove stop words
        logger.debug("Removing stop words")
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        logger.debug(f"Removed stop words, remaining: {len(words)} words")
        
        # Step 5: Lemmatize the words
        logger.debug("Lemmatizing words")
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
        # Return the processed text as a single string
        result = ' '.join(lemmatized_words)
        logger.info(f"Successfully preprocessed text, final length: {len(result)} characters")
        return result
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of text strings.

        Args:
            texts (List[str]): A list of text strings to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            List[str]: A list of preprocessed text strings, one for each input text.
        """        
        logger.info(f"BM25 preprocessing batch of {len(texts)} texts")
        processed = []
        
        for i, t in enumerate(texts):
            logger.debug(f"Processing text {i+1}/{len(texts)}")
            processed.append(self.preprocess(t))
            
        logger.info(f"Successfully preprocessed batch of {len(processed)} texts")
        return processed