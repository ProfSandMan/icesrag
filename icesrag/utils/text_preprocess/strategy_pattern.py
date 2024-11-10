from abc import ABC, abstractmethod
from typing import List

class TextPreprocessingStrategy(ABC):
    """
    Abstract base class for text preprocessing strategies.
    This class defines the common interface for different text preprocessing strategies.
    Concrete subclasses must implement the `preprocess` and `batch_preprocess` methods.
    """

    @abstractmethod
    def preprocess(self, text: str, **kwargs) -> str:
        """
        Process a single text string.

        This method will be implemented by subclasses to define how a single text string
        is preprocessed (e.g., cleaning, tokenization, stop word removal, lemmatization, etc.).

        Args:
            text (str): The input text to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            str: The preprocessed version of the input text.
        """
        pass
    
    @abstractmethod
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of text strings.

        This method will be implemented by subclasses to define how a list of text strings 
        is preprocessed. It typically calls the `preprocess` method on each individual text.

        Args:
            texts (List[str]): A list of text strings to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            List[str]: A list of preprocessed text strings, one for each input text.
        """
        pass

class TextPreprocessingEngine:
    """
    Engine to apply a text preprocessing strategy to a single text or a batch of texts.
    
    This class allows you to set different preprocessing strategies at runtime and 
    process individual or batches of text based on the chosen strategy.
    """

    def __init__(self, strategy: TextPreprocessingStrategy):
        """
        Initialize the TextPreprocessingEngine with a specific preprocessing strategy.

        Args:
            strategy (TextPreprocessingStrategy): The preprocessing strategy to use.
        """        
        self._strategy = strategy

    def set_strategy(self, strategy: TextPreprocessingStrategy):
        """
        Set or switch the preprocessing strategy at runtime.

        Args:
            strategy (TextPreprocessingStrategy): The new preprocessing strategy to use.
        """
        self._strategy = strategy
    
    def preprocess(self, text: str, **kwargs) -> str:
        """
        Process a single text string using the current strategy.

        This method delegates the task of processing the text to the `preprocess` method 
        of the strategy currently in use.

        Args:
            text (str): The input text to preprocess.
            **kwargs: Additional keyword arguments that may be used by the strategy's implementation.

        Returns:
            str: The preprocessed version of the input text.
        """
        return self._strategy.preprocess(text=text, **kwargs)
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of text strings using the current strategy.

        This method delegates the task of processing each text in the batch to the 
        `batch_preprocess` method of the strategy currently in use.

        Args:
            texts (List[str]): A list of text strings to preprocess.
            **kwargs: Additional keyword arguments that may be used by the strategy's implementation.

        Returns:
            List[str]: A list of preprocessed text strings, one for each input text.
        """
        return self._strategy.batch_preprocess(texts=texts, **kwargs)