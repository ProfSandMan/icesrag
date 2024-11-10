from abc import ABC, abstractmethod
from typing import List

class EmbeddingStrategy(ABC):
    """
    Abstract class representing the Strategy for generating vector embeddings.
    """
    
    @abstractmethod
    def __init__(self, model_name: str):
        """
        Initialize the embedding strategy with a specific model name.

        Args:
            model_name (str): The model name used for embeddings (e.g., 'all-MiniLM-L6-v2' or 'bert-base-uncased').
        """
        self.model_name = model_name

    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Method to embed a single text input into a vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The embedding of the input text as a list of floats.
        """
        pass

    @abstractmethod
    def batch_embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Method to embed a batch of texts into vectors.

        Args:
            texts (List[str]): A list of input texts to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, each a list of floats, corresponding to the input texts.
        """
        pass

class EmbeddingEngine:
    """
    Engine to apply an embedding strategy to a single text or a batch of texts.
    
    This class allows you to set different embedding strategies at runtime and 
    process individual or batches of text based on the chosen strategy.
    """

    def __init__(self, strategy: EmbeddingStrategy):
        """
        Initialize the EmbeddingEngine with a specific embedding strategy.

        Args:
            strategy (EmbeddingStrategy): The embedding strategy to use (e.g., SBERT or BERT).
        """
        self._strategy = strategy

    def set_strategy(self, strategy: EmbeddingStrategy):
        """
        Set or switch the embedding strategy at runtime.

        Args:
            strategy (EmbeddingStrategy): The new embedding strategy to use.
        """
        self._strategy = strategy
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Embed a single text string using the current strategy.

        This method delegates the task of embedding the text to the `embed` method 
        of the strategy currently in use.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding for the input text.
        """
        return self._strategy.embed(text=text, **kwargs)
    
    def batch_embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Embed a batch of text strings using the current strategy.

        This method delegates the task of embedding each text in the batch to the 
        `batch_embed` method of the strategy currently in use.

        Args:
            texts (List[str]): A list of text strings to embed.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """
        return self._strategy.batch_embed(texts=texts, **kwargs)