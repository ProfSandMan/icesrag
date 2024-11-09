from abc import ABC, abstractmethod
from typing import List

class TextPreprocessingStrategy(ABC):
    
    @abstractmethod
    def preprocess(self, text: str, **kwargs) -> str:
        """Process a single text string."""
        pass
    
    @abstractmethod
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Process a batch of text strings."""
        pass

class TextPreprocessingEngine:
    
    def __init__(self, strategy: TextPreprocessingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TextPreprocessingStrategy):
        """Allow runtime strategy switching."""
        self._strategy = strategy
    
    def preprocess(self, text: str, **kwargs) -> str:
        """Process a single text."""
        return self._strategy.preprocess(text=text, **kwargs)
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Process a batch of texts."""
        return self._strategy.batch_preprocess(texts=texts, **kwargs)