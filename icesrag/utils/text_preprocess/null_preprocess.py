from typing import List
import logging

from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingStrategy

logger = logging.getLogger(__name__)

class NullPreProcess(TextPreprocessingStrategy):

    def preprocess(self, text: str, **kwargs) -> str:
        """
        Process a single text string.

        Args:
            text (str): The input text to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            str: The preprocessed version of the input text.
        """        
        logger.debug(f"Null preprocessing text: {text[:50]}...")
        logger.info("No preprocessing applied (null strategy)")
        return text
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of text strings.

        Args:
            texts (List[str]): A list of text strings to preprocess.
            **kwargs: Additional keyword arguments that may be used by specific implementations.

        Returns:
            List[str]: A list of preprocessed text strings, one for each input text.
        """
        logger.info(f"Null preprocessing batch of {len(texts)} texts")
        logger.info("No preprocessing applied (null strategy)")
        return texts