from typing import List
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformEmbedder():
    """
    A general strategy using sentence_transformers for generating vector embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding strategy with a specific model name.

        Args:
            model_name (str): The model name used for embeddings (e.g., 'all-MiniLM-L6-v2' or 'bert-base-uncased').
                a. Options: 
                    all-MiniLM-L6-v2
                    paraphrase-MiniLM-L6-v2
                    paraphrase-distilroberta-base-v1
                    distilbert-base-nli-stsb-mean-tokens
                    paraphrase-xlm-r-multilingual-v1
                    bert-base-nli-mean-tokens
                    roberta-base-nli-stsb-mean-tokens
                    distilbert-base-uncased
                    msmarco-distilbert-base-v3
                    paraphrase-T5-large-v1
                    xlm-r-large-en-fr-it-de                
        """
        logger.info(f"Initializing SentenceTransformEmbedder with model: {model_name}")
        self.model_name_ = model_name
        self.model = None

    def process(self, text: str, **kwargs) -> List[float]:
        """
        Method to embed a single text input into a vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The embedding of the input text as a list of floats.
        """
        logger.debug(f"Processing text: {text[:50]}...")
        
        # Load pre-trained Sentence-BERT model if not already loaded
        if self.model is None:
            logger.debug(f"Loading model {self.model_name_}")
            self.model = SentenceTransformer(self.model_name_)
            logger.info("Model loaded successfully")

        # Generate sentence embeddings
        logger.debug("Generating embeddings")
        embeddings = self.model.encode([text])
        logger.info(f"Successfully generated embedding of dimension {len(embeddings[0])}")
        return embeddings[0]

    def batch_process(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Method to embed a batch of texts into vectors.

        Args:
            texts (List[str]): A list of input texts to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, each a list of floats, corresponding to the input texts.
        """
        logger.info(f"Processing batch of {len(texts)} texts")
        embeddings = []
        
        for i, t in enumerate(texts):
            logger.debug(f"Processing text {i+1}/{len(texts)}")
            embeddings.append(self.process(t))
            
        logger.info(f"Successfully processed batch of {len(embeddings)} texts")
        return embeddings