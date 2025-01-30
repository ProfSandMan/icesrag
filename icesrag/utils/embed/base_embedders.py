from typing import List

from sentence_transformers import SentenceTransformer


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
        self.model_name_ = model_name

    def process(self, text: str, **kwargs) -> List[float]:
        """
        Method to embed a single text input into a vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The embedding of the input text as a list of floats.
        """
        # Load pre-trained Sentence-BERT model
        model = SentenceTransformer(self.model_name_)

        # Generate sentence embeddings
        embeddings = model.encode([text])
        return embeddings[0]

    def batch_process(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Method to embed a batch of texts into vectors.

        Args:
            texts (List[str]): A list of input texts to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, each a list of floats, corresponding to the input texts.
        """
        embeddings = []
        for t in texts:
            embeddings.append(self.process(t))
        return embeddings