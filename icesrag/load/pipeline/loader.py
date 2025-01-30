from datetime import datetime
from typing import Dict, List

from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingEngine


def _create_ids(n:int) -> List[str]:
    """
    Creates unique ids across all chunks. 
    Required to ensure that all chunks carry the same ids across all strategies

    args:
        n (int): number of chunks

    returns:
        List[str]
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids = []
    for i in range(0, n):
        ids.append(f"{i}_{now}")
    return ids    

class CompositeLoader():
    def __init__(self, strategies: List[Dict]) -> None:
        """
        Performs all actions to prepare and load chunks for future retrieval

        Args:
            strategies (Dict):# TODO: add documentation 
        strategy keys (in order): name, type, preprocessing (optional list), embedding (optional class), 
            package, load
        """
        self.strategies_ = strategies
        self.packages_ = []

    def prepare(self, 
                chunks: List[str], 
                metadatas: List[Dict]):
        """
        Performs all preparation steps for each strategy

        Args:
            chunks (List[str]): the text chunks to prepare
            metadatas (List[Dict]): the associated metadata with each chunk

        Notes:
            original chunks are stored as 'documents' which will remain untransformed
            if any preprocessing occurs, the transformed chunks will be passed through to embeddings (if included)
                or stored as retrieval objecs (for non vector strategies)

        Returns:
            None
        """
        ids = _create_ids(len(chunks))

        for strategy in self.strategies_:
            package = {'chunks': chunks} # retain untransformed values
            # Perform all preprocessing to chunks
            if 'preprocess' in strategy.keys():
                if isinstance(strategy['preprocess'], TextPreprocessingEngine):
                    strategy['preprocess'] = [strategy['preprocess']]
                for preprocessor in strategy['preprocess']:
                    chunks = preprocessor.batch_preprocess(chunks)

            # Perform all embeddings
            if 'embed' in strategy.keys():
                embeddings = strategy['embed'].batch_process(chunks)
                package['embeddings'] = embeddings
            else:
                package['embeddings'] = chunks

            # Build packages
            package['ids'] = ids
            package['metadatas'] = metadatas
            package = strategy['package'].batch_process(**package)
            self.packages_.append(package)

    def load(self):
        """
        Pushes all of the prepared packages to their respective stores

        Args:
            None

        Returns:
            None
        """
        assert len(self.packages_) > 0, "No data has been prepared"
        for i, strategy in enumerate(self.strategies_):
            strategy['store'].add(self.packages_[i])

    def prepare_load(self, 
                chunks: List[str], 
                metadatas: List[Dict]):
        """
        Performs all preparation steps for each strategy and stores the data

        Args:
            chunks (List[str]): the text chunks to prepare
            metadatas (List[Dict]): the associated metadata with each chunk

        Notes:
            original chunks are stored as 'documents' which will remain untransformed
            if any preprocessing occurs, the transformed chunks will be passed through to embeddings (if included)
                or stored as retrieval objecs (for non vector strategies)

        Returns:
            None
        """
        self.prepare(chunks, metadatas)
        self.load()