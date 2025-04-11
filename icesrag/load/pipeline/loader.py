from datetime import datetime
from typing import Dict, List
import logging

from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingEngine

logger = logging.getLogger(__name__)

def _create_ids(n:int) -> List[str]:
    """
    Creates unique ids across all chunks. 
    Required to ensure that all chunks carry the same ids across all strategies

    args:
        n (int): number of chunks

    returns:
        List[str]
    """
    logger.info(f"Creating {n} unique IDs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids = []
    for i in range(0, n):
        id = f"{i}_{now}"
        ids.append(id)
        logger.debug(f"Generated ID: {id}")
    logger.info(f"Successfully generated {len(ids)} IDs")
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
        logger.info(f"Initializing CompositeLoader with {len(strategies)} strategies")
        self.strategies_ = strategies
        self.packages_ = []
        for i, strategy in enumerate(strategies):
            logger.debug(f"Strategy {i+1}: {strategy.get('name', 'unnamed')}")

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
        logger.info(f"Preparing {len(chunks)} chunks for {len(self.strategies_)} strategies")
        ids = _create_ids(len(chunks))

        for i, strategy in enumerate(self.strategies_):
            logger.info(f"Processing strategy {i+1}/{len(self.strategies_)}: {strategy.get('name', 'unnamed')}")
            package = {'chunks': chunks} # retain untransformed values
            
            # Perform all preprocessing to chunks
            if 'preprocess' in strategy.keys():
                logger.debug("Applying preprocessing steps")
                if isinstance(strategy['preprocess'], TextPreprocessingEngine):
                    strategy['preprocess'] = [strategy['preprocess']]
                for j, preprocessor in enumerate(strategy['preprocess']):
                    logger.debug(f"Applying preprocessor {j+1}/{len(strategy['preprocess'])}")
                    chunks = preprocessor.batch_preprocess(chunks)
                    logger.debug(f"Preprocessing step {j+1} complete")

            # Perform all embeddings
            if 'embed' in strategy.keys():
                logger.debug("Generating embeddings")
                embeddings = strategy['embed'].batch_process(chunks)
                package['embeddings'] = embeddings
                logger.debug(f"Generated {len(embeddings)} embeddings")
            else:
                logger.debug("No embedding strategy specified, using raw chunks")
                package['embeddings'] = chunks

            # Build packages
            logger.debug("Building package")
            package['ids'] = ids
            package['metadatas'] = metadatas
            package = strategy['package'].batch_process(**package)
            self.packages_.append(package)
            logger.info(f"Strategy {i+1} preparation complete")

    def load(self):
        """
        Pushes all of the prepared packages to their respective stores

        Args:
            None

        Returns:
            None
        """
        logger.info("Loading prepared packages into stores")
        assert len(self.packages_) > 0, "No data has been prepared"
        for i, strategy in enumerate(self.strategies_):
            logger.info(f"Loading package {i+1}/{len(self.packages_)} into store")
            strategy['store'].add(self.packages_[i])
            logger.debug(f"Successfully loaded package {i+1}")

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
        logger.info("Starting prepare_load operation")
        self.prepare(chunks, metadatas)
        self.load()
        logger.info("prepare_load operation complete")