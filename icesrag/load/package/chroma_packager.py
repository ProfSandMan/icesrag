from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from icesrag.load.package.strategy_pattern import PackageStrategy

logger = logging.getLogger(__name__)

class PackageChroma(PackageStrategy):
    """
    Class to package documents, embeddings, and metadata for chromadb
    """

    def create_ids(self, chunks: Union[str, List[str]], 
                   tag: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Create unique IDs for one or more document chunks.
        
        Args:
            chunks (Union[str, List[str]]): A single chunk or a list of chunks for which to generate IDs.
            tag (Optional[Union[str, List[str]]]): An optional tag to prepend to the IDs.
            
        Returns:
            List[str]: A list of generated unique IDs.
        """
        logger.info("Creating Chroma IDs")
        ids = []
        if isinstance(chunks, str):
            chunks = [chunks]
        for i, chunk in enumerate(chunks):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if tag is None:
                id = f"{i}_{now}"
            else:
                if isinstance(tag, list):
                    id = f"{tag[i]} - {i} -{now}"
                else:
                    id = f"{tag} - {i} - {now}"
            ids.append(id)
        logger.info(f"Generated {len(ids)} Chroma IDs")
        return ids

    def process(self, chunk: str, 
                metadata: Dict,
                embedding: Optional[List[float]] = None, 
                tag: Optional[str] = None) -> Any:
        """
        Package a single chunk of text along with its embedding, metadata, and ID.
        
        Args:
            chunk (str): The chunk of text.
            metadata (Dict): Metadata associated with the chunk.
            embedding (Optional[List[float]]): The embedding associated with the chunk.
            tag (Optional[str]): An optional tag to prepend to the IDs.
        
        Returns:
            Dict: A dictionary containing the chunk, embedding, metadata, and ID.
        """
        logger.info("Processing single chunk for Chroma")
        ids = self.create_ids(chunks=chunk, tag=tag)
        d = {'documents':[chunk],
             'metadatas':[metadata],
             'ids':ids}
        if embedding is not None:
            d['embeddings'] = [embedding]
            logger.debug("Added embedding to package")
        logger.debug(f"Created package with ID: {ids[0]}")
        return d

    def batch_process(self, 
                      chunks: List[str], 
                      metadatas: List[Dict], 
                      ids: Optional[List[List[float]]] = None,
                      embeddings: Optional[List[List[float]]] = None, 
                      tag: Optional[List[str]] = None) -> Any:
        """
        Package a batch of chunks, embeddings, metadata, and IDs.
        
        Args:
            chunks (List[str]): A list of document chunks.
            metadatas (List[Dict]): A list of metadata dictionaries, each corresponding to a chunk.
            ids (Optional[List[List[str]]]): unique ids for each chunks
            embeddings (Optional[List[List[float]]]): A list of embeddings, each corresponding to a chunk.
            tag (Optional[List[str]]): An optional tag to prepend to the IDs.
        
        Returns:
            List[Dict]: A list of dictionaries, each containing a chunk, embedding, metadata, and ID.
        """
        logger.info(f"Processing batch of {len(chunks)} chunks for Chroma")
        documents = []
        embeds = []
        mdata = []
        if ids is None:
            ids = self.create_ids(chunks=chunks, tag=tag)
        if embeddings is not None:
            assert len(chunks) == len(embeddings) == len(metadatas), "chunks, embeddings, and metadatas must be the same length!"
            logger.debug("Validated lengths of chunks, embeddings, and metadatas")
        else:
            assert len(chunks) == len(metadatas), "chunks and metadatas must be the same length!"
            logger.debug("Validated lengths of chunks and metadatas")

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            mdata.append(metadatas[i])
            if embeddings is not None:
                embeds.append(embeddings[i])
            logger.debug(f"Processed chunk {i+1}/{len(chunks)}")

        d = {'documents':documents,
             'metadatas':metadatas,
             'ids':ids}
        
        if embeddings is not None:
            d['embeddings'] = embeddings
            logger.debug("Added embeddings to package")
        
        logger.info(f"Successfully processed batch of {len(chunks)} chunks")
        return d