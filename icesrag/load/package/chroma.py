from typing import List, Dict, Union, Optional, Any
from icesrag.load.package.strategy_pattern import PackageStrategy
from datetime import datetime

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
        ids = []
        if isinstance(chunks, str):
            chunks = [chunks]
        for i, chunk in enumerate(chunks):
            now = datetime.now().strftime("%H:%M:%S")
            if tag is None:
                id = f"{i}_{now}"
            else:
                if isinstance(tag, list):
                    id = f"{tag[i]}_{i}_{now}"
                else:
                    id = f"{tag}_{i}_{now}"
            ids.append(id)
        return ids

    def package(self, chunk: str, 
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
        ids = self.create_ids(chunks=chunk, tag=tag)
        d = {'documents':[chunk],
             'metadatas':[metadata],
             'ids':ids}
        if embedding is not None:
            d['embeddings'] = [embedding]
        return d

    def batch_package(self, 
                      chunks: List[str], 
                      metadatas: List[Dict], 
                      embeddings: Optional[List[List[float]]] = None, 
                      tag: Optional[List[str]] = None) -> Any:
        """
        Package a batch of chunks, embeddings, metadata, and IDs.
        
        Args:
            chunks (List[str]): A list of document chunks.
            metadatas (List[Dict]): A list of metadata dictionaries, each corresponding to a chunk.
            embeddings (Optional[List[List[float]]]): A list of embeddings, each corresponding to a chunk.
            tag (Optional[List[str]]): An optional tag to prepend to the IDs.
        
        Returns:
            List[Dict]: A list of dictionaries, each containing a chunk, embedding, metadata, and ID.
        """
        documents = []
        embeds = []
        mdata = []
        ids = self.create_ids(chunks=chunks, tag=tag)
        assert len(chunks) == len(embeddings) == len(metadatas), "chunks, embeddings, and metadatas must be the same length!"

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            mdata.append(metadatas[i])
            if embeddings is not None:
                embeds.append(embeddings[i])

        d = {'documents':documents,
             'metadatas':metadatas,
             'ids':ids}
        
        if embeddings is not None:
            d['embeddings'] = embeddings
        
        return d