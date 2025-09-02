from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

import chromadb
from chromadb.utils import embedding_functions


class OpenAIEmbeddingModel(Enum):
    ADA_002 = "text-embedding-ada-002"
    SMALL_3 = "text-embedding-3-small"
    LARGE_3 = "text-embedding-3-large"

def get_chromadb_client(db_name:str)-> chromadb.ClientAPI:
    client = chromadb.PersistentClient(path=f"./local_dbs/{db_name}")
    return client

def remove_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    if collection_name in client.list_collections():
        print(f"Removing existing collection {collection_name}")
        client.delete_collection(collection_name)
    else:
        print(f"Collection {collection_name} does not exist nothing to remove")

@dataclass
class VectorDBItem:
    id: str
    text: str
    metadata: Optional[dict] = None
    distance: Optional[float] = None
    embedding: Optional[list[float]] = None





class VectorCollection:

    def __init__(
        self,
        name: str,
        client: chromadb.ClientAPI,
        token: str,
        metadata: Optional[dict] = None,
        embedding_model: OpenAIEmbeddingModel = OpenAIEmbeddingModel.ADA_002
    ):
        self.name = name
        self.metadata = metadata
        self.chromadb_collection = client.create_collection(
            name,
            embedding_function=self._embedding_function(token, embedding_model),
            metadata=metadata,
            get_or_create=True,
        )


    @staticmethod
    def _embedding_function(token: str, model: OpenAIEmbeddingModel):
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=token, model_name=model.value
        )


    def add_item(self, document, id,  metadata=None) -> None:
        self.chromadb_collection.add(
            documents=[document], metadatas=[metadata], ids=[id]
        )

    def get_item(self, id: str) -> Optional[VectorDBItem]:
        response = self.chromadb_collection.get(id, include=["documents", "metadatas", "embeddings"])
        if len(response["ids"]) == 0:
            return None
        return VectorDBItem(
            id=response["ids"][0],
            text=response["documents"][0],
            metadata=response["metadatas"][0],
            embedding=response["embeddings"][0]
        )

    def similar_items(self, input_text: str, n_results: int = 10) -> List[VectorDBItem]:
        response = self.chromadb_collection.query(
            query_texts=[input_text], n_results=n_results, include=["documents", "metadatas", "embeddings", "distances"]
        )
        items = []
        for id, document, metadata, distance, embedding in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            response["embeddings"][0],
        ):
            items.append(
                VectorDBItem(
                    id=id,
                    text=document,
                    metadata=metadata,
                    distance=distance,
                    embedding=embedding,
                )
            )
        return items