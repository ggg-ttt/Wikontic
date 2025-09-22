from typing import List, Tuple, Set, Dict
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
from pymongo import MongoClient, UpdateOne
import torch
from dotenv import load_dotenv, find_dotenv
import os 

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
_ = load_dotenv(find_dotenv())



class EntityAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]

class PropertyAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(self, triplets_db):
        self.db = triplets_db

        self.entity_aliases_collection_name = 'entity_aliases'
        self.property_aliases_collection_name = 'property_aliases'

        self.property_vector_index_name = 'property_aliases'
        self.entities_vector_index_name = 'entity_aliases'

        self.initial_triplets_collection_name = 'initial_triplets'
        self.triplets_collection_name = 'triplets'
        self.filtered_triplets_collection_name = 'filtered_triplets'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY"))
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        # self.model = AutoModel.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY")).to(self.device)
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)


    def get_embedding(self, text):

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        if not text or not isinstance(text, str):
            return None

        try:
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**inputs.to(self.device))
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            return embeddings.detach().cpu().tolist()[0]
        
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            return None


    def retrieve_similar_properties(
        self,
        target_relation: str,
        sample_id: str,
        k: int = 10
    ) -> List[str]: # List of property labels
        """
        Retrieve and rank properties that match given relation.
        
        Args:
            target_relation: The relation to search for
            k: Number of results to return
            
        Returns:
            List of property labels
        """
        
        collection = self.db.get_collection(self.property_aliases_collection_name)
        query_embedding = self.get_embedding(target_relation)

        query_k = k * 2
        max_attempts = 5  # 
        attempt = 0
        unique_ranked_properties: List[str] = []
        
        while len(unique_ranked_properties) < k and attempt < max_attempts:

            pipeline = [{
                "$vectorSearch": {
                    "index": self.property_vector_index_name, 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150 ,  
                    "limit": query_k if query_k < 150 else 150,  
                    # "filter": {
                    #                 "sample_id": {"$eq": sample_id},
                    #             },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "label": 1,
                    # "alias": 1
                    # "score": {"$meta": "vectorSearchScore"} 
                }
            }
            ]

            similar_properties = collection.aggregate(pipeline)

            for prop in similar_properties: 
                if prop['label'] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop['label'])
                if len(unique_ranked_properties) == k:
                    break
            
            query_k *= 2
            attempt += 1
        
        return unique_ranked_properties
    
    def retrieve_similar_entities(
        self,
        target_entity: str,
        sample_id: str,
        k: int = 10
    ) -> List[str]: # List of entity labels
        """
        Retrieve and rank entities that match given entity.
        
        Args:
            target_entity: The entity to search for
            k: Number of results to return
            
        Returns:
            List of entity labels
        """
        
        collection = self.db.get_collection(self.entity_aliases_collection_name)
        query_embedding = self.get_embedding(target_entity)

        query_k = k * 2
        max_attempts = 5  # 
        attempt = 0
        unique_ranked_entities: List[str] = []
        
        while len(unique_ranked_entities) < k and attempt < max_attempts:

            pipeline = [{
                "$vectorSearch": {
                    "index": self.entities_vector_index_name, 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150,  
                    "limit": query_k if query_k < 150 else 150,  
                    "filter": {
                                    "sample_id": {"$eq": sample_id},
                                },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "label": 1,
                    # "score": {"$meta": "vectorSearchScore"} 
                }
            }
            ]

            similar_entities = collection.aggregate(pipeline)

            for entity in similar_entities: 
                if entity['label'] not in unique_ranked_entities:
                    unique_ranked_entities.append(entity['label'])
                if len(unique_ranked_entities) == k:
                    break
            
            query_k *= 2
            attempt += 1
        
        return unique_ranked_entities



    def add_entity(self, entity_name, alias, sample_id):
        collection = self.db.get_collection(self.entity_aliases_collection_name)
        if not collection.find_one({"label": entity_name, "alias": alias, "sample_id": sample_id}):

            collection.insert_one({
                    "label": entity_name, 
                    "alias": alias, 
                    "sample_id": sample_id,
                    "alias_text_embedding": self.get_embedding(alias)
                })
    
    def add_property(self, property_name, alias, sample_id):
        collection = self.db.get_collection(self.property_aliases_collection_name)
        if not collection.find_one({"label": property_name, "alias": alias}):
            collection.insert_one({
                "label": property_name, 
                "alias": alias, 
                # "sample_id": sample_id,
                "alias_text_embedding": self.get_embedding(alias)
            })

    def add_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.triplets_collection_name)
        
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        
        if operations:
            collection.bulk_write(operations)

    
    def add_filtered_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.filtered_triplets_collection_name)
        
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        
        if operations:
            collection.bulk_write(operations)

    def add_initial_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.initial_triplets_collection_name)
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)
