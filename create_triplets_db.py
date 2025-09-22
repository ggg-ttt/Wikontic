from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import pymongo

from typing import List
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import time
import argparse
import logging
import os 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client

def create_search_index_for_entities(db, collection_name='entity_aliases', embedding_field_name="alias_text_embedding", index_name='entity_aliases'):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },

                    "sample_id": {
                        # "type": "number"
                        "type": "token"

                    }
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)  
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")
    
def create_search_index_for_properties(db, collection_name='property_aliases', embedding_field_name='alias_text_embedding', index_name='property_aliases'):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                },
            }
        },
        name=index_name,
    )
    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)  
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create collections and indexes for the dynamic triplets database')
    parser.add_argument('--mongo_uri', type=str, default="mongodb://localhost:27018/?directConnection=true")
    parser.add_argument('--db_name', type=str, default="triplets_db")
    parser.add_argument('--entity_aliases_collection', type=str, default="entity_aliases",
                        help='Collection name for entity aliases')
    parser.add_argument('--property_aliases_collection', type=str, default="property_aliases",
                        help='Collection name for property aliases')
    parser.add_argument('--triplets_collection', type=str, default="triplets",
                        help='Collection name for triplets')
    parser.add_argument('--initial_triplets_collection', type=str, default="initial_triplets",
                        help='Collection name for initial triplets')
    parser.add_argument('--filtered_triplets_collection', type=str, default="filtered_triplets",
                        help='Collection name for filtered triplets')
    parser.add_argument('--entity_aliases_index', type=str, default="entity_aliases",
                        help='Index name for entities')
    parser.add_argument('--property_aliases_index', type=str, default="property_aliases",
                        help='Index name for property aliases')
    parser.add_argument('--drop_collections', type=bool, default=False,
                        help='Drop existing collections')
                        
    args = parser.parse_args()
    mongo_uri = args.mongo_uri

    db_name = args.db_name
    mongo_client = get_mongo_client(mongo_uri)
    db = mongo_client.get_database(db_name)

    # Drop all existing collections
    if args.drop_collections:
        logger.info("Dropping existing collections...")
        for collection_name in db.list_collection_names():
            logger.info(f"Dropping collection: {collection_name}")
            db.drop_collection(collection_name)
        logger.info("Successfully dropped all existing collections")
    
    db.create_collection(args.entity_aliases_collection)
    db.create_collection(args.property_aliases_collection)

    db.create_collection(args.triplets_collection)
    db.create_collection(args.initial_triplets_collection)
    db.create_collection(args.filtered_triplets_collection)

    logger.info("Collections created successfully")
    db.entity_aliases.create_index([("sample_id", 1)])
    db.entity_aliases.create_index([("label", 1)])

    db.property_aliases.create_index([("sample_id", 1)])
    db.property_aliases.create_index([("label", 1)])

    db.triplets.create_index([("sample_id", 1)])
    db.initial_triplets.create_index([("sample_id", 1)])
    db.filtered_triplets.create_index([("sample_id", 1)])
    logger.info("Indexes created successfully")
    create_search_index_for_entities(db, collection_name=args.entity_aliases_collection, embedding_field_name="alias_text_embedding", index_name=args.entity_aliases_index)
    create_search_index_for_properties(db, collection_name=args.property_aliases_collection, embedding_field_name="alias_text_embedding", index_name=args.property_aliases_index)
    logger.info("Search index created successfully")
    logger.info("All indexes created successfully")
