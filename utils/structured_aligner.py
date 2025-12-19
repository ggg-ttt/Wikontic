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

@dataclass
class PropertyConstraints:
    subject_properties: Set[str]
    object_properties: Set[str]


class EntityAlias(BaseModel):
    _id: int
    label: str
    entity_type: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(self, ontology_db, triplets_db):
        """
        初始化 Aligner 类。

        设置数据库连接、集合名称、向量索引名称，并加载用于生成文本嵌入的 tokenizer 和模型。
        模型默认使用 'facebook/contriever' 并尝试加载到 GPU。

        Args:
            ontology_db: 本体数据库连接对象。
            triplets_db: 三元组数据库连接对象。
        """
        self.ontology_db = ontology_db
        self.triplets_db = triplets_db


        self.entity_type_collection_name = 'entity_types'
        self.entity_type_aliases_collection_name = 'entity_type_aliases'
        self.property_collection_name = 'properties'
        self.property_aliases_collection_name = 'property_aliases'

        self.entity_type_vector_index_name = 'entity_type_aliases'
        self.property_vector_index_name = 'property_aliases'

        self.entity_aliases_collection_name = 'entity_aliases'
        self.triplets_collection_name = 'triplets'
        self.filtered_triplets_collection_name = 'filtered_triplets'
        self.ontology_filtered_triplets_collection_name = 'ontology_filtered_triplets'
        self.initial_triplets_collection_name = 'initial_triplets'
        self.entities_vector_index_name = 'entity_aliases'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY"))
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        # self.model = AutoModel.from_pretrained('facebook/contriever', token=os.getenv("HF_KEY")).to(self.device)
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)


    def get_embedding(self, text):
        """
        获取输入文本的向量嵌入 (Embedding)。

        使用预加载的 Contriever 模型对文本进行编码，并采用平均池化 (mean pooling) 策略获取句子级别的向量。

        Args:
            text (str): 需要获取嵌入的文本字符串。

        Returns:
            List[float] or None: 文本的向量表示（浮点数列表）。如果输入无效或发生错误，返回 None。
        """
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
        
    
    def _get_unique_similar_entity_types(
        self,
        target_entity_type: str,
        k: int = 5,
        max_attempts: int = 10
    ) -> List[str]:
        """
        获取与目标实体类型最相似的 k 个唯一实体类型 ID。

        这是一个内部辅助函数。它通过向量搜索在本体数据库中查找相似的实体类型别名。
        由于搜索是在别名级别进行的，可能会返回重复的实体 ID，因此函数会循环查询直到收集到 k 个唯一的实体 ID。

        Args:
            target_entity_type (str): 目标实体类型的文本描述。
            k (int): 需要返回的相似实体类型数量。默认为 5。
            max_attempts (int): 最大尝试次数，防止无限循环。默认为 10。

        Returns:
            List[str]: 相似实体类型的 ID 列表。
        """
        # retrieve k most similar entity types to the given triplet
        # using the entity type index
        # return the wikidata ids of the most similar entity types
        
        query_k = k * 2
        attempt = 0
        unique_ranked_entities: List[str] = []
        query_embedding = self.get_embedding(target_entity_type)
        collection = self.ontology_db.get_collection(self.entity_type_aliases_collection_name)

        # as we search among aliases, there can be duplicated original entitites
        # and as we want K unique entities in result, we querying the index until we get exactly K unique entities
        while len(unique_ranked_entities) < k and attempt < max_attempts:
            search_pipeline = [{
                    "$vectorSearch": {
                    "index": self.entity_type_vector_index_name, #
                    "queryVector": query_embedding, 
                    "path": 'alias_text_embedding', 
                    "numCandidates": 150 if query_k < 150 else query_k, 
                    "limit": query_k 
                    }
                }, 
                {
                    "$project": {
                        "_id": 0,
                        "entity_type_id": 1

                    }
                }
            ]
            result = collection.aggregate(search_pipeline)
            for res in result: 
                if res['entity_type_id'] not in unique_ranked_entities:
                    unique_ranked_entities.append(res['entity_type_id'])
                if len(unique_ranked_entities) == k:
                    break
            query_k *= 2
            attempt += 1
        
        return unique_ranked_entities
    
    
    def retrieve_similar_entity_types(
        self,
        triplet: Dict[str, str],
        k: int = 10
    ) -> Tuple[List[str], List[str]]:

        """
        检索与给定三元组中的主语和宾语类型相似的实体类型。

        分别对三元组中的 'subject_type' 和 'object_type' 调用 `_get_unique_similar_entity_types`。

        Args:
            triplet (Dict[str, str]): 包含 'subject_type' 和可选 'object_type' 的三元组字典。
            k (int): 每个类型需要检索的相似数量。默认为 10。

        Returns:
            Tuple[List[str], List[str]]: 包含两个列表的元组，第一个是相似的主语类型 ID 列表，第二个是相似的宾语类型 ID 列表。
        """
        # collection = self.db.get_collection(self.entity_type_aliases_collection_name)

        # exact_match_subj_id = collection.find_one({"alias_label": triplet['subject_type']}, {"alias_label": 1, "entity_type_id": 1, "_id": 0})
        
        # print("EM ", triplet['subject_type'], exact_match_subj_id)

        # if exact_match_subj_id:
        #     similar_subject_types = [exact_match_subj_id['entity_type_id']]
        # else:
        #     # Get similar types for subject
        similar_subject_types = self._get_unique_similar_entity_types(
            target_entity_type=triplet['subject_type'],
            k=k
        )
        if 'object_type' in triplet:
            # exact_match_obj_id = collection.find_one({"alias_label": triplet['object_type']}, {"alias_label": 1, "entity_type_id": 1, "_id": 0})
            
            # print("EM ", exact_match_obj_id, triplet['object_type'])
            # if exact_match_obj_id:
            #     similar_object_types = [exact_match_obj_id['entity_type_id']]
            # else:
            similar_object_types = self._get_unique_similar_entity_types(
                target_entity_type=triplet['object_type'],
                k=k
            )
        else: 
            similar_object_types = []
        # print(similar_subject_types, similar_object_types)
        return similar_subject_types, similar_object_types
    

    def _get_valid_property_ids_by_entity_type(
        self,
        entity_type: str,
        is_object: bool = True
    ) -> Tuple[Set[str], Set[str]]:
        """
        获取指定实体类型的有效属性 ID。

        查询本体数据库，获取该实体类型及其所有父类型（层级结构）所允许的属性。
        区分该实体类型是作为三元组的宾语 (Object) 还是主语 (Subject) 来处理直接属性和逆向属性。

        Args:
            entity_type (str): 实体类型 ID。
            is_object (bool): 指示该实体类型在三元组中是否作为宾语。默认为 True。
                              如果为 True，返回 (object_props, subject_props) 作为 (direct, inverse)。
                              如果为 False，返回 (subject_props, object_props) 作为 (direct, inverse)。

        Returns:
            Tuple[Set[str], Set[str]]: 包含两个集合的元组 (direct_props, inverse_props)。
        """
        """
        Get direct and inverse properties for an entity type.
        
        Args:
            entity_type: The entity type to look up
            is_object: Whether this is an object type in triplet (True) or a subject type (False)
        """

        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        
        # Get extended types including supertypes
        extended_types = [entity_type, 'ANY']
        hirerarchy = collection.find_one({"entity_type_id": entity_type}, {"parent_type_ids": 1, "_id": 0})
        extended_types.extend(hirerarchy['parent_type_ids'])
        
        pipeline = [
            {"$match": {"entity_type_id": {"$in": extended_types}}},
            {
                "$group": {
                    "_id": None,
                    "subject_ids": {"$addToSet": {"$ifNull": ["$valid_subject_property_ids", []]}},
                    "object_ids": {"$addToSet": {"$ifNull": ["$valid_object_property_ids", []]}}
                }
            },
            {
                "$project": {
                    "subject_ids": {"$reduce": {
                        "input": "$subject_ids",
                        "initialValue": [],
                        "in": {"$setUnion": ["$$value", "$$this"]}
                    }},
                    "object_ids": {"$reduce": {
                        "input": "$object_ids",
                        "initialValue": [],
                        "in": {"$setUnion": ["$$value", "$$this"]}
                    }}
                }
            }
        ]
        result = collection.aggregate(pipeline)

        result_data = next(result, {})

        subject_props = result_data.get("subject_ids", [])
        object_props = result_data.get("object_ids", [])

        if is_object:
            direct_props = set(object_props)
            inverse_props = set(subject_props)
        else:
            direct_props = set(subject_props)
            inverse_props = set(object_props)

        return direct_props, inverse_props
    

    def _get_ranked_properties(
        self,
        prop_2_direction: Dict[str, List[str]], # mapping of property_ids to their direction that can be used in the specified context
        target_property: str,
        k: int
    ) -> List[Tuple[str, str]]: # List of tuples (<property_id>, <property_direction>)
        """
        根据与目标关系的相似度对候选属性进行排序。

        这是一个内部辅助函数。它在给定的候选属性集合 (`prop_2_direction`) 中，
        使用向量搜索找到与 `target_property` 语义最相似的属性。

        Args:
            prop_2_direction (Dict[str, List[str]]): 候选属性 ID 到其方向列表（'direct' 或 'inverse'）的映射。
            target_property (str): 目标关系的文本描述。
            k (int): 需要返回的排序后的属性数量。

        Returns:
            List[Tuple[str, str]]: 排序后的属性列表，每个元素为 (property_id, direction) 元组。
        """
        """
        Rank properties based on similarity to target relation.
        """
        collection = self.ontology_db.get_collection(self.property_aliases_collection_name)
        query_embedding = self.get_embedding(target_property)
        props = list(prop_2_direction.keys())
        # print("PROP2DIRECTION", prop_2_direction)
        query_k = k * 2
        max_attempts = 5  # Prevent infinite loops
        attempt = 0
        unique_ranked_properties: List[str] = []
        
        while len(unique_ranked_properties) < k and attempt < max_attempts:

            pipeline = [{
                "$vectorSearch": {
                    "index": self.property_vector_index_name, 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150 if query_k < 150 else query_k,  
                    "limit": query_k,  
                    "filter": {"relation_id": {"$in": props}},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "relation_id": 1,
                    # "score": {"$meta": "vectorSearchScore"} 
                }
            }
            ]

            similar_properties = collection.aggregate(pipeline)

            # print(list(similar_properties))

            for prop in similar_properties: 
                if prop['relation_id'] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop['relation_id'])
                if len(unique_ranked_properties) == k:
                    break
            
            query_k *= 2
            attempt += 1

        # print("UNIQUE RANKED PROPERTIES", unique_ranked_properties)
        # print("PROP 2 DIRECTION", prop_2_direction)
        # taking into account directions of properties 
        unique_ranked_properties_with_direction = []
        for prop_id in unique_ranked_properties:
            for direction in prop_2_direction[prop_id]:
                unique_ranked_properties_with_direction.append((prop_id, direction))
        # print("UNIQUE RANKED PROPERTIES WITH DIRECTION", unique_ranked_properties_with_direction)
            # if len(unique_ranked_properties_with_direction) >= k:
            #     break
        # print("UNIQUE RANKED PROPERTIES WITH DIRECTION", unique_ranked_properties_with_direction)
        return unique_ranked_properties_with_direction


    def retrieve_properties_for_entity_type(
        self,
        target_relation: str,  # relation from triplet
        object_types: List[str],
        subject_types: List[str],
        k: int = 10
    ) -> List[Tuple[str, str]]: # List of tuples (<property_id>, <property_direction>)
        """
        检索并排序与给定实体类型和关系匹配的属性。

        首先根据主语类型和宾语类型获取所有合法的属性约束（交集），
        然后在这些合法属性中搜索与 `target_relation` 最相似的属性。

        Args:
            target_relation (str): 三元组中的关系描述。
            object_types (List[str]): 可能的宾语实体类型 ID 列表。
            subject_types (List[str]): 可能的主语实体类型 ID 列表。
            k (int): 返回结果的数量。默认为 10。

        Returns:
            List[Tuple[str, str]]: 排序后的属性列表，每个元素为 (property_id, direction) 元组。
        """
        """
        Retrieve and rank properties that match given entity types and relation.
        
        Args:
            target_relation: The relation to search for
            object_types: List of valid object types
            subject_types: List of valid subject types
            k: Number of results to return
            
        Returns:
            List of tuples (<property_id>, <property_direction>)
        """
        # Initialize property constraints
        direct_props = PropertyConstraints(set(), set())
        inverse_props = PropertyConstraints(set(), set())

        # Collect object type properties
        for obj_type in object_types:
            obj_direct, obj_inverse = self._get_valid_property_ids_by_entity_type(obj_type, is_object=True)
            direct_props.object_properties.update(obj_direct)
            inverse_props.subject_properties.update(obj_inverse)
            
        # Collect subject type properties
        for subj_type in subject_types:
            subj_direct, subj_inverse = self._get_valid_property_ids_by_entity_type(subj_type, is_object=False)
            direct_props.subject_properties.update(subj_direct)
            inverse_props.object_properties.update(subj_inverse)

        # Find valid properties that satisfy both subject and object constraints
        valid_direct = direct_props.subject_properties & direct_props.object_properties
        valid_inverse = inverse_props.subject_properties & inverse_props.object_properties

        prop_id_2_direction = {prop_id: ["direct"] for prop_id in valid_direct}
        for prop_id in valid_inverse:
            if prop_id in prop_id_2_direction:
                prop_id_2_direction[prop_id].append("inverse")
            else: 
                prop_id_2_direction[prop_id] = ["inverse"]
        
        return self._get_ranked_properties(prop_id_2_direction, target_relation, k)


    def retrieve_properties_labels_and_constraints(self, property_id_list: List[str]) -> Dict[str, Dict[str, str]]:
        """
        根据属性 ID 列表检索属性的详细信息。

        查询本体数据库，获取每个属性的标签 (label) 以及有效的主语类型 ID 和宾语类型 ID。

        Args:
            property_id_list (List[str]): 属性 ID 列表。

        Returns:
            Dict[str, Dict[str, str]]: 属性 ID 到属性详细信息的映射字典。
                                       详细信息包括 'label', 'valid_subject_type_ids', 'valid_object_type_ids'。
        """
        collection = self.ontology_db.get_collection(self.property_collection_name)

        
        pipeline = [
            {"$match": {"property_id": {"$in": property_id_list}}},
            {
                "$project": {
                    "_id": 0,
                    "property_id": 1,
                    "label": 1,
                    "valid_subject_type_ids": 1,
                    "valid_object_type_ids": 1
                }
            }
        ]
        result = collection.aggregate(pipeline)

        result_dict = {
            item["property_id"]: {
                "label": item["label"],
                "valid_subject_type_ids": item["valid_subject_type_ids"],
                "valid_object_type_ids": item["valid_object_type_ids"],
            }
            for item in result
        }

        return result_dict
    

    def retrieve_entity_type_labels(self, entity_type_ids: List[str]):
        """
        根据实体类型 ID 列表检索实体类型的标签。

        Args:
            entity_type_ids (List[str]): 实体类型 ID 列表。

        Returns:
            Dict[str, str]: 实体类型 ID 到其标签 (label) 的映射字典。
        """
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        pipeline = [
            {"$match": {"entity_type_id": {"$in": entity_type_ids}}},
            {
                "$project": {
                    "_id": 0,
                    "entity_type_id": 1,
                    "label": 1,

                }
            }
        ]
        result = collection.aggregate(pipeline)

        result_dict = {
            item["entity_type_id"]: item["label"]
            for item in result
        }

        return result_dict


    def retrieve_entity_type_hierarchy(self, entity_type: str) -> List[str]:
        """
        检索指定实体类型的层级结构（包括自身和父类型）。

        Args:
            entity_type (str): 实体类型的标签 (label)。

        Returns:
            List[str]: 包含该实体类型 ID 及其所有父类型 ID 的列表。
        """
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        # print(entity_type)
        entity_id_parent_types = collection.find_one({"label": entity_type}, {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0})
        parent_type_id_labels = collection.find({"entity_type_id": {"$in": entity_id_parent_types['parent_type_ids']}}, {"_id": 0, "label": 1, "entity_type_id": 1})
        # ????!!!!
        if entity_id_parent_types:        
            extended_types = [entity_id_parent_types['entity_type_id']] + [item['entity_type_id'] for item in parent_type_id_labels]

        return extended_types


    def retrieve_entity_by_type(self, entity_name, entity_type, sample_id, k=10):
        """
        根据实体名称、类型和样本 ID 检索实体别名。

        首先获取实体类型及其父类型，然后在三元组数据库中进行向量搜索，
        查找符合类型约束和样本 ID 约束的相似实体别名。

        Args:
            entity_name (str): 实体名称（用于生成查询向量）。
            entity_type (str): 实体类型标签。
            sample_id (str): 样本 ID 约束。
            k (int): 返回结果的数量。默认为 10。

        Returns:
            Dict[str, str]: 别名 (alias) 到标签 (label) 的映射字典。
        """
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one({"label": entity_type}, {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0})        
        extended_types = [entity_id_parent_types['entity_type_id']] + entity_id_parent_types['parent_type_ids']
        extended_types = [elem['label'] for elem in collection.find({"entity_type_id": {"$in": extended_types}}, {"_id": 0, "label": 1, "entity_type_id": 1})]

        # print(extended_types)
    
        collection = self.triplets_db.get_collection(self.entity_aliases_collection_name)

        query_embedding = self.get_embedding(entity_name)
        # print(sample_id)
        pipeline = [{
            "$vectorSearch": {
                    "index": self.entities_vector_index_name, 
                    "queryVector": query_embedding,  
                    "path": "alias_text_embedding", 
                    "numCandidates": 150 if k < 150 else k,  
                    "limit": k,  
                    "filter": {"entity_type": {"$in": extended_types},
                               "sample_id": {"$eq": sample_id}
                               },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "label": 1,
                    "alias": 1
                }
            }
            ]

        result = collection.aggregate(pipeline)
        result_dict = {item['alias']: item['label'] for item in result}

        # print(result_dict)

        return result_dict

    def add_entity(self, entity_name, alias, entity_type, sample_id):
        """
        向数据库添加新的实体别名信息。

        如果数据库中不存在相同的记录，则插入新的实体别名文档，包括其向量嵌入。

        Args:
            entity_name (str): 实体标准名称 (label)。
            alias (str): 实体别名。
            entity_type (str): 实体类型。
            sample_id (str): 样本 ID。
        """
        # collection = self.db.get_collection(self.entity_type_collection_name)
        # entity_type_id = collection.find_one({"label": entity_type}, {"_id": 0, "entity_type_id": 1})['entity_type_id']

        collection = self.triplets_db.get_collection(self.entity_aliases_collection_name)
        if not collection.find_one({"label": entity_name, "entity_type": entity_type, "alias": alias, "sample_id": sample_id}):
            collection.insert_one({
                    "label": entity_name, 
                    "entity_type": entity_type,
                    "alias": alias, 
                    "sample_id": sample_id,
                    "alias_text_embedding": self.get_embedding(alias)
                })
            

    def add_triplets(self, triplets_list, sample_id):
        """
        批量添加三元组到数据库的 'triplets' 集合。

        使用 `UpdateOne` 操作进行 upsert（更新或插入），基于三元组内容和 sample_id 进行去重。

        Args:
            triplets_list (List[Dict]): 三元组字典列表。
            sample_id (str): 样本 ID，将被添加到每个三元组中。
        """
        collection = self.triplets_db.get_collection(self.triplets_collection_name)
        
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "subject_type": triple["subject_type"], "object_type": triple["object_type"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        
        if operations:
            collection.bulk_write(operations)

    
    def add_filtered_triplets(self, triplets_list, sample_id):
        """
        批量添加过滤后的三元组到数据库的 'filtered_triplets' 集合。

        使用 `UpdateOne` 操作进行 upsert。

        Args:
            triplets_list (List[Dict]): 三元组字典列表。
            sample_id (str): 样本 ID。
        """
        collection = self.triplets_db.get_collection(self.filtered_triplets_collection_name)
        
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "subject_type": triple["subject_type"], "object_type": triple["object_type"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        
        if operations:
            collection.bulk_write(operations)

    def add_ontology_filtered_triplets(self, triplets_list, sample_id):
        """
        批量添加经过本体过滤的三元组到数据库的 'ontology_filtered_triplets' 集合。

        使用 `UpdateOne` 操作进行 upsert。

        Args:
            triplets_list (List[Dict]): 三元组字典列表。
            sample_id (str): 样本 ID。
        """
        collection = self.triplets_db.get_collection(self.ontology_filtered_triplets_collection_name)
        
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "subject_type": triple["subject_type"], "object_type": triple["object_type"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        
        if operations:
            collection.bulk_write(operations)

    def add_initial_triplets(self, triplets_list, sample_id):
        """
        批量添加初始三元组到数据库的 'initial_triplets' 集合。

        使用 `UpdateOne` 操作进行 upsert。

        Args:
            triplets_list (List[Dict]): 三元组字典列表。
            sample_id (str): 样本 ID。
        """
        collection = self.triplets_db.get_collection(self.initial_triplets_collection_name)
        operations = []
        for triple in triplets_list:
            triple['sample_id'] = sample_id
            filter_query = {"subject": triple["subject"], "relation": triple["relation"], 
                            "object": triple["object"], "subject_type": triple["subject_type"], "object_type": triple["object_type"], "sample_id": triple["sample_id"]}
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)


    def retrieve_similar_entity_names(self, entity_name: str, k: int = 10, sample_id: str = None) -> List[Dict[str, str]]:
        """
        检索与给定实体名称相似的实体名称。

        使用向量搜索在实体别名集合中查找相似项。如果提供了 `sample_id`，则会添加过滤条件。

        Args:
            entity_name (str): 待查询的实体名称。
            k (int): 返回结果的数量。默认为 10。
            sample_id (str, optional): 可选的样本 ID 过滤条件。

        Returns:
            List[Dict[str, str]]: 包含相似实体信息的字典列表，每个字典包含 'entity' (label)。
        """
        # print(f"Searching for similar entities to: {entity_name}, k: {k}, sample_id: {sample_id}")
        embedded_query = self.get_embedding(entity_name)
        collection = self.triplets_db.get_collection(self.entity_aliases_collection_name)
        # print(f"Available search indexes: {list(collection.list_search_indexes())}")
        
        # First try to search with sample_id filter if provided
        if sample_id:
            pipeline = [{
                "$vectorSearch": {
                        "index": self.entities_vector_index_name, 
                        "queryVector": embedded_query,  
                        "path": "alias_text_embedding", 
                        "numCandidates": 150,  
                        "limit": k,  
                        "filter": {
                                    "sample_id": {"$eq": sample_id},
                                },
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "label": 1, 
                        "entity_type": 1
                    }
                }
                ]
        else:
            pipeline = [{
                "$vectorSearch": {
                        "index": self.entities_vector_index_name, 
                        "queryVector": embedded_query,  
                        "path": "alias_text_embedding", 
                        "numCandidates": 150,  
                        "limit": k,  
                        # "filter": {
                        #             # "entity_type": {"$eq": "Q483394"},
                        #             # "sample_id": {"$eq": sample_id},
                        #         },
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "label": 1, 
                        "entity_type": 1
                    }
                }
                ]

        # Debug: print the pipeline to ensure it's constructed as expected
        # print("Aggregation pipeline:", pipeline)
        result = collection.aggregate(pipeline)
        # Convert the cursor to a list so we can print and reuse it
        result_list = list(result)
        # print("Aggregation result:", result_list)
        # If result_list is empty, possible reasons:
        # - The vector index or collection is empty or misconfigured
        # - The filter (e.g., sample_id) is too restrictive or doesn't match any documents
        # - The embedding query is malformed or not compatible with the index
        # - The 'alias_text_embedding' field is missing or not indexed
        # - The vector search index name is incorrect
        # - The database connection or permissions are incorrect
        # result_dict = [{'entity': item['label'], 'entity_type': item['entity_type']} for item in result_list]
        result_dict = [{'entity': item['label']} for item in result_list]

        return result_dict
