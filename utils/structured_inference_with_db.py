from unidecode import unidecode
import re
import warnings
import tenacity
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger('StructuredInferenceWithDB')
logger.setLevel(logging.DEBUG)

class StructuredInferenceWithDB:
    """
    结构化推理类，结合本体约束进行知识图谱三元组提取和问答

    该类是Wikontic项目的核心组件，负责：
    1. 从文本中提取三元组并进行本体对齐
    2. 利用Wikidata本体约束验证三元组的语义一致性
    3. 基于知识图谱进行智能问答
    4. 实现实体名称的标准化和去重
    """

    def __init__(self, extractor, aligner, triplets_db):
        """
        初始化结构化推理器

        参数:
            extractor: LLM三元组提取器，负责从文本中提取初步三元组
            aligner: 结构化对齐器，负责本体对齐和实体名称优化
            triplets_db: 三元组数据库连接，用于存储和检索知识图谱数据
        """
        self.extractor = extractor
        self.aligner = aligner
        self.triplets_db = triplets_db

    def extract_triplets_with_ontology_filtering(self, text, sample_id, source_text_id=None):
        """
        使用本体约束从文本中提取和优化知识图谱三元组

        这是本类的核心方法，实现了完整的三元组提取和本体对齐流程：
        1. 使用LLM从文本中提取初始三元组
        2. 通过本体对齐优化实体类型和关系
        3. 基于Wikidata约束验证三元组有效性
        4. 实现实体名称的标准化和去重
        5. 根据验证结果分类存储三元组

        参数:
            text (str): 要提取三元组的输入文本
            sample_id (str): 样本ID，用于数据追踪和管理
            source_text_id (str, optional): 源文本ID，用于多文档场景

        返回:
            tuple: 包含四个三元组列表的元组
                - initial_triplets: LLM初始提取的原始三元组
                - final_triplets: 通过所有验证的最终有效三元组
                - filtered_triplets: 处理过程中出现异常的三元组
                - ontology_filtered_triplets: 被本体约束过滤掉的三元组
        """
        # 重置提取器状态，准备处理新的文本
        self.extractor.reset_tokens()
        self.extractor.reset_messages()

        self.extractor.reset_error_state()
        extracted_triplets = self.extractor.extract_triplets_from_text(text)

        # 记录初始提取的三元组，包含token使用情况和元数据
        initial_triplets = []
        for triplet in extracted_triplets['triplets']:
            triplet['prompt_token_num'], triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
            triplet['source_text_id'] = source_text_id
            triplet['sample_id'] = sample_id
            initial_triplets.append(triplet.copy())

        # 初始化最终结果容器，用于分类存储不同处理结果的三元组
        final_triplets = []          # 通过所有验证的有效三元组
        filtered_triplets = []       # 处理过程中出现异常的三元组
        ontology_filtered_triplets = []  # 被本体约束过滤掉的三元组

        for triplet in extracted_triplets['triplets']:
            self.extractor.reset_tokens()
            backbone_triplet = triplet.copy()
            try:
                logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))
                
                # ___________________________ 第一步：优化实体类型 ___________________________
                # 从Wikidata本体中获取候选的实体类型ID
                subj_type_ids, obj_type_ids = self.get_candidate_entity_type_ids(triplet)

                # 将实体类型ID转换为可读的标签名称
                entity_type_id_2_label = self.get_candidate_entity_labels(
                    subj_type_ids=subj_type_ids, obj_type_ids=obj_type_ids
                )
                # 建立标签到ID的反向映射
                entity_type_label_2_id = {entity_label: entity_id for entity_id, entity_label in entity_type_id_2_label.items()}

                # 获取候选的实体类型标签列表
                candidate_subject_types = [entity_type_id_2_label[t] for t in subj_type_ids]
                candidate_object_types = [entity_type_id_2_label[t] for t in obj_type_ids]

                # 如果三元组的类型已在候选集合中，则无需优化
                if triplet['subject_type'] in candidate_subject_types and triplet['object_type'] in candidate_object_types:
                    refined_subject_type, refined_object_type = triplet['subject_type'], triplet['object_type']
                else:
                    # 如果主体类型已在候选集合中，只优化客体类型
                    if triplet['subject_type'] in candidate_subject_types:
                        candidate_subject_types = [triplet['subject_type']]
                    # 如果客体类型已在候选集合中，只优化主体类型
                    if triplet['object_type'] in candidate_object_types:
                        candidate_object_types = [triplet['object_type']]

                    # 调用LLM优化实体类型
                    refined_subject_type, refined_object_type = self.refine_entity_types(
                        text=text, triplet=triplet, candidate_subject_types=candidate_subject_types, candidate_object_types=candidate_object_types
                    )

                # ___________________________ 第二步：优化关系名称和方向 ___________________________
                # 只有当优化后的类型都在候选集合中时，才进行关系优化
                if refined_subject_type in candidate_subject_types and refined_object_type in candidate_object_types:
                    refined_subject_type_id = entity_type_label_2_id[refined_subject_type]
                    refined_object_type_id = entity_type_label_2_id[refined_object_type]

                    # 获取兼容的候选属性及其约束信息
                    relation_direction_candidate_pairs, prop_2_label_and_constraint = self.get_candidate_entity_properties(
                        triplet=triplet, subj_type_ids=[refined_subject_type_id], obj_type_ids=[refined_object_type_id]
                    )
                    candidate_relations = [prop_2_label_and_constraint[p[0]]['label'] for p in relation_direction_candidate_pairs]

                    # 如果关系已在候选集合中，则无需优化
                    if triplet['relation'] in candidate_relations:
                        refined_relation = triplet['relation']
                    else:
                        # 调用LLM优化关系名称
                        refined_relation = self.refine_relation(
                            text=text, triplet=triplet, candidate_relations=candidate_relations
                        )
                # 如果类型不在候选集合中，则不进行关系优化
                else:
                    refined_relation = triplet['relation']
                    prop_2_label_and_constraint = {}
                    candidate_relations = []

                # 确定关系的方向（正向或反向）
                if refined_relation in candidate_relations:
                    refined_relation_id_candidates = [p_id for p_id in prop_2_label_and_constraint if prop_2_label_and_constraint[p_id]['label'] == refined_relation]
                    refined_relation_id = refined_relation_id_candidates[0]
                    refined_relation_directions = [p[1] for p in relation_direction_candidate_pairs if p[0] == refined_relation_id]
                    refined_relation_direction = 'direct' if 'direct' in refined_relation_directions else 'inverse'

                    # 如果是反向关系，需要交换主体和客体
                    if refined_relation_direction == 'inverse':
                        refined_subject_type_id, refined_object_type_id = refined_object_type_id, refined_subject_type_id
                        refined_subject_type, refined_object_type = refined_object_type, refined_subject_type
                        candidate_subject_types, candidate_object_types = candidate_object_types, candidate_subject_types
                else:
                    refined_relation_direction = 'direct'
                    
                # ___________________________ 第三步：优化实体名称 ___________________________
                # 构建优化后的三元组骨干，考虑关系方向
                backbone_triplet = {
                    "subject": triplet['subject'] if refined_relation_direction == 'direct' else triplet['object'],
                    "relation": refined_relation,
                    "object": triplet['object'] if refined_relation_direction == 'direct' else triplet['subject'],
                    "subject_type": refined_subject_type,
                    "object_type": refined_object_type,
                }

                # 保留原始限定词信息
                backbone_triplet['qualifiers'] = triplet['qualifiers']

                # 基于类型约束优化实体名称
                if refined_subject_type in candidate_subject_types:
                    refined_subject = self.refine_entity_name(text, backbone_triplet, sample_id, is_object=False)
                else:
                    refined_subject = triplet['subject']
                if refined_object_type in candidate_object_types:
                    refined_object = self.refine_entity_name(text, backbone_triplet, sample_id, is_object=True)
                else:
                    refined_object = triplet['object']

                logger.log(logging.DEBUG, "Original subject name: %s\n%s" % (str(backbone_triplet['subject']), "-" * 100))
                logger.log(logging.DEBUG, "Original object name: %s\n%s" % (str(backbone_triplet['object']), "-" * 100))
                logger.log(logging.DEBUG, "Refined subject name: %s\n%s" % (str(refined_subject), "-" * 100))
                logger.log(logging.DEBUG, "Refined object name: %s\n%s" % (str(refined_object), "-" * 100))

                # 将优化后的实体名称更新到三元组中
                backbone_triplet['subject'] = refined_subject
                backbone_triplet['object'] = refined_object

                # 添加token使用情况和元数据信息
                backbone_triplet['prompt_token_num'], backbone_triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                backbone_triplet['source_text_id'] = source_text_id
                backbone_triplet['sample_id'] = sample_id

                # ___________________________ 第四步：本体约束验证 ___________________________
                # 验证优化后的三元组是否符合Wikidata本体约束
                backbone_triplet_valid, backbone_triplet_exception_msg = self.validate_backbone(
                    backbone_triplet['subject_type'],
                    backbone_triplet['object_type'],
                    backbone_triplet['relation'],
                    candidate_subject_types,
                    candidate_object_types,
                    candidate_relations,
                    prop_2_label_and_constraint
                )

                if backbone_triplet_valid:
                    # 验证通过，添加到最终有效三元组列表
                    final_triplets.append(backbone_triplet.copy())
                    logger.log(logging.DEBUG, "Final triplet: %s\n%s" % (str(backbone_triplet), "-" * 100))
                else:
                    # 验证失败，记录详细的错误信息并添加到本体过滤列表
                    logger.log(logging.ERROR, "Final triplet is ontology filtered: %s\n%s" % (str(backbone_triplet), "-" * 100))
                    logger.log(logging.ERROR, "Exception: %s" % (str(backbone_triplet_exception_msg)))
                    logger.log(logging.ERROR, "Refined relation: %s" % (str(refined_relation)))
                    logger.log(logging.ERROR, "Refined subject type: %s" % (str(refined_subject_type)))
                    logger.log(logging.ERROR, "Refined object type: %s" % (str(refined_object_type)))
                    logger.log(logging.ERROR, "Candidate subject types: %s" % (str(candidate_subject_types)))
                    logger.log(logging.ERROR, "Candidate object types: %s" % (str(candidate_object_types)))
                    logger.log(logging.ERROR, "Candidate relations: %s" % (str(candidate_relations)))
                    logger.log(logging.ERROR, "Prop 2 label and constraint: %s" % (str(prop_2_label_and_constraint)))

                    # 添加详细的过滤信息，用于后续分析和调试
                    backbone_triplet['candidate_subject_types'] = candidate_subject_types
                    backbone_triplet['candidate_object_types'] = candidate_object_types
                    backbone_triplet['candidate_relations'] = candidate_relations
                    backbone_triplet['exception_text'] = backbone_triplet_exception_msg
                    ontology_filtered_triplets.append(backbone_triplet.copy())

            except Exception as e:
                # 处理过程中发生异常，记录错误信息并添加到异常过滤列表
                backbone_triplet['prompt_token_num'], backbone_triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                backbone_triplet['source_text_id'] = source_text_id
                backbone_triplet['sample_id'] = sample_id
                backbone_triplet['exception_text'] = str(e)
                filtered_triplets.append(backbone_triplet.copy())
                logger.log(logging.INFO, "Filtered triplet: %s\n%s" % (str(backbone_triplet), "-" * 100))
                logger.log(logging.INFO, "Exception: %s" % (str(e)))

        # 返回四个分类的三元组列表
        return initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets

    def get_candidate_entity_type_ids(
        self, triplet: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        """
        获取候选主体和客体实体类型ID

        基于三元组中的实体名称和类型信息，通过语义相似度检索
        从Wikidata本体中查找最匹配的实体类型ID

        参数:
            triplet (Dict[str, str]): 包含subject、object、subject_type、object_type的三元组字典

        返回:
            Tuple[List[str], List[str]]:
                - 第一个列表: 主体实体的候选类型ID列表
                - 第二个列表: 客体实体的候选类型ID列表
        """
        subj_type_ids, obj_type_ids = self.aligner.retrieve_similar_entity_types(
            triplet=triplet
        )
        return subj_type_ids, obj_type_ids


    def get_candidate_entity_labels(
        self,
        subj_type_ids: List[str],
        obj_type_ids: List[str]
    ) -> Dict[str, dict]:
        """
        获取主体和客体实体类型的标签映射

        将实体类型的Wikidata ID转换为可读的标签名称，
        建立ID到标签的映射关系，用于后续的LLM交互

        参数:
            subj_type_ids (List[str]): 主体实体的候选类型ID列表
            obj_type_ids (List[str]): 客体实体的候选类型ID列表

        返回:
            Dict[str, dict]: 实体类型ID到标签的映射字典
                key: 实体类型ID (如 "Q5", "Q6256")
                value: 对应的标签名称 (如 "human", "country")
        """
        entity_type_id_2_label = self.aligner.retrieve_entity_type_labels(
            subj_type_ids + obj_type_ids
        )
        return entity_type_id_2_label

    def refine_entity_types(self, text, triplet, candidate_subject_types, candidate_object_types):
        """
        使用LLM优化实体类型

        当LLM初始提取的实体类型不在候选类型集合中时，
        调用LLM根据原始文本和候选类型列表重新选择最合适的实体类型

        参数:
            text (str): 原始输入文本，提供上下文信息
            triplet (Dict): 当前处理的三元组，包含subject、object、subject_type、object_type
            candidate_subject_types (List[str]): 主体实体的候选类型标签列表
            candidate_object_types (List[str]): 客体实体的候选类型标签列表

        返回:
            Tuple[str, str]: 优化后的主体和客体实体类型标签
                - 第一个元素: 优化后的主体类型
                - 第二个元素: 优化后的客体类型
        """
        self.extractor.reset_error_state()
        refined_entity_types = self.extractor.refine_entity_types(
            text=text, triplet=triplet, candidate_subject_types=candidate_subject_types, candidate_object_types=candidate_object_types
        )
        return refined_entity_types['subject_type'], refined_entity_types['object_type']
    

    def refine_relation(self, text, triplet, candidate_relations):
        """
        使用LLM优化关系名称

        当LLM初始提取的关系不在候选关系集合中时，
        调用LLM根据原始文本和候选关系列表重新选择最合适的关系

        参数:
            text (str): 原始输入文本，提供上下文信息
            triplet (Dict): 当前处理的三元组，包含subject、object、relation
            candidate_relations (List[str]): 候选关系标签列表

        返回:
            str: 优化后的关系标签
        """
        self.extractor.reset_error_state()
        refined_relation = self.extractor.refine_relation(
            text=text, triplet=triplet, candidate_relations=candidate_relations
        )
        return refined_relation['relation']
    
    def get_candidate_entity_properties(
        self,
        triplet: Dict[str, str],
        subj_type_ids: List[str],
        obj_type_ids: List[str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, dict]]:
        """
        获取候选实体属性及其标签和约束信息

        基于给定的主体和客体实体类型，从Wikidata本体中检索
        兼容的属性，包括属性的方向信息（正向/反向）和约束规则

        参数:
            triplet (Dict[str, str]): 当前处理的三元组，包含关系信息
            subj_type_ids (List[str]): 主体实体的类型ID列表
            obj_type_ids (List[str]): 客体实体的类型ID列表

        返回:
            Tuple[List[Tuple[str, str]], Dict[str, dict]]:
                - 第一个元素: 属性ID和方向的元组列表
                    元组格式: (property_id, direction)
                    direction为"direct"（正向）或"inverse"（反向）
                - 第二个元素: 属性详细信息字典
                    key: property_id
                    value: {
                        "label": 属性标签,
                        "valid_subject_type_ids": 有效主体类型ID列表,
                        "valid_object_type_ids": 有效客体类型ID列表
                    }
        """
        # Get the list of tuples (<property_id>, <property_direction>)
        properties: List[Tuple[str, str]] = self.aligner.retrieve_properties_for_entity_type(
            target_relation=triplet['relation'],
            object_types=obj_type_ids,
            subject_types=subj_type_ids,
            k=10
        )
        # Get dict {<prop_id>:
        #           {"label": <prop_label>,
        #           "valid_subject_type_ids": <valid_subject_type_ids>,
        #           "valid_object_type_ids": <valid_object_type_ids>}}
        prop_2_label_and_constraint = self.aligner.retrieve_properties_labels_and_constraints(
            property_id_list=[p[0] for p in properties]
        )
        return properties, prop_2_label_and_constraint

    
    def validate_backbone(
        self,
        refined_subject_type: str,
        refined_object_type: str,
        refined_relation: str,
        candidate_subject_types: List[str],
        candidate_object_types: List[str],
        candidate_relations: List[str],
        prop_2_label_and_constraint: Dict[str, dict]
    ):
        """
        验证三元组骨干是否符合本体约束

        这是本体过滤的核心函数，检查优化后的三元组是否满足Wikidata的约束规则：
        1. 检查类型兼容性：主体和客体类型是否在候选集合中
        2. 检查关系兼容性：关系是否在候选关系集合中
        3. 检查属性约束：实体类型层次结构是否与属性约束匹配

        参数:
            refined_subject_type (str): 优化后的主体实体类型
            refined_object_type (str): 优化后的客体实体类型
            refined_relation (str): 优化后的关系标签
            candidate_subject_types (List[str]): 主体候选类型列表
            candidate_object_types (List[str]): 客体候选类型列表
            candidate_relations (List[str]): 候选关系列表
            prop_2_label_and_constraint (Dict[str, dict]): 属性约束信息字典

        返回:
            Tuple[bool, str]:
                - 第一个元素: 验证是否通过 (True=通过, False=失败)
                - 第二个元素: 错误信息，验证失败时提供详细原因
        """

        exception_msg = ''
        if refined_relation not in candidate_relations:
            exception_msg += "Refined relation not in candidate relations\n"
        if refined_subject_type not in candidate_subject_types:
            exception_msg += "Refined subject type not in candidate subject types\n"
        if refined_object_type not in candidate_object_types:
            exception_msg += "Refined object type not in candidate object types\n"
        
        if exception_msg != '':
            return False, exception_msg
        
        else:

            # logger.log(logging.DEBUG, "Prop 2 label and constraint: %s\n%s" % (str(prop_2_label_and_constraint), "-" * 100))
            subject_type_hierarchy = self.aligner.retrieve_entity_type_hierarchy(refined_subject_type)
            object_type_hierarchy = self.aligner.retrieve_entity_type_hierarchy(refined_object_type)

            # logger.log(logging.DEBUG, "Subject type hierarchy for entity type %s: %s\n%s" % (str(refined_subject_type), str(subject_type_hierarchy), "-" * 100))
            # logger.log(logging.DEBUG, "Object type hierarchy for entity type %s: %s\n%s" % (str(refined_object_type), str(object_type_hierarchy), "-" * 100))
            prop_subject_type_ids = [prop_2_label_and_constraint[prop]['valid_subject_type_ids'] for prop in prop_2_label_and_constraint if prop_2_label_and_constraint[prop]['label'] == refined_relation][0]
            prop_object_type_ids = [prop_2_label_and_constraint[prop]['valid_object_type_ids'] for prop in prop_2_label_and_constraint if prop_2_label_and_constraint[prop]['label'] == refined_relation][0]

            if prop_subject_type_ids == ['ANY']:
                prop_subject_type_ids = subject_type_hierarchy
            if prop_object_type_ids == ['ANY']:
                prop_object_type_ids = object_type_hierarchy

            if any([t in subject_type_hierarchy for t in prop_subject_type_ids]) and any([t in object_type_hierarchy for t in prop_object_type_ids]):
                return True, exception_msg
            else:
                exception_msg += 'Triplet backbone violates property constraints\n'
                return False, exception_msg
    

    def refine_entity_name(self, text, triplet, sample_id, is_object=False):
        """
        使用类型约束优化实体名称

        基于实体类型约束和语义相似度检索，对实体名称进行标准化和去重：
        1. 检索同类型下的相似实体名称
        2. 优先选择完全匹配的实体名称
        3. 使用LLM从候选名称中选择最合适的
        4. 将优化后的实体名称添加到数据库中

        特殊处理：时间实体(Q186408)和数量实体(Q309314)不进行名称优化

        参数:
            text (str): 原始输入文本，提供上下文信息
            triplet (Dict): 当前处理的三元组
            sample_id (str): 样本ID，用于数据管理
            is_object (bool): 是否为客体实体，默认为False（主体实体）

        返回:
            str: 优化后的实体名称
        """
        self.extractor.reset_error_state()
        if is_object:
            entity = unidecode(triplet['object'])
            entity_type = triplet['object_type']
            entity_hierarchy = self.aligner.retrieve_entity_type_hierarchy(
                entity_type
            )
        else:
            entity = unidecode(triplet['subject'])
            entity_type = triplet['subject_type']
            entity_hierarchy = []

        # do not change time or quantity entities (of objects!)
        if any([t in ['Q186408', 'Q309314'] for t in entity_hierarchy]):
            updated_entity = entity
        else:
            # if not time or quantity entities -> retrieve similar entities by type and name similarity
            similar_entities = self.aligner.retrieve_entity_by_type(
                entity_name=entity,
                entity_type=entity_type,
                sample_id=sample_id
            )
            # if there are similar entities -> refine entity name
            if len(similar_entities) > 0:
                # if exact match found -> return the exact match
                if entity in similar_entities:
                    updated_entity = similar_entities[entity]
                else:
                    # if not exact match -> refine entity name
                    updated_entity = self.extractor.refine_entity(
                        text=text,
                        triplet=triplet,
                        candidates=list(similar_entities.values()),
                        is_object=is_object
                    )
                    # unidecode the updated entity
                    updated_entity = unidecode(updated_entity)
                    # if the updated entity is None (meaning that LLM didn't find any similar entities) 
                    # -> return the original entity
                    if re.sub(r'[^\w\s]', '', updated_entity) == 'None':
                        updated_entity = entity
            else:
                # if no similar entities -> return the original entity
                updated_entity = entity
        
        self.aligner.add_entity(
            entity_name=updated_entity,
            alias=entity,
            entity_type=entity_type,
            sample_id=sample_id
        )

        return updated_entity

    def identify_relevant_entities_from_question(self, question, sample_id='0'):
        """
        从问题中识别相关实体

        处理流程：
        1. 使用LLM从问题中提取实体名称
        2. 通过语义相似度检索查找数据库中的相似实体
        3. 优先选择完全匹配的实体
        4. 使用LLM从候选实体中选择最相关的

        参数:
            question (str): 用户的问题文本
            sample_id (str): 样本ID，用于限定搜索范围，默认为'0'

        返回:
            List[Dict]: 相关实体信息列表，每个元素包含：
                - entity: 实体名称
                - entity_type: 实体类型
                - 其他相关信息
        """
        entities = self.extractor.extract_entities_from_question(question)
        # print(entities)
        identified_entities = []
        chosen_entities = []

        if isinstance(entities, dict):
            entities = [entities]

        for ent in entities:
            similar_entities = self.aligner.retrieve_similar_entity_names(
                entity_name=ent, k=10, sample_id=sample_id
            )
            print("Similar entities: ", similar_entities)

            exact_entity_match = [e for e in similar_entities if e['entity'] == ent]
            if len(exact_entity_match) > 0:
                chosen_entities.extend(exact_entity_match)
            else:
                identified_entities.extend(similar_entities)
        

        print("Identified entities: ", identified_entities)
        print("Chosen entities: ", chosen_entities)

        chosen_entities.extend(
            self.extractor.identify_relevant_entities(
                question=question, entity_list=identified_entities
            )
        )
        print("Chosen entities after identification: ", chosen_entities)
        return chosen_entities

    def answer_question(self, question, relevant_entities, sample_id='0', use_filtered_triplets=False, use_qualifiers=False):
        """
        基于知识图谱回答用户问题

        处理流程：
        1. 从相关实体中提取实体名称
        2. 构建MongoDB查询条件，查找包含这些实体的三元组
        3. 多轮扩展搜索，包含新发现的实体
        4. 使用LLM基于检索到的三元组回答问题

        参数:
            question (str): 用户的问题文本
            relevant_entities (List[Dict]): 相关实体列表
            sample_id (str): 样本ID，限定搜索范围，默认为'0'
            use_filtered_triplets (bool): 是否使用过滤的三元组，默认False
            use_qualifiers (bool): 是否包含限定词信息，默认False

        返回:
            Tuple[List[Dict], str]:
                - 第一个元素: 支撑答案的三元组列表
                - 第二个元素: LLM生成的答案文本
        """
        print("Chosen relevant entities: ", relevant_entities)
        # entity_set = {(e['entity'], e['entity_type']) for e in relevant_entities}
        entity_set = {e['entity'] for e in relevant_entities}
        entities4search = list(entity_set)
        or_conditions = []

        for val in entities4search:
            # typ_hierarchy = self.aligner.retrieve_entity_type_hirerarchy(typ)
            # print("Typ hierarchy: ", typ_hierarchy)
            or_conditions.append({'$and': [{'subject': val}]})
            or_conditions.append({'$and': [{'object': val}]})

        entities4search = list(entity_set)
        # print("Entities4search: ", entities4search)
        for _ in range(5):
            or_conditions = []
            for ent in entities4search:
                # print(ent)
                or_conditions.append({'$and': [{'subject': ent}]})
                or_conditions.append({'$and': [{'object': ent}]})

            pipeline = [
                {
                    '$match': {
                        'sample_id': sample_id,
                        '$or': or_conditions
                    }
                }
            ]
            # print(self.triplets_db.get_collection(self.aligner.triplets_collection_name))
            results = list(self.triplets_db.get_collection(self.aligner.triplets_collection_name).aggregate(pipeline))
            for doc in results:
                entities4search.append(doc['subject'])
                entities4search.append(doc['object'])
                if use_qualifiers:
                    for q in doc['qualifiers']:
                        entities4search.append(q['object'])
                    
            if use_filtered_triplets:
                filtered_results = list(self.triplets_db.get_collection(self.aligner.ontology_filtered_triplets_collection_name).aggregate(pipeline))
                for doc in filtered_results:
                    entities4search.append(doc['subject'])
                    entities4search.append(doc['object'])
                    if use_qualifiers:
                        for q in doc['qualifiers']:
                            entities4search.append(q['object'])

            entities4search = list(set(entities4search))

        # print(results)
        if use_qualifiers:
            supporting_triplets = [
            {
                "subject": item['subject'],
                "relation": item['relation'],
                "object": item['object'],
                "qualifiers": item['qualifiers']
            }
            for item in results
        ]
        else:
            supporting_triplets = [
                {
                    "subject": item['subject'],
                    "relation": item['relation'],
                    "object": item['object']
                }
                for item in results
            ]
        logger.log(logging.DEBUG, "Supporting triplets: %s\n%s" % (str(supporting_triplets), "-" * 100))

        ans = self.extractor.answer_question(
            question=question, triplets=supporting_triplets
        )
        return supporting_triplets, ans

    
    def answer_with_qa_collapsing(self, question, sample_id='0', max_attempts=5, use_qualifiers=False, use_filtered_triplets=False):
        """
        使用问题分解和折叠的方式回答复杂问题

        对于复杂的多跳问题，采用迭代分解的方法：
        1. 将复杂问题分解为简单子问题
        2. 逐个回答子问题，构建中间答案序列
        3. 将中间答案作为上下文，继续回答后续子问题
        4. 最终整合所有信息形成完整答案

        参数:
            question (str): 用户的复杂问题
            sample_id (str): 样本ID，限定搜索范围，默认为'0'
            max_attempts (int): 最大迭代次数，默认5次
            use_qualifiers (bool): 是否使用限定词，默认False
            use_filtered_triplets (bool): 是否使用过滤三元组，默认False

        返回:
            str: 最终生成的完整答案
        """
        collapsed_question_answer = ''
        collapsed_question_sequence = []
        collapsed_answer_sequence = []
        # supporting_triplets_sequence = []

        logger.log(logging.DEBUG, "Question: %s" % (str(question)))
        collapsed_question = self.extractor.decompose_question(question)

        for i in range(max_attempts):
            extracted_entities = self.extractor.extract_entities_from_question(collapsed_question)
            logger.log(logging.DEBUG, "Collapsed question: %s" % (str(collapsed_question)))
            logger.log(logging.DEBUG, "Extracted entities: %s" % (str(extracted_entities)))

            if len(collapsed_question_answer) > 0:
                extracted_entities.append(collapsed_question_answer)
            
            entities4search = []
            for ent in extracted_entities:
                similar_entities = self.aligner.retrieve_similar_entity_names(
                    entity_name=ent, k=10, sample_id=sample_id
                )
                similar_entities = [e['entity'] for e in similar_entities]
                entities4search.extend(similar_entities)
            
            entities4search = list(set(entities4search))
            logger.log(logging.DEBUG, "Similar entities: %s" % (str(entities4search)))
            

            # if len(extracted_entities) == 0:
            #     return collapsed_question_answer

            or_conditions = []
            for ent in entities4search:
                or_conditions.append({'$and': [{'subject': ent}]})
                or_conditions.append({'$and': [{'object': ent}]})
                if use_qualifiers:
                    or_conditions.append({'$and': [{'qualifiers.object': ent}]})

            pipeline = [
                {
                    '$match': {
                        'sample_id': sample_id,
                        '$or': or_conditions
                    }
                }
            ]
            results = list(self.triplets_db.get_collection(self.aligner.triplets_collection_name).aggregate(pipeline))

            if use_filtered_triplets:
                logger.log(logging.DEBUG, "Using filtered triplets")
                filtered_results = list(self.triplets_db.get_collection(self.aligner.ontology_filtered_triplets_collection_name).aggregate(pipeline))
                logger.log(logging.DEBUG, "Filtered results: %s" % (str(len(filtered_results))))
                results.extend(filtered_results)

            if use_qualifiers:
                supporting_triplets = [

                    {
                        "subject": item['subject'],
                        "relation": item['relation'],
                        "object": item['object'],
                        "qualifiers": item['qualifiers']
                    }
                    for item in results
                ]
            else:
                supporting_triplets = [
                    {
                        "subject": item['subject'],
                        "relation": item['relation'],
                        "object": item['object']
                    }
                    for item in results
                ]

            # if len(supporting_triplets) == 0:
            #     return collapsed_question_answer

            # supporting_triplets_sequence.append(supporting_triplets)

            logger.log(logging.DEBUG, 'Supporting triplets length: %s' % (str(len(supporting_triplets))))
            # logger.log(logging.DEBUG, "Supporting triplets: %s\n%s" % (str(supporting_triplets), "-" * 100))

            collapsed_question_answer = self.extractor.answer_question(collapsed_question, supporting_triplets)
            collapsed_question_sequence.append(collapsed_question)
            collapsed_answer_sequence.append(collapsed_question_answer)

            # if len(collapsed_question_answer) == 0:
            #     return collapsed_question_answer

            logger.log(logging.DEBUG, "Collapsed question: %s" % (str(collapsed_question)))
            logger.log(logging.DEBUG, "Collapsed question answer: %s" % (str(collapsed_question_answer)))


            is_answered = self.extractor.check_if_question_is_answered(question, collapsed_question_sequence, collapsed_answer_sequence)
            question_answer_sequence = list(zip(collapsed_question_sequence, collapsed_answer_sequence))
            logger.log(logging.DEBUG, 'Collapsed question-answer sequence: %s' % (str(question_answer_sequence)))

            if is_answered == 'NOT FINAL':
                collapsed_question = self.extractor.collapse_question(original_question=question, question=collapsed_question, answer=collapsed_question_answer)
                continue
            else:
                return is_answered
        
        logger.log(logging.DEBUG, "Final answer: %s" % (str(collapsed_question_answer)))
        return collapsed_question_answer


    # def build_candidate_triplet_backbones(
    #     self,
    #     triplet,
    #     property_direction_pairs,
    #     prop_2_label_and_constraint,
    #     entity_type_id_2_label,
    #     subj_type_ids,
    #     obj_type_ids
    # ):
    #     """
    #     为LLM优化构建候选三元组骨干
    #
    #     基于属性约束和方向信息，构建符合本体规则的候选三元组集合，
    #     为后续的LLM优化提供结构化的选择空间
    #
    #     参数:
    #         triplet: 原始三元组信息
    #         property_direction_pairs: 属性ID和方向的配对列表
    #         prop_2_label_and_constraint: 属性约束信息字典
    #         entity_type_id_2_label: 实体类型ID到标签的映射
    #         subj_type_ids: 主体类型ID列表
    #         obj_type_ids: 客体类型ID列表
    #
    #     返回:
    #         List[Dict]: 候选三元组骨干列表，每个包含：
    #             - subject: 主体名称
    #             - relation: 关系标签
    #             - object: 客体名称
    #             - subject_types: 主体候选类型列表
    #             - object_types: 客体候选类型列表
    #     """
    #     backbone_candidates = []
    #     for prop_id, prop_direction in property_direction_pairs:
    #         # for each property identify valid subject and object ids as well as its label
    #         prop_valid_subject_type_ids = prop_2_label_and_constraint[prop_id]['valid_subject_type_ids']
    #         prop_valid_object_type_ids = prop_2_label_and_constraint[prop_id]['valid_object_type_ids']
    #         property_label = prop_2_label_and_constraint[prop_id]['label']

    #         # intersect subject and object type ids similar to ones from input 
    #         # and valid property type id identified from input 
    #         if prop_direction == 'direct':
    #             # do not intersect if property doesn't have any constraints
    #             subject_types = set(subj_type_ids) & set(prop_valid_subject_type_ids) if len(prop_valid_subject_type_ids) > 0 else subj_type_ids

    #             object_types = set(obj_type_ids) & set(prop_valid_object_type_ids) if len(prop_valid_object_type_ids) > 0 else obj_type_ids
    #         else:
    #             # do not intersect if property doesn't have any constraints
    #             subject_types = set(obj_type_ids) & set(prop_valid_subject_type_ids) if len(prop_valid_subject_type_ids) > 0 else obj_type_ids
    #             object_types = set(subj_type_ids) & set(prop_valid_object_type_ids) if len(prop_valid_object_type_ids) > 0 else subj_type_ids

    #         backbone_candidates.append({
    #             "subject": triplet['subject'] if prop_direction == 'direct' else triplet['object'],
    #             "relation": property_label,
    #             "object": triplet['object'] if prop_direction == 'direct' else triplet['subject'],
    #             "subject_types": [entity_type_id_2_label[t] for t in subject_types],
    #             "object_types": [entity_type_id_2_label[t] for t in object_types]
    #         })
    #     return backbone_candidates

    
    # def refine_backbone_with_llm(self, text, triplet, candidate_triplets):
    #     """
    #     使用LLM优化三元组骨干的关系和实体类型
    #
    #     从候选三元组骨干集合中，使用LLM选择最优的关系和实体类型组合，
    #     确保结果既符合原始文本语义，又满足本体约束
    #
    #     参数:
    #         text: 原始文本上下文
    #         triplet: 原始三元组信息
    #         candidate_triplets: 候选三元组骨干列表
    #
    #     返回:
    #         Dict: LLM优化后的三元组骨干
    #     """
    #     backbone_triplet = self.extractor.refine_relation_and_entity_types(
    #         text=text,
    #         triplet=triplet,
    #         candidate_triplets=candidate_triplets,
    #     )
    #     return backbone_triplet

    # def validate_backbone_triplet(
    #     self,
    #     backbone_triplet,
    #     candidate_triplets

    # ):
    #     """
    #     验证选定的三元组骨干是否在有效集合中
    #
    #     检查LLM选择的三元组骨干的关系和实体类型组合
    #     是否在候选三元组集合的允许范围内
    #
    #     参数:
    #         backbone_triplet: 待验证的三元组骨干
    #         candidate_triplets: 候选三元组列表
    #
    #     返回:
    #         bool: 验证结果，True表示有效，False表示无效
    #     """
    #     property_2_candidate_entities = {item['relation']: {"subject_types": item["subject_types"],
    #                                                         "object_types": item['object_types'] }
    #                                                         for item in candidate_triplets}
        
    #     if backbone_triplet['relation'] in property_2_candidate_entities.keys():
    #         candidate = property_2_candidate_entities[backbone_triplet['relation']]
            
    #         if backbone_triplet['subject_type'] in candidate['subject_types'] and backbone_triplet['object_type'] in candidate['object_types']:
    #             return True
        
    #     return False