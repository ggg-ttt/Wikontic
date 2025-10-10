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
    def __init__(self, extractor, aligner, triplets_db):
        self.extractor = extractor
        self.aligner = aligner
        self.triplets_db = triplets_db

    def extract_triplets_with_ontology_filtering(self, text, sample_id, source_text_id=None):
        """
        Extract and refine knowledge graph triplets from text using LLM.

        Args:
            text (str): Input text to extract triplets from

        Returns:
            tuple: (initial_triplets, final_triplets, filtered_triplets)
        """
        self.extractor.reset_tokens()
        self.extractor.reset_messages()

        self.extractor.reset_error_state()
        extracted_triplets = self.extractor.extract_triplets_from_text(text)
        
        initial_triplets = []
        for triplet in extracted_triplets['triplets']:
            triplet['prompt_token_num'], triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
            triplet['source_text_id'] = source_text_id
            triplet['sample_id'] = sample_id
            initial_triplets.append(triplet.copy())

        final_triplets = []
        filtered_triplets = []
        ontology_filtered_triplets = []

        for triplet in extracted_triplets['triplets']:
            self.extractor.reset_tokens()
            backbone_triplet = triplet.copy()
            try:
                logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))
                
                # ___________________________ Refine entity types  ___________________________
                subj_type_ids, obj_type_ids = self.get_candidate_entity_type_ids(triplet)

                entity_type_id_2_label = self.get_candidate_entity_labels(
                    subj_type_ids=subj_type_ids, obj_type_ids=obj_type_ids
                )
                entity_type_label_2_id = {entity_label: entity_id for entity_id, entity_label in entity_type_id_2_label.items()}

                candidate_subject_types = [entity_type_id_2_label[t] for t in subj_type_ids]
                candidate_object_types = [entity_type_id_2_label[t] for t in obj_type_ids]

                # no need to refine if the triplet's types are in the candidate types
                if triplet['subject_type'] in candidate_subject_types and triplet['object_type'] in candidate_object_types:
                    refined_subject_type, refined_object_type = triplet['subject_type'], triplet['object_type']
                else:
                    # if the triplet's subject type is in the candidate types, then only refine the subject type
                    if triplet['subject_type'] in candidate_subject_types:
                        candidate_subject_types = [triplet['subject_type']]
                    # if the triplet's object type is in the candidate types, then only refine the object type
                    if triplet['object_type'] in candidate_object_types:
                        candidate_object_types = [triplet['object_type']]
                
                    refined_subject_type, refined_object_type = self.refine_entity_types(
                        text=text, triplet=triplet, candidate_subject_types=candidate_subject_types, candidate_object_types=candidate_object_types
                    )

                # ___________________________ Refine relation ___________________________
                # if refined subject and object types are in the candidate types, then refine the relation
                if refined_subject_type in candidate_subject_types and refined_object_type in candidate_object_types:
                    refined_subject_type_id = entity_type_label_2_id[refined_subject_type]
                    refined_object_type_id = entity_type_label_2_id[refined_object_type]

                    relation_direction_candidate_pairs, prop_2_label_and_constraint = self.get_candidate_entity_properties(
                        triplet=triplet, subj_type_ids=[refined_subject_type_id], obj_type_ids=[refined_object_type_id] 
                    )
                    candidate_relations = [prop_2_label_and_constraint[p[0]]['label'] for p in relation_direction_candidate_pairs]

                    # no need to refine if the triplet's relation is in the candidate relations
                    if triplet['relation'] in candidate_relations:
                        refined_relation = triplet['relation']
                    else:
                        refined_relation = self.refine_relation(
                            text=text, triplet=triplet, candidate_relations=candidate_relations
                        )
                # if refined subject and object types are not in the candidate types, then do not refine the relation
                else:
                    refined_relation = triplet['relation']
                    prop_2_label_and_constraint = {}
                    candidate_relations = []
                
                # if refined relation is in the candidate relations, then refine the relation direction
                if refined_relation in candidate_relations:
                    refined_relation_id_candidates = [p_id for p_id in prop_2_label_and_constraint if prop_2_label_and_constraint[p_id]['label'] == refined_relation]
                    refined_relation_id = refined_relation_id_candidates[0]
                    refined_relation_directions = [p[1] for p in relation_direction_candidate_pairs if p[0] == refined_relation_id]
                    refined_relation_direction = 'direct' if 'direct' in refined_relation_directions else 'inverse'

                    if refined_relation_direction == 'inverse':
                        refined_subject_type_id, refined_object_type_id = refined_object_type_id, refined_subject_type_id
                        refined_subject_type, refined_object_type = refined_object_type, refined_subject_type
                        candidate_subject_types, candidate_object_types = candidate_object_types, candidate_subject_types
                else:
                    refined_relation_direction = 'direct'
                    
                # ___________________________ Refine entity names ___________________________
                backbone_triplet = {
                    "subject": triplet['subject'] if refined_relation_direction == 'direct' else triplet['object'],
                    "relation": refined_relation,
                    "object": triplet['object'] if refined_relation_direction == 'direct' else triplet['subject'],
                    "subject_type": refined_subject_type,
                    "object_type": refined_object_type,
                }
            
                backbone_triplet['qualifiers'] = triplet['qualifiers']
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

                # final_triplet = {
                #     "subject": refined_subject,
                #     "relation": backbone_triplet['relation'],
                #     "object": refined_object,
                #     "subject_type": backbone_triplet['subject_type'],
                #     "object_type": backbone_triplet['object_type'],
                #     "qualifiers": backbone_triplet['qualifiers']
                # }
                backbone_triplet['subject'] = refined_subject
                backbone_triplet['object'] = refined_object

                backbone_triplet['prompt_token_num'], backbone_triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                backbone_triplet['source_text_id'] = source_text_id
                backbone_triplet['sample_id'] = sample_id
                
                # ___________________________ Validate backbone triplet ___________________________
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
                    final_triplets.append(backbone_triplet.copy())
                    logger.log(logging.DEBUG, "Final triplet: %s\n%s" % (str(backbone_triplet), "-" * 100))
                else:
                    logger.log(logging.ERROR, "Final triplet is ontology filtered: %s\n%s" % (str(backbone_triplet), "-" * 100))
                    logger.log(logging.ERROR, "Exception: %s" % (str(backbone_triplet_exception_msg)))
                    logger.log(logging.ERROR, "Refined relation: %s" % (str(refined_relation)))
                    logger.log(logging.ERROR, "Refined subject type: %s" % (str(refined_subject_type)))
                    logger.log(logging.ERROR, "Refined object type: %s" % (str(refined_object_type)))
                    logger.log(logging.ERROR, "Candidate subject types: %s" % (str(candidate_subject_types)))
                    logger.log(logging.ERROR, "Candidate object types: %s" % (str(candidate_object_types)))
                    logger.log(logging.ERROR, "Candidate relations: %s" % (str(candidate_relations)))
                    logger.log(logging.ERROR, "Prop 2 label and constraint: %s" % (str(prop_2_label_and_constraint)))
                    backbone_triplet['candidate_subject_types'] = candidate_subject_types
                    backbone_triplet['candidate_object_types'] = candidate_object_types
                    backbone_triplet['candidate_relations'] = candidate_relations
                    backbone_triplet['exception_text'] = backbone_triplet_exception_msg
                    ontology_filtered_triplets.append(backbone_triplet.copy())

            except Exception as e:
                backbone_triplet['prompt_token_num'], backbone_triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                backbone_triplet['source_text_id'] = source_text_id
                backbone_triplet['sample_id'] = sample_id
                backbone_triplet['exception_text'] = str(e)
                filtered_triplets.append(backbone_triplet.copy())
                logger.log(logging.INFO, "Filtered triplet: %s\n%s" % (str(backbone_triplet), "-" * 100))
                logger.log(logging.INFO, "Exception: %s" % (str(e)))

        return initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets

    def get_candidate_entity_type_ids(
        self, triplet: Dict[str, str]
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieve candidate subject and object entity type IDs.
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
        Retrieve entity type labels for subject and object types.
        """
        entity_type_id_2_label = self.aligner.retrieve_entity_type_labels(
            subj_type_ids + obj_type_ids
        )
        return entity_type_id_2_label

    def refine_entity_types(self, text, triplet, candidate_subject_types, candidate_object_types):
        """
        Refine entity types using LLM.
        """
        self.extractor.reset_error_state()
        refined_entity_types = self.extractor.refine_entity_types(
            text=text, triplet=triplet, candidate_subject_types=candidate_subject_types, candidate_object_types=candidate_object_types
        )
        return refined_entity_types['subject_type'], refined_entity_types['object_type']
    

    def refine_relation(self, text, triplet, candidate_relations):
        """
        Refine relation using LLM.
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
        Retrieve candidate properties and their labels/constraints.
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
        Check if the selected backbone_triplet's types and relation are in the valid sets.
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
        Refine entity names using type constraints.
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
    #     Build candidate triplet backbones for LLM refinement.
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
    #     Refine relation and entity types using LLM.
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
    #     Check if the selected backbone_triplet's types and relation are in the valid sets.
    #     """
    #     property_2_candidate_entities = {item['relation']: {"subject_types": item["subject_types"],
    #                                                         "object_types": item['object_types'] }
    #                                                         for item in candidate_triplets}
        
    #     if backbone_triplet['relation'] in property_2_candidate_entities.keys():
    #         candidate = property_2_candidate_entities[backbone_triplet['relation']]
            
    #         if backbone_triplet['subject_type'] in candidate['subject_types'] and backbone_triplet['object_type'] in candidate['object_types']:
    #             return True
        
    #     return False