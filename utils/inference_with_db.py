from unidecode import unidecode
import re
import warnings
import tenacity
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger('InferenceWithDB')
logger.setLevel(logging.DEBUG)
    
class InferenceWithDB:
    def __init__(self, extractor, aligner, triplets_db, initial_triplets_db=None):
        self.extractor = extractor
        self.aligner = aligner
        self.db = triplets_db
        self.initial_triplets_db = initial_triplets_db


    def extract_triplets(self, text, sample_id, source_text_id=None):
        """
        Extract triplets from text using LLM.
        """
        self.extractor.reset_tokens()
        self.extractor.reset_messages()
        self.extractor.reset_error_state()

        initial_triplets = []

        if self.initial_triplets_db is None:
            logger.info("Initial triplets db is None")
            extracted_triplets = self.extractor.extract_triplets_from_text(text)
            extracted_triplets = extracted_triplets['triplets']
            

            for triplet in extracted_triplets:
                triplet['prompt_token_num'], triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                triplet['source_text_id'] = source_text_id
                triplet['sample_id'] = sample_id
                initial_triplets.append(triplet.copy())
        else:
            logger.info("Initial triplets db is not None")
            extracted_triplets = list(self.initial_triplets_db.get_collection("initial_triplets").find({"sample_id": sample_id, "source_text_id": source_text_id}))
            for triplet in extracted_triplets:
                initial_triplets.append(triplet.copy())
        
        final_triplets = []
        filtered_triplets = []

        for triplet in extracted_triplets:
            self.extractor.reset_tokens()
            try:
                logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))
                refined_subject = self.refine_entity_name(text, triplet, sample_id, is_object=False)
                refined_object = self.refine_entity_name(text, triplet, sample_id, is_object=True)

                triplet['subject'] = refined_subject
                triplet['object'] = refined_object

                refined_relation = self.refine_relation_name(text, triplet, sample_id)
                triplet['relation'] = refined_relation

                final_triplets.append(triplet)
                logger.log(logging.DEBUG, "Final triplet: %s\n%s" % (str(triplet), "-" * 100))
                logger.log(logging.DEBUG, "Refined subject: %s\n%s" % (str(refined_subject), "-" * 100))
                logger.log(logging.DEBUG, "Refined object: %s\n%s" % (str(refined_object), "-" * 100))
                logger.log(logging.DEBUG, "Refined relation: %s\n%s" % (str(refined_relation), "-" * 100))

            except Exception as e:
                triplet['exception_text'] = str(e)
                triplet['prompt_token_num'], triplet['completion_token_num'] = self.extractor.calculate_used_tokens()
                triplet['sample_id'] = sample_id
                filtered_triplets.append(triplet)
                logger.log(logging.INFO, "Filtered triplet: %s\n%s" % (str(triplet), "-" * 100))
                logger.log(logging.INFO, "Exception: %s" % (str(e)))

        return initial_triplets, final_triplets, filtered_triplets


    def refine_entity_name(self, text, triplet, sample_id, is_object=False):

        """
        Refine entity names using type constraints.
        """
        self.extractor.reset_error_state()
        if is_object:
            entity = triplet['object']
        else:
            entity = triplet['subject']

        if isinstance(entity, str):
            entity = unidecode(entity)

        similar_entities = self.aligner.retrieve_similar_entities(
            target_entity=entity,
            sample_id=sample_id
        )
        # if there are similar entities -> refine entity name
        # if no similar entities -> return the original entity
        # if exact match found -> return the exact match
        if len(similar_entities) == 0 or entity in similar_entities:
            updated_entity = entity
        else:
            # if not exact match -> refine entity name
            updated_entity = self.extractor.refine_entity(
                text=text,
                triplet=triplet,
                candidates=similar_entities,
                is_object=is_object
            )
            # unidecode the updated entity
            updated_entity = unidecode(updated_entity)
            # if the updated entity is None (meaning that LLM didn't find any similar entities) 
            # -> return the original entity
            if re.sub(r'[^\w\s]', '', updated_entity) == 'None':
                updated_entity = entity

        self.aligner.add_entity(
            entity_name=updated_entity,
            alias=entity,
            sample_id=sample_id
        )

        return updated_entity


    def refine_relation_name(self, text, triplet, sample_id):
        """
        Refine relation names using LLM.
        """
        self.extractor.reset_error_state()

        relation = unidecode(triplet['relation'])

        similar_relations: List[str] = self.aligner.retrieve_similar_properties(
            target_relation=relation,
            sample_id=sample_id
        )
        if len(similar_relations) == 0 or relation in similar_relations:
            updated_relation = relation
        else:
            updated_relation = self.extractor.refine_relation_wo_entity_types(
                text=text,
                triplet=triplet,
                candidate_relations=similar_relations
            )
            
            updated_relation = unidecode(updated_relation)

            if re.sub(r'[^\w\s]', '', updated_relation) == 'None':
                updated_relation = relation

        self.aligner.add_property(
            property_name=updated_relation,
            alias=relation,
            sample_id=sample_id
        )

        return updated_relation

    def identify_relevant_entities_from_question(self, question, sample_id):
        """
        Identify relevant entities from question using LLM.
        """
        self.extractor.reset_error_state()

        entities = self.extractor.extract_entities_from_question(question)
        print("Entities: ", entities)

        if isinstance(entities, dict):
            entities = [entities]

        relevant_entities = []
        linked_entities = []

        for entity in entities:
            similar_entities = self.aligner.retrieve_similar_entities(
                target_entity=entity,
                sample_id=sample_id
            )
            print("Similar entities: ", similar_entities)
            exact_entity_match = [e for e in similar_entities if e == entity]
            if len(exact_entity_match) > 0:
                relevant_entities.extend(exact_entity_match)
            else:
                linked_entities.extend(similar_entities)
        print("Linked entities: ", linked_entities)
        identified_entities = self.extractor.identify_relevant_entities_wo_types(question, linked_entities)
        relevant_entities.extend(
            [e['entity'] for e in identified_entities]
        )
        return relevant_entities

    
    def answer_question(self, question, relevant_entities, sample_id='0', use_qualifiers=False):
        print("Initial relevant entities: ", relevant_entities)
        entities4search = list(set(relevant_entities))
        or_conditions = []

        for ent in entities4search:
            # typ_hierarchy = self.aligner.retrieve_entity_type_hirerarchy(typ)
            # print("Typ hierarchy: ", typ_hierarchy)
            or_conditions.append({'$and': [{'subject': ent}]})
            or_conditions.append({'$and': [{'object': ent}]})
            if use_qualifiers:
                or_conditions.append({'$and': [{'qualifiers.object': ent}]})

        # print("Entities4search: ", entities4search)
        for _ in range(5):
            or_conditions = []
            for ent in entities4search:
                # print(ent)
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
            # print(self.triplets_db.get_collection(self.aligner.triplets_collection_name))
            results = list(self.db.get_collection(self.aligner.triplets_collection_name).aggregate(pipeline))

            for doc in results:
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
        
    def answer_with_qa_collapsing(self, question, sample_id='0', max_attempts=5, use_qualifiers=False):
        collapsed_question_answer = ''
        collapsed_question_sequence = []
        collapsed_answer_sequence = []

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
                similar_entities = self.aligner.retrieve_similar_entities(
                    target_entity=ent, k=10, sample_id=sample_id
                )
                # similar_entities = [e['entity'] for e in similar_entities]
                entities4search.extend(similar_entities)
            
            entities4search = list(set(entities4search))
            logger.log(logging.DEBUG, "Similar entities: %s" % (str(entities4search)))
            

            if len(extracted_entities) == 0:
                return collapsed_question_answer

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
            results = list(self.db.get_collection(self.aligner.triplets_collection_name).aggregate(pipeline))

            if use_qualifiers:
                supporting_triplets = [
                    # only use the qualifiers that are related to the subject or object
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

            if len(supporting_triplets) == 0:
                return collapsed_question_answer

            logger.log(logging.DEBUG, 'Supporting triplets length: %s' % (str(len(supporting_triplets))))
            logger.log(logging.DEBUG, "Supporting triplets: %s\n%s" % (str(supporting_triplets), "-" * 100))

            collapsed_question_answer = self.extractor.answer_question(collapsed_question, supporting_triplets)
            collapsed_question_sequence.append(collapsed_question)
            collapsed_answer_sequence.append(collapsed_question_answer)

            if len(collapsed_question_answer) == 0:
                return collapsed_question_answer

            logger.log(logging.DEBUG, "Collapsed question: %s" % (str(collapsed_question)))
            logger.log(logging.DEBUG, "Collapsed question answer: %s" % (str(collapsed_question_answer)))

            is_answered = self.extractor.check_if_question_is_answered(question, collapsed_question_sequence, collapsed_answer_sequence)
            if is_answered == 'YES':
                return collapsed_question_answer
            else:
                collapsed_question = self.extractor.collapse_question(original_question=question, question=collapsed_question, answer=collapsed_question_answer)
                continue
        
        logger.log(logging.DEBUG, "Final answer: %s" % (str(collapsed_question_answer)))
        return collapsed_question_answer
