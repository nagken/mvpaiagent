"""
Document Reasoning Agent - Third agent in the ADK pipeline
Specializes in logical reasoning, inference, and knowledge synthesis
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from utils.adk_framework import BaseAgent, AgentResult, AgentStatus
from utils.document_processor import DocumentChunk, DocumentMetadata


class ReasoningAgent(BaseAgent):
    """
    Agent responsible for logical reasoning and knowledge synthesis.
    
    Capabilities:
    - Logical inference and deduction from document content
    - Knowledge graph construction and relationship mapping
    - Question answering and fact verification
    - Hypothesis generation and testing
    - Causal relationship identification
    - Contradiction detection and resolution
    - Multi-document reasoning and synthesis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ReasoningAgent", config)
        
        # Agent-specific configuration
        self.reasoning_mode = config.get('reasoning_mode', 'comprehensive')
        self.inference_depth = config.get('inference_depth', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.max_reasoning_chains = config.get('max_reasoning_chains', 10)
        self.output_dir = config.get('output_dir', 'output/reasoning')
        
        # Knowledge graph configuration
        self.kg_node_types = config.get('kg_node_types', ['concept', 'entity', 'fact', 'rule'])
        self.kg_relationship_types = config.get('kg_relationship_types', 
            ['related_to', 'part_of', 'causes', 'supports', 'contradicts'])
        
        # Reasoning strategies
        self.reasoning_strategies = [
            'deductive',
            'inductive',
            'abductive',
            'analogical',
            'causal'
        ]
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} with {self.reasoning_mode} reasoning mode")
    
    async def process(self, input_data: Any, session_id: str) -> AgentResult:
        """
        Main processing method for document reasoning.
        
        Args:
            input_data: Output from DocumentAnalysisAgent containing analysis results
            session_id: Unique session identifier
            
        Returns:
            AgentResult with comprehensive reasoning results
        """
        start_time = time.time()
        
        try:
            # Pre-process input
            processed_input = await self.pre_process(input_data, session_id)
            
            # Validate input from analysis agent
            if not await self.validate_input(processed_input):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="Invalid input from analysis agent"
                )
            
            # Extract analysis data
            analysis_data = self._extract_analysis_data(processed_input)
            
            if not analysis_data:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="No valid analysis data found in input"
                )
            
            # Build knowledge graph from analysis
            knowledge_graph = await self._build_knowledge_graph(analysis_data, session_id)
            
            # Perform logical reasoning
            reasoning_results = await self._perform_logical_reasoning(
                knowledge_graph, analysis_data, session_id
            )
            
            # Generate inferences and insights
            inferences = await self._generate_inferences(
                knowledge_graph, reasoning_results, session_id
            )
            
            # Detect contradictions and inconsistencies
            contradictions = await self._detect_contradictions(
                knowledge_graph, analysis_data, session_id
            )
            
            # Generate hypotheses and predictions
            hypotheses = await self._generate_hypotheses(
                knowledge_graph, reasoning_results, session_id
            )
            
            # Perform question answering preparation
            qa_knowledge = await self._prepare_qa_knowledge_base(
                knowledge_graph, analysis_data, session_id
            )
            
            # Generate reasoning chains for complex queries
            reasoning_chains = await self._generate_reasoning_chains(
                knowledge_graph, reasoning_results, session_id
            )
            
            # Save reasoning results
            await self._save_reasoning_results(
                knowledge_graph, reasoning_results, inferences, contradictions,
                hypotheses, qa_knowledge, reasoning_chains, session_id
            )
            
            # Prepare output for next agent
            output = {
                'session_id': session_id,
                'knowledge_graph': knowledge_graph,
                'reasoning_results': reasoning_results,
                'inferences': inferences,
                'contradictions': contradictions,
                'hypotheses': hypotheses,
                'qa_knowledge_base': qa_knowledge,
                'reasoning_chains': reasoning_chains,
                'reasoning_metadata': {
                    'agent_name': self.name,
                    'reasoning_mode': self.reasoning_mode,
                    'processing_time': time.time() - start_time,
                    'knowledge_graph_nodes': len(knowledge_graph.get('nodes', [])),
                    'knowledge_graph_edges': len(knowledge_graph.get('edges', [])),
                    'inferences_generated': len(inferences),
                    'contradictions_found': len(contradictions),
                    'hypotheses_generated': len(hypotheses)
                },
                'previous_agent_data': processed_input  # Pass through for next agent
            }
            
            self.logger.info(
                f"Reasoning completed with {output['reasoning_metadata']['knowledge_graph_nodes']} nodes, "
                f"{output['reasoning_metadata']['inferences_generated']} inferences, "
                f"and {output['reasoning_metadata']['contradictions_found']} contradictions"
            )
            
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                output=output,
                execution_time=time.time() - start_time,
                session_id=session_id,
                metadata={
                    'knowledge_graph_size': output['reasoning_metadata']['knowledge_graph_nodes'],
                    'inferences_count': output['reasoning_metadata']['inferences_generated'],
                    'reasoning_mode': self.reasoning_mode
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, session_id)
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input from analysis agent"""
        try:
            # Handle orchestrator wrapper
            if isinstance(input_data, dict) and 'original_input' in input_data:
                actual_input = input_data['previous_results']['DocumentAnalysisAgent'].output
            else:
                actual_input = input_data
            
            # Check for required fields from analysis agent
            required_fields = ['document_analysis', 'session_id']
            if not all(field in actual_input for field in required_fields):
                return False
            
            analysis_data = actual_input['document_analysis']
            if not isinstance(analysis_data, dict) or not analysis_data:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_analysis_data(self, input_data: Any) -> Dict[str, Any]:
        """Extract analysis data from previous agent output"""
        try:
            # Handle orchestrator wrapper
            if isinstance(input_data, dict) and 'original_input' in input_data:
                analysis_output = input_data['previous_results']['DocumentAnalysisAgent'].output
            else:
                analysis_output = input_data
            
            return {
                'document_analysis': analysis_output['document_analysis'],
                'cross_document_analysis': analysis_output.get('cross_document_analysis', {}),
                'ai_insights': analysis_output.get('ai_insights', {}),
                'semantic_embeddings': analysis_output.get('semantic_embeddings', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract analysis data: {str(e)}")
            return {}
    
    async def _build_knowledge_graph(
        self, 
        analysis_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Build a knowledge graph from analysis results"""
        knowledge_graph = {
            'session_id': session_id,
            'nodes': [],
            'edges': [],
            'node_index': {},  # For quick lookups
            'statistics': {}
        }
        
        node_counter = 0
        
        # Extract entities, concepts, and facts from each document
        for doc_path, doc_analysis in analysis_data['document_analysis'].items():
            doc_id = Path(doc_path).stem
            
            # Add document node
            doc_node = {
                'id': f"doc_{node_counter}",
                'type': 'document',
                'label': Path(doc_path).name,
                'properties': {
                    'file_path': doc_path,
                    'document_type': doc_analysis.get('classification', {}).get('document_type', 'unknown'),
                    'sentiment': doc_analysis.get('sentiment', {}).get('overall_sentiment', 'neutral'),
                    'topic_count': len(doc_analysis.get('topics', {}).get('main_topics', []))
                }
            }
            knowledge_graph['nodes'].append(doc_node)
            knowledge_graph['node_index'][doc_node['id']] = doc_node
            node_counter += 1
            
            # Add entity nodes
            entities = doc_analysis.get('entities', {}).get('document_entities', {})
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_node = {
                        'id': f"entity_{node_counter}",
                        'type': 'entity',
                        'subtype': entity_type.lower(),
                        'label': entity['text'],
                        'properties': {
                            'confidence': entity.get('confidence', 0.0),
                            'entity_type': entity_type,
                            'source_document': doc_id
                        }
                    }
                    knowledge_graph['nodes'].append(entity_node)
                    knowledge_graph['node_index'][entity_node['id']] = entity_node
                    
                    # Add edge from document to entity
                    edge = {
                        'id': f"edge_{len(knowledge_graph['edges'])}",
                        'source': doc_node['id'],
                        'target': entity_node['id'],
                        'type': 'contains',
                        'properties': {
                            'confidence': entity.get('confidence', 0.0)
                        }
                    }
                    knowledge_graph['edges'].append(edge)
                    node_counter += 1
            
            # Add topic/concept nodes
            topics = doc_analysis.get('topics', {}).get('main_topics', [])
            for topic in topics:
                topic_node = {
                    'id': f"topic_{node_counter}",
                    'type': 'concept',
                    'label': topic,
                    'properties': {
                        'source_document': doc_id,
                        'topic_type': 'main_topic'
                    }
                }
                knowledge_graph['nodes'].append(topic_node)
                knowledge_graph['node_index'][topic_node['id']] = topic_node
                
                # Add edge from document to topic
                edge = {
                    'id': f"edge_{len(knowledge_graph['edges'])}",
                    'source': doc_node['id'],
                    'target': topic_node['id'],
                    'type': 'discusses',
                    'properties': {}
                }
                knowledge_graph['edges'].append(edge)
                node_counter += 1
            
            # Add keyword nodes
            keywords = doc_analysis.get('keywords', [])
            for keyword, importance in keywords[:5]:  # Top 5 keywords
                keyword_node = {
                    'id': f"keyword_{node_counter}",
                    'type': 'concept',
                    'label': keyword,
                    'properties': {
                        'importance': importance,
                        'source_document': doc_id,
                        'keyword_type': 'extracted'
                    }
                }
                knowledge_graph['nodes'].append(keyword_node)
                knowledge_graph['node_index'][keyword_node['id']] = keyword_node
                
                # Add edge from document to keyword
                edge = {
                    'id': f"edge_{len(knowledge_graph['edges'])}",
                    'source': doc_node['id'],
                    'target': keyword_node['id'],
                    'type': 'emphasizes',
                    'properties': {
                        'importance': importance
                    }
                }
                knowledge_graph['edges'].append(edge)
                node_counter += 1
        
        # Add cross-document relationships
        await self._add_cross_document_relationships(knowledge_graph, analysis_data)
        
        # Calculate statistics
        knowledge_graph['statistics'] = {
            'total_nodes': len(knowledge_graph['nodes']),
            'total_edges': len(knowledge_graph['edges']),
            'node_types': self._count_node_types(knowledge_graph['nodes']),
            'edge_types': self._count_edge_types(knowledge_graph['edges'])
        }
        
        self.logger.info(
            f"Built knowledge graph with {knowledge_graph['statistics']['total_nodes']} nodes "
            f"and {knowledge_graph['statistics']['total_edges']} edges"
        )
        
        return knowledge_graph
    
    async def _add_cross_document_relationships(
        self, 
        knowledge_graph: Dict[str, Any], 
        analysis_data: Dict[str, Any]
    ):
        """Add relationships between entities/concepts across documents"""
        cross_analysis = analysis_data.get('cross_document_analysis', {})
        
        # Add similarity relationships
        doc_relationships = cross_analysis.get('document_relationships', [])
        for relationship in doc_relationships:
            doc1_name = relationship['document1']
            doc2_name = relationship['document2']
            similarity = relationship['similarity_score']
            
            # Find document nodes
            doc1_node = None
            doc2_node = None
            
            for node in knowledge_graph['nodes']:
                if node['type'] == 'document':
                    if node['label'] == doc1_name:
                        doc1_node = node
                    elif node['label'] == doc2_name:
                        doc2_node = node
            
            if doc1_node and doc2_node:
                edge = {
                    'id': f"edge_{len(knowledge_graph['edges'])}",
                    'source': doc1_node['id'],
                    'target': doc2_node['id'],
                    'type': 'similar_to',
                    'properties': {
                        'similarity_score': similarity,
                        'relationship_type': relationship.get('relationship_type', 'unknown')
                    }
                }
                knowledge_graph['edges'].append(edge)
        
        # Add common entity relationships
        # Find entities with the same label across documents
        entity_groups = {}
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                label = node['label'].lower()
                if label not in entity_groups:
                    entity_groups[label] = []
                entity_groups[label].append(node)
        
        # Create relationships between same entities in different documents
        for label, entities in entity_groups.items():
            if len(entities) > 1:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        edge = {
                            'id': f"edge_{len(knowledge_graph['edges'])}",
                            'source': entities[i]['id'],
                            'target': entities[j]['id'],
                            'type': 'same_as',
                            'properties': {
                                'entity_label': label,
                                'cross_document': True
                            }
                        }
                        knowledge_graph['edges'].append(edge)
    
    def _count_node_types(self, nodes: List[Dict]) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node in nodes:
            node_type = node['type']
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_edge_types(self, edges: List[Dict]) -> Dict[str, int]:
        """Count edges by type"""
        counts = {}
        for edge in edges:
            edge_type = edge['type']
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    async def _perform_logical_reasoning(
        self, 
        knowledge_graph: Dict[str, Any], 
        analysis_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Perform various types of logical reasoning"""
        reasoning_results = {
            'deductive_reasoning': await self._perform_deductive_reasoning(knowledge_graph),
            'inductive_reasoning': await self._perform_inductive_reasoning(knowledge_graph),
            'abductive_reasoning': await self._perform_abductive_reasoning(knowledge_graph),
            'analogical_reasoning': await self._perform_analogical_reasoning(knowledge_graph),
            'causal_reasoning': await self._perform_causal_reasoning(knowledge_graph, analysis_data)
        }
        
        # Combine all reasoning results
        reasoning_results['combined_insights'] = await self._combine_reasoning_insights(reasoning_results)
        
        return reasoning_results
    
    async def _perform_deductive_reasoning(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deductive reasoning using logical rules"""
        deductions = []
        
        # Rule: If two documents are similar AND they both contain the same entities, 
        # then those entities are likely important
        similar_docs = [
            edge for edge in knowledge_graph['edges'] 
            if edge['type'] == 'similar_to' and edge['properties']['similarity_score'] > 0.5
        ]
        
        for edge in similar_docs:
            doc1_id = edge['source']
            doc2_id = edge['target']
            
            # Find common entities
            doc1_entities = [
                e for e in knowledge_graph['edges'] 
                if e['source'] == doc1_id and e['type'] == 'contains'
            ]
            doc2_entities = [
                e for e in knowledge_graph['edges'] 
                if e['source'] == doc2_id and e['type'] == 'contains'
            ]
            
            # Find same entities
            common_entities = []
            for e1 in doc1_entities:
                for e2 in doc2_entities:
                    entity1 = knowledge_graph['node_index'][e1['target']]
                    entity2 = knowledge_graph['node_index'][e2['target']]
                    if entity1['label'].lower() == entity2['label'].lower():
                        common_entities.append(entity1['label'])
            
            if common_entities:
                deduction = {
                    'type': 'entity_importance',
                    'premise': f"Documents are similar (score: {edge['properties']['similarity_score']:.2f})",
                    'conclusion': f"Common entities are important: {', '.join(common_entities)}",
                    'confidence': min(0.9, edge['properties']['similarity_score'] + 0.2),
                    'entities': common_entities
                }
                deductions.append(deduction)
        
        return {
            'deductions': deductions,
            'rule_count': 1,
            'confidence_avg': sum(d['confidence'] for d in deductions) / len(deductions) if deductions else 0
        }
    
    async def _perform_inductive_reasoning(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inductive reasoning to find patterns"""
        patterns = []
        
        # Pattern: Find entities that appear frequently across documents
        entity_frequency = {}
        for edge in knowledge_graph['edges']:
            if edge['type'] == 'contains':
                entity_node = knowledge_graph['node_index'][edge['target']]
                entity_label = entity_node['label'].lower()
                entity_frequency[entity_label] = entity_frequency.get(entity_label, 0) + 1
        
        # Find high-frequency entities
        total_docs = len([n for n in knowledge_graph['nodes'] if n['type'] == 'document'])
        frequent_entities = [
            (entity, freq) for entity, freq in entity_frequency.items() 
            if freq > 1 and freq / total_docs > 0.3
        ]
        
        if frequent_entities:
            pattern = {
                'type': 'frequent_entity_pattern',
                'pattern': 'Entities appearing in multiple documents are likely central themes',
                'evidence': frequent_entities,
                'confidence': 0.8,
                'generalization': 'Cross-document entity frequency indicates thematic importance'
            }
            patterns.append(pattern)
        
        # Pattern: Document type clustering
        doc_types = {}
        for node in knowledge_graph['nodes']:
            if node['type'] == 'document':
                doc_type = node['properties'].get('document_type', 'unknown')
                if doc_type not in doc_types:
                    doc_types[doc_type] = []
                doc_types[doc_type].append(node['label'])
        
        if len(doc_types) > 1:
            pattern = {
                'type': 'document_type_pattern',
                'pattern': 'Documents cluster by type with shared characteristics',
                'evidence': doc_types,
                'confidence': 0.7,
                'generalization': 'Document type determines content patterns'
            }
            patterns.append(pattern)
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns),
            'confidence_avg': sum(p['confidence'] for p in patterns) / len(patterns) if patterns else 0
        }
    
    async def _perform_abductive_reasoning(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Perform abductive reasoning to find best explanations"""
        explanations = []
        
        # Observation: Why are certain entities mentioned together?
        entity_co_occurrence = {}
        
        # Find entities that appear in the same document
        for node in knowledge_graph['nodes']:
            if node['type'] == 'document':
                doc_entities = [
                    knowledge_graph['node_index'][edge['target']]['label']
                    for edge in knowledge_graph['edges']
                    if edge['source'] == node['id'] and edge['type'] == 'contains'
                ]
                
                # Create pairs
                for i in range(len(doc_entities)):
                    for j in range(i + 1, len(doc_entities)):
                        pair = tuple(sorted([doc_entities[i], doc_entities[j]]))
                        entity_co_occurrence[pair] = entity_co_occurrence.get(pair, 0) + 1
        
        # Find frequently co-occurring entities
        frequent_pairs = [
            (pair, count) for pair, count in entity_co_occurrence.items() 
            if count > 1
        ]
        
        for (entity1, entity2), count in frequent_pairs:
            explanation = {
                'observation': f"Entities '{entity1}' and '{entity2}' appear together {count} times",
                'hypothesis': f"There is a logical relationship between {entity1} and {entity2}",
                'explanation': 'Entities co-occur because they are contextually related or causally connected',
                'confidence': min(0.9, 0.5 + (count * 0.2)),
                'type': 'entity_relationship'
            }
            explanations.append(explanation)
        
        return {
            'explanations': explanations,
            'explanation_count': len(explanations),
            'confidence_avg': sum(e['confidence'] for e in explanations) / len(explanations) if explanations else 0
        }
    
    async def _perform_analogical_reasoning(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analogical reasoning to find structural similarities"""
        analogies = []
        
        # Find documents with similar entity structures
        doc_structures = {}
        
        for node in knowledge_graph['nodes']:
            if node['type'] == 'document':
                # Get entity types in this document
                entity_types = []
                for edge in knowledge_graph['edges']:
                    if edge['source'] == node['id'] and edge['type'] == 'contains':
                        entity_node = knowledge_graph['node_index'][edge['target']]
                        entity_types.append(entity_node['subtype'])
                
                entity_type_signature = tuple(sorted(set(entity_types)))
                if entity_type_signature not in doc_structures:
                    doc_structures[entity_type_signature] = []
                doc_structures[entity_type_signature].append(node['label'])
        
        # Find analogous structures
        for signature, docs in doc_structures.items():
            if len(docs) > 1 and len(signature) > 1:
                analogy = {
                    'type': 'structural_analogy',
                    'pattern': f"Documents with entity pattern: {', '.join(signature)}",
                    'analogous_documents': docs,
                    'analogy': f"These documents share the same informational structure",
                    'confidence': 0.7,
                    'implications': 'Documents with similar entity patterns likely serve similar purposes'
                }
                analogies.append(analogy)
        
        return {
            'analogies': analogies,
            'analogy_count': len(analogies),
            'confidence_avg': sum(a['confidence'] for a in analogies) / len(analogies) if analogies else 0
        }
    
    async def _perform_causal_reasoning(
        self, 
        knowledge_graph: Dict[str, Any], 
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform causal reasoning to identify cause-effect relationships"""
        causal_chains = []
        
        # Use AI to identify potential causal relationships in text
        for doc_path, doc_analysis in analysis_data['document_analysis'].items():
            # Look for causal indicators in topics and entities
            topics = doc_analysis.get('topics', {}).get('main_topics', [])
            entities = []
            
            entity_dict = doc_analysis.get('entities', {}).get('document_entities', {})
            for entity_list in entity_dict.values():
                entities.extend([e['text'] for e in entity_list])
            
            if len(entities) > 1:
                # Generate causal analysis using AI
                causal_analysis = await self._analyze_causal_relationships(
                    Path(doc_path).name, topics, entities
                )
                
                if causal_analysis.get('causal_relationships'):
                    causal_chains.extend(causal_analysis['causal_relationships'])
        
        return {
            'causal_chains': causal_chains,
            'chain_count': len(causal_chains),
            'confidence_avg': sum(c.get('confidence', 0) for c in causal_chains) / len(causal_chains) if causal_chains else 0
        }
    
    async def _analyze_causal_relationships(
        self, 
        document_name: str, 
        topics: List[str], 
        entities: List[str]
    ) -> Dict[str, Any]:
        """Use AI to analyze potential causal relationships"""
        try:
            prompt = f"""
            Analyze potential causal relationships in a document about these topics and entities:
            
            Document: {document_name}
            Topics: {', '.join(topics)}
            Entities: {', '.join(entities[:10])}  # Limit for prompt size
            
            Identify potential cause-effect relationships. Return JSON:
            {{
                "causal_relationships": [
                    {{
                        "cause": "entity or concept that causes",
                        "effect": "entity or concept that is affected", 
                        "relationship_type": "direct/indirect/potential",
                        "confidence": 0.8,
                        "evidence": "reasoning for this relationship"
                    }}
                ]
            }}
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.2)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {'causal_relationships': []}
                
        except Exception as e:
            self.logger.warning(f"Causal relationship analysis failed: {str(e)}")
            return {'causal_relationships': []}
    
    async def _combine_reasoning_insights(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from all reasoning types"""
        combined_insights = {
            'key_deductions': reasoning_results['deductive_reasoning'].get('deductions', []),
            'identified_patterns': reasoning_results['inductive_reasoning'].get('patterns', []),
            'best_explanations': reasoning_results['abductive_reasoning'].get('explanations', []),
            'structural_analogies': reasoning_results['analogical_reasoning'].get('analogies', []),
            'causal_relationships': reasoning_results['causal_reasoning'].get('causal_chains', [])
        }
        
        # Calculate overall reasoning confidence
        confidences = []
        for reasoning_type, results in reasoning_results.items():
            if reasoning_type != 'combined_insights' and 'confidence_avg' in results:
                confidences.append(results['confidence_avg'])
        
        combined_insights['overall_reasoning_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        combined_insights['reasoning_completeness'] = len([c for c in confidences if c > 0]) / len(self.reasoning_strategies)
        
        return combined_insights
    
    async def _generate_inferences(
        self, 
        knowledge_graph: Dict[str, Any], 
        reasoning_results: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate new inferences from reasoning results"""
        inferences = []
        
        # Inference from entity frequency
        frequent_entities = {}
        for edge in knowledge_graph['edges']:
            if edge['type'] == 'contains':
                entity_node = knowledge_graph['node_index'][edge['target']]
                entity_label = entity_node['label']
                frequent_entities[entity_label] = frequent_entities.get(entity_label, 0) + 1
        
        # Top entities
        top_entities = sorted(frequent_entities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for entity, frequency in top_entities:
            if frequency > 1:
                inference = {
                    'type': 'importance_inference',
                    'inference': f"'{entity}' is a central theme (appears {frequency} times)",
                    'confidence': min(0.9, 0.4 + (frequency * 0.2)),
                    'evidence': f"Entity frequency across {frequency} documents",
                    'implications': f"'{entity}' likely plays a key role in the document collection"
                }
                inferences.append(inference)
        
        # Inference from reasoning patterns
        patterns = reasoning_results.get('inductive_reasoning', {}).get('patterns', [])
        for pattern in patterns:
            inference = {
                'type': 'pattern_inference',
                'inference': pattern['generalization'],
                'confidence': pattern['confidence'],
                'evidence': pattern['pattern'],
                'implications': f"This pattern suggests underlying structure in the data"
            }
            inferences.append(inference)
        
        # Inference from causal relationships
        causal_chains = reasoning_results.get('causal_reasoning', {}).get('causal_chains', [])
        if causal_chains:
            inference = {
                'type': 'causal_inference',
                'inference': f"Found {len(causal_chains)} potential causal relationships",
                'confidence': reasoning_results['causal_reasoning'].get('confidence_avg', 0.5),
                'evidence': f"Causal analysis identified {len(causal_chains)} cause-effect pairs",
                'implications': 'Documents contain information about causal processes or mechanisms'
            }
            inferences.append(inference)
        
        return inferences
    
    async def _detect_contradictions(
        self, 
        knowledge_graph: Dict[str, Any], 
        analysis_data: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Detect contradictions and inconsistencies"""
        contradictions = []
        
        # Check for sentiment contradictions
        doc_sentiments = {}
        for node in knowledge_graph['nodes']:
            if node['type'] == 'document':
                sentiment = node['properties'].get('sentiment', 'neutral')
                doc_name = node['label']
                doc_sentiments[doc_name] = sentiment
        
        # Check if similar documents have very different sentiments
        similar_edges = [
            edge for edge in knowledge_graph['edges']
            if edge['type'] == 'similar_to' and edge['properties']['similarity_score'] > 0.7
        ]
        
        for edge in similar_edges:
            doc1_node = knowledge_graph['node_index'][edge['source']]
            doc2_node = knowledge_graph['node_index'][edge['target']]
            
            sentiment1 = doc1_node['properties'].get('sentiment', 'neutral')
            sentiment2 = doc2_node['properties'].get('sentiment', 'neutral')
            
            # Check for sentiment contradiction
            if (sentiment1 == 'positive' and sentiment2 == 'negative') or \
               (sentiment1 == 'negative' and sentiment2 == 'positive'):
                
                contradiction = {
                    'type': 'sentiment_contradiction',
                    'description': f"Similar documents have opposite sentiments",
                    'document1': doc1_node['label'],
                    'document2': doc2_node['label'],
                    'sentiment1': sentiment1,
                    'sentiment2': sentiment2,
                    'similarity_score': edge['properties']['similarity_score'],
                    'severity': 'moderate',
                    'implications': 'Documents may represent different perspectives on the same topic'
                }
                contradictions.append(contradiction)
        
        # Check for entity classification contradictions
        # (This would be more sophisticated with actual entity resolution)
        entity_classifications = {}
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                label = node['label'].lower()
                subtype = node['subtype']
                
                if label in entity_classifications:
                    if entity_classifications[label] != subtype:
                        contradiction = {
                            'type': 'entity_classification_contradiction',
                            'description': f"Entity '{node['label']}' classified as both {entity_classifications[label]} and {subtype}",
                            'entity': node['label'],
                            'classification1': entity_classifications[label],
                            'classification2': subtype,
                            'severity': 'low',
                            'implications': 'Entity may have multiple roles or classification uncertainty'
                        }
                        contradictions.append(contradiction)
                else:
                    entity_classifications[label] = subtype
        
        return contradictions
    
    async def _generate_hypotheses(
        self, 
        knowledge_graph: Dict[str, Any], 
        reasoning_results: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses for further investigation"""
        hypotheses = []
        
        # Hypothesis from frequent entity co-occurrence
        entity_pairs = {}
        
        for node in knowledge_graph['nodes']:
            if node['type'] == 'document':
                doc_entities = [
                    knowledge_graph['node_index'][edge['target']]['label']
                    for edge in knowledge_graph['edges']
                    if edge['source'] == node['id'] and edge['type'] == 'contains'
                ]
                
                for i in range(len(doc_entities)):
                    for j in range(i + 1, len(doc_entities)):
                        pair = tuple(sorted([doc_entities[i], doc_entities[j]]))
                        entity_pairs[pair] = entity_pairs.get(pair, 0) + 1
        
        # Generate hypotheses for frequent pairs
        for (entity1, entity2), frequency in entity_pairs.items():
            if frequency > 1:
                hypothesis = {
                    'type': 'relationship_hypothesis',
                    'hypothesis': f"'{entity1}' and '{entity2}' have a significant relationship",
                    'evidence': f"Co-occur in {frequency} documents",
                    'testable_prediction': f"If this relationship exists, they should appear together in related future documents",
                    'confidence': min(0.8, 0.3 + (frequency * 0.2)),
                    'research_questions': [
                        f"What is the nature of the relationship between {entity1} and {entity2}?",
                        f"In what contexts do {entity1} and {entity2} typically appear together?"
                    ]
                }
                hypotheses.append(hypothesis)
        
        # Hypothesis from causal relationships
        causal_chains = reasoning_results.get('causal_reasoning', {}).get('causal_chains', [])
        if len(causal_chains) > 2:
            hypothesis = {
                'type': 'causal_system_hypothesis',
                'hypothesis': 'The documents describe a complex causal system with multiple interconnected factors',
                'evidence': f'Identified {len(causal_chains)} causal relationships',
                'testable_prediction': 'Additional documents should reveal more connections in this causal network',
                'confidence': 0.7,
                'research_questions': [
                    'What are the primary drivers in this causal system?',
                    'Are there feedback loops or recursive relationships?',
                    'What are the ultimate outcomes of this causal chain?'
                ]
            }
            hypotheses.append(hypothesis)
        
        # Hypothesis from document clustering
        doc_count = len([n for n in knowledge_graph['nodes'] if n['type'] == 'document'])
        if doc_count > 3:
            hypothesis = {
                'type': 'collection_purpose_hypothesis',
                'hypothesis': 'This document collection was assembled for a specific purpose or project',
                'evidence': f'Coherent collection of {doc_count} related documents',
                'testable_prediction': 'Document metadata or content should reveal common project or theme',
                'confidence': 0.6,
                'research_questions': [
                    'What was the original purpose of assembling these documents?',
                    'Are there missing documents that would complete the collection?',
                    'What decision-making process do these documents support?'
                ]
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _prepare_qa_knowledge_base(
        self, 
        knowledge_graph: Dict[str, Any], 
        analysis_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Prepare knowledge base optimized for question answering"""
        qa_knowledge = {
            'entities': {},
            'facts': [],
            'relationships': [],
            'topics': {},
            'document_summaries': {},
            'searchable_content': {}
        }
        
        # Extract entities with context
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                entity_info = {
                    'label': node['label'],
                    'type': node['subtype'],
                    'confidence': node['properties'].get('confidence', 0.0),
                    'source_documents': [],
                    'related_entities': [],
                    'contexts': []
                }
                
                # Find source documents
                for edge in knowledge_graph['edges']:
                    if edge['target'] == node['id'] and edge['type'] == 'contains':
                        source_doc = knowledge_graph['node_index'][edge['source']]
                        entity_info['source_documents'].append(source_doc['label'])
                
                qa_knowledge['entities'][node['label']] = entity_info
        
        # Extract relationships
        for edge in knowledge_graph['edges']:
            if edge['type'] not in ['contains', 'discusses', 'emphasizes']:
                source_node = knowledge_graph['node_index'][edge['source']]
                target_node = knowledge_graph['node_index'][edge['target']]
                
                relationship = {
                    'source': source_node['label'],
                    'target': target_node['label'],
                    'type': edge['type'],
                    'confidence': edge['properties'].get('confidence', 0.0),
                    'properties': edge['properties']
                }
                qa_knowledge['relationships'].append(relationship)
        
        # Extract document summaries
        for doc_path, doc_analysis in analysis_data['document_analysis'].items():
            doc_name = Path(doc_path).name
            
            summary = {
                'filename': doc_name,
                'document_type': doc_analysis.get('classification', {}).get('document_type', 'unknown'),
                'main_topics': doc_analysis.get('topics', {}).get('main_topics', []),
                'key_entities': [
                    entity['text'] for entity_list in doc_analysis.get('entities', {}).get('document_entities', {}).values()
                    for entity in entity_list[:3]  # Top 3 entities per type
                ],
                'sentiment': doc_analysis.get('sentiment', {}).get('overall_sentiment', 'neutral'),
                'keywords': [kw[0] for kw in doc_analysis.get('keywords', [])[:5]]
            }
            
            qa_knowledge['document_summaries'][doc_name] = summary
        
        # Prepare searchable content index
        qa_knowledge['searchable_content'] = {
            'entity_index': list(qa_knowledge['entities'].keys()),
            'topic_index': list(set(
                topic for summary in qa_knowledge['document_summaries'].values()
                for topic in summary['main_topics']
            )),
            'keyword_index': list(set(
                keyword for summary in qa_knowledge['document_summaries'].values()
                for keyword in summary['keywords']
            ))
        }
        
        return qa_knowledge
    
    async def _generate_reasoning_chains(
        self, 
        knowledge_graph: Dict[str, Any], 
        reasoning_results: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate reasoning chains for complex query answering"""
        reasoning_chains = []
        
        # Chain 1: Entity -> Document -> Related Entities
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity' and node['properties'].get('confidence', 0) > 0.7:
                
                # Find documents containing this entity
                containing_docs = [
                    edge for edge in knowledge_graph['edges']
                    if edge['target'] == node['id'] and edge['type'] == 'contains'
                ]
                
                if containing_docs:
                    # Find other entities in the same documents
                    related_entities = []
                    for doc_edge in containing_docs:
                        doc_id = doc_edge['source']
                        other_entities = [
                            knowledge_graph['node_index'][edge['target']]['label']
                            for edge in knowledge_graph['edges']
                            if edge['source'] == doc_id and edge['type'] == 'contains' and edge['target'] != node['id']
                        ]
                        related_entities.extend(other_entities)
                    
                    if related_entities:
                        chain = {
                            'type': 'entity_context_chain',
                            'start_entity': node['label'],
                            'reasoning_path': [
                                f"Entity '{node['label']}' appears in documents",
                                f"These documents also contain: {', '.join(set(related_entities)[:5])}",
                                f"Therefore, '{node['label']}' is contextually related to these entities"
                            ],
                            'conclusion': f"'{node['label']}' operates in the context of {', '.join(set(related_entities)[:3])}",
                            'confidence': node['properties'].get('confidence', 0.5)
                        }
                        reasoning_chains.append(chain)
        
        # Chain 2: Similar Documents -> Common Themes -> Insights
        similar_edges = [
            edge for edge in knowledge_graph['edges']
            if edge['type'] == 'similar_to' and edge['properties']['similarity_score'] > 0.5
        ]
        
        for edge in similar_edges:
            doc1 = knowledge_graph['node_index'][edge['source']]
            doc2 = knowledge_graph['node_index'][edge['target']]
            
            chain = {
                'type': 'similarity_reasoning_chain',
                'start_documents': [doc1['label'], doc2['label']],
                'reasoning_path': [
                    f"Documents '{doc1['label']}' and '{doc2['label']}' are similar (score: {edge['properties']['similarity_score']:.2f})",
                    "Similar documents likely discuss related topics or share common themes",
                    "Analysis of their common elements reveals shared focus areas"
                ],
                'conclusion': f"These documents represent related perspectives on common themes",
                'confidence': edge['properties']['similarity_score']
            }
            reasoning_chains.append(chain)
        
        return reasoning_chains[:self.max_reasoning_chains]  # Limit number of chains
    
    async def _save_reasoning_results(
        self, 
        knowledge_graph: Dict[str, Any],
        reasoning_results: Dict[str, Any],
        inferences: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        qa_knowledge: Dict[str, Any],
        reasoning_chains: List[Dict[str, Any]],
        session_id: str
    ):
        """Save comprehensive reasoning results"""
        try:
            output_path = Path(self.output_dir) / session_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save knowledge graph
            kg_file = output_path / "knowledge_graph.json"
            with open(kg_file, 'w') as f:
                json.dump(knowledge_graph, f, indent=2)
            
            # Save reasoning results
            reasoning_file = output_path / "reasoning_results.json"
            with open(reasoning_file, 'w') as f:
                json.dump(reasoning_results, f, indent=2)
            
            # Save inferences
            inferences_file = output_path / "inferences.json"
            with open(inferences_file, 'w') as f:
                json.dump(inferences, f, indent=2)
            
            # Save contradictions
            contradictions_file = output_path / "contradictions.json"
            with open(contradictions_file, 'w') as f:
                json.dump(contradictions, f, indent=2)
            
            # Save hypotheses
            hypotheses_file = output_path / "hypotheses.json"
            with open(hypotheses_file, 'w') as f:
                json.dump(hypotheses, f, indent=2)
            
            # Save QA knowledge base
            qa_file = output_path / "qa_knowledge_base.json"
            with open(qa_file, 'w') as f:
                json.dump(qa_knowledge, f, indent=2)
            
            # Save reasoning chains
            chains_file = output_path / "reasoning_chains.json"
            with open(chains_file, 'w') as f:
                json.dump(reasoning_chains, f, indent=2)
            
            # Generate reasoning summary report
            summary_report = await self._generate_reasoning_summary_report(
                knowledge_graph, reasoning_results, inferences, contradictions, hypotheses
            )
            
            summary_file = output_path / "reasoning_summary_report.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2)
            
            self.logger.info(f"Saved reasoning results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save reasoning results: {str(e)}")
    
    async def _generate_reasoning_summary_report(
        self,
        knowledge_graph: Dict[str, Any],
        reasoning_results: Dict[str, Any],
        inferences: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate executive summary report of reasoning"""
        return {
            'executive_summary': {
                'knowledge_graph_nodes': knowledge_graph['statistics']['total_nodes'],
                'knowledge_graph_edges': knowledge_graph['statistics']['total_edges'],
                'inferences_generated': len(inferences),
                'contradictions_found': len(contradictions),
                'hypotheses_proposed': len(hypotheses),
                'reasoning_confidence': reasoning_results.get('combined_insights', {}).get('overall_reasoning_confidence', 0),
                'reasoning_completeness': reasoning_results.get('combined_insights', {}).get('reasoning_completeness', 0)
            },
            'key_insights': {
                'most_confident_inference': max(inferences, key=lambda x: x['confidence'])['inference'] if inferences else None,
                'critical_contradictions': [c for c in contradictions if c.get('severity') == 'high'],
                'testable_hypotheses': [h for h in hypotheses if h['confidence'] > 0.6]
            },
            'reasoning_methodology': {
                'strategies_applied': self.reasoning_strategies,
                'reasoning_mode': self.reasoning_mode,
                'inference_depth': self.inference_depth
            },
            'processing_metadata': {
                'reasoning_timestamp': time.time(),
                'agent_name': self.name,
                'session_id': knowledge_graph.get('session_id')
            }
        }


async def test_reasoning_agent():
    """Test function for Document Reasoning Agent"""
    # This would be called with output from analysis agent
    print("Reasoning Agent test would run here with analysis output")


if __name__ == "__main__":
    asyncio.run(test_reasoning_agent())