"""
Document Response Agent - Fourth and final agent in the ADK pipeline
Specializes in generating intelligent responses, summaries, and insights
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from utils.adk_framework import BaseAgent, AgentResult, AgentStatus


class ResponseAgent(BaseAgent):
    """
    Agent responsible for generating intelligent responses and insights.
    
    Capabilities:
    - Natural language query answering
    - Document summarization and synthesis
    - Insight generation and reporting
    - Interactive Q&A preparation
    - Executive summary creation
    - Actionable recommendations
    - Multi-format output generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ResponseAgent", config)
        
        # Agent-specific configuration
        self.response_mode = config.get('response_mode', 'comprehensive')
        self.output_formats = config.get('output_formats', ['json', 'markdown', 'summary'])
        self.max_response_length = config.get('max_response_length', 2000)
        self.include_citations = config.get('include_citations', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.output_dir = config.get('output_dir', 'output/responses')
        
        # Response templates and styles
        self.response_styles = {
            'executive': 'High-level strategic insights for executives',
            'technical': 'Detailed technical analysis for specialists',
            'analytical': 'Comprehensive analytical breakdown',
            'conversational': 'Natural, accessible explanations'
        }
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} with {self.response_mode} response mode")
    
    async def process(self, input_data: Any, session_id: str) -> AgentResult:
        """
        Main processing method for response generation.
        
        Args:
            input_data: Output from ReasoningAgent containing reasoning results
            session_id: Unique session identifier
            
        Returns:
            AgentResult with comprehensive response package
        """
        start_time = time.time()
        
        try:
            # Pre-process input
            processed_input = await self.pre_process(input_data, session_id)
            
            # Validate input from reasoning agent
            if not await self.validate_input(processed_input):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="Invalid input from reasoning agent"
                )
            
            # Extract reasoning data
            reasoning_data = self._extract_reasoning_data(processed_input)
            
            if not reasoning_data:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="No valid reasoning data found in input"
                )
            
            # Generate comprehensive document intelligence report
            intelligence_report = await self._generate_intelligence_report(reasoning_data, session_id)
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(reasoning_data, session_id)
            
            # Generate actionable insights
            actionable_insights = await self._generate_actionable_insights(reasoning_data, session_id)
            
            # Prepare Q&A system
            qa_system = await self._prepare_qa_system(reasoning_data, session_id)
            
            # Generate different format outputs
            output_formats = await self._generate_output_formats(
                intelligence_report, executive_summary, actionable_insights, session_id
            )
            
            # Create searchable knowledge index
            knowledge_index = await self._create_knowledge_index(reasoning_data, session_id)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(reasoning_data, session_id)
            
            # Save all outputs
            await self._save_response_outputs(
                intelligence_report, executive_summary, actionable_insights,
                qa_system, output_formats, knowledge_index, recommendations, session_id
            )
            
            # Prepare final output package
            output = {
                'session_id': session_id,
                'intelligence_report': intelligence_report,
                'executive_summary': executive_summary,
                'actionable_insights': actionable_insights,
                'qa_system': qa_system,
                'output_formats': output_formats,
                'knowledge_index': knowledge_index,
                'recommendations': recommendations,
                'response_metadata': {
                    'agent_name': self.name,
                    'response_mode': self.response_mode,
                    'processing_time': time.time() - start_time,
                    'generated_insights': len(actionable_insights),
                    'recommendations_count': len(recommendations),
                    'qa_entries': len(qa_system.get('qa_pairs', [])),
                    'output_formats': list(output_formats.keys())
                },
                'complete_pipeline_data': processed_input  # Full pipeline history
            }
            
            self.logger.info(
                f"Response generation completed with {len(actionable_insights)} insights "
                f"and {len(recommendations)} recommendations"
            )
            
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                output=output,
                execution_time=time.time() - start_time,
                session_id=session_id,
                metadata={
                    'insights_generated': len(actionable_insights),
                    'recommendations_count': len(recommendations),
                    'response_mode': self.response_mode,
                    'pipeline_complete': True
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, session_id)
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input from reasoning agent"""
        try:
            # Handle orchestrator wrapper
            if isinstance(input_data, dict) and 'original_input' in input_data:
                actual_input = input_data['previous_results']['ReasoningAgent'].output
            else:
                actual_input = input_data
            
            # Check for required fields from reasoning agent
            required_fields = ['knowledge_graph', 'reasoning_results', 'session_id']
            if not all(field in actual_input for field in required_fields):
                return False
            
            knowledge_graph = actual_input['knowledge_graph']
            if not isinstance(knowledge_graph, dict) or not knowledge_graph.get('nodes'):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_reasoning_data(self, input_data: Any) -> Dict[str, Any]:
        """Extract reasoning data from previous agent outputs"""
        try:
            # Handle orchestrator wrapper to get all agent results
            if isinstance(input_data, dict) and 'original_input' in input_data:
                return {
                    'reasoning_output': input_data['previous_results']['ReasoningAgent'].output,
                    'analysis_output': input_data['previous_results']['DocumentAnalysisAgent'].output,
                    'ingestion_output': input_data['previous_results']['DocumentIngestionAgent'].output,
                    'original_input': input_data['original_input']
                }
            else:
                # Direct input from reasoning agent only
                return {
                    'reasoning_output': input_data,
                    'analysis_output': input_data.get('previous_agent_data', {}),
                    'ingestion_output': {},
                    'original_input': {}
                }
            
        except Exception as e:
            self.logger.error(f"Failed to extract reasoning data: {str(e)}")
            return {}
    
    async def _generate_intelligence_report(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive document intelligence report"""
        reasoning_output = reasoning_data['reasoning_output']
        analysis_output = reasoning_data.get('analysis_output', {})
        
        # Extract key data
        knowledge_graph = reasoning_output['knowledge_graph']
        reasoning_results = reasoning_output['reasoning_results']
        inferences = reasoning_output.get('inferences', [])
        contradictions = reasoning_output.get('contradictions', [])
        hypotheses = reasoning_output.get('hypotheses', [])
        
        # Document collection overview
        doc_nodes = [n for n in knowledge_graph['nodes'] if n['type'] == 'document']
        entity_nodes = [n for n in knowledge_graph['nodes'] if n['type'] == 'entity']
        
        intelligence_report = {
            'report_header': {
                'title': 'Document Intelligence Analysis Report',
                'session_id': session_id,
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(doc_nodes),
                'processing_pipeline': 'ADK Multi-Agent System'
            },
            
            'collection_overview': {
                'document_count': len(doc_nodes),
                'document_types': self._analyze_document_types(doc_nodes),
                'content_diversity': self._calculate_content_diversity(knowledge_graph),
                'collection_coherence': self._assess_collection_coherence(reasoning_results)
            },
            
            'entity_analysis': {
                'total_entities': len(entity_nodes),
                'entity_distribution': self._analyze_entity_distribution(entity_nodes),
                'key_entities': self._identify_key_entities(entity_nodes, knowledge_graph),
                'entity_relationships': len([e for e in knowledge_graph['edges'] if e['type'] in ['same_as', 'related_to']])
            },
            
            'knowledge_insights': {
                'reasoning_confidence': reasoning_results.get('combined_insights', {}).get('overall_reasoning_confidence', 0),
                'key_inferences': inferences[:5],  # Top 5 inferences
                'logical_contradictions': len(contradictions),
                'research_hypotheses': len(hypotheses),
                'knowledge_graph_density': len(knowledge_graph['edges']) / len(knowledge_graph['nodes']) if knowledge_graph['nodes'] else 0
            },
            
            'thematic_analysis': await self._perform_thematic_analysis(reasoning_data),
            
            'strategic_findings': await self._identify_strategic_findings(reasoning_data),
            
            'quality_assessment': {
                'data_completeness': self._assess_data_completeness(reasoning_data),
                'analysis_depth': self._assess_analysis_depth(reasoning_results),
                'confidence_metrics': self._calculate_confidence_metrics(inferences, reasoning_results)
            }
        }
        
        return intelligence_report
    
    def _analyze_document_types(self, doc_nodes: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of document types"""
        type_counts = {}
        for doc in doc_nodes:
            doc_type = doc['properties'].get('document_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts
    
    def _calculate_content_diversity(self, knowledge_graph: Dict[str, Any]) -> float:
        """Calculate content diversity score"""
        # Based on variety of entity types and topics
        entity_types = set()
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                entity_types.add(node['subtype'])
        
        # Diversity = unique entity types / total possible types
        max_types = len(['person', 'organization', 'location', 'date', 'concept', 'monetary', 'product'])
        return len(entity_types) / max_types if max_types > 0 else 0
    
    def _assess_collection_coherence(self, reasoning_results: Dict[str, Any]) -> float:
        """Assess how coherent the document collection is"""
        # Based on reasoning confidence and pattern detection
        patterns = reasoning_results.get('inductive_reasoning', {}).get('patterns', [])
        analogies = reasoning_results.get('analogical_reasoning', {}).get('analogies', [])
        
        coherence_factors = [
            len(patterns) * 0.1,  # Pattern diversity
            len(analogies) * 0.2,  # Structural similarity
            reasoning_results.get('combined_insights', {}).get('overall_reasoning_confidence', 0)
        ]
        
        return min(1.0, sum(coherence_factors) / len(coherence_factors))
    
    def _analyze_entity_distribution(self, entity_nodes: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of entity types"""
        type_counts = {}
        for entity in entity_nodes:
            entity_type = entity['subtype']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _identify_key_entities(self, entity_nodes: List[Dict], knowledge_graph: Dict[str, Any]) -> List[Dict]:
        """Identify most important entities"""
        # Count entity connections and frequency
        entity_importance = {}
        
        for entity in entity_nodes:
            # Count connections
            connections = len([
                e for e in knowledge_graph['edges'] 
                if e['source'] == entity['id'] or e['target'] == entity['id']
            ])
            
            entity_importance[entity['id']] = {
                'entity': entity,
                'connections': connections,
                'confidence': entity['properties'].get('confidence', 0),
                'importance_score': connections * entity['properties'].get('confidence', 0)
            }
        
        # Sort by importance
        sorted_entities = sorted(
            entity_importance.values(), 
            key=lambda x: x['importance_score'], 
            reverse=True
        )
        
        return [
            {
                'name': item['entity']['label'],
                'type': item['entity']['subtype'],
                'connections': item['connections'],
                'confidence': item['confidence']
            }
            for item in sorted_entities[:10]  # Top 10
        ]
    
    async def _perform_thematic_analysis(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep thematic analysis"""
        analysis_output = reasoning_data.get('analysis_output', {})
        reasoning_output = reasoning_data['reasoning_output']
        
        # Extract cross-document themes
        cross_analysis = analysis_output.get('cross_document_analysis', {})
        common_themes = cross_analysis.get('common_themes', [])
        
        # Use AI to generate thematic insights
        thematic_insights = await self._generate_thematic_insights_with_ai(
            common_themes, reasoning_output
        )
        
        return {
            'primary_themes': common_themes[:5] if common_themes else [],
            'theme_evolution': self._analyze_theme_evolution(common_themes),
            'thematic_coherence': len(common_themes) / max(1, len(reasoning_output['knowledge_graph']['nodes'])),
            'ai_thematic_insights': thematic_insights
        }
    
    async def _generate_thematic_insights_with_ai(
        self, 
        common_themes: List[Dict], 
        reasoning_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered thematic insights"""
        try:
            if not common_themes:
                return {'insights': [], 'summary': 'No clear themes identified'}
            
            themes_summary = {
                'top_themes': [theme['theme'] for theme in common_themes[:5]],
                'theme_prevalence': [theme['prevalence'] for theme in common_themes[:5]],
                'total_inferences': len(reasoning_output.get('inferences', [])),
                'knowledge_graph_size': len(reasoning_output['knowledge_graph']['nodes'])
            }
            
            prompt = f"""
            Analyze these document themes and provide strategic insights:
            
            Themes Analysis:
            {json.dumps(themes_summary, indent=2)}
            
            Generate thematic insights in JSON format:
            {{
                "central_narrative": "What story do these themes tell together?",
                "thematic_gaps": ["What important themes might be missing?"],
                "strategic_implications": ["What do these themes suggest for decision-making?"],
                "theme_relationships": ["How do the themes interconnect?"],
                "business_relevance": "How relevant are these themes for business decisions?"
            }}
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.3)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._fallback_thematic_insights(common_themes)
                
        except Exception as e:
            self.logger.warning(f"AI thematic insights generation failed: {str(e)}")
            return self._fallback_thematic_insights(common_themes)
    
    def _fallback_thematic_insights(self, common_themes: List[Dict]) -> Dict[str, Any]:
        """Fallback thematic insights"""
        return {
            'central_narrative': 'Document collection covers multiple related business topics',
            'thematic_gaps': ['Additional context may be needed for complete understanding'],
            'strategic_implications': ['Themes suggest need for integrated analysis'],
            'theme_relationships': ['Themes appear to be interconnected through common entities'],
            'business_relevance': 'moderate',
            'fallback_mode': True
        }
    
    def _analyze_theme_evolution(self, common_themes: List[Dict]) -> str:
        """Analyze how themes evolve across documents"""
        if not common_themes:
            return "No thematic evolution detected"
        
        if len(common_themes) < 2:
            return "Single dominant theme across documents"
        
        # Simple analysis based on theme prevalence
        high_prevalence = [t for t in common_themes if t['prevalence'] > 0.7]
        medium_prevalence = [t for t in common_themes if 0.3 < t['prevalence'] <= 0.7]
        
        if len(high_prevalence) > 1:
            return "Multiple dominant themes with high consistency"
        elif len(medium_prevalence) > 2:
            return "Diverse thematic landscape with moderate consistency"
        else:
            return "Mixed thematic content with low consistency"
    
    async def _identify_strategic_findings(self, reasoning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strategic findings and insights"""
        reasoning_output = reasoning_data['reasoning_output']
        analysis_output = reasoning_data.get('analysis_output', {})
        
        strategic_findings = []
        
        # Finding 1: Key entity relationships
        key_relationships = reasoning_output.get('reasoning_results', {}).get('combined_insights', {}).get('causal_relationships', [])
        if key_relationships:
            finding = {
                'type': 'causal_relationships',
                'title': 'Critical Causal Relationships Identified',
                'description': f'Analysis revealed {len(key_relationships)} potential cause-effect relationships',
                'implications': 'Understanding these relationships can inform strategic decision-making',
                'confidence': 0.8,
                'actionable': True
            }
            strategic_findings.append(finding)
        
        # Finding 2: Knowledge gaps
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            finding = {
                'type': 'knowledge_gaps',
                'title': 'Information Inconsistencies Detected',
                'description': f'Found {len(contradictions)} contradictions requiring clarification',
                'implications': 'These inconsistencies may indicate incomplete information or conflicting sources',
                'confidence': 0.7,
                'actionable': True
            }
            strategic_findings.append(finding)
        
        # Finding 3: Research opportunities
        hypotheses = reasoning_output.get('hypotheses', [])
        testable_hypotheses = [h for h in hypotheses if h['confidence'] > 0.6]
        if testable_hypotheses:
            finding = {
                'type': 'research_opportunities',
                'title': 'High-Confidence Research Hypotheses',
                'description': f'{len(testable_hypotheses)} testable hypotheses with high confidence',
                'implications': 'These represent clear opportunities for further investigation',
                'confidence': 0.9,
                'actionable': True
            }
            strategic_findings.append(finding)
        
        # Finding 4: Collection completeness
        doc_count = len([n for n in reasoning_output['knowledge_graph']['nodes'] if n['type'] == 'document'])
        if doc_count > 5:
            finding = {
                'type': 'collection_analysis',
                'title': 'Comprehensive Document Collection',
                'description': f'Analysis of {doc_count} documents provides substantial knowledge base',
                'implications': 'Collection appears sufficient for reliable insights and decision support',
                'confidence': 0.8,
                'actionable': False
            }
            strategic_findings.append(finding)
        
        return strategic_findings
    
    def _assess_data_completeness(self, reasoning_data: Dict[str, Any]) -> float:
        """Assess completeness of the data and analysis"""
        reasoning_output = reasoning_data['reasoning_output']
        
        completeness_factors = [
            min(1.0, len(reasoning_output['knowledge_graph']['nodes']) / 20),  # Node density
            min(1.0, len(reasoning_output.get('inferences', [])) / 10),  # Inference count
            1.0 if reasoning_output.get('reasoning_results') else 0.0,  # Reasoning results exist
            1.0 if reasoning_output.get('qa_knowledge_base') else 0.0,  # QA base exists
        ]
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _assess_analysis_depth(self, reasoning_results: Dict[str, Any]) -> str:
        """Assess the depth of analysis performed"""
        reasoning_types = [
            'deductive_reasoning',
            'inductive_reasoning', 
            'abductive_reasoning',
            'analogical_reasoning',
            'causal_reasoning'
        ]
        
        completed_types = sum(1 for rt in reasoning_types if rt in reasoning_results and reasoning_results[rt])
        
        if completed_types >= 4:
            return "comprehensive"
        elif completed_types >= 2:
            return "moderate"
        else:
            return "basic"
    
    def _calculate_confidence_metrics(
        self, 
        inferences: List[Dict], 
        reasoning_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence metrics"""
        inference_confidences = [inf['confidence'] for inf in inferences if 'confidence' in inf]
        
        return {
            'average_inference_confidence': sum(inference_confidences) / len(inference_confidences) if inference_confidences else 0,
            'high_confidence_inferences_ratio': len([c for c in inference_confidences if c > 0.7]) / len(inference_confidences) if inference_confidences else 0,
            'overall_reasoning_confidence': reasoning_results.get('combined_insights', {}).get('overall_reasoning_confidence', 0)
        }
    
    async def _create_executive_summary(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Create executive summary for business leaders"""
        reasoning_output = reasoning_data['reasoning_output']
        
        # Key metrics
        knowledge_graph = reasoning_output['knowledge_graph']
        doc_count = len([n for n in knowledge_graph['nodes'] if n['type'] == 'document'])
        entity_count = len([n for n in knowledge_graph['nodes'] if n['type'] == 'entity'])
        inference_count = len(reasoning_output.get('inferences', []))
        
        executive_summary = {
            'summary_header': {
                'title': 'Executive Summary - Document Intelligence Analysis',
                'session_id': session_id,
                'analysis_scope': f'{doc_count} documents analyzed',
                'key_metrics': {
                    'documents_processed': doc_count,
                    'entities_identified': entity_count,
                    'insights_generated': inference_count
                }
            },
            
            'key_findings': await self._generate_executive_key_findings(reasoning_data),
            
            'strategic_recommendations': await self._generate_executive_recommendations(reasoning_data),
            
            'risk_assessment': await self._assess_risks_and_opportunities(reasoning_data),
            
            'next_steps': await self._recommend_next_steps(reasoning_data),
            
            'confidence_assessment': {
                'overall_confidence': reasoning_output.get('reasoning_results', {}).get('combined_insights', {}).get('overall_reasoning_confidence', 0),
                'data_quality': 'high' if entity_count > 20 else 'medium' if entity_count > 10 else 'basic',
                'recommendation_reliability': 'high' if inference_count > 5 else 'medium'
            }
        }
        
        return executive_summary
    
    async def _generate_executive_key_findings(self, reasoning_data: Dict[str, Any]) -> List[str]:
        """Generate key findings for executives"""
        reasoning_output = reasoning_data['reasoning_output']
        analysis_output = reasoning_data.get('analysis_output', {})
        
        findings = []
        
        # Top inferences
        inferences = reasoning_output.get('inferences', [])
        high_conf_inferences = [inf for inf in inferences if inf.get('confidence', 0) > 0.7]
        
        if high_conf_inferences:
            findings.append(f"High-confidence analysis identified {len(high_conf_inferences)} key business insights")
        
        # Document coherence
        knowledge_graph = reasoning_output['knowledge_graph']
        doc_count = len([n for n in knowledge_graph['nodes'] if n['type'] == 'document'])
        
        if doc_count > 3:
            similar_docs = len([e for e in knowledge_graph['edges'] if e['type'] == 'similar_to'])
            if similar_docs > 0:
                findings.append(f"Document collection shows strong thematic coherence with {similar_docs} similarity relationships")
        
        # Entity concentration
        entity_dist = {}
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                entity_type = node['subtype']
                entity_dist[entity_type] = entity_dist.get(entity_type, 0) + 1
        
        if entity_dist:
            dominant_type = max(entity_dist.items(), key=lambda x: x[1])
            findings.append(f"Analysis reveals focus on {dominant_type[0]} entities ({dominant_type[1]} identified)")
        
        # Contradictions as risks
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            findings.append(f"Identified {len(contradictions)} information inconsistencies requiring attention")
        
        return findings[:5]  # Top 5 findings
    
    async def _generate_executive_recommendations(self, reasoning_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate strategic recommendations for executives"""
        reasoning_output = reasoning_data['reasoning_output']
        
        recommendations = []
        
        # Recommendation from hypotheses
        hypotheses = reasoning_output.get('hypotheses', [])
        high_conf_hypotheses = [h for h in hypotheses if h['confidence'] > 0.7]
        
        if high_conf_hypotheses:
            recommendations.append({
                'priority': 'high',
                'action': 'Investigate high-confidence hypotheses',
                'rationale': f'{len(high_conf_hypotheses)} hypotheses show strong potential for actionable insights',
                'timeline': 'short-term'
            })
        
        # Recommendation from contradictions
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            recommendations.append({
                'priority': 'medium',
                'action': 'Resolve information contradictions',
                'rationale': 'Inconsistent information may impact decision quality',
                'timeline': 'medium-term'
            })
        
        # Recommendation from knowledge gaps
        knowledge_graph = reasoning_output['knowledge_graph']
        entity_coverage = len([n for n in knowledge_graph['nodes'] if n['type'] == 'entity'])
        
        if entity_coverage < 15:
            recommendations.append({
                'priority': 'medium',
                'action': 'Expand information collection',
                'rationale': 'Limited entity coverage suggests additional sources may provide value',
                'timeline': 'long-term'
            })
        
        return recommendations
    
    async def _assess_risks_and_opportunities(self, reasoning_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Assess risks and opportunities"""
        reasoning_output = reasoning_data['reasoning_output']
        
        risks = []
        opportunities = []
        
        # Risks from contradictions
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            risks.append(f"Information inconsistencies ({len(contradictions)} identified) may lead to poor decisions")
        
        # Risks from low confidence
        inferences = reasoning_output.get('inferences', [])
        low_conf_inferences = [inf for inf in inferences if inf.get('confidence', 0) < 0.5]
        
        if len(low_conf_inferences) > len(inferences) / 2:
            risks.append("High proportion of low-confidence insights may indicate data quality issues")
        
        # Opportunities from hypotheses
        hypotheses = reasoning_output.get('hypotheses', [])
        if hypotheses:
            opportunities.append(f"Research opportunities identified ({len(hypotheses)} testable hypotheses)")
        
        # Opportunities from reasoning depth
        reasoning_confidence = reasoning_output.get('reasoning_results', {}).get('combined_insights', {}).get('overall_reasoning_confidence', 0)
        if reasoning_confidence > 0.7:
            opportunities.append("High reasoning confidence enables reliable strategic planning")
        
        return {
            'risks': risks,
            'opportunities': opportunities
        }
    
    async def _recommend_next_steps(self, reasoning_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Recommend next steps"""
        next_steps = [
            {
                'step': 'Review high-confidence insights',
                'description': 'Examine the most reliable findings for immediate action',
                'owner': 'Strategy Team',
                'timeframe': '1-2 weeks'
            },
            {
                'step': 'Validate key hypotheses',
                'description': 'Test the most promising hypotheses with additional research',
                'owner': 'Research Team',
                'timeframe': '4-6 weeks'
            },
            {
                'step': 'Address information gaps',
                'description': 'Collect additional data to resolve contradictions and fill gaps',
                'owner': 'Data Team',
                'timeframe': '2-4 weeks'
            }
        ]
        
        return next_steps
    
    async def _generate_actionable_insights(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate specific actionable insights"""
        reasoning_output = reasoning_data['reasoning_output']
        actionable_insights = []
        
        # Process high-confidence inferences
        inferences = reasoning_output.get('inferences', [])
        high_conf_inferences = [inf for inf in inferences if inf.get('confidence', 0) > 0.7]
        
        for inference in high_conf_inferences[:5]:  # Top 5
            insight = {
                'type': 'inference_insight',
                'title': f"Strategic Insight: {inference['inference'][:50]}...",
                'description': inference['inference'],
                'evidence': inference.get('evidence', ''),
                'confidence': inference['confidence'],
                'actionability': 'high',
                'recommended_action': await self._generate_action_from_inference(inference),
                'business_impact': await self._assess_business_impact(inference)
            }
            actionable_insights.append(insight)
        
        # Process causal relationships
        causal_chains = reasoning_output.get('reasoning_results', {}).get('causal_reasoning', {}).get('causal_chains', [])
        for causal in causal_chains[:3]:  # Top 3
            insight = {
                'type': 'causal_insight',
                'title': f"Causal Relationship: {causal.get('cause', 'Unknown')} → {causal.get('effect', 'Unknown')}",
                'description': f"Identified relationship between {causal.get('cause')} and {causal.get('effect')}",
                'evidence': causal.get('evidence', ''),
                'confidence': causal.get('confidence', 0.5),
                'actionability': 'medium',
                'recommended_action': f"Monitor {causal.get('cause')} to predict changes in {causal.get('effect')}",
                'business_impact': 'Enables proactive management of cause-effect relationships'
            }
            actionable_insights.append(insight)
        
        # Process knowledge gaps as opportunities
        contradictions = reasoning_output.get('contradictions', [])
        for contradiction in contradictions[:2]:  # Top 2
            insight = {
                'type': 'gap_insight',
                'title': f"Information Gap: {contradiction['description'][:50]}...",
                'description': contradiction['description'],
                'evidence': f"Contradiction severity: {contradiction.get('severity', 'unknown')}",
                'confidence': 0.8,
                'actionability': 'high',
                'recommended_action': 'Collect additional information to resolve contradiction',
                'business_impact': 'Resolving gaps improves decision quality'
            }
            actionable_insights.append(insight)
        
        return actionable_insights
    
    async def _generate_action_from_inference(self, inference: Dict[str, Any]) -> str:
        """Generate specific action recommendation from inference"""
        inference_type = inference.get('type', 'unknown')
        
        if inference_type == 'importance_inference':
            return "Focus strategic attention on this key theme"
        elif inference_type == 'pattern_inference':
            return "Leverage identified pattern for process optimization"
        elif inference_type == 'causal_inference':
            return "Monitor causal factors for predictive insights"
        else:
            return "Incorporate insight into strategic planning"
    
    async def _assess_business_impact(self, inference: Dict[str, Any]) -> str:
        """Assess potential business impact of insight"""
        confidence = inference.get('confidence', 0)
        
        if confidence > 0.8:
            return "High potential for strategic advantage"
        elif confidence > 0.6:
            return "Moderate impact on business decisions"
        else:
            return "Low-to-medium impact, requires validation"
    
    async def _prepare_qa_system(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Prepare intelligent Q&A system"""
        reasoning_output = reasoning_data['reasoning_output']
        qa_knowledge = reasoning_output.get('qa_knowledge_base', {})
        
        # Generate common questions and answers
        qa_pairs = await self._generate_common_qa_pairs(reasoning_data)
        
        # Create searchable indexes
        search_indexes = {
            'entity_index': qa_knowledge.get('searchable_content', {}).get('entity_index', []),
            'topic_index': qa_knowledge.get('searchable_content', {}).get('topic_index', []),
            'keyword_index': qa_knowledge.get('searchable_content', {}).get('keyword_index', [])
        }
        
        # Query processing capabilities
        query_capabilities = {
            'supported_query_types': [
                'entity_lookup',
                'document_search',
                'relationship_queries',
                'summary_requests',
                'comparison_queries'
            ],
            'response_formats': ['detailed', 'summary', 'bullet_points']
        }
        
        return {
            'qa_pairs': qa_pairs,
            'search_indexes': search_indexes,
            'query_capabilities': query_capabilities,
            'knowledge_base': qa_knowledge,
            'session_id': session_id
        }
    
    async def _generate_common_qa_pairs(self, reasoning_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate common question-answer pairs"""
        reasoning_output = reasoning_data['reasoning_output']
        knowledge_graph = reasoning_output['knowledge_graph']
        
        qa_pairs = []
        
        # Document overview questions
        doc_count = len([n for n in knowledge_graph['nodes'] if n['type'] == 'document'])
        qa_pairs.append({
            'question': 'How many documents were analyzed?',
            'answer': f'The analysis processed {doc_count} documents.',
            'confidence': 1.0,
            'category': 'overview'
        })
        
        # Entity questions
        entities = [n for n in knowledge_graph['nodes'] if n['type'] == 'entity']
        if entities:
            top_entity = max(entities, key=lambda x: x['properties'].get('confidence', 0))
            qa_pairs.append({
                'question': 'What are the most important entities identified?',
                'answer': f'Key entities include {top_entity["label"]} and others, with {len(entities)} total entities identified.',
                'confidence': 0.8,
                'category': 'entities'
            })
        
        # Insights questions
        inferences = reasoning_output.get('inferences', [])
        if inferences:
            qa_pairs.append({
                'question': 'What are the key insights from the analysis?',
                'answer': f'Analysis generated {len(inferences)} insights, including patterns in entity relationships and document themes.',
                'confidence': 0.9,
                'category': 'insights'
            })
        
        # Contradictions questions
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            qa_pairs.append({
                'question': 'Are there any contradictions in the information?',
                'answer': f'Yes, {len(contradictions)} contradictions were identified that may require clarification.',
                'confidence': 0.9,
                'category': 'quality'
            })
        
        return qa_pairs
    
    async def _generate_output_formats(
        self, 
        intelligence_report: Dict[str, Any],
        executive_summary: Dict[str, Any],
        actionable_insights: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, str]:
        """Generate different output formats"""
        formats = {}
        
        # JSON format (already structured)
        formats['json'] = json.dumps({
            'intelligence_report': intelligence_report,
            'executive_summary': executive_summary,
            'actionable_insights': actionable_insights
        }, indent=2)
        
        # Markdown format
        formats['markdown'] = await self._generate_markdown_output(
            intelligence_report, executive_summary, actionable_insights
        )
        
        # Summary format
        formats['summary'] = await self._generate_summary_output(
            intelligence_report, executive_summary, actionable_insights
        )
        
        return formats
    
    async def _generate_markdown_output(
        self,
        intelligence_report: Dict[str, Any],
        executive_summary: Dict[str, Any],
        actionable_insights: List[Dict[str, Any]]
    ) -> str:
        """Generate markdown formatted output"""
        markdown = f"""
# Document Intelligence Analysis Report

## Executive Summary

**Analysis Scope:** {executive_summary['summary_header']['analysis_scope']}

### Key Metrics
- Documents Processed: {executive_summary['summary_header']['key_metrics']['documents_processed']}
- Entities Identified: {executive_summary['summary_header']['key_metrics']['entities_identified']}
- Insights Generated: {executive_summary['summary_header']['key_metrics']['insights_generated']}

### Key Findings
"""
        
        for finding in executive_summary.get('key_findings', []):
            markdown += f"- {finding}\n"
        
        markdown += "\n## Intelligence Report\n\n"
        markdown += f"**Total Documents:** {intelligence_report['collection_overview']['document_count']}\n"
        markdown += f"**Entity Analysis:** {intelligence_report['entity_analysis']['total_entities']} entities identified\n"
        markdown += f"**Reasoning Confidence:** {intelligence_report['knowledge_insights']['reasoning_confidence']:.2f}\n"
        
        markdown += "\n## Actionable Insights\n\n"
        
        for i, insight in enumerate(actionable_insights[:5], 1):
            markdown += f"### {i}. {insight['title']}\n\n"
            markdown += f"**Description:** {insight['description']}\n\n"
            markdown += f"**Confidence:** {insight['confidence']:.2f}\n\n"
            markdown += f"**Recommended Action:** {insight['recommended_action']}\n\n"
        
        return markdown
    
    async def _generate_summary_output(
        self,
        intelligence_report: Dict[str, Any],
        executive_summary: Dict[str, Any],
        actionable_insights: List[Dict[str, Any]]
    ) -> str:
        """Generate concise summary output"""
        summary = f"""DOCUMENT INTELLIGENCE SUMMARY

Analyzed: {executive_summary['summary_header']['key_metrics']['documents_processed']} documents
Entities: {executive_summary['summary_header']['key_metrics']['entities_identified']} identified
Insights: {executive_summary['summary_header']['key_metrics']['insights_generated']} generated

TOP FINDINGS:
"""
        
        for i, finding in enumerate(executive_summary.get('key_findings', [])[:3], 1):
            summary += f"{i}. {finding}\n"
        
        summary += f"\nCONFIDENCE LEVEL: {intelligence_report['knowledge_insights']['reasoning_confidence']:.1%}\n"
        
        if actionable_insights:
            summary += f"\nTOP RECOMMENDATION: {actionable_insights[0]['recommended_action']}\n"
        
        return summary
    
    async def _create_knowledge_index(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Create searchable knowledge index"""
        reasoning_output = reasoning_data['reasoning_output']
        knowledge_graph = reasoning_output['knowledge_graph']
        
        # Build comprehensive index
        index = {
            'entities': {},
            'documents': {},
            'concepts': {},
            'relationships': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'session_id': session_id,
                'total_nodes': len(knowledge_graph['nodes']),
                'total_edges': len(knowledge_graph['edges'])
            }
        }
        
        # Index entities
        for node in knowledge_graph['nodes']:
            if node['type'] == 'entity':
                index['entities'][node['label']] = {
                    'type': node['subtype'],
                    'confidence': node['properties'].get('confidence', 0),
                    'node_id': node['id']
                }
            elif node['type'] == 'document':
                index['documents'][node['label']] = {
                    'document_type': node['properties'].get('document_type', 'unknown'),
                    'sentiment': node['properties'].get('sentiment', 'neutral'),
                    'node_id': node['id']
                }
            elif node['type'] == 'concept':
                index['concepts'][node['label']] = {
                    'concept_type': node['properties'].get('topic_type', 'general'),
                    'node_id': node['id']
                }
        
        # Index relationships
        for edge in knowledge_graph['edges']:
            source_node = knowledge_graph['node_index'].get(edge['source'])
            target_node = knowledge_graph['node_index'].get(edge['target'])
            
            if source_node and target_node:
                index['relationships'].append({
                    'source': source_node['label'],
                    'target': target_node['label'],
                    'relationship': edge['type'],
                    'confidence': edge['properties'].get('confidence', 0)
                })
        
        return index
    
    async def _generate_recommendations(
        self, 
        reasoning_data: Dict[str, Any], 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations"""
        reasoning_output = reasoning_data['reasoning_output']
        
        recommendations = []
        
        # Data quality recommendations
        contradictions = reasoning_output.get('contradictions', [])
        if contradictions:
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high',
                'title': 'Resolve Information Contradictions',
                'description': f'Address {len(contradictions)} identified contradictions to improve data reliability',
                'specific_actions': [
                    'Review source documents for conflicting information',
                    'Establish data validation protocols',
                    'Create conflict resolution procedures'
                ],
                'expected_outcome': 'Improved data quality and decision confidence',
                'timeframe': '2-4 weeks'
            })
        
        # Research recommendations
        hypotheses = reasoning_output.get('hypotheses', [])
        high_conf_hypotheses = [h for h in hypotheses if h['confidence'] > 0.6]
        if high_conf_hypotheses:
            recommendations.append({
                'category': 'research',
                'priority': 'medium',
                'title': 'Investigate High-Confidence Hypotheses',
                'description': f'Test {len(high_conf_hypotheses)} promising hypotheses for strategic insights',
                'specific_actions': [
                    'Design validation studies for top hypotheses',
                    'Collect additional supporting data',
                    'Conduct focused research initiatives'
                ],
                'expected_outcome': 'Validated insights for strategic planning',
                'timeframe': '4-8 weeks'
            })
        
        # Process improvement recommendations
        reasoning_confidence = reasoning_output.get('reasoning_results', {}).get('combined_insights', {}).get('overall_reasoning_confidence', 0)
        if reasoning_confidence > 0.7:
            recommendations.append({
                'category': 'process_improvement',
                'priority': 'low',
                'title': 'Leverage High-Quality Analysis Framework',
                'description': 'Current analysis framework shows high reliability for future use',
                'specific_actions': [
                    'Document successful analysis methodology',
                    'Create reusable analysis templates',
                    'Train team on effective analysis techniques'
                ],
                'expected_outcome': 'Repeatable high-quality analysis capability',
                'timeframe': '1-2 weeks'
            })
        
        # Knowledge management recommendations
        entity_count = len([n for n in reasoning_output['knowledge_graph']['nodes'] if n['type'] == 'entity'])
        if entity_count > 20:
            recommendations.append({
                'category': 'knowledge_management',
                'priority': 'medium',
                'title': 'Establish Knowledge Management System',
                'description': 'Rich entity data suggests value in systematic knowledge management',
                'specific_actions': [
                    'Implement knowledge graph database',
                    'Create entity relationship tracking',
                    'Establish knowledge update procedures'
                ],
                'expected_outcome': 'Systematic organizational knowledge management',
                'timeframe': '6-12 weeks'
            })
        
        return recommendations
    
    async def _save_response_outputs(
        self, 
        intelligence_report: Dict[str, Any],
        executive_summary: Dict[str, Any],
        actionable_insights: List[Dict[str, Any]],
        qa_system: Dict[str, Any],
        output_formats: Dict[str, str],
        knowledge_index: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        session_id: str
    ):
        """Save all response outputs"""
        try:
            output_path = Path(self.output_dir) / session_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save intelligence report
            report_file = output_path / "intelligence_report.json"
            with open(report_file, 'w') as f:
                json.dump(intelligence_report, f, indent=2)
            
            # Save executive summary
            summary_file = output_path / "executive_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(executive_summary, f, indent=2)
            
            # Save actionable insights
            insights_file = output_path / "actionable_insights.json"
            with open(insights_file, 'w') as f:
                json.dump(actionable_insights, f, indent=2)
            
            # Save QA system
            qa_file = output_path / "qa_system.json"
            with open(qa_file, 'w') as f:
                json.dump(qa_system, f, indent=2)
            
            # Save knowledge index
            index_file = output_path / "knowledge_index.json"
            with open(index_file, 'w') as f:
                json.dump(knowledge_index, f, indent=2)
            
            # Save recommendations
            rec_file = output_path / "recommendations.json"
            with open(rec_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            # Save output formats
            for format_name, content in output_formats.items():
                format_file = output_path / f"output.{format_name}"
                with open(format_file, 'w') as f:
                    f.write(content)
            
            # Create final comprehensive report
            final_report = {
                'session_summary': {
                    'session_id': session_id,
                    'completion_time': datetime.now().isoformat(),
                    'intelligence_report': intelligence_report,
                    'executive_summary': executive_summary,
                    'actionable_insights': actionable_insights,
                    'recommendations': recommendations,
                    'processing_pipeline': 'ADK Multi-Agent System - Complete'
                }
            }
            
            final_file = output_path / "FINAL_REPORT.json"
            with open(final_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info(f"Saved all response outputs to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save response outputs: {str(e)}")


async def test_response_agent():
    """Test function for Document Response Agent"""
    # This would be called with output from reasoning agent
    print("Response Agent test would run here with reasoning output")


if __name__ == "__main__":
    asyncio.run(test_response_agent())