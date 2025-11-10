"""
Main Application - Agentic Document Intelligence System
Orchestrates the complete multi-agent pipeline using Google's Agent Development Kit
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

# Import the ADK framework and agents
from utils.adk_framework import AgentOrchestrator, AgentStatus
from utils.document_processor import DocumentProcessor
from agents.ingestion_agent import DocumentIngestionAgent
from agents.analysis_agent import DocumentAnalysisAgent
from agents.reasoning_agent import ReasoningAgent
from agents.response_agent import ResponseAgent


class AgenticDocumentIntelligence:
    """
    Main application class for the Agentic Document Intelligence System.
    
    This system uses Google's Agent Development Kit (ADK) to create an autonomous
    document processing and analysis pipeline with 4 specialized agents:
    
    1. DocumentIngestionAgent - Document processing and validation
    2. DocumentAnalysisAgent - Deep content analysis and entity extraction
    3. ReasoningAgent - Logical reasoning and knowledge graph construction
    4. ResponseAgent - Intelligent response generation and insights
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the main application"""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize orchestrator and agents
        self.orchestrator = AgentOrchestrator(self.config)
        self.agents = self._initialize_agents()
        
        # Register agents with orchestrator
        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Agentic Document Intelligence System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            # Fallback configuration
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available"""
        return {
            'google_ai': {
                'model': 'gemini-1.5-pro',
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'agents': {
                'ingestion': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'supported_formats': ['pdf', 'docx', 'txt'],
                    'output_dir': 'output/ingestion'
                },
                'analysis': {
                    'analysis_depth': 'comprehensive',
                    'confidence_threshold': 0.7,
                    'entity_types': ['PERSON', 'ORGANIZATION', 'DATE', 'LOCATION', 'CONCEPT'],
                    'output_dir': 'output/analysis'
                },
                'reasoning': {
                    'reasoning_mode': 'comprehensive',
                    'inference_depth': 3,
                    'confidence_threshold': 0.6,
                    'max_reasoning_chains': 10,
                    'output_dir': 'output/reasoning'
                },
                'response': {
                    'response_mode': 'comprehensive',
                    'output_formats': ['json', 'markdown', 'summary'],
                    'max_response_length': 2000,
                    'include_citations': True,
                    'confidence_threshold': 0.5,
                    'output_dir': 'output/responses'
                }
            },
            'pipeline': {
                'max_execution_time': 3600,  # 1 hour
                'save_intermediate_results': True,
                'parallel_processing': False
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/agentic_intelligence.log')
            ]
        )
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with their configurations"""
        agents = {}
        
        # Initialize each agent with its specific config
        agents['ingestion'] = DocumentIngestionAgent(self.config['agents']['ingestion'])
        agents['analysis'] = DocumentAnalysisAgent(self.config['agents']['analysis'])
        agents['reasoning'] = ReasoningAgent(self.config['agents']['reasoning'])
        agents['response'] = ResponseAgent(self.config['agents']['response'])
        
        return agents
    
    async def process_documents(
        self, 
        input_path: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process documents through the complete intelligence pipeline.
        
        Args:
            input_path: Path to document(s) to process
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Complete processing results
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting document processing pipeline for session {session_id}")
            self.logger.info(f"Input path: {input_path}")
            
            # Validate input path
            input_path_obj = Path(input_path)
            if not input_path_obj.exists():
                raise ValueError(f"Input path does not exist: {input_path}")
            
            # Define the processing pipeline
            pipeline_config = {
                'agents': ['DocumentIngestionAgent', 'DocumentAnalysisAgent', 'ReasoningAgent', 'ResponseAgent'],
                'mode': 'sequential',  # Process in order
                'save_intermediate': self.config['pipeline']['save_intermediate_results']
            }
            
            # Execute the pipeline
            results = await self.orchestrator.execute_pipeline(
                input_data=input_path,
                pipeline_config=pipeline_config,
                session_id=session_id
            )
            
            # Process results
            if results.status == AgentStatus.COMPLETED:
                self.logger.info(f"Pipeline completed successfully in {time.time() - start_time:.2f} seconds")
                
                # Extract final results
                final_output = results.output
                
                # Save complete session results
                await self._save_session_results(final_output, session_id)
                
                # Generate session summary
                session_summary = await self._generate_session_summary(
                    final_output, session_id, time.time() - start_time
                )
                
                return {
                    'session_id': session_id,
                    'status': 'success',
                    'execution_time': time.time() - start_time,
                    'results': final_output,
                    'session_summary': session_summary,
                    'output_location': f'output/responses/{session_id}'
                }
            
            else:
                self.logger.error(f"Pipeline failed with status: {results.status}")
                return {
                    'session_id': session_id,
                    'status': 'error',
                    'execution_time': time.time() - start_time,
                    'error': results.error,
                    'results': None
                }
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                'session_id': session_id,
                'status': 'error',
                'execution_time': time.time() - start_time,
                'error': str(e),
                'results': None
            }
    
    async def _save_session_results(self, results: Dict[str, Any], session_id: str):
        """Save complete session results"""
        try:
            session_dir = Path('output') / 'sessions' / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Save complete results
            results_file = session_dir / 'complete_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Session results saved to {session_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session results: {str(e)}")
    
    async def _generate_session_summary(
        self, 
        results: Dict[str, Any], 
        session_id: str, 
        execution_time: float
    ) -> Dict[str, Any]:
        """Generate a summary of the processing session"""
        try:
            # Extract key metrics from final results
            response_output = results.get('response_metadata', {})
            
            summary = {
                'session_id': session_id,
                'execution_time': execution_time,
                'processing_pipeline': 'ADK Multi-Agent System',
                'agents_executed': ['DocumentIngestionAgent', 'DocumentAnalysisAgent', 'ReasoningAgent', 'ResponseAgent'],
                'key_metrics': {
                    'documents_processed': response_output.get('documents_processed', 0),
                    'entities_identified': response_output.get('entities_identified', 0),
                    'insights_generated': response_output.get('generated_insights', 0),
                    'recommendations_count': response_output.get('recommendations_count', 0)
                },
                'output_formats_generated': response_output.get('output_formats', []),
                'processing_quality': {
                    'pipeline_complete': response_output.get('pipeline_complete', False),
                    'data_quality': 'high' if execution_time < 300 else 'medium',
                    'analysis_depth': 'comprehensive'
                },
                'output_locations': {
                    'main_results': f'output/responses/{session_id}',
                    'session_data': f'output/sessions/{session_id}',
                    'agent_outputs': {
                        'ingestion': f'output/ingestion/{session_id}',
                        'analysis': f'output/analysis/{session_id}',
                        'reasoning': f'output/reasoning/{session_id}',
                        'response': f'output/responses/{session_id}'
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate session summary: {str(e)}")
            return {
                'session_id': session_id,
                'execution_time': execution_time,
                'error': 'Failed to generate summary'
            }
    
    async def query_knowledge(
        self, 
        session_id: str, 
        query: str
    ) -> Dict[str, Any]:
        """
        Query the knowledge base from a previous processing session.
        
        Args:
            session_id: Session ID from previous processing
            query: Natural language query
            
        Returns:
            Query results
        """
        try:
            # Load session results
            session_file = Path('output') / 'sessions' / session_id / 'complete_results.json'
            
            if not session_file.exists():
                return {
                    'status': 'error',
                    'error': f'Session {session_id} not found'
                }
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Extract QA system
            qa_system = session_data.get('qa_system', {})
            knowledge_index = session_data.get('knowledge_index', {})
            
            # Simple query processing (would be more sophisticated in production)
            query_result = await self._process_query(query, qa_system, knowledge_index)
            
            return {
                'status': 'success',
                'query': query,
                'session_id': session_id,
                'result': query_result
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _process_query(
        self, 
        query: str, 
        qa_system: Dict[str, Any], 
        knowledge_index: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a natural language query against the knowledge base"""
        query_lower = query.lower()
        
        # Check pre-generated QA pairs
        qa_pairs = qa_system.get('qa_pairs', [])
        for qa in qa_pairs:
            if any(word in qa['question'].lower() for word in query_lower.split()):
                return {
                    'type': 'direct_answer',
                    'answer': qa['answer'],
                    'confidence': qa['confidence'],
                    'source': 'pre_generated_qa'
                }
        
        # Search entities
        entities = knowledge_index.get('entities', {})
        matching_entities = [
            entity for entity in entities.keys() 
            if entity.lower() in query_lower
        ]
        
        if matching_entities:
            entity_info = entities[matching_entities[0]]
            return {
                'type': 'entity_info',
                'answer': f"Information about {matching_entities[0]}: Type - {entity_info['type']}, Confidence - {entity_info['confidence']:.2f}",
                'confidence': entity_info['confidence'],
                'source': 'entity_index'
            }
        
        # Search documents
        documents = knowledge_index.get('documents', {})
        matching_docs = [
            doc for doc in documents.keys() 
            if any(word in doc.lower() for word in query_lower.split())
        ]
        
        if matching_docs:
            doc_info = documents[matching_docs[0]]
            return {
                'type': 'document_info',
                'answer': f"Document {matching_docs[0]} is a {doc_info['document_type']} with {doc_info['sentiment']} sentiment",
                'confidence': 0.7,
                'source': 'document_index'
            }
        
        # Fallback
        return {
            'type': 'no_match',
            'answer': 'I could not find specific information about that query in the processed documents.',
            'confidence': 0.3,
            'source': 'fallback',
            'suggestions': 'Try asking about specific entities, documents, or topics that were mentioned in the analysis.'
        }
    
    def get_session_list(self) -> List[Dict[str, Any]]:
        """Get list of all processing sessions"""
        try:
            sessions_dir = Path('output') / 'sessions'
            if not sessions_dir.exists():
                return []
            
            sessions = []
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    results_file = session_dir / 'complete_results.json'
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            session_info = {
                                'session_id': session_dir.name,
                                'processing_time': results.get('response_metadata', {}).get('processing_time', 0),
                                'documents_processed': len(results.get('complete_pipeline_data', {}).get('original_input', [])),
                                'status': 'completed',
                                'created_at': session_dir.stat().st_ctime
                            }
                            sessions.append(session_info)
                        except Exception as e:
                            self.logger.warning(f"Could not read session {session_dir.name}: {e}")
            
            return sorted(sessions, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get session list: {str(e)}")
            return []


async def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Agentic Document Intelligence System')
    parser.add_argument('input_path', help='Path to document(s) to process')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--session-id', help='Optional session ID')
    parser.add_argument('--query', help='Query a previous session (requires --session-id)')
    parser.add_argument('--list-sessions', action='store_true', help='List all processing sessions')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = AgenticDocumentIntelligence(args.config)
    
    if args.list_sessions:
        # List all sessions
        sessions = system.get_session_list()
        print("\nProcessing Sessions:")
        print("-" * 50)
        for session in sessions:
            print(f"Session ID: {session['session_id']}")
            print(f"Documents: {session['documents_processed']}")
            print(f"Status: {session['status']}")
            print(f"Duration: {session['processing_time']:.2f}s")
            print("-" * 50)
        return
    
    if args.query and args.session_id:
        # Query existing session
        result = await system.query_knowledge(args.session_id, args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {result['result']['answer']}")
        print(f"Confidence: {result['result']['confidence']:.2f}")
        return
    
    # Process documents
    print(f"\nStarting Agentic Document Intelligence System")
    print(f"Input: {args.input_path}")
    
    result = await system.process_documents(args.input_path, args.session_id)
    
    if result['status'] == 'success':
        print(f"\nProcessing completed successfully!")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        print(f"Session ID: {result['session_id']}")
        print(f"Results saved to: {result['output_location']}")
        
        # Print summary
        summary = result['session_summary']
        print(f"\nProcessing Summary:")
        print(f"   Documents processed: {summary['key_metrics']['documents_processed']}")
        print(f"   Entities identified: {summary['key_metrics']['entities_identified']}")
        print(f"   Insights generated: {summary['key_metrics']['insights_generated']}")
        print(f"   Recommendations: {summary['key_metrics']['recommendations_count']}")
        
    else:
        print(f"\nProcessing failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())