"""
Document Ingestion Agent - First agent in the ADK pipeline
Specializes in document intake, validation, and initial processing
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from utils.adk_framework import BaseAgent, AgentResult, AgentStatus
from utils.document_processor import DocumentProcessor, DocumentMetadata, DocumentChunk


class DocumentIngestionAgent(BaseAgent):
    """
    Agent responsible for document ingestion and initial processing.
    
    Capabilities:
    - Document validation and format checking
    - Text extraction from multiple formats (PDF, DOCX, TXT)
    - Metadata extraction and document fingerprinting
    - Content chunking for downstream processing
    - Batch processing for multiple documents
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DocumentIngestionAgent", config)
        
        # Initialize document processor
        self.processor = DocumentProcessor(config.get('processing', {}))
        
        # Agent-specific configuration
        self.max_concurrent = config.get('max_concurrent_documents', 5)
        self.supported_formats = config.get('supported_formats', ['.pdf', '.docx', '.txt'])
        self.output_dir = config.get('output_dir', 'output/ingestion')
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} with support for {self.supported_formats}")
    
    async def process(self, input_data: Any, session_id: str) -> AgentResult:
        """
        Main processing method for document ingestion.
        
        Args:
            input_data: Can be a single file path, list of file paths, or dict with files and metadata
            session_id: Unique session identifier
            
        Returns:
            AgentResult with processed documents, metadata, and chunks
        """
        start_time = time.time()
        
        try:
            # Pre-process input
            processed_input = await self.pre_process(input_data, session_id)
            
            # Validate input format
            if not await self.validate_input(processed_input):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="Invalid input format"
                )
            
            # Extract file paths from input
            file_paths = self._extract_file_paths(processed_input)
            
            # Validate all documents before processing
            validation_results = await self._validate_documents(file_paths)
            valid_files = [fp for fp, is_valid, _ in validation_results if is_valid]
            
            if not valid_files:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="No valid documents found for processing"
                )
            
            # Process documents
            processing_results = await self._process_documents(valid_files, session_id)
            
            # Generate summary and insights
            ingestion_summary = await self._generate_ingestion_summary(
                processing_results, validation_results, session_id
            )
            
            # Save results to disk
            await self._save_results(processing_results, ingestion_summary, session_id)
            
            # Prepare output for next agent
            output = {
                'session_id': session_id,
                'processed_documents': processing_results,
                'ingestion_summary': ingestion_summary,
                'total_documents': len(valid_files),
                'total_chunks': sum(len(chunks) for _, chunks in processing_results.values() if chunks),
                'validation_results': validation_results,
                'agent_metadata': {
                    'agent_name': self.name,
                    'processing_time': time.time() - start_time,
                    'supported_formats': self.supported_formats
                }
            }
            
            self.logger.info(
                f"Successfully processed {len(valid_files)} documents "
                f"with {output['total_chunks']} total chunks"
            )
            
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                output=output,
                execution_time=time.time() - start_time,
                session_id=session_id,
                metadata={
                    'documents_processed': len(valid_files),
                    'total_chunks': output['total_chunks'],
                    'validation_failures': len(file_paths) - len(valid_files)
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, session_id)
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data structure and content"""
        try:
            # Handle different input formats
            if isinstance(input_data, str):
                # Single file path
                return Path(input_data).exists()
            elif isinstance(input_data, list):
                # List of file paths
                return all(isinstance(fp, str) for fp in input_data)
            elif isinstance(input_data, dict):
                # Dictionary with metadata
                if 'files' in input_data:
                    return isinstance(input_data['files'], (str, list))
                elif 'original_input' in input_data:
                    # Input from orchestrator
                    return await self.validate_input(input_data['original_input'])
            
            return False
            
        except Exception:
            return False
    
    def _extract_file_paths(self, input_data: Any) -> List[str]:
        """Extract file paths from various input formats"""
        if isinstance(input_data, str):
            return [input_data]
        elif isinstance(input_data, list):
            return input_data
        elif isinstance(input_data, dict):
            if 'files' in input_data:
                files = input_data['files']
                return [files] if isinstance(files, str) else files
            elif 'original_input' in input_data:
                return self._extract_file_paths(input_data['original_input'])
        
        return []
    
    async def _validate_documents(self, file_paths: List[str]) -> List[Tuple[str, bool, Optional[str]]]:
        """Validate all documents before processing"""
        validation_results = []
        
        for file_path in file_paths:
            is_valid, error_msg = await self.processor.validate_document(file_path)
            validation_results.append((file_path, is_valid, error_msg))
            
            if not is_valid:
                self.logger.warning(f"Validation failed for {file_path}: {error_msg}")
            else:
                self.logger.debug(f"Validation passed for {file_path}")
        
        return validation_results
    
    async def _process_documents(self, file_paths: List[str], session_id: str) -> Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]]:
        """Process validated documents"""
        self.logger.info(f"Processing {len(file_paths)} documents for session {session_id}")
        
        # Use batch processing for efficiency
        results = await self.processor.batch_process_documents(file_paths)
        
        # Filter out failed processing results
        successful_results = {k: v for k, v in results.items() if v is not None}
        
        failed_count = len(results) - len(successful_results)
        if failed_count > 0:
            self.logger.warning(f"{failed_count} documents failed processing")
        
        return successful_results
    
    async def _generate_ingestion_summary(
        self, 
        processing_results: Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]],
        validation_results: List[Tuple[str, bool, Optional[str]]],
        session_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive ingestion summary"""
        
        total_files = len(validation_results)
        valid_files = sum(1 for _, is_valid, _ in validation_results if is_valid)
        processed_files = len(processing_results)
        
        # Calculate statistics
        total_chunks = sum(len(chunks) for _, chunks in processing_results.values())
        total_size = sum(metadata.file_size for metadata, _ in processing_results.values())
        total_pages = sum(
            metadata.page_count for metadata, _ in processing_results.values() 
            if metadata.page_count is not None
        )
        
        # File type distribution
        file_types = {}
        for metadata, _ in processing_results.values():
            mime_type = metadata.mime_type
            file_types[mime_type] = file_types.get(mime_type, 0) + 1
        
        # Average chunk size
        all_chunks = [chunk for _, chunks in processing_results.values() for chunk in chunks]
        avg_chunk_size = sum(len(chunk.content) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        # Gemini-powered insights
        insights = await self._generate_ai_insights(processing_results)
        
        summary = {
            'session_id': session_id,
            'timestamp': time.time(),
            'statistics': {
                'total_files_submitted': total_files,
                'valid_files': valid_files,
                'processed_files': processed_files,
                'total_chunks': total_chunks,
                'total_size_bytes': total_size,
                'total_pages': total_pages,
                'average_chunk_size': avg_chunk_size
            },
            'file_types': file_types,
            'validation_failures': [
                {'file': fp, 'error': err} 
                for fp, is_valid, err in validation_results 
                if not is_valid
            ],
            'processing_success_rate': processed_files / valid_files if valid_files > 0 else 0,
            'ai_insights': insights,
            'documents': [
                {
                    'filename': metadata.filename,
                    'hash': metadata.file_hash,
                    'chunks': len(chunks),
                    'size': metadata.file_size,
                    'pages': metadata.page_count
                }
                for metadata, chunks in processing_results.values()
            ]
        }
        
        return summary
    
    async def _generate_ai_insights(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini to generate insights about the document collection"""
        try:
            # Prepare summary for AI analysis
            doc_summaries = []
            for file_path, (metadata, chunks) in processing_results.items():
                # Get first few chunks as sample content
                sample_content = ' '.join([
                    chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                    for chunk in chunks[:3]
                ])
                
                doc_summaries.append({
                    'filename': metadata.filename,
                    'type': metadata.mime_type,
                    'pages': metadata.page_count,
                    'sample_content': sample_content
                })
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze the following collection of {len(doc_summaries)} documents and provide insights:
            
            Documents:
            {json.dumps(doc_summaries, indent=2)}
            
            Provide analysis in JSON format with these insights:
            1. content_themes: Main themes or topics across documents
            2. document_complexity: Assessment of document complexity (simple/medium/complex)
            3. language_style: Writing style and formality level
            4. potential_relationships: How documents might relate to each other
            5. processing_recommendations: Recommendations for downstream analysis
            
            Return only valid JSON.
            """
            
            # Generate insights using Gemini
            response = await self.generate_gemini_response(prompt, temperature=0.3)
            
            # Parse JSON response
            try:
                insights = json.loads(response)
                return insights
            except json.JSONDecodeError:
                self.logger.warning("Could not parse AI insights JSON, using fallback")
                return self._generate_fallback_insights(processing_results)
                
        except Exception as e:
            self.logger.warning(f"AI insights generation failed: {str(e)}")
            return self._generate_fallback_insights(processing_results)
    
    def _generate_fallback_insights(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic insights without AI"""
        total_docs = len(processing_results)
        
        # Simple heuristics
        avg_chunks = sum(len(chunks) for _, chunks in processing_results.values()) / total_docs
        complexity = "simple" if avg_chunks < 5 else "medium" if avg_chunks < 15 else "complex"
        
        return {
            'content_themes': ['General business documents'],
            'document_complexity': complexity,
            'language_style': 'formal',
            'potential_relationships': ['Documents may be part of same project or domain'],
            'processing_recommendations': [
                'Proceed with standard analysis pipeline',
                f'Consider batching for {total_docs} documents'
            ],
            'fallback_mode': True
        }
    
    async def _save_results(
        self, 
        processing_results: Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]],
        summary: Dict[str, Any],
        session_id: str
    ):
        """Save processing results to disk"""
        try:
            output_path = Path(self.output_dir) / session_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            summary_file = output_path / "ingestion_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save document metadata
            metadata_file = output_path / "document_metadata.json"
            metadata_list = [metadata.to_dict() for metadata, _ in processing_results.values()]
            with open(metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
            # Save chunks
            chunks_file = output_path / "document_chunks.json"
            all_chunks = [
                chunk.to_dict() 
                for _, chunks in processing_results.values() 
                for chunk in chunks
            ]
            with open(chunks_file, 'w') as f:
                json.dump(all_chunks, f, indent=2)
            
            self.logger.info(f"Saved ingestion results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            # Don't fail the whole process if saving fails
    
    async def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """Get current processing status for a session"""
        return {
            'agent': self.name,
            'status': self.status.value,
            'session_id': session_id,
            'supported_formats': self.supported_formats,
            'max_concurrent': self.max_concurrent
        }
    
    async def cleanup_session(self, session_id: str):
        """Cleanup resources for a completed session"""
        try:
            # Could implement cleanup of temporary files, etc.
            self.logger.info(f"Cleanup completed for session {session_id}")
        except Exception as e:
            self.logger.error(f"Cleanup failed for session {session_id}: {str(e)}")


# Utility functions for the Ingestion Agent
def create_test_documents(output_dir: str = "data/test_documents"):
    """Create sample test documents for testing the ingestion agent"""
    import lorem
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create sample text file
    with open(Path(output_dir) / "sample_report.txt", 'w') as f:
        f.write("Sample Business Report\n")
        f.write("=" * 20 + "\n\n")
        f.write(lorem.paragraph() + "\n\n")
        f.write("Key Findings:\n")
        for i in range(3):
            f.write(f"- {lorem.sentence()}\n")
    
    print(f"Created test documents in {output_dir}")


async def test_ingestion_agent():
    """Test function for the Document Ingestion Agent"""
    # Sample configuration
    config = {
        'google_api_key': os.getenv('GOOGLE_API_KEY', 'test-key'),
        'gemini_model': 'gemini-1.5-pro',
        'processing': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_file_size_mb': 100
        },
        'max_concurrent_documents': 3,
        'output_dir': 'output/test_ingestion'
    }
    
    # Create agent
    agent = DocumentIngestionAgent(config)
    
    # Create test session
    session_id = f"test_session_{int(time.time())}"
    
    # Test with sample file paths (you'll need to create these)
    test_files = [
        "data/test_documents/sample_report.txt"
    ]
    
    try:
        result = await agent.process(test_files, session_id)
        print(f"Test result: {result.status}")
        print(f"Output keys: {list(result.output.keys()) if result.output else 'None'}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_ingestion_agent())