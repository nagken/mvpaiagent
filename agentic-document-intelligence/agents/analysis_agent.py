"""
Document Analysis Agent - Second agent in the ADK pipeline
Specializes in deep content analysis, entity extraction, and semantic understanding
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from utils.adk_framework import BaseAgent, AgentResult, AgentStatus
from utils.document_processor import DocumentChunk, DocumentMetadata


class DocumentAnalysisAgent(BaseAgent):
    """
    Agent responsible for deep document analysis and understanding.
    
    Capabilities:
    - Content semantic analysis and topic modeling
    - Named entity recognition and extraction
    - Sentiment analysis and tone detection
    - Key concept identification and relationships
    - Document classification and categorization
    - Cross-document similarity analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DocumentAnalysisAgent", config)
        
        # Agent-specific configuration
        self.analysis_depth = config.get('analysis_depth', 'comprehensive')
        self.entity_types = config.get('entity_types', ['PERSON', 'ORGANIZATION', 'DATE', 'LOCATION', 'CONCEPT'])
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.output_dir = config.get('output_dir', 'output/analysis')
        
        # Vector database configuration (would integrate with ChromaDB)
        self.embedding_model = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.name} with {self.analysis_depth} analysis depth")
    
    async def process(self, input_data: Any, session_id: str) -> AgentResult:
        """
        Main processing method for document analysis.
        
        Args:
            input_data: Output from DocumentIngestionAgent containing processed documents
            session_id: Unique session identifier
            
        Returns:
            AgentResult with comprehensive analysis results
        """
        start_time = time.time()
        
        try:
            # Pre-process input
            processed_input = await self.pre_process(input_data, session_id)
            
            # Validate input from ingestion agent
            if not await self.validate_input(processed_input):
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="Invalid input from ingestion agent"
                )
            
            # Extract documents and chunks from input
            documents_data = self._extract_documents_data(processed_input)
            
            if not documents_data:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.ERROR,
                    output=None,
                    execution_time=time.time() - start_time,
                    session_id=session_id,
                    error="No valid documents found in input"
                )
            
            # Perform comprehensive analysis
            analysis_results = await self._perform_comprehensive_analysis(
                documents_data, session_id
            )
            
            # Generate cross-document insights
            cross_document_analysis = await self._analyze_cross_document_relationships(
                analysis_results, session_id
            )
            
            # Create semantic embeddings for future retrieval
            embeddings_data = await self._generate_semantic_embeddings(
                documents_data, session_id
            )
            
            # Generate AI-powered insights
            ai_insights = await self._generate_ai_analysis_insights(
                analysis_results, cross_document_analysis, session_id
            )
            
            # Save analysis results
            await self._save_analysis_results(
                analysis_results, cross_document_analysis, embeddings_data, session_id
            )
            
            # Prepare output for next agent
            output = {
                'session_id': session_id,
                'document_analysis': analysis_results,
                'cross_document_analysis': cross_document_analysis,
                'semantic_embeddings': embeddings_data,
                'ai_insights': ai_insights,
                'analysis_metadata': {
                    'agent_name': self.name,
                    'analysis_depth': self.analysis_depth,
                    'processing_time': time.time() - start_time,
                    'documents_analyzed': len(documents_data),
                    'total_entities': sum(
                        len(result['entities']) for result in analysis_results.values()
                    )
                },
                'previous_agent_data': processed_input  # Pass through for next agent
            }
            
            self.logger.info(
                f"Analysis completed for {len(documents_data)} documents "
                f"with {output['analysis_metadata']['total_entities']} entities extracted"
            )
            
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                output=output,
                execution_time=time.time() - start_time,
                session_id=session_id,
                metadata={
                    'documents_analyzed': len(documents_data),
                    'entities_extracted': output['analysis_metadata']['total_entities'],
                    'analysis_depth': self.analysis_depth
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, session_id)
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input from ingestion agent"""
        try:
            # Handle orchestrator wrapper
            if isinstance(input_data, dict) and 'original_input' in input_data:
                actual_input = input_data['previous_results']['DocumentIngestionAgent'].output
            else:
                actual_input = input_data
            
            # Check for required fields from ingestion agent
            required_fields = ['processed_documents', 'session_id']
            if not all(field in actual_input for field in required_fields):
                return False
            
            processed_docs = actual_input['processed_documents']
            if not isinstance(processed_docs, dict) or not processed_docs:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_documents_data(self, input_data: Any) -> Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]]:
        """Extract documents and chunks from ingestion agent output"""
        try:
            # Handle orchestrator wrapper
            if isinstance(input_data, dict) and 'original_input' in input_data:
                ingestion_output = input_data['previous_results']['DocumentIngestionAgent'].output
            else:
                ingestion_output = input_data
            
            return ingestion_output['processed_documents']
            
        except Exception as e:
            self.logger.error(f"Failed to extract documents data: {str(e)}")
            return {}
    
    async def _perform_comprehensive_analysis(
        self, 
        documents_data: Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]],
        session_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive analysis on each document"""
        analysis_results = {}
        
        for file_path, (metadata, chunks) in documents_data.items():
            self.logger.info(f"Analyzing document: {metadata.filename}")
            
            # Combine all chunks for document-level analysis
            full_text = ' '.join([chunk.content for chunk in chunks])
            
            # Perform various analyses
            document_analysis = {
                'metadata': metadata.to_dict(),
                'document_hash': metadata.file_hash,
                'chunk_count': len(chunks),
                'total_length': len(full_text),
                'entities': await self._extract_entities(full_text, chunks),
                'topics': await self._analyze_topics(full_text),
                'sentiment': await self._analyze_sentiment(full_text),
                'keywords': await self._extract_keywords(full_text),
                'classification': await self._classify_document(full_text, metadata),
                'chunk_analysis': await self._analyze_chunks(chunks),
                'linguistic_features': await self._analyze_linguistic_features(full_text),
                'conceptual_structure': await self._analyze_conceptual_structure(full_text)
            }
            
            analysis_results[file_path] = document_analysis
        
        return analysis_results
    
    async def _extract_entities(self, text: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Extract named entities using Gemini AI"""
        try:
            prompt = f"""
            Extract named entities from the following text. Return a JSON object with these categories:
            - PERSON: Names of people
            - ORGANIZATION: Company names, institutions, agencies
            - LOCATION: Cities, countries, addresses, geographical locations  
            - DATE: Dates, time periods, years
            - CONCEPT: Important concepts, technologies, methodologies
            - MONETARY: Dollar amounts, financial figures
            - PRODUCT: Product names, service names
            
            For each entity, include the text and confidence score (0-1).
            
            Text to analyze:
            {text[:4000]}...  # Truncate for API limits
            
            Return only valid JSON in this format:
            {{
                "PERSON": [{{"text": "John Smith", "confidence": 0.95}}],
                "ORGANIZATION": [{{"text": "Apple Inc.", "confidence": 0.90}}],
                ...
            }}
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.1)
            
            try:
                entities = json.loads(response)
                
                # Add chunk-level entity mapping
                chunk_entities = []
                for i, chunk in enumerate(chunks):
                    chunk_entities.append({
                        'chunk_id': chunk.chunk_id,
                        'chunk_index': i,
                        'entities_found': await self._find_entities_in_chunk(chunk.content, entities)
                    })
                
                return {
                    'document_entities': entities,
                    'chunk_entities': chunk_entities,
                    'entity_counts': {
                        category: len(entity_list) 
                        for category, entity_list in entities.items()
                    }
                }
                
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse entity extraction JSON")
                return self._fallback_entity_extraction(text)
                
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {str(e)}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback entity extraction using simple heuristics"""
        # Simple pattern-based extraction
        import re
        
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'LOCATION': [],
            'DATE': [],
            'CONCEPT': [],
            'MONETARY': [],
            'PRODUCT': []
        }
        
        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        monetary_entities = re.findall(money_pattern, text)
        entities['MONETARY'] = [{'text': entity, 'confidence': 0.8} for entity in monetary_entities]
        
        # Extract years/dates
        year_pattern = r'\b(19|20)\d{2}\b'
        date_entities = re.findall(year_pattern, text)
        entities['DATE'] = [{'text': entity, 'confidence': 0.7} for entity in date_entities]
        
        return {
            'document_entities': entities,
            'chunk_entities': [],
            'entity_counts': {k: len(v) for k, v in entities.items()},
            'fallback_mode': True
        }
    
    async def _find_entities_in_chunk(self, chunk_text: str, document_entities: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """Find which entities appear in a specific chunk"""
        chunk_entities = {}
        
        for category, entity_list in document_entities.items():
            found_entities = []
            for entity in entity_list:
                if entity['text'].lower() in chunk_text.lower():
                    found_entities.append(entity['text'])
            chunk_entities[category] = found_entities
        
        return chunk_entities
    
    async def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """Analyze main topics and themes using Gemini"""
        try:
            prompt = f"""
            Analyze the main topics and themes in this document. Return a JSON object with:
            - main_topics: List of 3-5 primary topics/themes
            - topic_distribution: Estimated percentage distribution of content across topics
            - subject_matter: Overall subject matter category
            - complexity_level: Assessment of content complexity (basic/intermediate/advanced)
            
            Text to analyze:
            {text[:3000]}...
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.2)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._fallback_topic_analysis(text)
                
        except Exception as e:
            self.logger.warning(f"Topic analysis failed: {str(e)}")
            return self._fallback_topic_analysis(text)
    
    def _fallback_topic_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback topic analysis using keyword frequency"""
        words = text.lower().split()
        word_freq = {}
        
        # Count meaningful words (filter short words and common words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'main_topics': [word for word, _ in top_words],
            'topic_distribution': {'general': 100},
            'subject_matter': 'general',
            'complexity_level': 'intermediate',
            'fallback_mode': True
        }
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze document sentiment and tone using Gemini"""
        try:
            prompt = f"""
            Analyze the sentiment and tone of this document. Return a JSON object with:
            - overall_sentiment: positive/negative/neutral
            - sentiment_score: float from -1 (very negative) to 1 (very positive)
            - tone: formal/informal/academic/business/technical/conversational
            - emotional_indicators: list of emotional cues found
            - confidence: confidence in the analysis (0-1)
            
            Text to analyze:
            {text[:2000]}...
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.1)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._fallback_sentiment_analysis()
                
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            return self._fallback_sentiment_analysis()
    
    def _fallback_sentiment_analysis(self) -> Dict[str, Any]:
        """Fallback sentiment analysis"""
        return {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'tone': 'business',
            'emotional_indicators': [],
            'confidence': 0.5,
            'fallback_mode': True
        }
    
    async def _extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract key terms and phrases"""
        try:
            prompt = f"""
            Extract the 10 most important keywords and key phrases from this text.
            Return a JSON array of objects with "keyword" and "importance" (0-1 score).
            
            Text:
            {text[:3000]}...
            
            Format: [{{"keyword": "artificial intelligence", "importance": 0.95}}, ...]
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.1)
            
            try:
                keywords_data = json.loads(response)
                return [(item['keyword'], item['importance']) for item in keywords_data]
            except (json.JSONDecodeError, KeyError):
                return self._fallback_keyword_extraction(text)
                
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {str(e)}")
            return self._fallback_keyword_extraction(text)
    
    def _fallback_keyword_extraction(self, text: str) -> List[Tuple[str, float]]:
        """Fallback keyword extraction using frequency"""
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 4:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort and normalize
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        max_freq = max(freq for _, freq in sorted_words) if sorted_words else 1
        
        return [(word, freq / max_freq) for word, freq in sorted_words]
    
    async def _classify_document(self, text: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Classify document type and category"""
        try:
            prompt = f"""
            Classify this document into categories. Return JSON with:
            - document_type: contract/report/manual/article/proposal/other
            - domain: business/technical/legal/academic/financial/other
            - formality: formal/informal
            - purpose: informational/instructional/persuasive/analytical
            - audience: technical/business/general
            
            Document metadata: {metadata.filename}
            Text sample: {text[:2000]}...
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.1)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._fallback_document_classification(metadata)
                
        except Exception as e:
            self.logger.warning(f"Document classification failed: {str(e)}")
            return self._fallback_document_classification(metadata)
    
    def _fallback_document_classification(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Fallback document classification based on filename"""
        filename = metadata.filename.lower()
        
        if 'report' in filename:
            doc_type = 'report'
        elif 'contract' in filename:
            doc_type = 'contract'
        elif 'manual' in filename:
            doc_type = 'manual'
        else:
            doc_type = 'other'
        
        return {
            'document_type': doc_type,
            'domain': 'business',
            'formality': 'formal',
            'purpose': 'informational',
            'audience': 'business',
            'fallback_mode': True
        }
    
    async def _analyze_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Analyze individual chunks for granular insights"""
        chunk_analyses = []
        
        for chunk in chunks:
            # Basic chunk analysis
            analysis = {
                'chunk_id': chunk.chunk_id,
                'length': len(chunk.content),
                'word_count': len(chunk.content.split()),
                'sentence_count': len([s for s in chunk.content.split('.') if s.strip()]),
                'readability_score': self._calculate_readability(chunk.content),
                'has_numbers': any(c.isdigit() for c in chunk.content),
                'has_caps': any(c.isupper() for c in chunk.content),
                'density_score': len(chunk.content.split()) / len(chunk.content) if chunk.content else 0
            }
            
            chunk_analyses.append(analysis)
        
        return chunk_analyses
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (0-1, higher = more readable)"""
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Simple heuristic: shorter words and sentences = more readable
        readability = 1.0 - min(1.0, (avg_word_length - 3) / 10 + (avg_sentence_length - 10) / 30)
        return max(0.0, readability)
    
    async def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features of the text"""
        words = text.split()
        
        return {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'long_words_ratio': sum(1 for word in words if len(word) > 6) / len(words) if words else 0,
            'sentences': len([s for s in text.split('.') if s.strip()]),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()])
        }
    
    async def _analyze_conceptual_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the conceptual structure and flow of ideas"""
        # Simple analysis of text structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            'structure_type': 'sequential' if len(paragraphs) > 3 else 'simple',
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            'has_headers': any(line.isupper() or line.startswith('#') for line in text.split('\n')),
            'has_lists': '-' in text or '•' in text or any(line.strip().startswith(('1.', '2.', '3.')) for line in text.split('\n')),
            'conceptual_depth': 'shallow' if len(paragraphs) < 5 else 'moderate' if len(paragraphs) < 15 else 'deep'
        }
    
    async def _analyze_cross_document_relationships(
        self, 
        analysis_results: Dict[str, Dict[str, Any]], 
        session_id: str
    ) -> Dict[str, Any]:
        """Analyze relationships and similarities between documents"""
        documents = list(analysis_results.keys())
        
        if len(documents) < 2:
            return {
                'similarity_matrix': {},
                'clusters': [],
                'common_themes': [],
                'document_relationships': [],
                'relationship_summary': 'Insufficient documents for relationship analysis'
            }
        
        # Calculate pairwise similarities
        similarities = await self._calculate_document_similarities(analysis_results)
        
        # Find common themes
        common_themes = await self._find_common_themes(analysis_results)
        
        # Detect document clusters
        clusters = await self._cluster_documents(analysis_results, similarities)
        
        # Generate relationship insights
        relationships = await self._generate_relationship_insights(
            analysis_results, similarities, common_themes
        )
        
        return {
            'similarity_matrix': similarities,
            'clusters': clusters,
            'common_themes': common_themes,
            'document_relationships': relationships,
            'relationship_summary': f"Analyzed {len(documents)} documents with {len(common_themes)} common themes"
        }
    
    async def _calculate_document_similarities(
        self, 
        analysis_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate similarity scores between all document pairs"""
        similarities = {}
        documents = list(analysis_results.keys())
        
        for i, doc1 in enumerate(documents):
            similarities[doc1] = {}
            for j, doc2 in enumerate(documents):
                if i == j:
                    similarities[doc1][doc2] = 1.0
                elif doc2 in similarities and doc1 in similarities[doc2]:
                    similarities[doc1][doc2] = similarities[doc2][doc1]
                else:
                    # Calculate similarity based on keywords, topics, entities
                    sim_score = await self._compute_similarity_score(
                        analysis_results[doc1], analysis_results[doc2]
                    )
                    similarities[doc1][doc2] = sim_score
        
        return similarities
    
    async def _compute_similarity_score(self, doc1_analysis: Dict, doc2_analysis: Dict) -> float:
        """Compute similarity score between two documents"""
        # Keyword similarity
        keywords1 = set(kw[0] for kw in doc1_analysis.get('keywords', []))
        keywords2 = set(kw[0] for kw in doc2_analysis.get('keywords', []))
        keyword_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0
        
        # Topic similarity
        topics1 = set(doc1_analysis.get('topics', {}).get('main_topics', []))
        topics2 = set(doc2_analysis.get('topics', {}).get('main_topics', []))
        topic_sim = len(topics1 & topics2) / len(topics1 | topics2) if topics1 | topics2 else 0
        
        # Entity similarity (focus on organizations and concepts)
        entities1 = set()
        entities2 = set()
        
        for category in ['ORGANIZATION', 'CONCEPT']:
            doc1_entities = doc1_analysis.get('entities', {}).get('document_entities', {}).get(category, [])
            doc2_entities = doc2_analysis.get('entities', {}).get('document_entities', {}).get(category, [])
            
            entities1.update(e['text'] for e in doc1_entities)
            entities2.update(e['text'] for e in doc2_entities)
        
        entity_sim = len(entities1 & entities2) / len(entities1 | entities2) if entities1 | entities2 else 0
        
        # Weighted average
        return 0.4 * keyword_sim + 0.3 * topic_sim + 0.3 * entity_sim
    
    async def _find_common_themes(self, analysis_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find themes that appear across multiple documents"""
        # Collect all themes/topics from all documents
        all_themes = {}
        
        for doc_path, analysis in analysis_results.items():
            topics = analysis.get('topics', {}).get('main_topics', [])
            for topic in topics:
                if topic not in all_themes:
                    all_themes[topic] = {'documents': [], 'count': 0}
                all_themes[topic]['documents'].append(doc_path)
                all_themes[topic]['count'] += 1
        
        # Filter themes that appear in multiple documents
        common_themes = [
            {
                'theme': theme,
                'document_count': data['count'],
                'documents': data['documents'],
                'prevalence': data['count'] / len(analysis_results)
            }
            for theme, data in all_themes.items()
            if data['count'] > 1
        ]
        
        return sorted(common_themes, key=lambda x: x['prevalence'], reverse=True)
    
    async def _cluster_documents(
        self, 
        analysis_results: Dict[str, Dict[str, Any]], 
        similarities: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Group similar documents into clusters"""
        # Simple clustering based on similarity threshold
        threshold = 0.3
        documents = list(analysis_results.keys())
        clusters = []
        clustered = set()
        
        for doc in documents:
            if doc in clustered:
                continue
                
            cluster = {'documents': [doc], 'similarity_scores': {}}
            clustered.add(doc)
            
            # Find similar documents
            for other_doc in documents:
                if other_doc != doc and other_doc not in clustered:
                    sim_score = similarities[doc][other_doc]
                    if sim_score >= threshold:
                        cluster['documents'].append(other_doc)
                        cluster['similarity_scores'][other_doc] = sim_score
                        clustered.add(other_doc)
            
            # Only add clusters with multiple documents
            if len(cluster['documents']) > 1:
                cluster['avg_similarity'] = sum(cluster['similarity_scores'].values()) / len(cluster['similarity_scores'])
                clusters.append(cluster)
        
        return clusters
    
    async def _generate_relationship_insights(
        self, 
        analysis_results: Dict[str, Dict[str, Any]],
        similarities: Dict[str, Dict[str, float]],
        common_themes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate insights about document relationships"""
        relationships = []
        
        # Find highest similarity pairs
        doc_pairs = []
        for doc1, sim_dict in similarities.items():
            for doc2, sim_score in sim_dict.items():
                if doc1 != doc2 and sim_score > 0.3:  # Only meaningful similarities
                    doc_pairs.append((doc1, doc2, sim_score))
        
        # Sort by similarity
        doc_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Create relationship descriptions
        for doc1, doc2, sim_score in doc_pairs[:5]:  # Top 5 relationships
            relationship = {
                'document1': Path(doc1).name,
                'document2': Path(doc2).name,
                'similarity_score': sim_score,
                'relationship_type': self._classify_relationship_type(sim_score),
                'common_elements': await self._find_common_elements(
                    analysis_results[doc1], analysis_results[doc2]
                )
            }
            relationships.append(relationship)
        
        return relationships
    
    def _classify_relationship_type(self, similarity_score: float) -> str:
        """Classify the type of relationship based on similarity score"""
        if similarity_score > 0.7:
            return "highly_related"
        elif similarity_score > 0.5:
            return "moderately_related"
        elif similarity_score > 0.3:
            return "somewhat_related"
        else:
            return "weakly_related"
    
    async def _find_common_elements(self, doc1_analysis: Dict, doc2_analysis: Dict) -> Dict[str, List[str]]:
        """Find specific common elements between two documents"""
        common_elements = {
            'keywords': [],
            'topics': [],
            'entities': []
        }
        
        # Common keywords
        keywords1 = set(kw[0] for kw in doc1_analysis.get('keywords', []))
        keywords2 = set(kw[0] for kw in doc2_analysis.get('keywords', []))
        common_elements['keywords'] = list(keywords1 & keywords2)
        
        # Common topics
        topics1 = set(doc1_analysis.get('topics', {}).get('main_topics', []))
        topics2 = set(doc2_analysis.get('topics', {}).get('main_topics', []))
        common_elements['topics'] = list(topics1 & topics2)
        
        # Common entities
        entities1 = set()
        entities2 = set()
        
        for category in ['ORGANIZATION', 'PERSON', 'CONCEPT']:
            doc1_entities = doc1_analysis.get('entities', {}).get('document_entities', {}).get(category, [])
            doc2_entities = doc2_analysis.get('entities', {}).get('document_entities', {}).get(category, [])
            
            entities1.update(e['text'] for e in doc1_entities)
            entities2.update(e['text'] for e in doc2_entities)
        
        common_elements['entities'] = list(entities1 & entities2)
        
        return common_elements
    
    async def _generate_semantic_embeddings(
        self, 
        documents_data: Dict[str, Tuple[DocumentMetadata, List[DocumentChunk]]], 
        session_id: str
    ) -> Dict[str, Any]:
        """Generate semantic embeddings for documents and chunks"""
        # This would integrate with ChromaDB or similar vector database
        embeddings_data = {
            'session_id': session_id,
            'embedding_model': self.embedding_model,
            'documents': {},
            'chunks': {},
            'total_embeddings': 0
        }
        
        # For now, create placeholder embeddings structure
        for file_path, (metadata, chunks) in documents_data.items():
            doc_id = metadata.file_hash
            
            embeddings_data['documents'][doc_id] = {
                'filename': metadata.filename,
                'file_path': file_path,
                'embedding_id': f"doc_embed_{doc_id}",
                'chunk_count': len(chunks)
            }
            
            # Chunk embeddings
            for chunk in chunks:
                embeddings_data['chunks'][chunk.chunk_id] = {
                    'document_id': doc_id,
                    'chunk_index': chunk.chunk_index,
                    'embedding_id': f"chunk_embed_{chunk.chunk_id}",
                    'content_length': len(chunk.content)
                }
                embeddings_data['total_embeddings'] += 1
        
        self.logger.info(f"Generated {embeddings_data['total_embeddings']} embedding references")
        
        return embeddings_data
    
    async def _generate_ai_analysis_insights(
        self, 
        analysis_results: Dict[str, Dict[str, Any]],
        cross_document_analysis: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Generate high-level AI insights about the analysis"""
        try:
            # Prepare summary for AI analysis
            summary_data = {
                'total_documents': len(analysis_results),
                'common_themes': cross_document_analysis.get('common_themes', []),
                'relationship_count': len(cross_document_analysis.get('document_relationships', [])),
                'sample_classifications': [
                    result.get('classification', {}).get('document_type', 'unknown')
                    for result in list(analysis_results.values())[:3]
                ]
            }
            
            prompt = f"""
            Analyze this document analysis summary and provide strategic insights:
            
            Analysis Summary:
            {json.dumps(summary_data, indent=2)}
            
            Provide insights in JSON format:
            {{
                "key_findings": ["insight1", "insight2", "insight3"],
                "document_collection_assessment": "assessment of the overall collection",
                "recommended_next_steps": ["step1", "step2", "step3"],
                "complexity_assessment": "simple/moderate/complex",
                "strategic_value": "high/medium/low",
                "potential_use_cases": ["use case 1", "use case 2"]
            }}
            
            Return only valid JSON.
            """
            
            response = await self.generate_gemini_response(prompt, temperature=0.3)
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._fallback_ai_insights(summary_data)
                
        except Exception as e:
            self.logger.warning(f"AI insights generation failed: {str(e)}")
            return self._fallback_ai_insights(summary_data)
    
    def _fallback_ai_insights(self, summary_data: Dict) -> Dict[str, Any]:
        """Fallback AI insights"""
        return {
            'key_findings': [
                f"Analyzed {summary_data['total_documents']} documents",
                f"Found {len(summary_data['common_themes'])} common themes",
                "Analysis completed successfully"
            ],
            'document_collection_assessment': 'Standard business document collection',
            'recommended_next_steps': [
                'Proceed with reasoning analysis',
                'Prepare for query answering',
                'Consider additional context'
            ],
            'complexity_assessment': 'moderate',
            'strategic_value': 'medium',
            'potential_use_cases': ['Information retrieval', 'Content summarization'],
            'fallback_mode': True
        }
    
    async def _save_analysis_results(
        self, 
        analysis_results: Dict[str, Dict[str, Any]],
        cross_document_analysis: Dict[str, Any],
        embeddings_data: Dict[str, Any],
        session_id: str
    ):
        """Save comprehensive analysis results"""
        try:
            output_path = Path(self.output_dir) / session_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save document analysis results
            analysis_file = output_path / "document_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Save cross-document analysis
            cross_analysis_file = output_path / "cross_document_analysis.json"
            with open(cross_analysis_file, 'w') as f:
                json.dump(cross_document_analysis, f, indent=2)
            
            # Save embeddings metadata
            embeddings_file = output_path / "semantic_embeddings.json"
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            # Generate analysis summary report
            summary_report = await self._generate_analysis_summary_report(
                analysis_results, cross_document_analysis, embeddings_data
            )
            
            summary_file = output_path / "analysis_summary_report.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2)
            
            self.logger.info(f"Saved analysis results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {str(e)}")
    
    async def _generate_analysis_summary_report(
        self,
        analysis_results: Dict[str, Dict[str, Any]],
        cross_document_analysis: Dict[str, Any],
        embeddings_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary report of analysis"""
        total_entities = sum(
            sum(analysis.get('entities', {}).get('entity_counts', {}).values())
            for analysis in analysis_results.values()
        )
        
        return {
            'executive_summary': {
                'documents_analyzed': len(analysis_results),
                'total_entities_extracted': total_entities,
                'common_themes_found': len(cross_document_analysis.get('common_themes', [])),
                'document_relationships': len(cross_document_analysis.get('document_relationships', [])),
                'embeddings_generated': embeddings_data.get('total_embeddings', 0)
            },
            'key_insights': cross_document_analysis.get('ai_insights', {}),
            'processing_metadata': {
                'analysis_timestamp': time.time(),
                'agent_name': self.name,
                'analysis_depth': self.analysis_depth
            }
        }


async def test_analysis_agent():
    """Test function for Document Analysis Agent"""
    # This would be called with output from ingestion agent
    print("Analysis Agent test would run here with ingestion output")


if __name__ == "__main__":
    asyncio.run(test_analysis_agent())