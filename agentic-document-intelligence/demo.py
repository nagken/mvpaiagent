"""
Demo Script for Agentic Document Intelligence System
Demonstrates the capabilities of the ADK multi-agent system
"""

import asyncio
import os
import json
from pathlib import Path
from main import AgenticDocumentIntelligence


class ADKDemo:
    """Demonstration class for the Agentic Document Intelligence System"""
    
    def __init__(self):
        self.system = AgenticDocumentIntelligence()
        self.demo_docs_dir = Path("data/demo_documents")
        self.setup_demo_environment()
    
    def setup_demo_environment(self):
        """Setup demonstration environment with sample documents"""
        self.demo_docs_dir.mkdir(parents=True, exist_ok=True)
        self.create_sample_documents()
    
    def create_sample_documents(self):
        """Create sample documents for demonstration"""
        
        # Sample business report
        business_report = """
# Quarterly Business Report - Q3 2024

## Executive Summary
Our company achieved significant milestones in Q3 2024, with revenue growth of 15% year-over-year. 
Key performance indicators show strong market position and customer satisfaction improvements.

## Financial Performance
- Total Revenue: $2.4M (up 15% from Q3 2023)
- Operating Expenses: $1.8M 
- Net Profit: $600K
- Cash Flow: Positive $450K

## Market Analysis
The technology sector continues to show robust growth, particularly in artificial intelligence and cloud computing services. 
Our main competitors include TechCorp Inc., Innovation Labs, and Digital Solutions Ltd.

## Key Achievements
1. Launched new AI-powered product suite
2. Expanded to 3 new geographic markets
3. Increased customer base by 25%
4. Improved customer satisfaction scores to 4.2/5.0

## Strategic Initiatives
- Investment in machine learning capabilities
- Partnership with Global Tech Partners
- Enhanced cybersecurity infrastructure
- Sustainability program implementation

## Risks and Challenges
- Increased competition in AI market
- Supply chain disruptions
- Regulatory changes in data privacy
- Talent acquisition in specialized roles

## Outlook for Q4 2024
We anticipate continued growth with projected revenue of $2.7M and expansion into European markets.
Focus areas include product innovation, customer experience, and operational efficiency.
"""
        
        # Sample technical document
        technical_doc = """
# System Architecture Documentation

## Overview
This document describes the microservices architecture for our cloud-based AI platform.
The system supports real-time data processing, machine learning inference, and API management.

## Architecture Components

### Frontend Services
- React-based web application
- Mobile application (iOS/Android)
- Administrator dashboard
- API gateway (Kong)

### Backend Services
- User Authentication Service (Node.js)
- Data Processing Service (Python/FastAPI)
- Machine Learning Service (TensorFlow Serving)
- Notification Service (Go)
- Database Management (PostgreSQL, Redis)

### Infrastructure
- Kubernetes orchestration on AWS
- Docker containerization
- Prometheus monitoring
- ElasticSearch logging
- SSL/TLS encryption

## Security Measures
- OAuth 2.0 authentication
- Role-based access control (RBAC)
- API rate limiting
- Data encryption at rest and in transit
- Regular security audits by CyberSec Corp

## Performance Metrics
- 99.9% uptime SLA
- < 200ms API response time
- Support for 10,000 concurrent users
- Auto-scaling based on demand

## Data Flow
1. Client requests via API gateway
2. Authentication and authorization
3. Request routing to appropriate microservice
4. Data processing and ML inference
5. Response delivery and logging

## Monitoring and Alerting
- Real-time metrics dashboard
- Automated alerting for system issues
- Performance analytics
- Cost optimization tracking
"""
        
        # Sample research document
        research_doc = """
# Market Research Report: AI Technology Adoption

## Research Methodology
This study analyzes artificial intelligence adoption trends across 500 enterprises 
conducted between January and March 2024. Data collected through surveys, interviews, 
and market analysis by Research Analytics Inc.

## Key Findings

### Adoption Rates
- 78% of enterprises have adopted AI in some capacity
- 45% report significant ROI from AI investments
- 23% plan major AI initiatives in next 12 months

### Industry Breakdown
- Financial Services: 85% adoption rate
- Healthcare: 72% adoption rate  
- Manufacturing: 68% adoption rate
- Retail: 65% adoption rate
- Education: 45% adoption rate

### Technology Preferences
1. Machine Learning Platforms (67%)
2. Natural Language Processing (54%)
3. Computer Vision (43%)
4. Robotic Process Automation (38%)
5. Predictive Analytics (71%)

### Implementation Challenges
- Data quality issues (62%)
- Lack of skilled personnel (58%)
- Integration complexity (51%)
- Cost considerations (47%)
- Regulatory compliance (39%)

### Success Factors
- Executive sponsorship and support
- Clear business objectives alignment
- Phased implementation approach
- Investment in training and change management
- Partnership with experienced vendors like AI Solutions Group

## Market Projections
The global AI market is projected to reach $1.8 trillion by 2030, with compound 
annual growth rate of 37.5%. Key growth drivers include cloud adoption, 
digital transformation initiatives, and increasing data availability.

## Recommendations
1. Develop comprehensive AI strategy aligned with business goals
2. Invest in data infrastructure and governance
3. Build internal AI capabilities through training
4. Start with pilot projects to demonstrate value
5. Establish partnerships with technology vendors
6. Address ethical AI and bias concerns proactively

## Conclusion
AI adoption continues to accelerate across industries, with early adopters 
gaining competitive advantages. Organizations should act quickly to develop 
AI capabilities while carefully managing implementation challenges.
"""
        
        # Write sample documents
        sample_docs = {
            "business_report_q3_2024.txt": business_report,
            "technical_architecture.txt": technical_doc,
            "ai_market_research.txt": research_doc
        }
        
        for filename, content in sample_docs.items():
            doc_path = self.demo_docs_dir / filename
            with open(doc_path, 'w') as f:
                f.write(content)
        
        print(f"Created {len(sample_docs)} sample documents in {self.demo_docs_dir}")
    
    async def run_demo(self):
        """Run the complete demonstration"""
        print("\nAgentic Document Intelligence System Demo")
        print("=" * 60)
        
        # Demo 1: Process sample documents
        print("\nDemo 1: Processing Sample Documents")
        print("-" * 40)
        
        demo_result = await self.system.process_documents(str(self.demo_docs_dir))
        
        if demo_result['status'] == 'success':
            session_id = demo_result['session_id']
            
            print(f"Processing completed successfully!")
            print(f"Session ID: {session_id}")
            print(f"Processing time: {demo_result['execution_time']:.2f} seconds")
            
            # Display key results
            summary = demo_result['session_summary']
            print(f"\nResults Summary:")
            print(f"   Documents processed: {summary['key_metrics']['documents_processed']}")
            print(f"   Entities identified: {summary['key_metrics']['entities_identified']}")
            print(f"   Insights generated: {summary['key_metrics']['insights_generated']}")
            print(f"   Recommendations: {summary['key_metrics']['recommendations_count']}")
            
            # Demo 2: Query the knowledge base
            print(f"\nDemo 2: Querying Knowledge Base")
            print("-" * 40)
            
            sample_queries = [
                "What companies are mentioned in the documents?",
                "What are the key financial metrics?",
                "Tell me about AI technology trends",
                "What are the main challenges discussed?"
            ]
            
            for query in sample_queries:
                print(f"\nQuery: {query}")
                query_result = await self.system.query_knowledge(session_id, query)
                
                if query_result['status'] == 'success':
                    result = query_result['result']
                    print(f"Answer: {result['answer']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                else:
                    print(f"Query failed: {query_result['error']}")
            
            # Demo 3: Show generated outputs
            print(f"\nDemo 3: Generated Outputs")
            print("-" * 40)
            
            output_dir = Path(f"output/responses/{session_id}")
            if output_dir.exists():
                print(f"Output directory: {output_dir}")
                for file in output_dir.iterdir():
                    if file.is_file():
                        print(f"   {file.name} ({file.stat().st_size} bytes)")
            
            # Demo 4: Show intelligence report sample
            print(f"\nDemo 4: Intelligence Report Sample")
            print("-" * 40)
            
            intelligence_file = output_dir / "intelligence_report.json"
            if intelligence_file.exists():
                try:
                    with open(intelligence_file, 'r') as f:
                        intelligence_data = json.load(f)
                    
                    report_header = intelligence_data.get('report_header', {})
                    print(f"Report Title: {report_header.get('title', 'N/A')}")
                    print(f"Generated: {report_header.get('generated_at', 'N/A')}")
                    
                    collection_overview = intelligence_data.get('collection_overview', {})
                    print(f"Document Types: {collection_overview.get('document_types', {})}")
                    
                    entity_analysis = intelligence_data.get('entity_analysis', {})
                    print(f"Total Entities: {entity_analysis.get('total_entities', 0)}")
                    
                except Exception as e:
                    print(f"Could not read intelligence report: {e}")
            
            return session_id
            
        else:
            print(f"Demo failed: {demo_result['error']}")
            return None
    
    async def show_advanced_features(self, session_id: str):
        """Demonstrate advanced features"""
        print(f"\nAdvanced Features Demo")
        print("=" * 40)
        
        # Load session data to show advanced insights
        try:
            session_file = Path(f'output/sessions/{session_id}/complete_results.json')
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Show reasoning insights
                reasoning_metadata = session_data.get('reasoning_metadata', {})
                print(f"Knowledge Graph: {reasoning_metadata.get('knowledge_graph_nodes', 0)} nodes, {reasoning_metadata.get('knowledge_graph_edges', 0)} edges")
                print(f"Inferences: {reasoning_metadata.get('inferences_generated', 0)} generated")
                print(f"Contradictions: {reasoning_metadata.get('contradictions_found', 0)} found")
                print(f"Hypotheses: {reasoning_metadata.get('hypotheses_generated', 0)} proposed")
                
                # Show actionable insights sample
                actionable_insights = session_data.get('actionable_insights', [])
                if actionable_insights:
                    print(f"\nSample Actionable Insight:")
                    insight = actionable_insights[0]
                    print(f"   Title: {insight.get('title', 'N/A')}")
                    print(f"   Confidence: {insight.get('confidence', 0):.2f}")
                    print(f"   Action: {insight.get('recommended_action', 'N/A')}")
                
                # Show recommendations sample
                recommendations = session_data.get('recommendations', [])
                if recommendations:
                    print(f"\nSample Recommendation:")
                    rec = recommendations[0]
                    print(f"   Category: {rec.get('category', 'N/A')}")
                    print(f"   Priority: {rec.get('priority', 'N/A')}")
                    print(f"   Title: {rec.get('title', 'N/A')}")
                    print(f"   Timeframe: {rec.get('timeframe', 'N/A')}")
                
        except Exception as e:
            print(f"Could not load advanced features: {e}")
    
    def show_architecture_overview(self):
        """Show system architecture overview"""
        print(f"\nSystem Architecture Overview")
        print("=" * 50)
        print("""
┌─────────────────────────────────────────────────────────────┐
│                    ADK Multi-Agent Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents  ->  Orchestrator  ->  Intelligence             │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Ingestion   │ → │ Analysis    │ → │ Reasoning   │ →     │
│  │ Agent       │   │ Agent       │   │ Agent       │       │
│  │             │   │             │   │             │       │
│  │• Validation │   │• NLP        │   │• Logic      │       │
│  │• Chunking   │   │• Entities   │   │• Knowledge  │       │
│  │• Processing │   │• Topics     │   │• Inference  │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│                                                             │
│                    ┌─────────────┐                         │
│                  → │ Response    │ -> Reports & Insights   │
│                    │ Agent       │                         │
│                    │             │                         │
│                    │• Synthesis  │                         │
│                    │• Q&A        │                         │
│                    │• Insights   │                         │
│                    └─────────────┘                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Powered by Google's Agent Development Kit (ADK)           │
│  Gemini 1.5-pro for AI reasoning                           │
│  Knowledge graph construction                               │
│  Multi-format output generation                            │
└─────────────────────────────────────────────────────────────┘
        """)


async def run_complete_demo():
    """Run the complete demonstration"""
    demo = ADKDemo()
    
    # Show architecture
    demo.show_architecture_overview()
    
    # Run main demo
    session_id = await demo.run_demo()
    
    if session_id:
        # Show advanced features
        await demo.show_advanced_features(session_id)
        
        print(f"\nDemo completed successfully!")
        print(f"Session ID: {session_id}")
        print(f"Full results available in: output/responses/{session_id}")
        print(f"\nTry running queries with: python main.py --query 'your question' --session-id {session_id}")
    
    print(f"\nThank you for trying the Agentic Document Intelligence System!")


if __name__ == "__main__":
    asyncio.run(run_complete_demo())