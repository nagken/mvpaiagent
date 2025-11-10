"""
Google Agent Development Kit (ADK) Base Framework
Foundational classes for building agentic AI systems with Gemini
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import google.generativeai as genai
from google.cloud import aiplatform
from google.cloud import logging as cloud_logging


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str
    content: Any
    message_type: str
    timestamp: float
    session_id: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    """Standardized agent execution result"""
    agent_name: str
    status: AgentStatus
    output: Any
    execution_time: float
    session_id: str
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result


class BaseAgent(ABC):
    """Base class for all ADK agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{name}")
        self.session_id = None
        
        # Initialize Gemini
        genai.configure(api_key=config.get('google_api_key'))
        self.model = genai.GenerativeModel(
            config.get('gemini_model', 'gemini-1.5-pro')
        )
    
    @abstractmethod
    async def process(self, input_data: Any, session_id: str) -> AgentResult:
        """Main processing method - must be implemented by each agent"""
        pass
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data format and content"""
        return input_data is not None
    
    async def pre_process(self, input_data: Any, session_id: str) -> Any:
        """Pre-processing hook"""
        self.session_id = session_id
        self.status = AgentStatus.PROCESSING
        self.logger.info(f"Starting processing for session {session_id}")
        return input_data
    
    async def post_process(self, result: Any, session_id: str) -> AgentResult:
        """Post-processing hook"""
        self.status = AgentStatus.COMPLETED
        self.logger.info(f"Completed processing for session {session_id}")
        return result
    
    async def handle_error(self, error: Exception, session_id: str) -> AgentResult:
        """Error handling"""
        self.status = AgentStatus.ERROR
        self.logger.error(f"Error in {self.name}: {str(error)}")
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.ERROR,
            output=None,
            execution_time=0.0,
            session_id=session_id,
            error=str(error)
        )
    
    async def generate_gemini_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini model"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, 
                prompt,
                **kwargs
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            raise


class AgentOrchestrator:
    """Orchestrates multi-agent workflows using Google ADK patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.execution_graph: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("orchestrator")
        
        # Initialize Google Cloud Logging
        if config.get('use_cloud_logging', False):
            client = cloud_logging.Client()
            client.setup_logging()
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def define_workflow(self, workflow: Dict[str, List[str]]):
        """Define agent execution workflow/dependencies"""
        self.execution_graph = workflow
        self.logger.info(f"Defined workflow: {workflow}")
    
    async def execute_workflow(self, input_data: Any, session_id: str) -> Dict[str, AgentResult]:
        """Execute the complete multi-agent workflow"""
        results = {}
        execution_order = self._get_execution_order()
        
        self.logger.info(f"Executing workflow for session {session_id}")
        self.logger.info(f"Execution order: {execution_order}")
        
        for agent_name in execution_order:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                
                # Prepare input based on dependencies
                agent_input = self._prepare_agent_input(
                    agent_name, input_data, results
                )
                
                # Execute agent
                start_time = time.time()
                try:
                    result = await agent.process(agent_input, session_id)
                    result.execution_time = time.time() - start_time
                    results[agent_name] = result
                    
                    self.logger.info(
                        f"Agent {agent_name} completed in {result.execution_time:.2f}s"
                    )
                    
                except Exception as e:
                    error_result = await agent.handle_error(e, session_id)
                    results[agent_name] = error_result
                    
                    # Decide whether to continue or halt workflow
                    if self._should_halt_on_error(agent_name, e):
                        self.logger.error(f"Halting workflow due to error in {agent_name}")
                        break
        
        return results
    
    def _get_execution_order(self) -> List[str]:
        """Determine optimal agent execution order based on dependencies"""
        # Simple topological sort for now
        visited = set()
        order = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            
            # Visit dependencies first
            for dep in self.execution_graph.get(node, []):
                dfs(dep)
            
            order.append(node)
        
        for agent_name in self.agents.keys():
            dfs(agent_name)
        
        return order
    
    def _prepare_agent_input(self, agent_name: str, original_input: Any, 
                           previous_results: Dict[str, AgentResult]) -> Any:
        """Prepare input for an agent based on dependencies and previous results"""
        # Base case: use original input
        if not previous_results:
            return original_input
        
        # Combine original input with relevant previous results
        agent_input = {
            'original_input': original_input,
            'previous_results': previous_results,
            'metadata': {
                'agent_name': agent_name,
                'timestamp': time.time()
            }
        }
        
        return agent_input
    
    def _should_halt_on_error(self, agent_name: str, error: Exception) -> bool:
        """Determine if workflow should halt on agent error"""
        # For now, always halt on error
        # In production, this could be more sophisticated
        return True
    
    async def send_message(self, message: AgentMessage):
        """Send message between agents"""
        self.message_queue.append(message)
        self.logger.debug(f"Message queued: {message.sender} -> {message.receiver}")
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """Get current status of an agent"""
        agent = self.agents.get(agent_name)
        return agent.status if agent else None
    
    def get_workflow_summary(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Generate workflow execution summary"""
        total_time = sum(r.execution_time for r in results.values())
        success_count = sum(1 for r in results.values() if r.status == AgentStatus.COMPLETED)
        
        return {
            'total_agents': len(results),
            'successful_agents': success_count,
            'failed_agents': len(results) - success_count,
            'total_execution_time': total_time,
            'average_execution_time': total_time / len(results) if results else 0,
            'success_rate': success_count / len(results) if results else 0,
            'results': {name: result.to_dict() for name, result in results.items()}
        }


class ADKLogger:
    """Enhanced logging for ADK agents with Google Cloud integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format=self.config.get(
                'log_format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'logs/adk.log')),
                logging.StreamHandler()
            ]
        )
    
    def log_agent_execution(self, agent_name: str, result: AgentResult):
        """Log agent execution details"""
        logger = logging.getLogger(f"execution.{agent_name}")
        logger.info(f"Agent execution: {result.to_dict()}")
    
    def log_workflow_summary(self, summary: Dict[str, Any]):
        """Log workflow execution summary"""
        logger = logging.getLogger("workflow")
        logger.info(f"Workflow summary: {json.dumps(summary, indent=2)}")


# Utility functions for ADK
def create_session_id() -> str:
    """Generate unique session ID"""
    import uuid
    return f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


async def initialize_adk_system(config_path: str) -> AgentOrchestrator:
    """Initialize the complete ADK system"""
    config = load_config(config_path)
    
    # Setup logging
    adk_logger = ADKLogger(config.get('logging', {}))
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(config)
    
    return orchestrator