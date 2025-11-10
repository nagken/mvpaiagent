"""
orchestrator.py
---------------
Simple workflow orchestrator for the AI Product Classification MVP.
It executes each agent node in sequence and prints a DAG trace.

Later you can swap this for LangGraph or CrewAI orchestration.
"""

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.classifier_agent import ClassifierAgent
from agents.validator_agent import ValidatorAgent
from agents.feedback_agent import FeedbackAgent
from utils.vector_store import VectorStore
import yaml, json


class Orchestrator:
    """Minimal directed-graph orchestrator."""

    def __init__(self, cfg_path="config.yaml"):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.nodes = []
        self.edges = []

    def add_node(self, name, func):
        self.nodes.append({"name": name, "func": func})

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def run(self, input_payload):
        print("\nDAG Execution Trace")
        print("===================")
        print(f"Input: {input_payload['description'][:50]}...")
        print("-" * 50)
        results = { "input": input_payload }

        for node in self.nodes:
            name = node["name"]
            func = node["func"]
            print(f"Executing: {name}")
            try:
                results[name] = func(results)
                print(f"    {name} completed successfully")
            except Exception as e:
                print(f"    {name} failed: {e}")
                results[name] = {"error": str(e)}
            print()
        
        print("Workflow complete!")
        print("=" * 50)
        return results

    def print_dag_topology(self):
        """Print visual representation of the DAG."""
        print("\nAgent DAG Topology:")
        print("===================")
        for i, (src, dst) in enumerate(self.edges):
            connector = "└──" if i == len(self.edges) - 1 else "├──"
            print(f"  {src}")
            print(f"  {connector}→ {dst}")
        print()


# --- Agent wrapper functions -----------------------------------------------------------

def ingestion_step(context):
    """Load catalog data."""
    csv_path = "data/catalog_sample.csv"
    agent = IngestionAgent(csv_path)
    df = agent.run()
    return {"catalog": df, "query": context["input"]["description"]}


def retrieval_step(context):
    """Find similar products using vector search."""
    cfg = yaml.safe_load(open("config.yaml"))
    store = VectorStore("data/catalog_sample.csv", cfg["vector_db"]["path"])
    retriever = RetrievalAgent(store, cfg["retrieval"]["top_k"])
    records = retriever.run(context["query"])
    context_summary = retriever.get_context_summary(records)
    return {
        "retrievals": records, 
        "query": context["query"],
        "context_summary": context_summary
    }


def classification_step(context):
    """Classify product using LLM."""
    cfg = yaml.safe_load(open("config.yaml"))
    classifier = ClassifierAgent(cfg["llm_model"])
    prediction = classifier.run(context["query"], context["context_summary"])
    return {"prediction": prediction}


def validation_step(context):
    """Validate classification against catalog constraints."""
    validator = ValidatorAgent("data/catalog_sample.csv")
    result = validator.run(context["prediction"])
    return {"validated": result}


def feedback_step(context):
    """Log results for continuous learning."""
    feedback = FeedbackAgent("logs/agent_logs.jsonl")
    fb = feedback.run(context["validated"])
    return {"feedback": fb}


# --- Main orchestration function -------------------------------------------------------

def run_demo(description: str):
    """Run a complete classification demo with visual trace."""
    print("AI Product Classification Mini-Agent MVP Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()

    # Add agent nodes in execution order
    orchestrator.add_node("IngestionAgent", ingestion_step)
    orchestrator.add_node("RetrievalAgent", retrieval_step)
    orchestrator.add_node("ClassifierAgent", classification_step)
    orchestrator.add_node("ValidatorAgent", validation_step)
    orchestrator.add_node("FeedbackAgent", feedback_step)

    # Define edges for visualization (linear pipeline)
    orchestrator.add_edge("IngestionAgent", "RetrievalAgent")
    orchestrator.add_edge("RetrievalAgent", "ClassifierAgent")
    orchestrator.add_edge("ClassifierAgent", "ValidatorAgent")
    orchestrator.add_edge("ValidatorAgent", "FeedbackAgent")

    # Print DAG topology
    orchestrator.print_dag_topology()

    # Execute the pipeline
    payload = {"description": description}
    outputs = orchestrator.run(payload)

    # Display final results
    print_final_results(outputs)
    
    return outputs


def print_final_results(outputs):
    """Print formatted final results."""
    print("\nFinal Classification Results:")
    print("=" * 50)
    
    try:
        # Extract key results
        validation = outputs.get("ValidatorAgent", {}).get("validated", {})
        feedback = outputs.get("FeedbackAgent", {}).get("feedback", {})
        
        if "validated_prediction" in validation:
            pred = validation["validated_prediction"]
            print(f"Product Category: {pred.get('code_level_1', 'UNKNOWN')}")
            print(f"Subcategory: {pred.get('code_level_2', 'UNKNOWN')}")
            print(f"Vendor: {pred.get('vendor', 'UNKNOWN')}")
            print(f"Price Range: {pred.get('price_range', 'UNKNOWN')}")
            print(f"Confidence: {pred.get('confidence', 0):.2f}")
            print(f"Valid: {validation.get('is_valid', False)}")
            
            if validation.get("validation_errors"):
                print(f"Validation Issues:")
                for error in validation["validation_errors"]:
                    print(f"   - {error}")
        
        print(f"\nSession ID: {feedback.get('session_id', 'unknown')}")
        print(f"Logged to: {feedback.get('log_file', 'unknown')}")
        
    except Exception as e:
        print(f"Error displaying results: {e}")
        print("\nRaw outputs:")
        print(json.dumps(outputs, indent=2, default=str))


def interactive_demo():
    """Interactive demo mode."""
    print("AI Interactive Demo")
    print("=" * 40)
    print("Enter product descriptions to classify (type 'quit' to exit)")
    print()
    
    while True:
        try:
            description = input("Product description: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if len(description) < 10:
                print("Please enter a more detailed description (at least 10 characters)")
                continue
            
            print()
            run_demo(description)
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Demo error: {e}")


# --- Entry point ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line argument provided
        description = " ".join(sys.argv[1:])
        run_demo(description)
    else:
        # Example demo
        example_products = [
            "Dell Ultrasharp 27-inch 4K Monitor",
            "Microsoft Surface Pro 9 with Keyboard",
            "Cisco Catalyst 9300 Switch 48-Port",
            "Canon EOS R5 Professional Camera"
        ]
        
        print("Running example classifications:")
        print("=" * 50)
        
        for product in example_products:
            print(f"\n{'='*20} EXAMPLE {'='*20}")
            run_demo(product)
            print()
        
        print("\nStarting interactive mode...")
        print("(You can also run: python orchestrator.py 'Your product description')")
        print()
        interactive_demo()