#!/usr/bin/env python3
"""
Quick Demo Script for AI Product Classification MVP
===================================================

This script demonstrates the core functionality without requiring
a full FastAPI server setup. Perfect for quick testing and demos.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_demo():
    """Run a quick classification demo."""
    
    print("AI Product Classification MVP - Quick Demo")
    print("=" * 50)
    
    # Test products to classify
    test_products = [
        "Dell XPS 13 Plus Developer Edition Laptop with Ubuntu",
        "Apple iPad Pro 12.9-inch with M2 chip and Apple Pencil", 
        "NVIDIA GeForce RTX 4090 Ti Graphics Card for Gaming",
        "Cisco Catalyst 9300 48-port Gigabit Switch",
        "Sony WH-1000XM5 Wireless Noise Canceling Headphones"
    ]
    
    print("Testing with sample products:")
    for i, product in enumerate(test_products, 1):
        print(f"  {i}. {product}")
    print()
    
    # Check if we have the required setup
    try:
        # Check for config
        if not os.path.exists("config.yaml"):
            print("ERROR: config.yaml not found. Please run from project root directory.")
            return False
            
        # Check for data
        if not os.path.exists("data/catalog_sample.csv"):
            print("ERROR: Sample catalog not found. Please ensure data/catalog_sample.csv exists.")
            return False
            
        # Check for OpenAI key
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("   Please set it with: export OPENAI_API_KEY='your_key_here'")
            return False
            
        print("SUCCESS: Prerequisites check passed!")
        print()
        
        # Import and run orchestrator
        from orchestrator import run_demo
        
        # Run demo for first product
        print(f"Running classification for: {test_products[0]}")
        print()
        
        result = run_demo(test_products[0])
        
        print("\nDemo completed successfully!")
        print(f"To try other products, run: python orchestrator.py 'Product description'")
        print(f"To start API server, run: python main.py")
        
        return True
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("   Please install dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt") 
        print("3. Set OPENAI_API_KEY environment variable")
        print("4. Check that data/catalog_sample.csv exists")
        return False


def interactive_mode():
    """Run interactive classification mode."""
    print("\nInteractive Classification Mode")
    print("=" * 40)
    print("Enter product descriptions to classify (type 'quit' to exit)")
    
    while True:
        try:
            description = input("\nProduct description: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if len(description) < 10:
                print("ERROR: Please enter a more detailed description (at least 10 characters)")
                continue
                
            # Import and run 
            from orchestrator import run_demo
            run_demo(description)
            
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        description = " ".join(sys.argv[1:])
        print(f"Classifying: {description}")
        from orchestrator import run_demo
        run_demo(description)
    else:
        # Quick demo mode
        success = quick_demo()
        
        if success:
            # Offer interactive mode
            while True:
                choice = input("\nTry interactive mode? (y/n): ").lower().strip()
                if choice in ['y', 'yes']:
                    interactive_mode()
                    break
                elif choice in ['n', 'no']:
                    print("Thanks for trying the AI Classification MVP!")
                    break
                else:
                    print("Please enter 'y' or 'n'")