"""
Enhanced main entry point for PyAgenticRAG
Supports both CLI and programmatic usage
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import get_config, set_config, AppConfig
from utils.logging_setup import setup_logging, get_logger
from src.agentic_rag_system import AgenticRAGSystem, create_rag_system
from src.enhanced_agent_state import create_initial_state

def setup_directories():
    """Create necessary directories"""
    dirs = ["./data", "./logs", "./vector_db"]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)

def run_cli_mode():
    """Run in interactive CLI mode"""
    print("🚀 PyAgenticRAG - Advanced Agentic RAG System")
    print("=" * 50)
    
    try:
        # Initialize system
        print("Initializing system...")
        system = create_rag_system()
        print("✅ System initialized successfully!\n")
        
        # Interactive loop
        while True:
            print("\nOptions:")
            print("1. Ask a question")
            print("2. View system status")
            print("3. Refresh data sources")
            print("4. Exit")
            
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                query = input("\nEnter your question: ").strip()
                if query:
                    print(f"\n🔍 Processing query: {query}")
                    print("-" * 50)
                    
                    # Create state and process
                    initial_state = create_initial_state(query)
                    final_state = system.process_query(initial_state)
                    
                    # Display results
                    print("\n📝 Response:")
                    print(final_state["final_response"])
                    
                    # Display metadata
                    confidence = final_state.get("confidence_score", 0.0)
                    processing_time = final_state.get("processing_time", 0.0)
                    sources_count = len(final_state.get("retrieved_documents", []))
                    
                    print(f"\n📊 Metadata:")
                    print(f"   Confidence: {confidence:.2f}")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Sources used: {sources_count}")
                    
            elif choice == "2":
                info = system.get_system_info()
                print("\n📈 System Status:")
                print(f"   Status: {info['status']}")
                print(f"   Uptime: {info['uptime_seconds']:.1f} seconds")
                print(f"   Total queries: {info['total_queries']}")
                print(f"   Active sessions: {info['active_sessions']}")
                print(f"   LLM: {info['configuration']['llm_provider']}/{info['configuration']['llm_model']}")
                
            elif choice == "3":
                print("\n🔄 Refreshing data sources...")
                try:
                    system.refresh_data_sources()
                    print("✅ Data sources refreshed successfully!")
                except Exception as e:
                    print(f"❌ Failed to refresh data sources: {e}")
                    
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-4.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        return 1
    
    return 0

def run_single_query(query: str):
    """Run a single query and exit"""
    try:
        print(f"🔍 Processing query: {query}")
        
        # Initialize system
        system = create_rag_system()
        
        # Process query
        initial_state = create_initial_state(query)
        final_state = system.process_query(initial_state)
        
        # Output result
        print("\n" + "=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(final_state["final_response"])
        
        # Output metadata if verbose
        confidence = final_state.get("confidence_score", 0.0)
        processing_time = final_state.get("processing_time", 0.0)
        sources_count = len(final_state.get("retrieved_documents", []))
        
        print("\n" + "-" * 80)
        print(f"Confidence: {confidence:.2f} | Processing time: {processing_time:.2f}s | Sources: {sources_count}")
        print("-" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return 1

def run_web_server():
    """Run the web server"""
    try:
        from api.web_interface import main as run_api_server
        print("🌐 Starting web server...")
        run_api_server()
        return 0
    except ImportError:
        print("❌ Web server dependencies not installed. Install with: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"❌ Web server error: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PyAgenticRAG - Advanced Agentic RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["cli", "web", "query"], 
        default="cli",
        help="Run mode: cli (interactive), web (server), or query (single query)"
    )
    
    parser.add_argument(
        "--query", 
        type=str,
        help="Query to process (required for query mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup directories
        setup_directories()
        
        # Load configuration
        if args.config and Path(args.config).exists():
            config = AppConfig.from_yaml(args.config)
            set_config(config)
        else:
            config = get_config()
        
        # Override config with CLI args
        if args.data_path:
            config.data_sources.local_data_path = args.data_path
        
        # Setup logging
        setup_logging(
            level=args.log_level,
            file_path=config.logging.file_path
        )
        
        logger = get_logger(__name__)
        logger.info(f"Starting PyAgenticRAG in {args.mode} mode")
        
        # Run based on mode
        if args.mode == "cli":
            return run_cli_mode()
            
        elif args.mode == "web":
            return run_web_server()
            
        elif args.mode == "query":
            if not args.query:
                print("❌ --query is required for query mode")
                return 1
            return run_single_query(args.query)
        
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())