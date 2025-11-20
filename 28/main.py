#!/usr/bin/env python3
"""
File Exploration Agent - Main CLI Entry Point

Execute file exploration tasks using a DSPy ReAct-based agent.

Usage:
    uv run main.py --task "Analyze directory structure" --directory ../27
    uv run main.py --task "List all Python files" --directory . --max-iters 5
    uv run main.py --task "Find config files" --model artifact/agent_gepa_optimized_latest.json
"""

import argparse
import sys
from pathlib import Path

import dspy

from config import configure_lm, FAST_MODEL
from agent_module import FileExplorationAgent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="File Exploration Agent - Analyze directories using DSPy ReAct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze directory structure
  uv run main.py --task "Analyze the directory structure and create a report" --directory ../27

  # List specific file types
  uv run main.py --task "Find all Python files and count lines of code" --directory .

  # Use optimized model
  uv run main.py --task "Analyze project" --model artifact/agent_gepa_optimized_latest.json

  # Limit iterations
  uv run main.py --task "List main files" --directory . --max-iters 5
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description for the agent to execute"
    )

    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Working directory to explore (default: current directory)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to optimized model JSON file (optional, uses baseline if not specified)"
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum number of ReAct iterations (default: 10)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed execution trace (only show final report)"
    )

    return parser.parse_args()


def load_model(model_path: str) -> FileExplorationAgent:
    """
    Load an optimized agent model from JSON file.

    Args:
        model_path: Path to the model JSON file

    Returns:
        FileExplorationAgent instance with loaded parameters
    """
    path = Path(model_path)

    if not path.exists():
        print(f"❌ Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📦 Loading optimized model from: {model_path}")

    try:
        agent = FileExplorationAgent()
        agent.load(str(path))
        print("✅ Model loaded successfully")
        return agent
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main execution function."""
    args = parse_arguments()

    # Configure DSPy language model
    print("🔧 Configuring DSPy...")
    lm = configure_lm(FAST_MODEL, temperature=0.0)
    dspy.configure(lm=lm)
    print(f"✅ Using model: {FAST_MODEL}")

    # Load or create agent
    if args.model:
        agent = load_model(args.model)
        agent.max_iters = args.max_iters
        agent.verbose = not args.quiet
    else:
        print("🤖 Using baseline agent (no optimization)")
        agent = FileExplorationAgent(
            max_iters=args.max_iters,
            verbose=not args.quiet
        )

    # Resolve working directory
    working_dir = Path(args.directory).resolve()

    if not working_dir.exists():
        print(f"❌ Error: Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    if not working_dir.is_dir():
        print(f"❌ Error: Path is not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Execute agent
    try:
        result = agent(
            task=args.task,
            working_directory=str(working_dir)
        )

        # If quiet mode, print the final report
        if args.quiet:
            print("\n" + "=" * 80)
            print("FINAL REPORT")
            print("=" * 80 + "\n")
            print(result.report)
            print("\n" + "=" * 80 + "\n")

        # Exit successfully
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ Error during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
