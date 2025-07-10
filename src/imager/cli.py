"""Command-line interface for Imager."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from imager.utils.config import Config, load_config
from imager.utils.logging import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Imager: Open-source text-to-image generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate images")
    generate_parser.add_argument(
        "--prompt", "-p", 
        type=str, 
        required=True,
        help="Text prompt for image generation"
    )
    generate_parser.add_argument(
        "--model", "-m", 
        type=str, 
        default=None,
        help="Path to trained model checkpoint"
    )
    generate_parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="./outputs",
        help="Output directory"
    )
    generate_parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    generate_parser.add_argument(
        "--num-images", "-n", 
        type=int, 
        default=1,
        help="Number of images to generate"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument(
        "--model", "-m", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    eval_parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    # Add global arguments
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config if hasattr(args, 'config') else None)
        if args.seed is not None:
            config.seed = args.seed
        logger.info(f"Loaded configuration from {args.config if hasattr(args, 'config') else 'default'}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Execute command
    try:
        if args.command == "train":
            from imager.training.trainer import train
            if args.resume:
                config.training.resume_from_checkpoint = args.resume
            train(config)
            
        elif args.command == "generate":
            from imager.inference.pipeline import generate_images
            generate_images(
                prompt=args.prompt,
                config=config,
                model_path=args.model,
                output_dir=args.output,
                num_images=args.num_images
            )
            
        elif args.command == "eval":
            from imager.training.eval import evaluate
            evaluate(config, args.model)
            
        else:
            parser.print_help()
            return 1
            
        logger.info("Command completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
