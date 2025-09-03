#!/usr/bin/env python3
"""
Gemma-270M Master Pipeline
=========================

This script runs the complete pipeline:
1. Clean up old data files
2. Download and process fresh training data
3. Train the model
4. Run tests
5. Build examples

Usage:
    python run_pipeline.py [--config CONFIG] [--steps STEPS] [--clean-all]
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gemma_270m.config import ExperimentConfig, get_default_config
from gemma_270m.data import DataProcessor


def setup_logging():
    """Setup logging for the pipeline"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_banner(title: str, char: str = "="):
    """Print a banner for pipeline steps"""
    print(f"\n{char * 60}")
    print(f"🚀 {title.upper()}")
    print(f"{char * 60}\n")


def clean_data_files(data_dir: str = "data", logger=None):
    """Clean up old training artifacts (NOT the preprocessed dataset files)"""
    print_banner("STEP 1: CLEANING OLD TRAINING ARTIFACTS")
    
    # Clean checkpoints directory (actual training outputs)
    checkpoints_path = Path("checkpoints")
    if checkpoints_path.exists():
        print(f"🗑️  Cleaning training checkpoints: {checkpoints_path.resolve()}")
        shutil.rmtree(checkpoints_path)
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Cleaned training checkpoints directory")
        if logger:
            logger.info(f"Cleaned training checkpoints directory: {checkpoints_path}")
    
    # Clean old log files (but skip current pipeline.log as it's in use)
    for log_file in Path(".").glob("*.log"):
        if log_file.name != "pipeline.log":  # Don't delete current log file
            print(f"🗑️  Removing old log file: {log_file}")
            log_file.unlink()
            if logger:
                logger.info(f"Removed log file: {log_file}")
        else:
            print(f"ℹ️  Keeping current pipeline.log (in use)")
    
    # Keep the preprocessed data files - they are expensive to regenerate!
    data_path = Path(data_dir)
    if data_path.exists():
        train_bin = data_path / "train.bin"
        val_bin = data_path / "validation.bin"
        if train_bin.exists() and val_bin.exists():
            print(f"✅ Keeping existing preprocessed dataset files:")
            print(f"   - {train_bin} ({train_bin.stat().st_size / 1e6:.1f} MB)")
            print(f"   - {val_bin} ({val_bin.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"ℹ️  Preprocessed dataset files not found - will be created during data step")
    else:
        print(f"ℹ️  Data directory does not exist - will be created during data step")
    
    return True


def create_fresh_data(config: ExperimentConfig, logger=None):
    """Download and process fresh training data"""
    print_banner("STEP 2: CREATING FRESH TRAINING DATA")
    
    try:
        # Ensure output directory exists
        output_dir = Path(config.data.data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data processor
        processor = DataProcessor(
            tokenizer_name=config.data.tokenizer_name,
            num_proc=config.data.num_proc
        )
        
        print(f"📥 Processing dataset: {config.data.dataset_name}")
        print(f"🔧 Using tokenizer: {config.data.tokenizer_name}")
        print(f"⚙️  Parallel processes: {config.data.num_proc}")
        
        # Create binary dataset (this will download if needed, but won't reprocess if exists)
        train_path, val_path = processor.create_binary_dataset(
            dataset_name=config.data.dataset_name,
            output_dir=config.data.data_dir,
            force_reprocess=False  # Don't reprocess expensive data unless needed
        )
        
        # Update config paths
        config.data.train_path = train_path
        config.data.val_path = val_path
        
        print(f"✅ Fresh data created:")
        print(f"   - Training: {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
        print(f"   - Validation: {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")
        
        if logger:
            logger.info(f"Created fresh training data: {train_path}, {val_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating data: {e}")
        if logger:
            logger.error(f"Data creation failed: {e}")
        return False


def train_model(config: ExperimentConfig, logger=None):
    """Train the model with the given configuration"""
    print_banner("STEP 3: TRAINING MODEL")
    
    try:
        from gemma_270m.trainer import GemmaTrainer
        
        print(f"🧠 Model: {config.name}")
        print(f"📊 Architecture: {config.model.n_layers} layers, {config.model.emb_dim} dim")
        print(f"🎯 Training: {config.training.max_iters} iterations")
        print(f"💾 Batch size: {config.training.batch_size} (effective: {config.training.effective_batch_size})")
        print(f"📏 Sequence length: {config.training.block_size}")
        
        # Initialize trainer
        trainer = GemmaTrainer(config)
        
        print("🚀 Starting training...")
        start_time = time.time()
        
        # Train the model
        results = trainer.train()
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.1f}s ({training_time/3600:.2f}h)")
        print(f"   - Final train loss: {results['final_train_loss']:.4f}")
        print(f"   - Final val loss: {results['final_val_loss']:.4f}")
        print(f"   - Best val loss: {results['best_val_loss']:.4f}")
        print(f"   - Total steps: {results['total_steps']}")
        
        if logger:
            logger.info(f"Training completed: {results}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if logger:
            logger.error(f"Training failed: {e}")
        return False, None


def run_tests(logger=None):
    """Run the test suite"""
    print_banner("STEP 4: RUNNING TESTS")
    
    try:
        import subprocess
        
        test_files = [
            "tests/test_model.py",
            "tests/test_training.py",
        ]
        
        results = []
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"🧪 Running {test_file}...")
                result = subprocess.run([sys.executable, test_file], 
                                      capture_output=True, text=True)
                results.append((test_file, result.returncode == 0))
                
                if result.returncode == 0:
                    print(f"   ✅ {test_file} passed")
                else:
                    print(f"   ❌ {test_file} failed:")
                    print(f"      {result.stderr}")
            else:
                print(f"   ⚠️ {test_file} not found")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"✅ Tests completed: {passed}/{total} passed")
        
        if logger:
            logger.info(f"Tests completed: {passed}/{total} passed")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        if logger:
            logger.error(f"Testing failed: {e}")
        return False


def create_examples(config: ExperimentConfig, logger=None):
    """Create example applications"""
    print_banner("STEP 5: CREATING EXAMPLES")
    
    try:
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)
        
        # We'll create examples in the next step
        print("🚧 Examples creation will be implemented next...")
        print("   - Story generation")
        print("   - Chat interface") 
        print("   - Fine-tuning")
        print("   - Benchmarking")
        print("   - API server")
        print("   - Interactive playground")
        
        if logger:
            logger.info("Examples placeholder created")
        
        return True
        
    except Exception as e:
        print(f"❌ Examples creation failed: {e}")
        if logger:
            logger.error(f"Examples creation failed: {e}")
        return False


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Gemma-270M Pipeline")
    parser.add_argument("--config", "-c", default="configs/optimized_config.yaml",
                      help="Configuration file path")
    parser.add_argument("--steps", nargs="+", 
                      choices=["clean", "data", "train", "test", "examples"],
                      default=["clean", "data", "train", "test", "examples"],
                      help="Pipeline steps to run")
    parser.add_argument("--clean-all", action="store_true",
                      help="Clean all artifacts (checkpoints, logs, etc.)")
    parser.add_argument("--quick", action="store_true",
                      help="Quick training run (small model, fewer iterations)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print_banner("GEMMA-270M TRAINING PIPELINE", "🌟")
    print(f"⚙️  Configuration: {args.config}")
    print(f"📋 Steps: {args.steps}")
    print(f"🕒 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    try:
        if args.quick:
            from gemma_270m.config import get_small_config
            config = get_small_config()
            config.training.max_iters = 500  # Quick run
            print("🏃 Quick mode: Using small model with 500 iterations")
        else:
            config = ExperimentConfig.load(args.config)
            print(f"📖 Loaded config: {config.name}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        logger.error(f"Config loading failed: {e}")
        return False
    
    # Track results
    results = {}
    overall_start = time.time()
    
    # Step 1: Clean data files
    if "clean" in args.steps:
        results["clean"] = clean_data_files(config.data.data_dir, logger)
        if not results["clean"]:
            print("❌ Pipeline failed at cleaning step")
            return False
    
    # Step 2: Create fresh data
    if "data" in args.steps:
        results["data"] = create_fresh_data(config, logger)
        if not results["data"]:
            print("❌ Pipeline failed at data creation step")
            return False
    
    # Step 3: Train model
    if "train" in args.steps:
        train_success, train_results = train_model(config, logger)
        results["train"] = train_success
        if not train_success:
            print("❌ Pipeline failed at training step")
            return False
    
    # Step 4: Run tests
    if "test" in args.steps:
        results["test"] = run_tests(logger)
        # Don't fail pipeline if tests fail, just warn
        if not results["test"]:
            print("⚠️  Some tests failed, but continuing pipeline...")
    
    # Step 5: Create examples
    if "examples" in args.steps:
        results["examples"] = create_examples(config, logger)
        if not results["examples"]:
            print("❌ Pipeline failed at examples creation step")
            return False
    
    # Summary
    total_time = time.time() - overall_start
    print_banner("PIPELINE COMPLETED", "🎉")
    print(f"✅ Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"📊 Results:")
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {step.title()}")
    
    logger.info(f"Pipeline completed in {total_time:.1f}s with results: {results}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
