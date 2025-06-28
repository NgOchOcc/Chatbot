#!/usr/bin/env python3
"""
LLM Model Quantization Script using AutoAWQ
Supports: Gemma2, Llama3, Llama3.1, Llama3.2, and other compatible models
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelQuantizer:
    def __init__(self, model_path: str, quant_path: str, quant_config: dict = None):
        """
        Initialize the model quantizer
        
        Args:
            model_path: Path or HuggingFace model ID of the model to quantize
            quant_path: Path where quantized model will be saved
            quant_config: Quantization configuration dictionary
        """
        self.model_path = model_path
        self.quant_path = quant_path
        self.quant_config = quant_config or {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
        
    def load_calibration_dataset(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1", num_samples: int = 512):
        """
        Load calibration dataset for quantization
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset
            num_samples: Number of samples to use for calibration
            
        Returns:
            List of text samples
        """
        logger.info(f"Loading calibration dataset: {dataset_name}")
        
        try:
            if dataset_name == "wikitext":
                dataset = load_dataset(dataset_name, subset, split="train")
                # Filter out empty texts and take first num_samples
                texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][:num_samples]
            elif dataset_name == "c4":
                dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= num_samples:
                        break
                    if len(item["text"].strip()) > 50:
                        texts.append(item["text"])
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
                
            logger.info(f"Loaded {len(texts)} calibration samples")
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to simple text samples
            return self._get_fallback_samples()
    
    def _get_fallback_samples(self):
        """Fallback calibration samples if dataset loading fails"""
        return [
            "The quick brown fox jumps over the lazy dog. This is a common sentence used for testing.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing enables computers to understand and generate human language.",
            "Deep learning models have revolutionized computer vision and natural language understanding.",
            "Transformers are a type of neural network architecture that has become dominant in NLP."
        ] * 100  # Repeat to get more samples
    
    def quantize_model(self, use_calib_data: bool = False):
        """
        Quantize the model using AutoAWQ (zero-shot or with calibration)
        
        Args:
            use_calib_data: Whether to use calibration data or zero-shot quantization
        """
        try:
            logger.info(f"Starting quantization of model: {self.model_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model for quantization
            logger.info("Loading model for quantization...")
            model = AutoAWQForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            if use_calib_data:
                # Load and prepare calibration data
                logger.info("Loading calibration data...")
                calib_data = self.load_calibration_dataset()
                
                def prepare_calib_data(texts, tokenizer, max_length=512):
                    calib_inputs = []
                    for text in texts[:512]:
                        tokens = tokenizer(
                            text,
                            return_tensors="pt",
                            max_length=max_length,
                            truncation=True,
                            padding=False
                        )
                        calib_inputs.append(tokens)
                    return calib_inputs
                
                calib_inputs = prepare_calib_data(calib_data, tokenizer)
                
                # Quantize with calibration data
                logger.info("Starting model quantization with calibration data...")
                model.quantize(
                    tokenizer,
                    quant_config=self.quant_config,
                    max_model_len=8192,
                )
            else:
                # Zero-shot quantization (no calibration data needed)
                logger.info("Starting zero-shot model quantization...")
                model.quantize(
                    tokenizer,
                    quant_config=self.quant_config
                )
            
            # Save quantized model
            logger.info(f"Saving quantized model to: {self.quant_path}")
            model.save_quantized(self.quant_path)
            tokenizer.save_pretrained(self.quant_path)
            
            logger.info("Quantization completed successfully!")
            
            # Print model size comparison
            self._print_size_comparison()
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            raise
    
    def _print_size_comparison(self):
        """Print size comparison between original and quantized models"""
        try:
            if os.path.exists(self.quant_path):
                def get_folder_size(path):
                    total = 0
                    for dirpath, dirnames, filenames in os.walk(path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total += os.path.getsize(filepath)
                    return total / (1024**3)  # Convert to GB
                
                quant_size = get_folder_size(self.quant_path)
                logger.info(f"Quantized model size: {quant_size:.2f} GB")
                
        except Exception as e:
            logger.warning(f"Could not calculate model sizes: {e}")

def get_model_configs():
    """Get predefined configurations for popular models"""
    return {
        "gemma2-2b": "google/gemma-2-2b-it",
        "gemma2-9b": "google/gemma-2-9b-it",
        "gemma2-27b": "google/gemma-2-27b-it",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "llama3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"
    }

def main():
    parser = argparse.ArgumentParser(description="Quantize LLM models using AutoAWQ")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model path or HuggingFace model ID (or use preset: gemma2-2b, llama3-8b, etc.)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for quantized model")
    parser.add_argument("--w_bit", type=int, default=4, choices=[2, 3, 4, 8],
                       help="Weight quantization bits (default: 4)")
    parser.add_argument("--q_group_size", type=int, default=128, choices=[32, 64, 128],
                       help="Quantization group size (default: 128)")
    parser.add_argument("--use_calib", action="store_true", default=False,
                       help="Use calibration data (default: False - zero-shot quantization)")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "c4"],
                       help="Calibration dataset (only used with --use_calib)")
    parser.add_argument("--num_samples", type=int, default=512,
                       help="Number of calibration samples (only used with --use_calib)")
    parser.add_argument("--zero_point", action="store_true", default=True,
                       help="Use zero point quantization (default: True)")
    parser.add_argument("--version", type=str, default="GEMM", choices=["GEMM", "GEMV"],
                       help="Quantization version (default: GEMM)")
    
    args = parser.parse_args()
    
    # Check if model is a preset
    model_configs = get_model_configs()
    if args.model in model_configs:
        model_path = model_configs[args.model]
        logger.info(f"Using preset model: {args.model} -> {model_path}")
    else:
        model_path = args.model
    
    # Setup quantization config
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }
    
    logger.info(f"Quantization config: {quant_config}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize quantizer and run
    quantizer = ModelQuantizer(model_path, args.output, quant_config)
    
    try:
        # Quantize model (zero-shot by default, with calibration if requested)
        quantizer.quantize_model(use_calib_data=args.use_calib)
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())