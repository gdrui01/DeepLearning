"""
Verification script for generation models used in RL training pipeline.

This script helps verify and compare the output of different language models
using the same prompts from your PPO training pipeline. It's particularly useful
for understanding why base GPT-2 fails at instruction-following and comparing it
with instruction-tuned alternatives.

USAGE:
------
1. List available models:
   python verify_gpt2_generation.py --list_models

2. Test base GPT-2 (will show poor instruction-following):
   python verify_gpt2_generation.py --model_path gpt2

3. Compare GPT-2 with instruction-tuned model:
   python verify_gpt2_generation.py --compare --compare_models gpt2 facebook/opt-iml-1.3b

4. Test your fine-tuned model after RL training:
   python verify_gpt2_generation.py --model_path checkpoints/gpt2-ppo-method2

5. Compare multiple models at once:
   python verify_gpt2_generation.py --compare --compare_models gpt2 bigscience/bloomz-560m google/flan-t5-base

WHY THIS IS USEFUL:
------------------
- Base GPT-2 is NOT instruction-tuned → continues text instead of following instructions
- Your PPO training teaches it to follow instructions → this script verifies the improvement
- Compare with pre-trained instruction-tuned models to see what good performance looks like
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Default instruction template from the training pipeline
DEFAULT_INSTRUCTION = (
    "Make the following sentence harder to translate while keeping it grammatically correct and natural.\n\n"
    "'{seed}'\n\nRewrite it as one grammatical English sentence. Output only the edited sentence, no other text."
)

# Recommended instruction-tuned models (sorted by size)
INSTRUCTION_MODELS = {
    "distilgpt2": {"size": "82M", "instruction_tuned": False, "description": "Distilled GPT-2 (not instruction-tuned)"},
    "gpt2": {"size": "124M", "instruction_tuned": False, "description": "Base GPT-2 (not instruction-tuned)"},
    "facebook/opt-125m": {"size": "125M", "instruction_tuned": False, "description": "OPT base model (not instruction-tuned)"},
    "gpt2-medium": {"size": "355M", "instruction_tuned": False, "description": "Medium GPT-2 (not instruction-tuned)"},
    "google/flan-t5-small": {"size": "80M", "instruction_tuned": True, "description": "FLAN-T5 small (encoder-decoder, instruction-tuned)"},
    "google/flan-t5-base": {"size": "250M", "instruction_tuned": True, "description": "FLAN-T5 base (encoder-decoder, instruction-tuned)"},
    "cerebras/Cerebras-GPT-111M": {"size": "111M", "instruction_tuned": False, "description": "Cerebras GPT (not instruction-tuned)"},
    "facebook/opt-iml-1.3b": {"size": "1.3B", "instruction_tuned": True, "description": "OPT with instruction meta-learning (best for RL)"},
    "bigscience/bloomz-560m": {"size": "560M", "instruction_tuned": True, "description": "BLOOMZ 560M (multilingual, instruction-tuned)"},
}


def build_prompt(seed: str) -> str:
    """Build a prompt from a seed sentence using the training template."""
    return DEFAULT_INSTRUCTION.format(seed=seed)


def is_seq2seq_model(model_name: str) -> bool:
    """Check if model is a seq2seq model (like T5, BART)."""
    seq2seq_keywords = ["t5", "bart", "pegasus"]
    return any(keyword in model_name.lower() for keyword in seq2seq_keywords)


def load_model_and_tokenizer(model_path: str):
    """Load the model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Determine if seq2seq or causal LM
    if is_seq2seq_model(model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        is_seq2seq = False

    # Set pad token if not present (same as training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    print(f"Model type: {'Seq2Seq' if is_seq2seq else 'Causal LM'}")
    return model, tokenizer, device, is_seq2seq


def generate_response(model, tokenizer, device, prompt: str,
                     max_new_tokens: int = 40,
                     temperature: float = 0.9,
                     top_p: float = 0.95,
                     num_samples: int = 1,
                     is_seq2seq: bool = False):
    """Generate response(s) for a given prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generation parameters (matching training pipeline)
    gen_kwargs = {
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": num_samples,
    }

    # For causal LM models, include eos_token_id
    if not is_seq2seq:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode outputs
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses


def display_model_info():
    """Display available instruction-tuned models."""
    print("\nAvailable models for comparison:")
    print("=" * 100)
    print(f"{'Model Name':<35} {'Size':<10} {'Instruction-Tuned':<20} {'Description'}")
    print("-" * 100)
    for name, info in INSTRUCTION_MODELS.items():
        tuned = "✓ Yes" if info["instruction_tuned"] else "✗ No"
        print(f"{name:<35} {info['size']:<10} {tuned:<20} {info['description']}")
    print("=" * 100)
    print("\nRecommendations for RL on single GPU:")
    print("  1. facebook/opt-iml-1.3b      - Best instruction-tuned option (1.3B params)")
    print("  2. bigscience/bloomz-560m     - Smaller instruction-tuned alternative (560M params)")
    print("  3. google/flan-t5-base        - Good for testing (250M params, encoder-decoder)")
    print("  4. google/flan-t5-small       - Lightweight testing (80M params, encoder-decoder)")
    print("\nNote: Base models (gpt2, opt-125m) are NOT instruction-tuned and will not follow instructions well.")
    print("      Your PPO training aims to teach them instruction-following behavior!")
    print()


def test_single_model(model_name: str, seed_sentence: str, max_new_tokens: int,
                     temperature: float, top_p: float, num_samples: int,
                     show_prompt: bool):
    """Test a single model and display results."""
    model, tokenizer, device, is_seq2seq = load_model_and_tokenizer(model_name)
    prompt = build_prompt(seed_sentence)

    if show_prompt:
        print("\n" + "="*80)
        print("FULL PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

    print(f"\n{'='*80}")
    print(f"GENERATING {num_samples} RESPONSE(S)")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Seed sentence: '{seed_sentence}'")
    print(f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_new_tokens}")
    print(f"{'='*80}\n")

    responses = generate_response(
        model, tokenizer, device, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_samples=num_samples,
        is_seq2seq=is_seq2seq
    )

    # Display responses
    for i, response in enumerate(responses, 1):
        print(f"Sample {i}:")
        print(f"{'-'*80}")
        print(response)
        print(f"{'-'*80}\n")

        # Try to extract just the generated part (after the prompt)
        if prompt in response:
            generated_only = response[len(prompt):].strip()
            if generated_only:
                print(f"Generated text only:")
                print(f"  → {generated_only}")
                print()

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def compare_models(models: list, seed_sentence: str, max_new_tokens: int,
                  temperature: float, top_p: float, num_samples: int):
    """Compare multiple models side-by-side."""
    prompt = build_prompt(seed_sentence)

    print(f"\n{'='*100}")
    print("MODEL COMPARISON")
    print(f"{'='*100}")
    print(f"Seed sentence: '{seed_sentence}'")
    print(f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_new_tokens}")
    print(f"{'='*100}\n")

    all_results = {}

    for model_name in models:
        print(f"\n{'─'*100}")
        print(f"Loading: {model_name}")
        print(f"{'─'*100}")

        try:
            model, tokenizer, device, is_seq2seq = load_model_and_tokenizer(model_name)

            responses = generate_response(
                model, tokenizer, device, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_samples=num_samples,
                is_seq2seq=is_seq2seq
            )

            all_results[model_name] = responses

            # Display first sample
            print(f"\nSample output:")
            print(f"{'-'*100}")
            print(responses[0])
            print(f"{'-'*100}")

            # Extract generated part
            if prompt in responses[0]:
                generated_only = responses[0][len(prompt):].strip()
                if generated_only:
                    print(f"\nGenerated text only:")
                    print(f"  → {generated_only[:200]}{'...' if len(generated_only) > 200 else ''}")

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"ERROR loading {model_name}: {e}")
            all_results[model_name] = None

    # Summary comparison
    print(f"\n\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}\n")

    for model_name, responses in all_results.items():
        if responses is None:
            print(f"[{model_name}]")
            print(f"  Status: FAILED TO LOAD\n")
            continue

        print(f"[{model_name}]")
        # Get instruction-tuned status
        is_instruction = INSTRUCTION_MODELS.get(model_name, {}).get("instruction_tuned", "Unknown")
        print(f"  Instruction-tuned: {'Yes ✓' if is_instruction else 'No ✗'}")

        for i, response in enumerate(responses[:3], 1):  # Show up to 3 samples
            # Extract just generated part
            if prompt in response:
                generated = response[len(prompt):].strip()
            else:
                generated = response.strip()

            # Truncate if too long
            display_text = generated[:150] + "..." if len(generated) > 150 else generated
            print(f"  Sample {i}: {display_text}")
        print()

    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify GPT-2 model generation with training pipeline prompts and compare with instruction-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single model
  python verify_gpt2_generation.py --model_path gpt2

  # Compare base GPT-2 with instruction-tuned model
  python verify_gpt2_generation.py --compare --compare_models gpt2 facebook/opt-iml-1.3b

  # List available models
  python verify_gpt2_generation.py --list_models

  # Test with fine-tuned checkpoint
  python verify_gpt2_generation.py --model_path checkpoints/gpt2-ppo-method2
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt2",
        help="Path to the model (default: 'gpt2' for base model, or path to fine-tuned checkpoint)"
    )
    parser.add_argument(
        "--seed_sentence",
        type=str,
        default="He saw her duck.",
        help="Seed sentence to test with"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum number of new tokens to generate (default: 40)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to generate (default: 3)"
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Display the full prompt being sent to the model"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode to test multiple models"
    )
    parser.add_argument(
        "--compare_models",
        type=str,
        nargs="+",
        default=["gpt2", "google/flan-t5-small"],
        help="List of models to compare (default: gpt2 google/flan-t5-small)"
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available instruction-tuned models and exit"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        display_model_info()
        return

    # Comparison mode
    if args.compare:
        compare_models(
            models=args.compare_models,
            seed_sentence=args.seed_sentence,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples
        )
    else:
        # Single model test
        test_single_model(
            model_name=args.model_path,
            seed_sentence=args.seed_sentence,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            show_prompt=args.show_prompt
        )

    print(f"{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
