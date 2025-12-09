"""
Test script to verify the HuggingFace dataset loading works correctly
with the agentlans/high-quality-english-sentences dataset
"""

from datasets import load_dataset

# Test loading the dataset
print("Loading dataset...")
try:
    hs_dataset = load_dataset("agentlans/high-quality-english-sentences", split="train", streaming=True)
    print("✓ Dataset loaded successfully (streaming mode)")
    print(f"Dataset type: {type(hs_dataset)}")

    # Test accessing a few examples
    print("\nTesting data access...")
    sample_iter = iter(hs_dataset)
    for i in range(3):
        sample = next(sample_iter)
        print(f"\nSample {i+1}:")
        print(f"  Keys: {sample.keys()}")
        print(f"  Text type: {type(sample['text'])}")
        print(f"  Text content: {sample['text'][:100]}...")

except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()

# Test with filter (as used in the training script)
print("\n" + "="*80)
print("Testing with filter (as used in training script)...")

def filter_by_word_count(input):
    word_count = len(input['text'].split())  # Note: 'train' should be 'text'
    return word_count <= 200

try:
    hs_dataset = load_dataset("agentlans/high-quality-english-sentences", num_proc=4, split="train")
    print(f"✓ Dataset loaded: {len(hs_dataset)} samples")

    print("Applying filter...")
    filtered = hs_dataset.filter(filter_by_word_count, num_proc=4)
    print(f"✓ Filtered dataset: {len(filtered)} samples")

    # Test accessing the 'text' column
    print("\nTesting column access methods:")
    print(f"  Method 1 - filtered['text']: {type(filtered['text'])}, length: {len(filtered['text'])}")
    print(f"  First 3 texts: {filtered['text'][:3]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
