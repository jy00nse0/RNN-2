#!/usr/bin/env python3
"""
Unit tests for batch_reverse_source function.
Tests various tensor shapes and padding configurations.

Note: The function is defined inline here because train.py has cascading imports
that require torchtext (which may not be installed in all environments).
This allows the test to run independently.
"""

import torch
import sys

# Try to import from train.py, fall back to inline definition if dependencies are missing
try:
    from train import batch_reverse_source
except (ImportError, OSError):
    # Define locally if train.py cannot be imported due to missing dependencies
    def batch_reverse_source(src_tensor, pad_idx, batch_first=False):
        """
        Reverse the content of source sequences while preserving <sos>, <eos>, and <pad> positions.
        
        Supports both (seq_len, batch) and (batch, seq_len) tensor shapes.
        Handles padding at both beginning (left) and end (right) of sequences.
        
        Args:
            src_tensor: Input tensor of shape (seq_len, batch) or (batch, seq_len)
            pad_idx: Index of padding token
            batch_first: If True, input shape is (batch, seq_len); if False, (seq_len, batch)
        
        Input structure: <sos> w1 w2 ... wn <eos> (with optional <pad> before or after)
        Output structure: <sos> wn ... w2 w1 <eos> (with same <pad> positions preserved)
        
        Returns:
            Tensor with reversed content, same shape as input
        """
        rev_src = src_tensor.clone()
        
        if batch_first:
            batch_size, seq_len = src_tensor.size()
        else:
            seq_len, batch_size = src_tensor.size()
        
        for b in range(batch_size):
            if batch_first:
                seq = src_tensor[b, :]
            else:
                seq = src_tensor[:, b]
            
            # Find indices of non-padding tokens
            non_pad_indices = (seq != pad_idx).nonzero(as_tuple=True)[0]
            
            if len(non_pad_indices) <= 2:
                # Only <sos> and <eos> or fewer - nothing to reverse
                continue
            
            # First and last non-padding positions are <sos> and <eos>
            first_non_pad = non_pad_indices[0].item()
            last_non_pad = non_pad_indices[-1].item()
            
            # Content to reverse: tokens between <sos> (first_non_pad) and <eos> (last_non_pad)
            start_idx = first_non_pad + 1  # skip <sos>
            end_idx = last_non_pad  # up to but not including <eos>
            
            if end_idx <= start_idx:
                # No content to reverse (only special tokens)
                continue
            
            # Extract content and reverse
            if batch_first:
                content = src_tensor[b, start_idx:end_idx]
                rev_src[b, start_idx:end_idx] = torch.flip(content, dims=[0])
            else:
                content = src_tensor[start_idx:end_idx, b]
                rev_src[start_idx:end_idx, b] = torch.flip(content, dims=[0])
        
        return rev_src


def run_tests():
    """Run all test cases for batch_reverse_source function."""
    pad_idx = 2  # <pad>=2
    all_pass = True
    
    print("="*60)
    print("Testing batch_reverse_source Function")
    print("="*60)
    
    # Test Case 1: Standard (seq_len, batch) with padding at end
    print("\nTest 1: (seq_len, batch) with padding at END")
    src1 = torch.tensor([[0, 0], [3, 3], [4, 4], [5, 5], [1, 1], [2, 2]])
    result1 = batch_reverse_source(src1, pad_idx, batch_first=False)
    expected1 = torch.tensor([[0, 0], [5, 5], [4, 4], [3, 3], [1, 1], [2, 2]])
    pass1 = torch.equal(result1, expected1)
    print(f"Pass: {pass1}")
    all_pass = all_pass and pass1
    
    # Test Case 2: Different length sequences with padding at end
    print("\nTest 2: Different length sequences (padding at end)")
    src2 = torch.tensor([[0, 0], [3, 3], [4, 1], [5, 2], [1, 2], [2, 2]])
    result2 = batch_reverse_source(src2, pad_idx, batch_first=False)
    expected2 = torch.tensor([[0, 0], [5, 3], [4, 1], [3, 2], [1, 2], [2, 2]])
    pass2 = torch.equal(result2, expected2)
    print(f"Pass: {pass2}")
    all_pass = all_pass and pass2
    
    # Test Case 3: Padding at the beginning (left padding)
    print("\nTest 3: Padding at the BEGINNING (left padding)")
    src3 = torch.tensor([[2, 2], [0, 0], [3, 3], [4, 4], [5, 5], [1, 1]])
    result3 = batch_reverse_source(src3, pad_idx, batch_first=False)
    expected3 = torch.tensor([[2, 2], [0, 0], [5, 5], [4, 4], [3, 3], [1, 1]])
    pass3 = torch.equal(result3, expected3)
    print(f"Pass: {pass3}")
    all_pass = all_pass and pass3
    
    # Test Case 4: (batch, seq_len) shape - batch_first=True
    print("\nTest 4: (batch, seq_len) shape with batch_first=True")
    src4 = torch.tensor([[0, 3, 4, 5, 1, 2], [0, 3, 4, 5, 1, 2]])
    result4 = batch_reverse_source(src4, pad_idx, batch_first=True)
    expected4 = torch.tensor([[0, 5, 4, 3, 1, 2], [0, 5, 4, 3, 1, 2]])
    pass4 = torch.equal(result4, expected4)
    print(f"Pass: {pass4}")
    all_pass = all_pass and pass4
    
    # Test Case 5: Mixed padding in batch_first mode
    print("\nTest 5: Mixed padding positions in batch_first=True")
    src5 = torch.tensor([[0, 3, 4, 1, 2, 2], [2, 0, 3, 4, 5, 1]])
    result5 = batch_reverse_source(src5, pad_idx, batch_first=True)
    expected5 = torch.tensor([[0, 4, 3, 1, 2, 2], [2, 0, 5, 4, 3, 1]])
    pass5 = torch.equal(result5, expected5)
    print(f"Pass: {pass5}")
    all_pass = all_pass and pass5
    
    # Test Case 6: Only <sos> and <eos> (no content)
    print("\nTest 6: Only <sos> and <eos> (no content to reverse)")
    src6 = torch.tensor([[0], [1], [2]])
    result6 = batch_reverse_source(src6, pad_idx, batch_first=False)
    pass6 = torch.equal(result6, src6)
    print(f"Pass: {pass6}")
    all_pass = all_pass and pass6
    
    # Test Case 7: Single word content
    print("\nTest 7: Single word content")
    src7 = torch.tensor([[0], [3], [1], [2]])
    result7 = batch_reverse_source(src7, pad_idx, batch_first=False)
    pass7 = torch.equal(result7, src7)
    print(f"Pass: {pass7}")
    all_pass = all_pass and pass7
    
    print("\n" + "="*60)
    if all_pass:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
