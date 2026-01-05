#!/usr/bin/env python3
"""
Integration test for calculate_bleu.py with multi-bleu.perl
Tests the calculate_bleu_with_perl function
"""

import os
import sys
import tempfile
import subprocess

# Import the function we want to test
sys.path.insert(0, os.path.dirname(__file__))
from calculate_bleu import calculate_bleu_with_perl


def test_bleu_calculation():
    """Test BLEU calculation with multi-bleu.perl"""
    print("=" * 70)
    print("Testing multi-bleu.perl Integration")
    print("=" * 70)
    
    # Create temporary reference file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as ref_file:
        ref_path = ref_file.name
        ref_file.write("This is a test sentence .\n")
        ref_file.write("Another test sentence here .\n")
    
    try:
        # Test 1: Perfect match (BLEU = 100)
        print("\n1. Testing perfect match...")
        hypotheses = [
            "This is a test sentence .",
            "Another test sentence here ."
        ]
        
        result = calculate_bleu_with_perl(hypotheses, ref_path, lowercase=False)
        print(f"   Result: {result}")
        assert "BLEU = 100.00" in result, f"Expected BLEU = 100.00, got {result}"
        print("   ✅ Perfect match test passed")
        
        # Test 2: Partial match
        print("\n2. Testing partial match...")
        hypotheses = [
            "This is a different sentence .",
            "Another test sentence here ."
        ]
        
        result = calculate_bleu_with_perl(hypotheses, ref_path, lowercase=False)
        print(f"   Result: {result}")
        assert "BLEU" in result, f"Expected BLEU score, got {result}"
        # Should be around 63.40 based on our earlier test
        assert "63." in result or "64." in result or "62." in result, f"Expected ~63 BLEU, got {result}"
        print("   ✅ Partial match test passed")
        
        # Test 3: Lowercase mode
        print("\n3. Testing lowercase mode...")
        hypotheses = [
            "THIS IS A TEST SENTENCE .",
            "ANOTHER TEST SENTENCE HERE ."
        ]
        
        result = calculate_bleu_with_perl(hypotheses, ref_path, lowercase=True)
        print(f"   Result: {result}")
        assert "BLEU = 100.00" in result, f"Expected BLEU = 100.00 with lowercase, got {result}"
        print("   ✅ Lowercase mode test passed")
        
        print("\n" + "=" * 70)
        print("✅ All BLEU calculation tests passed!")
        print("=" * 70)
        return True
        
    finally:
        # Clean up
        if os.path.exists(ref_path):
            os.unlink(ref_path)


if __name__ == '__main__':
    try:
        success = test_bleu_calculation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
