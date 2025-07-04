#!/usr/bin/env python3
"""Final test to ensure all fixes are working."""

import sys
import os
import subprocess

# Add project root to Python path
sys.path.insert(0, '.')

def run_pytest():
    """Run pytest and return results."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "src/tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=120)
        
        print("Return code:", result.returncode)
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("ERROR: Tests timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run the test."""
    print("🧪 Final Test Run")
    print("=" * 30)
    
    success = run_pytest()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
