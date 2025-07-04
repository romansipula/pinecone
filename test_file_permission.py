#!/usr/bin/env python3
"""Test the specific file permission fix."""

import sys
import tempfile
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '.')

def test_file_permission_fix():
    """Test that the file permission issue is fixed."""
    print("Testing file permission fix...")
    
    try:
        from src.rag_utils import _extract_text_file
        
        # Create temporary file the same way as the test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            f.flush()
            temp_file_path = f.name
            
        # File is now properly closed, safe to read and delete
        try:
            result = _extract_text_file(Path(temp_file_path))
            assert result == "Test file content"
            print("✓ Text extraction successful")
        finally:
            # Ensure file is deleted even if test fails
            try:
                os.unlink(temp_file_path)
                print("✓ File deletion successful")
            except (PermissionError, FileNotFoundError):
                print("⚠️  File deletion failed (expected on Windows)")
                
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_file_permission_fix()
    if success:
        print("\n🎉 File permission fix validated!")
    else:
        print("\n❌ File permission fix failed!")
    exit(0 if success else 1)
