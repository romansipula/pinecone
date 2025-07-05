#!/usr/bin/env python3
"""
Development Environment Test Script
This script tests that all core dependencies are properly installed and working.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pinecone
        print("✓ pinecone-client imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pinecone: {e}")
        return False
    
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import openai: {e}")
        return False
    
    try:
        import langchain
        print("✓ langchain imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import langchain: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sentence-transformers: {e}")
        return False
    
    try:
        import pypdf
        print("✓ pypdf imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pypdf: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import python-dotenv: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration."""
    print("\nTesting environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file exists")
    else:
        print("✗ .env file not found")
        return False
    
    # Check if data directory exists
    data_dir = Path("src/data")
    if data_dir.exists():
        print("✓ data directory exists")
        # List data files
        data_files = list(data_dir.glob("*"))
        print(f"  Found {len(data_files)} files in data directory")
    else:
        print("✗ data directory not found")
        return False
    
    return True

def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "src/agents",
        "src/data",
        "src/prompts",
        "src/scripts",
        "src/tests"
    ]
    
    required_files = [
        "main.py",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        ".env.example"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path} directory exists")
        else:
            print(f"✗ {dir_path} directory missing")
            return False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} file exists")
        else:
            print(f"✗ {file_path} file missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 RAG Chatbot Development Environment Test")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Environment Configuration", test_environment),
        ("Project Structure", test_project_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! Development environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
