#!/usr/bin/env python3
"""Test script for sandbox tool functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sandbox_fusion import create_sandbox_tool

def test_sandbox():
    """Test basic sandbox functionality."""
    print("Testing sandbox tool...")

    # Create sandbox tool
    sandbox_tool = create_sandbox_tool()

    if not sandbox_tool:
        print("Failed to create sandbox tool. Check configuration.")
        return False

    # Test simple Python code
    test_code = "print('Hello, World!')\nprint(2 + 3)"

    print(f"Executing code: {test_code}")

    try:
        result = sandbox_tool.execute_code(test_code)
        print(f"Execution result: {result}")

        # Check if result contains expected output
        if "Hello, World!" in result and "5" in result:
            print("✓ Basic test passed!")
            return True
        else:
            print("✗ Basic test failed - unexpected output")
            return False

    except Exception as e:
        print(f"✗ Error during execution: {e}")
        return False

if __name__ == "__main__":
    success = test_sandbox()
    sys.exit(0 if success else 1)
