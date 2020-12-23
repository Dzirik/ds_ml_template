"""
Example script for usage in example make command which runs this script.
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(f"First input: {sys.argv[1]}")
        print(f"Second input: {sys.argv[2]}")
    else:
        print("No input was set.")
