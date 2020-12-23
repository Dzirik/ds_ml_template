"""
Example script for usage in example make command which runs this script.
"""
import sys

if __name__ == "__main__":
    print("This is example script!")

    if len(sys.argv) > 1:
        print(f"Is input: {sys.argv[1]}")
    else:
        print("No input was set.")

    print("=" * 100)
