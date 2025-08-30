#!/usr/bin/env python
"""
This script runs the anonymization demo to showcase the complete anonymization process.
"""

import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the demo script
from demo_anonymization import main

if __name__ == "__main__":
    main()