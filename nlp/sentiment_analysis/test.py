# This test file can help you understanding the file imports from different folders

import os
import sys
# Add the main directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#print(os.getcwd())
from nlp.comments_preprocessing import *
from preprocessing import *