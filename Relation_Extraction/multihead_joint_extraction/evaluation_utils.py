import sys
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_evaluation import Base_Evaluation

class Evaluation_Utils(Base_Evaluation):
    def __init__(self):
        self.name = "evaluation_utils"