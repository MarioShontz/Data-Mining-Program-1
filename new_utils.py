import numpy as np

"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""


def scale_data(data):
    """
    ensure every element is a float and normalize points to be between 0 and 1
    """
    return (data - data.min()) / (data.max() - data.min())
