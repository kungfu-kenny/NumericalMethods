import numpy as np
from pprint import pprint

size_matrix_maximum = 2
# from config import size_matrix_maximum

class SoleBasic:
    """
    Class which is dedicated to work with Arrays of the Basics
    It takes values, checks them on a minority
    """
    def __init__(self, **kwargs):
        self.produce_values(kwargs)

    @staticmethod
    def check_size(array_check:np.array) -> (set, bool):
        """
        Static method which is dedicated to work with size arrays and to check 
        is it compatible for now methods
        Input:  array_check = array which is 
        Output: size of selected matrix and boolean which signifies is it okay
        """
        value_size = array_check.size
        return value_size, len(value_size) <= size_matrix_maximum

    def produce_checking(self) -> bool:
        """
        Method which is dedicated to work with a checking of the array
        in cases of new values inside
        Input:  self.array_original = array which is main to perform operations
        Output: boolean value of the self.array_original
        """
        return 0

    @staticmethod
    def produce_minor(array_check:np.array) -> (int, bool):
        """
        Method which is dedicated to work with a minor of the array
        Input:  array_check = numerical array which needs checking
        Output: minor of the array
        """
        
        return 0, True

    @staticmethod
    def produce_determinant(array_check:np.array) -> (float, bool):
        """
        Method which is dedicated to produce determinant of the array
        Input:  array_check = numerical array which needs checking
        Output: determinant
        """
        return 0, False

    @staticmethod
    def produce_function_analysis(array_check:np.array) -> (float, bool):
        """
        Method which is dedicated to produce function analysis of selected array
        and to return all 
        Input:  array_check = list which is required to check values
        Output: we returned list 
        """
        return 0, True

    def produce_values(self, value_dictionary:dict) -> None:
        """
        Method which is dedicated to produce selected values from
        sented values to this class
        Input:  value_dictionary = dictionary with our values
        Output: values of all of it
        """
        print('==============================================')
        pprint(value_dictionary)
        print('==============================================')
        return 0

    def return_basic_answer(self, array_original, array_result) -> np.array:
        """
        Method which is dedicated to solve linear equations by NumPy
        Input:  array_original = input matrix 
                array_result = result of the value
        Output: linear equations which shows
        """
        array_original = self.array_original if self.array_original and not 
        if array_original and array_result:
            return np.linalg.solve(self.array_original, self.array_result)
        return np.zeros(size_matrix_maximum + 1, dtype=int)

#TODO add producing new values for the array, interface for insertion values

if __name__ == "__main__":
    array_original = np.array([[3, 2, -1], [2, -2, 4], [-1, +0.5, -1]], dtype=float)
    array_result = np.array([1, -2, 0], dtype=float)
    A = SoleBasic(array_original=array_original, array_result=array_result)
