import numpy as np
from sole_basics import SoleBasic

class SoleHaletsky:
    """
    class which is dedicated to solve the equation by Haletsky method
    """

    def __init__(self):
        self.array_original_test = np.array([[5,8,1], [3, -2, 6], [2, 1, -1]])
        self.array_result_test = np.array([2, -7, -5])
        self.size_min, self.size_max = 0, 5
        self.array_original_size = (3, 3)
        self.array_result_size = (1, 3)
        self.eps = 0.01
        self.array_original = self.produce_array(self.array_original_test, self.array_original_size, self.size_min, self.size_max)
        self.array_result = self.produce_array(self.array_result_test, self.array_result_size, self.size_min, self.size_max)

    def show_results(self, triangle_high:np.array, triangle_low:np.array, value_y:np.array, value_x:np.array) -> None:
        """
        Method which is dedicated to show results in cases of the shoving them
        Input:  all array which we are dedicated to work with
        Output: None
        """
        value_output = "We have successfully started to work with Haletsky Method\n" +\
                        "It uses triangle arrays to solve this linear equation\n" +\
                        "Your array:"
        print(value_output)
        pprint(self.array_original)
        print("Value answer:")
        pprint(self.array_result)
        print("Let's calculate triange arrays:")
        print("High triange array")
        pprint(triangle_high)
        print("Low triange array")
        pprint(triangle_low)
        print("Let's calculate values of the ")
        print(value_y)
        print("From what we can see, let's calculate results")
        print(value_x)
        print("Calculated results via the library")
        value_calculated = self.return_basic_answer()
        print(value_calculated)
        print(f"Check that everything is okay: {self.make_check_values(value_calculated, value_x)}")
        print('==========================================================')

    @staticmethod
    def produce_array(value_test:np.ndarray, value_size:set, value_min:int, value_max:int, value_reproduce:bool=False) -> np.ndarray:
        """
        Static method to return array as original
        Input:  value_reproduce = boolean which tells what value is more efficient to use
                value_test = our test value which we would use after for this values
                value_size = size of our values for it to make
                value_min = minimal value in the array
                value_max = maximum value in the array
        Output: np.array for calculations
        """
        return np.random.randint(value_min, value_max, size=value_size) if value_reproduce else value_test

    def produce_zeroes_array(self) -> (np.array, np.array):
        """
        Method to return two basic values of the 
        Input:  all 
        Output: two zeroes arrays
        """
        return np.zeros(self.array_original.shape, dtype=float), np.zeros(self.array_original.shape, dtype=float)

    def make_triange_division(self) -> (np.array, np.array):
        """
        Method to produce division of the original array in two of them
        Input:  previously calculated values of it
        Output: two arrays which are required for the future
        """
        calculated_b, calculated_c = self.produce_zeroes_array()
        for row, row_list in enumerate(self.array_original):
            for column, value in enumerate(row_list):
                if column == 0:
                    calculated_b[row, column] = self.array_original[row, column]
                if row == 0:
                    calculated_c[row, column] = self.array_original[row, column]/calculated_b[0,0]
                if row >= column > 0:
                    value_sum = sum(calculated_b[row, k]* calculated_c[k, column] for k in range(0, column, 1))
                    calculated_b[row, column] = self.array_original[row, column] - value_sum
                if column >= row > 0:
                    value_sum = sum(calculated_b[row, k]* calculated_c[k, column] for k in range(0, row, 1))
                    calculated_c[row, column] = (self.array_original[row, column] -  value_sum)/calculated_b[row, row]
        return calculated_b, calculated_c

    def produce_result(self) -> np.array:
        """
        Method which is dedicated to  produce values of the result
        Input:  all generated values which we hadd currently produced
        Output: answer list which is dedicated to all of that
        """
        try:
            triangle_low, triangle_high = self.make_triange_division()
            value_y, value_x = np.array([], dtype=float), np.array([], dtype=float)
            for i, value_row in enumerate(triangle_low):
                if i == 0:
                    value_y = np.append(value_y, self.array_result[i]/self.array_original[i, i])
                else:
                    value_sum = sum(triangle_low[i,k]*value_y[k] for k in range(i))
                    value_y = np.append(value_y, [(self.array_result[i] - value_sum)/triangle_low[i, i]])
            value_y_reverse = value_y[::-1]
            triangle_high_reversed = np.array([x[::-1] for x in triangle_high[::-1]])
            value_x = np.append(value_x, [value_y[-1]])
            for i, value_use in enumerate(triangle_high_reversed):
                if i > 0:
                    value_sum = sum(triangle_high_reversed[i, k]*value_x[k] for k in range(0, i))
                    value_x = np.append(value_x, value_y_reverse[i] - value_sum)
            value_x = value_x[::-1]
            self.show_results(triangle_high, triangle_low, value_y, value_x)
            return value_x
        except Exception as e:
            print("We faced some issues, please recheck your values")
            return np.array([])

    def return_basic_answer(self) -> np.array:
        """
        Method which is dedicated to produce usual results
        Input:  all previously inputted values
        Output: real answer calculated via the numpy
        """
        return np.linalg.solve(self.array_original, self.array_result)
        
    def make_check_values(self, value_calculated:np.array, value_returned:np.array) -> np.ndarray:
        """
        Method which is dedicated to check our values via the NumPy library
        Input:  previously calculated values
        Output: answer array which is required to work with
        """
        return all(abs(x-y) <= self.eps for x, y in zip(value_calculated, value_returned))