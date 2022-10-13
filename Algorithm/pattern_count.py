'''
by Yifan Guan
https://github.com/yifanguan/DatabaseRepair/blob/master/minority_detection/pattern_count.py

Pattern Count algorithm based on Coverage paper; adapted into Yuval's pattern detection algorithm
Bitvector count calculation

each attribute value has one bitarray, total number of bitarrays = sum of cardinalities of all attributes
length of each bitarray = # of different value combinations in a dataset
assume pattern has not somehow been transformed to the form ([0-9a-zA-Z] \ X)*          (i.e. currently supported maximum cardinality of an attribute is 10 + 26 + 25 = 61)
assume datafile cannot read directly with pandas
'''
from bitarray import bitarray
from bitarray.util import zeros
import pandas as pd
import numpy as np


class PatternCounter:
    '''
    Create a counter for each set of data (i.e. train data, test data)
    Create object, call parse_data(), then call pattern_count()
    add support for non-encoded dataset; that is, to deal with any-arbitrary attribute value for each attribute
    '''
    def __init__(self, dataframe, selected_attrs_names=None, selected_attrs_id=None, encoded=True):
        '''
        dataframe: df = pd.read_csv(self.filename, delimiter=', *', engine='python')
        filename: name of the data file
        selected_attrs_names: selected attributes considered in a lattice
        selected_attrs_id: id of selected attributes considered in a lattice
        both of them are in a iterable format (i.e. list) At least one of them should be provided
        encoded: whether the dataset has been processed
        '''
        self.empty = False
        self.dataframe = dataframe
        self.selected_attrs_names = selected_attrs_names
        self.selected_attrs_id = selected_attrs_id
        self.encoded = encoded
        self.num_attrs = -1
        self.cardinalities = None
        self.count_map = {} # map pattern (value combination) -> frequency in data
        self.occurences = [] # values of count_map for faster calculation
        self.dataBitVec = [] # list of bitarrays, each index of one bitarray corresponds to a
        self.num_unique_value_combinations = -1
        self.char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A',
            'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
        self.attr_value_map = None # map of maps, [attr_index][attr_value] stores the index of one attribute value within one attribute

    def parse_data(self):
        # open file, read corresponding columns
        df = self.dataframe
        if len(df) == 0:
            print("PatternCount original set is empty!")
            self.empty = True
            return
        if self.selected_attrs_names is not None:
            df = df[self.selected_attrs_names]
            self.num_attrs = len(self.selected_attrs_names)
        elif self.selected_attrs_id is not None:
            df = df.iloc[:, self.selected_attrs_id]
            self.num_attrs = len(self.selected_attrs_id)
        else:
            self.selected_attrs_names = df.columns.values.tolist()
            self.num_attrs = len(self.selected_attrs_names)
        data = df.values.astype(str) # 2d np array
        cardinalities = list(df.nunique())
        self.cardinalities = cardinalities

        # actual mapping for non-encoded dataset to keep track of locations of attribute-value bitarrays
        if not self.encoded:
            self.attr_value_map = {}
            for i in range(self.num_attrs):
                self.attr_value_map[i] = {}
                attr_values = np.unique(data[:,i])
                for j, attr_value in enumerate(attr_values):
                    self.attr_value_map[i][attr_value] = j

        if not self.encoded:
            for attr_value_list in data: # each row in data is a list containing attribute values
                value_combination = '|'.join(attr_value_list) # use | to seperate different string values
                if value_combination in self.count_map:
                    self.count_map[value_combination] += 1
                else:
                    self.count_map[value_combination] = 1
        else:
            # construct count map to count frequency of each value combination in a dataset
            for attr_value_list in data: # each row in data is a list containing attribute values
                value_combination = ''.join(attr_value_list)
                if value_combination in self.count_map:
                    self.count_map[value_combination] += 1
                else:
                    self.count_map[value_combination] = 1

        # create bitarrays
        uniqueValueCombinations = list(self.count_map.keys())
        for pattern in uniqueValueCombinations:
            self.occurences.append(self.count_map[pattern])
        self.num_unique_value_combinations = len(uniqueValueCombinations)

        zero_bitarray = zeros(self.num_unique_value_combinations) # allocate a bitarray with length = # of unique value combinations
        self.dataBitVec = [bitarray()] * sum(cardinalities)
        self.dataBitVec[0] = zero_bitarray
        for i in range(1, sum(cardinalities)):
            self.dataBitVec[i] = bitarray(zero_bitarray) # this is fast deep copy based on the reference manual

        if not self.encoded:
            for i in range(self.num_unique_value_combinations):
                value_combination = uniqueValueCombinations[i].split('|')
                for j in range(self.num_attrs):
                    self.dataBitVec[self.bitarray_index(value_combination[j], j)][i] = 1
        else:
            # fill bitarrays
            for i in range(self.num_unique_value_combinations):
                for j in range(self.num_attrs):
                    self.dataBitVec[self.bitarray_index(uniqueValueCombinations[i][j], j)][i] = 1


    def pattern_count(self, pattern):
        '''
        This is the function directly called by the search algorithm
        pattern is of the format: 0X100X, etc
        return the coverge/count of a pattern
        '''
        if self.empty:
            print("PatternCount original set is empty, so return count 0")
            return 0
        if not self.encoded: # dataset contains non-processed strings (i.e. not in the format of OXXX, etc)
            and_bitarray = bitarray(self.num_unique_value_combinations)
            and_bitarray.setall(1)
            attr_values = pattern.split('|')
            for i in range(self.num_attrs):
                if attr_values[i] != '': # this attribute value is deterministic, not X in encoded case
                    if attr_values[i] not in self.attr_value_map[i]: # if this attribute value is not shown/mentioned in a dataset, its count should be 0 (testcase 3)
                        return 0
                    and_bitarray = and_bitarray & self.dataBitVec[self.bitarray_index(attr_values[i], i)]

            count = 0
            for i, bit in enumerate(and_bitarray):
                if bit:
                    count += self.occurences[i]
            return count
        else:
            and_bitarray = bitarray(self.num_unique_value_combinations)
            and_bitarray.setall(1)
            for i in range(self.num_attrs):
                if pattern[i] != 'X':
                    and_bitarray = and_bitarray & self.dataBitVec[self.bitarray_index(pattern[i], i)]

            count = 0
            for i, bit in enumerate(and_bitarray):
                if bit:
                    count += self.occurences[i]
            return count


    def char_index(self, character):
        '''
        This is a private function used to convert char to index
        '''
        return self.char_list.index(character)

    def bitarray_index(self, attr_value, attr_index):
        '''
        can also dict to access bitarray; use attribute index and attribute value as keys
        attr_value: value of an attribute. For character encoded data, attribute value is a single character.
            For non-encoded data, attribute value is the orignal value
        attr_index: index of the attribute in all attributes
        '''
        if not self.encoded:
            return self.attr_value_map[attr_index][attr_value] + sum(self.cardinalities[:attr_index])
        return self.char_index(attr_value) + sum(self.cardinalities[:attr_index])


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st



def main():
    '''
    Test case to check for pattern count correctness
    '''
    file1 = '../InputData/test_data.txt'
    file1_data = pd.read_csv(file1, delimiter=', *', engine='python')
    pc = PatternCounter(file1_data)
    pc.parse_data()
    print(pc.pattern_count('2XXX')) # 6
    print(pc.pattern_count('XXXX')) # 13
    print(pc.pattern_count('2001')) # 4
    print(pc.pattern_count('X1XX')) # 3
    print(pc.pattern_count('X0XX')) # 7
    print(pc.pattern_count('X10X')) # 0
    print(pc.pattern_count('21XX')) # 0
    print(pc.pattern_count('X1X0')) # 2

    file2 = '../InputData/test_data2.txt'
    file2_data = pd.read_csv(file2, delimiter=', *', engine='python')
    pc2 = PatternCounter(file2_data, ['col1', 'col2', 'col3', 'col4'], encoded=False)
    pc2.parse_data()
    print(pc2.pattern_count('hehe|||')) # 6
    print(pc2.pattern_count('|||')) # 13
    print(pc2.pattern_count('hehe|haha|0|Yifan Guan')) # 4
    print(pc2.pattern_count('|123||')) # 3
    print(pc2.pattern_count('|haha||')) # 7
    print(pc2.pattern_count('|123|0|')) # 0
    print(pc2.pattern_count('hehe|123||')) # 0
    print(pc2.pattern_count('|123||Yifan')) # 2


    """
    pc3 = PatternCounter('../InputData/test_data3.txt', ['age', 'workclass', 'education', 'educational-num'], encoded=False)
    pc3.parse_data()
    print(pc3.pattern_count('|||37')) # 1
    print(pc3.pattern_count('|||33')) # 0
    """

    return 0


if __name__ == '__main__':
    main()