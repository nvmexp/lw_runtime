# A DecisionTreeGenerator is able to generate a DecisionTree by using given cases by the user



# There are two primary purposes:

#   1. add_case to add cases to be generated in decision tree

#   2. gen_tree to generate a DecisionTree that classifies each given case



import DecisionTree

import DecisionNode



def check_valid(a):

    # Must have "choice" parameter

    if(not("choice" in a)):

        raise KeyError

        

    # "choice" parameter must be string

    if(not(isinstance(a["choice"], str) or isinstance(a["choice"], tuple))):

        raise TypeError

        

    # Loop through all keys but "choice"

    for key in a["choosers"]:

        # Error if not int or float

        if(not(isinstance(a["choosers"][key], int) or isinstance(a["choosers"][key], float))):

            raise TypeError

            

    # Must have at least 1 chooser

    if(len(a["choosers"]) < 1):

        raise KeyError

            

def check_same(a, b):

    # Get keys in set form

    a_keys = set(a)

    b_keys = set(b)

    

    # Check matching keys

    if(a_keys != b_keys):

        raise KeyError

    

    # Check same type

    if( type(a["choice"]) != type(b["choice"]) ):

        raise TypeError



    # Check same length (if list)

    if(isinstance(a["choice"], tuple) and (len(a["choice"]) != len(b["choice"]))):

        print a

        print b

        raise ValueError

        

class DecisionTreeGenerator:

    def __init__(self, cases=[]):

        # Initialize cases to given (or empty list)

        self.cases = []

        

        # Add each of the given user cases

        for case in cases:

            self.add_case(case)

       

    def add_case(self, case):

        # Case must be valid

        check_valid(case)

            

        # If we have an existing case

        if(len(self.cases) > 0):

            # Ensure this new case matches existing precedent

            check_same(case, self.cases[0])

        

        # Add case to cases

        self.cases.append(case)

        return True

        

    def get_params(self):

        # Early exit if we don't have any cases

        if(len(self.cases) == 0):

            raise ValueError

      

        # Get params for first case (used for reference)

        params = [key for key in self.cases[0]["choosers"]]

            

        return params

        

    def get_multi_val(self):

        # Early exit if we don't have any cases

        if(len(self.cases) == 0):

            raise ValueError

           

        # If a list, return list count

        if(isinstance(self.cases[0]["choice"], tuple)):

            return len(self.cases[0]["choice"])

        # If not a list, return None

        else:

            return None

            

        # Something wrong with system if this happens

        raise SystemError

        

    def gen_tree(self):

        # Get params for cases

        param_names = self.get_params()



        # Determine if there are multiple values

        multi_count = self.get_multi_val()



        # Construct base tree

        tree = DecisionTree.DecisionTree(param_names, multi_count)

        

        # Construct nodes (relwrsively)

        tree.root = DecisionNode.gen_node(param_names, self.cases)



        return tree

