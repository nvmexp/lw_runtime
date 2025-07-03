# A DecisionNode can either be a decision node or a value node

# - Decision Node: Switches to left/right

# -     If case[d_param] <= value, go left; else go right

# - Value Node: Just contains a value (it is a leaf node)

# -     d_param should be NULL; this node should only be used for value



import math

import operator



class DecisionNode:

    def __init__(self, d_param, value, cases):

        # Assume no children to begin with

        self.left = None

        self.right = None



        # Save decision parameter to self

        self.d_param = d_param



        # Save value to self

        self.value = value



        # Save cases to self

        self.cases = cases



    # If this is a decision node, returns next node; returns None otherwise

    def get_next(self, case):

        if(self.is_decision()):

            if(case[self.d_param] <= self.value):

                return self.left

            else:

                return self.right

        

        raise ValueError



    # Returns true if this node is a decision node

    def is_decision(self):

        if(self.d_param == None):

            return False

            

        return True



def sort_uniq(l):

    return sorted(list(set(l)))

    

def calc_entropy(stats):

    return -1 * sum([(math.log(stat, 2) * stat) for stat in stats if stat != 0])



def gen_node(chooser_names, cases):

    # We need cases in order to split...

    if(len(cases) == 0):

        raise ValueError



    # Get all parameter options

    chooser_options = [ sort_uniq([val["choosers"][chooser] for val in cases]) for chooser in chooser_names ]

    

    # Get all class options

    class_options = sort_uniq([val["choice"] for val in cases])



    # If there is only class, return a Value node of that class

    if(len(class_options) == 1):

        return DecisionNode(None, cases[0]["choice"], cases)



    # If we don't have any parameter options...

    if(max( len(option) for option in chooser_options ) == 1):

        

        # Get all class counts

        class_counts = [ len([case for case in cases if case["choice"] == class_opt]) for class_opt in class_options ]

        

        # Get class with largest count

        max_class_index, max_class_count = max(enumerate(class_counts), key=operator.itemgetter(1))



        # Get name of best class

        best_class      = class_options[max_class_index]



        # Get percentage of this best class

        best_class_perc = float(100*max_class_count) / len(cases)



        # Print message of conflict to user

        print("Conflict between following cases: %s" % (cases))

        print("Resolving with class %s with %4.2f%% accuracy" % (best_class, best_class_perc))



        # Assume value node with most likely class

        return DecisionNode(None, best_class, cases)



    # Initiliaze best to None (we haven't seen any cases)

    best = None



    count = 0



    # Loop through each parameter

    for chooser_index in range(len(chooser_options)):



        # Get param name

        chooser = chooser_names[chooser_index]



        # Get param options

        options = chooser_options[chooser_index]

        

        # Loop through each option (value); exclude last because we split from current index to next index

        for option_index in range(len(options) - 1):



            # Split between current and next option

            split = (options[option_index] + options[option_index+1]) / 2

            

            # Get all cases in each side of the split

            group_a = [case for case in cases if case["choosers"][chooser] <= split]

            group_b = [case for case in cases if case["choosers"][chooser] > split]

            

            # Count number for each class in each side of split

            class_counts_a = [ len([case for case in group_a if case["choice"] == class_opt]) for class_opt in class_options ]

            class_counts_b = [ len([case for case in group_b if case["choice"] == class_opt]) for class_opt in class_options ]

            

            # Normalize to percentages

            percent_a = [ float(val) / sum(class_counts_a) for val in class_counts_a ]

            percent_b = [ float(val) / sum(class_counts_b) for val in class_counts_b ]



            # Get weights of each group            

            weight_a = float(len(group_a)) / len(cases)

            weight_b = float(len(group_b)) / len(cases)



            # Callwlate entropy from percentages

            entropy = weight_a * calc_entropy(percent_a) + weight_b * calc_entropy(percent_b)



            # Set this to best (if it is better or our first)

            if(best == None or best["entropy"] > entropy):

                best = {"entropy": entropy, "chooser": chooser, "split": split, "group_a": group_a, "group_b": group_b}



    # Create decision node with proper split

    result = DecisionNode(best["chooser"], best["split"], cases)



    # If entropy is very low (near 0), set up children to Value nodes (we know their value decisively)

    if(best["entropy"] < 1e-7):

        result.left = DecisionNode(None, best["group_a"][0]["choice"], best["group_a"])

        result.right = DecisionNode(None, best["group_b"][0]["choice"], best["group_b"])



    # If entropy is large, continue splitting groups

    else:

        result.left = gen_node(chooser_names, best["group_a"])

        result.right = gen_node(chooser_names, best["group_b"])



    return result

