# A DecisionTree is a Tree constructed with DecisionNodes

#   All branch nodes are decision nodes

#   All leaf nodes are value nodes

# There are two primary purposes:

#   1. Classify a given case to leaf node value

#   2. Print out DecisionTree using C syntax

import DecisionNode



class DecisionTree:

    def __init__(self, d_params, multi_value):

        # Assume no children to begin with

        self.root = None



        # Save decision parameters to self

        self.d_params = d_params

        

        # Save multi_value choice to self

        #   Determines if this tree's leaf nodes have multiple values

        self.multi_value = multi_value



    # If this is a decision node, returns next node

    def get_class(self, case):

        # Early exit if tree is empty

        if(self.root == None):

            raise ValueError

        

        # Start iteration at root

        lwr_node = self.root



        # While in decision node

        while(lwr_node.is_decision()):

            # Decide on next node

            lwr_node = lwr_node.get_next(case)

            

        # Return value once at leaf node

        return lwr_node.value





    def get_str(self, name="Function", out_name=None):

        # Start exploration at root

        to_explore = [(self.root, 1)]



        # Assume empty string to begin with

        result = ""





        # -- Write out function code --

        # While there are nodes to explore

        while(len(to_explore) > 0):

            # Get node and tab level

            node  = to_explore[0][0]

            tab   = to_explore[0][1]



            # Construct tab string (tab number of tab characters)

            tab_str = ("    " * tab)



            # Delete this node (it will be explored soon)

            del to_explore[0]



            # Just print out any strings we get

            if(isinstance(node, str)):

                result += "%s%s\n" % (tab_str, node)



            # If not a string, it should be a node

            else:

                # Construct insertion array (to be inserted at beginning of exploration)

                to_insert = []



                # For decision nodes

                if(node.is_decision()):

                    # Insert left hand if condition

                    to_insert.append(("if(%s <= %s){" % (str(node.d_param), str(node.value)), tab))



                    # Explore left node (will be done after printing if condition)

                    to_insert.append((node.left, tab+1))



                    # Insert closing bracket

                    to_insert.append(("}", tab))



                    # Insert right hand if condition

                    to_insert.append(("if(%s > %s){" % (str(node.d_param), str(node.value)), tab))

                    

                    # Explore right node (will be done after printing if condition)

                    to_insert.append((node.right, tab+1))



                    # Insert closing bracket

                    to_insert.append(("}", tab))



                else:

                    # If multi-value node

                    if(self.multi_value):

                        # Show number of elements this class has in train set

                        to_insert.append(("// %d elements" % len(node.cases), tab))

                        

                        to_insert.append(("\t// Case Indices: [%s]" % ",".join([str(case["index"]) for case in node.cases]), tab))



                        is_visited = {}

                        

                        mod_value_index = 0

                        

                        # Print each output value

                        for value_index in range(len(node.value)):

                            value_int = int(node.value[value_index])

                            

                            value_int_mod = value_int % 1000

                            

                            if not value_int_mod in is_visited:

                                to_insert.append(("result[%d] = %s;" % (mod_value_index, node.value[value_index]), tab))

                                mod_value_index += 1

                                is_visited[value_int_mod] = mod_value_index

                            

                        # Early exit (we have filled up result array)

                        to_insert.append(("return;", tab))

                    else:

                        # On a single-value node, simply return result

                        to_insert.append(("return %s; // %d elements" % (node.value, len(node.cases)), tab))



                # Insert into exploration (at beginning)

                to_explore[0:0] = to_insert



        return result

