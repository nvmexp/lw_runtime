from DecisionTreeGenerator import DecisionTreeGenerator



cases = [{"a": 30, "b": 2, "class": "Apple"}, {"a": 30, "b": 3, "class": "Apple"}, {"a": 3, "b": 4, "class": "False"}, {"a": 20, "b": 15, "class": "True"}, {"a": 50, "b": 10, "class": "False"}]



TreeGen = DecisionTreeGenerator(cases)



tree = TreeGen.gen_tree()



for case in cases:

    print(tree.get_class(case))



print(tree.get_str("Function", "lwdnnColwolutionFwdAlgo_t"))

