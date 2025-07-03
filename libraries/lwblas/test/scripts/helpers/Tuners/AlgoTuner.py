# AlgoTuner is responsible for heuristics based on algo chosen
# 
from DecisionTreeGen.DecisionTreeGenerator import DecisionTreeGenerator
import bisect

# Removes backdoor from string of flags
def scrub_algo(flags):
    resized_flags = [flag[:5] for flag in flags]

    algo_idx = resized_flags.index('-algo')

    if algo_idx == -1:
        raise Exception("[ALGO TUNING] Unable to find -algo in %s" % str(flags))

    return flags[:algo_idx] + flags[algo_idx+1:]

def get_default(val, default):
    if(val == '' or val == None):
        return default

    return val

def get_param_vals(run):
    case = {}

    case["n"] = int(run.parsed.srcDesc.dimA_0)
    case["c"] = int(run.parsed.srcDesc.dimA_1)
    case["h"] = int(run.parsed.srcDesc.dimA_2)
    case["w"] = int(run.parsed.srcDesc.dimA_3)
    case["d"] = int(get_default(run.parsed.srcDesc.dimA_4, '1'))

    case["k"] = int(run.parsed.filterDesc.dimA_0)
    case["r"] = int(run.parsed.filterDesc.dimA_2)
    case["s"] = int(run.parsed.filterDesc.dimA_3)
    case["t"] = int(get_default(run.parsed.filterDesc.dimA_4, '1'))

    case["u"] = int(run.parsed.colwDesc.strideA_0)
    case["v"] = int(run.parsed.colwDesc.strideA_1)
    case["x"] = int(get_default(run.parsed.colwDesc.strideA_2, '1'))

    case["pad_h"] = int(run.parsed.colwDesc.padA_0)
    case["pad_w"] = int(run.parsed.colwDesc.padA_1)
    case["pad_d"] = int(get_default(run.parsed.colwDesc.padA_2, '0'))

    case["dil_h"] = int(run.parsed.colwDesc.dilationA_0)
    case["dil_w"] = int(run.parsed.colwDesc.dilationA_1)
    case["dil_d"] = int(get_default(run.parsed.colwDesc.dilationA_2, '1'))

    return case

class AlgoTuner:
    def __init__(self, name, mapping, enum_type):
        # Initialize user-provided values
        self.name      = name
        self.mapping   = mapping
        self.enum_type = enum_type

        # Initialize class values
        self.flags_run  = {}
        self.tree       = None
        self.prev_valid = False
        self.params     = ["n", "c", "h", "w", "d", "k", "r", "s", "t", "u", "v", "x", "pad_h", "pad_w", "pad_d", "dil_h", "dil_w", "dil_d"]

    def add_run(self, run):
        scrubbed_flags = tuple(scrub_algo(run.flags))

        # Check that our case is valid (parsed correctly)
        if(run == None or run.parsed == None):
            return False

        # Check that our case is valid (parsed time correctly)
        if(run.parsed.time == None or run.parsed.time.seconds == None):
            return False

        # Check that our case is valid (parsed algo correctly)
        if(run.parsed.algo == None or run.parsed.algo.choice == None):
            return False

        if( not(scrubbed_flags in self.flags_run) ):
            self.flags_run[scrubbed_flags] = []

        prev_flags = [prev_run.flags for prev_run in self.flags_run[scrubbed_flags]]
        
        # Ignore duplicate flags
        if run.flags in prev_flags:
            return True

        prev_times = [float(prev_run.parsed.time.seconds) for prev_run in self.flags_run[scrubbed_flags]]

        insert_index = bisect.bisect_left(prev_times, float(run.parsed.time.seconds))

        self.flags_run[scrubbed_flags].insert(insert_index, run)

        return True
        
    def get_algos_mapped(self, algos):
        # Get algo choices ('0' for algo0, ...)        
        unmapped_algos = [i.parsed.algo.choice for i in algos]

        # Map algo choice to enum ('0' to LWDNN_COLW_..._FWD_IMPLICIT_GEMM, ...)
        mapped_algos = [self.mapping[i] for i in unmapped_algos]

        # Add any remaining mappings that aren't found
        for algo_map in self.mapping:
            if(algo_map not in unmapped_algos):
                mapped_algos.append(self.mapping[algo_map])

        return tuple(mapped_algos)

    def generate_heuristic(self):
        # In case this is an orphan tuner
        if(self.name != None):
            # Initialize list of cases to empty
            cases = []

            # Add each best case to cases
            for flags in self.flags_run:

                case  = get_param_vals(self.flags_run[flags][0])

                case["class"] = self.get_algos_mapped(self.flags_run[flags])

                cases.append(case)
    
            # Generate classifier
            tree_gen = DecisionTreeGenerator(cases)

            self.tree = tree_gen.gen_tree()
            
            self.prev_valid = True

    def get_class_list(self):
        if(self.name != None):
            result = "//\n//\n"

            # If we have changed the previous heuristic, update it
            if(not self.prev_valid):
                self.generate_heuristic()

            # Get best time for each cane
            class_dict = {}

            # Add each best case to cases
            for flags in self.flags_run:
                params  = get_param_vals(self.flags_run[flags][0])

                mapped_algos = self.get_algos_mapped(self.flags_run[flags])

                if mapped_algos not in class_dict:
                    class_dict[mapped_algos] = [{"flags": flags, "params": params}]
                else:
                    class_dict[mapped_algos].append({"flags": flags, "params": params})


            # Output all best classes
            for class_val in class_dict:
                cases_in_class = class_dict[class_val]
        
                # Output current case number and count ex: "// 2 (213)"
                result += "// %s (%d)\n" % (class_val, len(cases_in_class))

                # Get spacing (based on maximum possible string)
                spacer = max([len(case["flags"]) for case in cases_in_class])

                # Get spacing for classifications (based on maximum possible string)
                spacer_class = max([len(str(case["params"])) for case in cases_in_class])

                # Loop through each case in this class
                for case in class_dict[class_val]:

                    flags = " ".join(case["flags"])

                    # Get classification value (from generated heuristic)
                    classified = self.classify(case["params"])

                    # Get spacer; (max-current) number of spaces
                    space_diff_params = ' '*(spacer-len(flags))

                    space_diff_class  = ' '*(spacer_class-len(str(case)))

                    # Assume no warning by default
                    warning = ""

                    # Output warning if classification is incorrect
                    if(class_val != classified):
                        warning = " -> %s" % str(classified)

                    # Output final message ex: -b -n1 -c3 -k96 -h2 -w2 [20, 30, 40, 0] -> 2
                    result += "//     %s%s %s%s%s\n" % (flags, space_diff_params, str(case["params"]), space_diff_class, warning)

                result += "//\n//\n"

            return result

        return None

    def classify(self, case):
        # If heuristic is no longer valid, update it
        if( not self.prev_valid ):
            self.generate_heuristic()

        # Return classification value
        return self.tree.get_class(case) 

    def output_heuristic(self, path):
        # Orphan tuners shouldn't run heuristics
        if(self.name != None):
            # If previous heuristic is not valid, remake it
            if(not self.prev_valid):
                self.generate_heuristic()

            # Some info to add to the output
            HEADER_TAIL = "/////////////////////////////////////////////////////////////////\n"

            dont_edit = "// DO NOT EDIT THIS FILE!\n"

            auto_gen = "// It was automatically generated by heurcheck (heurgen.py)\n"

            func_signature = ""

            for param in self.params:
                func_signature += "int " + str(param) + ", "

            func_header = "static void %s(%s%s *result){\n" % (self.name, func_signature, self.enum_type);

            out_file = open("%s/%s.h" % (path, self.name), 'w')
            out_file.write(HEADER_TAIL)
            out_file.write(dont_edit)
            out_file.write(auto_gen)
            out_file.write(self.get_class_list())
            out_file.write(HEADER_TAIL)
            out_file.write("\n\n");

            out_file.write(func_header)
            out_file.write(self.tree.get_str())
            out_file.write("\n}\n")
            out_file.close()
