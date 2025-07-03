from collections       import namedtuple
from re                import compile, search
from utility           import is_ignored_line, flags_from_descs_str, cross_flags, get_flags_list_intersection, OrderedDefaultDict, flags_match_a_in_b_lists
from Flags             import Flags
from sys               import exc_info
from copy              import deepcopy
from lwstomTest_layer  import testToLayer

layer_pat = compile('\s*"(.*)"\s*=\s*(.*?)\s*$')
    
Layer      = namedtuple("Layer",    "base_name split_name split_diff_flags test_name test_diff_flags flags")

def get_layers_from_str(layer_file_content, label_db, file_name):
    lines = layer_file_content.splitlines()

    layers_dict = OrderedDefaultDict(list)
    
    for (line_index, line) in enumerate(lines):
        # Ignore commented or empty lines
        if is_ignored_line(line):
            continue
        
        try:
            match_line = layer_pat.match(line)
            
            if match_line == None:
                raise Exception("Invalid layer description given")
            
            (layer_name, layer_descs) = match_line.groups()
            
            layer_descs_flags = flags_from_descs_str(layer_descs, label_db)
            
            layers_dict[layer_name] = layer_descs_flags

        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()

            # Re-raise exception with line info
            raise t, Exception("[LAYER PARSING] %s (at %s:%s)" % (e.message, file_name, line_index+1)), tb
            
    return layers_dict
    
def get_layers_from_file(layer_file_name, whitelist_flags_list, whitelist_layer_name, global_flags_list, split_flag_keys, label_db, lwstom_test):
    if split_flag_keys == None:
        split_flag_keys = []
    
    if lwstom_test:
        layer_read = testToLayer(lwstom_test)
        layers_dict = get_layers_from_str(layer_read, label_db, layer_file_name)
    else:
        with open(layer_file_name) as layer_file:
            layers_dict = get_layers_from_str(layer_file.read(), label_db, layer_file_name)
        
    scrubbed_layers_dict = OrderedDefaultDict(list)
    
    for layer_name in layers_dict:
        flags_list = layers_dict[layer_name]
        
        for flags in cross_flags(flags_list, global_flags_list):
            for sub_flag in flags.get_sub_flags():

                # Skip if whitelist_flags do not match
                if not flags_match_a_in_b_lists(whitelist_flags_list, [sub_flag]):
                    continue
                    
                # Skip if whitelist layer name does not match
                if search(whitelist_layer_name, layer_name) == None:
                    continue
                    
                scrubbed_layers_dict[layer_name].append(sub_flag)
                
    return resolve_layers(scrubbed_layers_dict, split_flag_keys)
    
def resolve_layers(layers_dict, split_flag_keys):
    layers = []
    
    for layer_name in layers_dict:
        flags_list = layers_dict[layer_name]
        
        intersecting_flag_keys = list(get_flags_list_intersection(flags_list))
        
        split_flags = Flags()
        
        for split_flag_key in split_flag_keys:
            if split_flag_key in intersecting_flag_keys:
                split_flags[split_flag_key] = ('', )
        
        unique_flags = Flags()
        
        for unique_key in intersecting_flag_keys:
            if not (unique_key in split_flag_keys):
                unique_flags[unique_key] = ('', )
            
        for flags in flags_list:
            split_diff_flags = flags - (flags - split_flags)
            test_diff_flags  = flags - (flags - unique_flags)
            
            split_name = layer_name + split_diff_flags.get_str(prefix='_', delimiter='')
            test_name  = split_name + test_diff_flags.get_str(prefix='_', delimiter='')

            layers.append(Layer(layer_name, split_name, split_diff_flags, test_name, test_diff_flags, flags))
            
    return layers
        
    
if __name__ == "__main__":
    test_str = '''"layer_name1" = n: 1
    
                  "layer_name2" = c: 2 * d: 4 * label_import'''
                  
    label_db = {"label_import": [Flags()]}
    
    print get_layers_from_str(test_str, label_db, "test_layer_name")
    
