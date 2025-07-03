from collections import namedtuple
from re          import compile
from utility     import is_ignored_line, split_and_strip, flags_from_descs_str
from Flags       import Flags
from sys         import exc_info

label_name  = compile('\s*<(.*?)(?::(.*))?>\s*$')

def get_label_name(line):
    name_match = label_name.match(line)
    
    if name_match:
        return name_match.groups()[0]
        
    return None
           
def is_label_valid(label_filters_str, filter_db):
    # Empty configs are always valid
    if label_filters_str == None:
        return True
    
    is_included = False
    is_excluded = False
    
    any_include = False
    
    for label_filter in split_and_strip(label_filters_str, ','):
        # Ignore empty labels
        if len(label_filter) == 0:
            continue
            
        label_filter_is_include = label_filter[0] != '!'
        
        label_filter_name = label_filter if label_filter_is_include else label_filter[1:]
        
        if label_filter_is_include:
            any_include = True
            
        if label_filter_is_include and label_filter_name in filter_db:
            is_included = True
                
        if not label_filter_is_include and label_filter_name in filter_db:
            is_excluded = True
        
    return (not is_excluded) and (not any_include or is_included)
    
def get_labels_from_str(label_file_content, config_db, file_name):
    labels = {}
    
    labels["exclude"] = [Flags()]
    labels["exclude"][0]["*exclude"] = ("",)
    
    lines = label_file_content.splitlines()
    
    lwr_label_name = None
    lwr_label_valid = None
    
    for (line_index, line) in enumerate(lines):
        # Ignore commented or empty lines
        if is_ignored_line(line):
            continue
        
        try:
            # Handle case of label name
            if label_name.match(line):
                (lwr_label_name, lwr_label_filters) = label_name.match(line).groups()
                
                if is_label_valid(lwr_label_filters, config_db):
                    lwr_label_valid = True
                        
                else:
                    lwr_label_valid = False
                
                # Error out if it already exists
                if(label_name in labels):
                    raise Exception("Label \"%s\" is already defined" % label_name)

                # Initialize label generator
                labels[lwr_label_name] = []
                
            # Handle case of flag descriptors/imports
            else:
                for new_flag in flags_from_descs_str(line, labels):
                    if lwr_label_valid:
                        labels[lwr_label_name].append(new_flag)
                    
        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()

            # Re-raise exception with line info
            raise t, Exception("[LABEL PARSING] %s (at %s:%s)" % (e.message, file_name, line_index+1)), tb
            
    return labels
    
def get_labels_from_file(label_file_name, config_db):
    with open(label_file_name) as label_file:
        return get_labels_from_str(label_file.read(), config_db, label_file_name)

if __name__ == "__main__":
    test_str = '''
                   <label_import>
                   a: 2
                   
                   <label_pre:filter_one>
                   b: 2 | label_import
                   c: 2 | label_import
                   
                   <label_post>
                   d: 2 | label_pre'''
                  
    print get_labels_from_str(test_str, "test_file_name")
