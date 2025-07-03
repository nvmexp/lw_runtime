from csv      import reader
from StringIO import StringIO
from re       import compile
from Flags    import Flags

# Regex name match (capture a range of ASCII values)
re_name_match = "[\x21-\x7E]+"

empty_pat   = compile('\s*$')
comment_pat = compile('\s*//.*$')

def is_empty_line(line):
    return empty_pat.match(line) != None
    
def is_comment_line(line):
    return comment_pat.match(line) != None
    
def is_ignored_line(line):
    if is_empty_line(line) or is_comment_line(line):
        return True
        
    return False
   
def split_and_strip(string, split_by):
    return [val.strip() for val in string.split(split_by)]
    
def strip_exclude_quotes(string):
    result = ""
    
    in_quotes = False
    
    for char_idx, char in enumerate(string):
        if char == '"' and (char_idx == 0 or string[char_idx-1] != '\\'):
            in_quotes = not in_quotes
        
        if not char.isspace() or in_quotes:
            result += char
            
    return result
        
    
def split_comma(string):
    if string == None:
        return []
        
    stripped_string = strip_exclude_quotes(string)
    
    stripped_split = next(reader(StringIO(stripped_string)), [''])

    result = [val for val in stripped_split]

    return result

def split_space(string):
    if string == None:
        return []

    unstripped_result = next(reader(StringIO(string), delimiter=' '), [''])

    result = [val.strip() for val in unstripped_result if val.strip() != ""]

    return result

def get_shell_list(flags):
    return split_space(str(flags))
    
def cross_flags(flags_bases, flags_overrides):
    result = []
    
    for flags_base in flags_bases:
        for flags_override in flags_overrides:
            result.append(flags_base + flags_override)
            
    return result

def flags_from_descs_str(string, label_db):
    if string == None:
        return [Flags()]
    
    # Case is assumed to be descriptors (and error if not)
    split_by_bar = split_and_strip(string, '*')
    
    flags = [Flags()]
    
    for val in split_by_bar:
        if ':' in val:
            (desc_key, desc_val) = split_and_strip(val, ':')
            
            for flags_idx in range(len(flags)):
                flags[flags_idx][desc_key] = tuple(split_comma(desc_val))
            
        elif is_empty_line(val):
            # Ignore empty areas between bars
            pass
            
        else:
            cross_label_name = val
            
            if not (cross_label_name in label_db):
                raise Exception("Label \"%s\" not found" % cross_label_name)
                
            flags = cross_flags(flags, label_db[cross_label_name])
        
    return flags


def flags_match_a_in_b(flags_a, flags_b):
    if flags_a.next_multi_flag():
        raise Exception("Error matching multi-flag a:" + repr(flags_a))
        
    if flags_b.next_multi_flag():
        raise Exception("Error matching multi-flag b:" + repr(flags_b))
        
    if flags_a.key_count() == 0:
        return True
        
    for flag_key in flags_a:
        if not (flag_key in flags_b):
            return False
            
        if flags_a[flag_key] != flags_b[flag_key]:
            return False
    
    return True

def flags_match_a_in_b_lists(flags_list_a, flags_list_b):
    
    for flags_a in flags_list_a:
        for flags_b in flags_list_b:
            for sub_flags_a in flags_a.get_sub_flags():
                for sub_flags_b in flags_b.get_sub_flags():
                    if flags_match_a_in_b(sub_flags_a, sub_flags_b):
                        return True
                        
                        
    return False


def get_flags_intersection(flags_a, flags_b):
    intersecting_flags = set()
    
    for flag in flags_a:
        if not (flag in flags_b) or flags_a[flag] != flags_b[flag]:
            intersecting_flags.add(flag)
            
    for flag in flags_b:
        if not (flag in flags_a) or flags_b[flag] != flags_a[flag]:
            intersecting_flags.add(flag)
            
    return intersecting_flags

def get_flags_list_intersection(flags_list):
    
    if flags_list == None:
        raise Exception("None is not a valid flags list")
        
    if len(flags_list) == 0:
        raise Exception("Empty flags list given")
    
    intersecting_flags = set()
    
    base_flags = flags_list[0]
    
    for flags in flags_list[1:]:
        intersecting_flags = intersecting_flags.union(get_flags_intersection(base_flags, flags))
        
    return intersecting_flags
    
    
from collections import OrderedDict, defaultdict

class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory
