# This module defines the class "Flags"
# See bottom __main__() function for explanations of how to use the class.

from collections import deque
from copy        import deepcopy

class Flags:
    def __init__(self, flags=None, key_order=None):
        # Set values to given or default to empty if none given (list or dictionary)
        self.flags_dict  = flags if flags else {}
        self.key_order   = key_order if key_order else []
        
    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise Exception("Following key is not a string:", key)
            
        if not isinstance(value, tuple):
            raise Exception("Following value is not a tuple:", value)
            
        if len(value) == 0:
            raise Exception("Empty list is not allowed for value:",value)
            
        for tup_elem in value:
            if not isinstance(tup_elem, basestring):
                raise Exception("Given value contains non-string element:", list_elem)
                
        self.flags_dict[key] = value

        if not (key in self.key_order):
            self.key_order.append(key)
        
    def __getitem__(self, key):
        if not isinstance(key, basestring):
            raise Exception("Following key is not a string:", key)
            
        if key in self.flags_dict:
            return self.flags_dict[key]
        
        raise Exception("Unable to find key %s in %s" % (key, str(self.flags_dict)))
        
    def __iter__(self):
        return iter(self.key_order)
        
    def __contains__(self, key):
        return key in self.flags_dict

    def next_multi_flag(self):
        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                return key
                
        return None
        
    def get_multi_keys(self):
        multi_keys = []
        
        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                multi_keys.append(key)
        
        return multi_keys
        
    def key_count(self):
        return len(self.key_order)
        
    def is_multiple(self):
        if self.next_multi_flag():
            return True
            
        return False

    def get_sub_flags(self):
        result_flags = []
        
        to_parse = deque()
        
        to_parse.append(self)
        
        while len(to_parse) > 0:
            lwr_elem = to_parse.pop()
            
            next_multi = lwr_elem.next_multi_flag()
            
            if next_multi:
                for flag_val in lwr_elem.flags_dict[next_multi]:
                    new_flags = deepcopy(lwr_elem.flags_dict)
            
                    new_flags[next_multi] = (flag_val, )
                    
                    to_parse.append( Flags(new_flags, self.key_order) )
                    
            else:
                result_flags.append(lwr_elem)
        
        return reversed(result_flags)

    def get_flags_for_keys(self, keys):
        result = Flags()

        for key in self.key_order:
            if key in keys:
                result[key] = self[key]

        return result

    def get_multi_flags(self):
        result = []

        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                result.append(key)

        return result

    def get_str(self, prefix='-', delimiter=' '):
        if self.next_multi_flag():
            raise Exception("Cannot obtain string of key %s in flags: %s" % (self.next_multi_flag(), str(self.flags_dict)))
            
        return ("%s" % delimiter).join(["%s%s%s" % (prefix,key,self.flags_dict[key][0]) for key in self.key_order])
        
    def __str__(self):
        return self.get_str()
        
    def __repr__(self):
        return "Flags(%s, %s)" % (str(self.flags_dict), str(self.key_order))
    
    def __add__(self, other):
        result = deepcopy(self)
        
        for other_flag in other:
            result[other_flag] = other[other_flag]
            
        return result

    def __sub__(self, other):
        result = Flags()

        for flag in self:
            if not (flag in other):
                result[flag] = self[flag]

        return result
        
def match_str(a, b):
    if a != b:
        print("\t[ERROR] Mismatch of %s and %s" % (a, b))
        return False
        
    print("\t[SUCCESS] Correct match of %s" % a)
    return True    
    
if __name__ == "__main__":
    
    mismatch_count = 0
    
    ##########################################################
    ## SINGLE TEST
    ##########################################################
    print("=== SINGLE TEST===")
    
    flags = Flags()
    
    flags["R"] = ("colw",)
    flags["R"] = ("dgrad",)

    flags["n"] = ("128",)
    flags["n"] = ("64",)
    
    flags["c"] = ("128",)
    
    golden_str = "-Rdgrad -n64 -c128"

    if not match_str(str(flags), golden_str):
        mismatch_count += 1
    
    ##########################################################
    ## MULTI TEST
    ##########################################################
    print("\n\n=== MULTI TEST===")
    
    flags = Flags()
    flags = Flags()
    
    flags["R"] = ("colw","dgrad")
    flags["n"] = tuple([str(i) for i in [1, 2]])
    
    golden_strings = ["-Rcolw -n1", "-Rcolw -n2", "-Rdgrad -n1", "-Rdgrad -n2"]

    for (sub_flag, golden_str) in zip(flags.get_sub_flags(), golden_strings):
        if not match_str(str(sub_flag), golden_str):
            mismatch_count += 1
        
        
    ##########################################################
    ## ERROR CASE: SET INT KEY
    ##########################################################
    print("\n\n=== SET INT KEY ===")
    
    flags = Flags()
    
    try:
        flags[0] = "blank"
        mismatch_count += 1
        print("[ERROR] Integer key should not be settable")
    except:
        print("[SUCCESS] Integer key is not settable")
        
    ##########################################################
    ## ERROR CASE: SET EMPTY TUPLE
    ##########################################################
    print("\n\n=== SET EMPTY TUPLE ===")
    
    flags = Flags()
    
    try:
        flags["blank"] = ()
        mismatch_count += 1
        print("[ERROR] Empty tuple value should not be settable")
    except:
        print("[SUCCESS] Empty tuple value is not settable")
        
    ##########################################################
    ## ERROR CASE: SET INT VALUE
    ##########################################################
    print("\n\n=== SET INT VALUE ===")
    
    flags = Flags()
    
    try:
        flags["blank"] = 0
        mismatch_count += 1
        print("[ERROR] Integer value should not be settable")
    except:
        print("[SUCCESS] Integer value is not settable")
        
    ##########################################################
    ## ERROR CASE: SET VALUE TO TUPLE OF INTS
    ##########################################################
    print("\n\n=== SET VALUE TO TUPLE OF INTS ===")
    
    flags = Flags()
    
    try:
        flags["blank"] = (0, 1)
        mismatch_count += 1
        print("[ERROR] Tuple of ints value should not be settable")
    except:
        print("[SUCCESS] Tuple of ints value is not settable")
        
    ##########################################################
    ## ERROR CASE: GET INT KEY
    ##########################################################
    print("\n\n=== GET INT KEY ===")
    
    flags = Flags()
    
    try:
        flags[0] = ("blank", )
        test_get = flags[0]
        mismatch_count += 1
        print("[ERROR] Int key should not be gettable")
    except:
        print("[SUCCESS] Int key is not gettable")


    print("\n\nERROR COUNT: %d" % mismatch_count)
