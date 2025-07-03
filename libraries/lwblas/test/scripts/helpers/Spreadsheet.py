import collections # For namedtuple
import re          # For regular expressions
import os          # For system path manipulations

from collections import OrderedDict
from utility import split_comma, get_shell_list
from sys import exc_info
from lwdnn_interface import parse_algoSweep

PerfRun        = collections.namedtuple("PerfRun", "layer results")
       
def wrap_val(val, none_val, bound):
    
    if val == None:
        return none_val
    
    if val < 0:
        return bound + val
        
    return val
    
def get_range_from_slice(slice_val, bound):
    start = wrap_val(slice_val.start, 0, bound)
    
    stop = wrap_val(slice_val.stop, bound, bound)
        
    step = slice_val.step
    
    if step == None:
        step = 1
    elif step < 0:
        start, stop = stop, start
        
        step *= -1
        
    return range(start, stop, step)
    
def get_range(slice_or_idx, bound):
    if isinstance(slice_or_idx, slice):
        return get_range_from_slice(slice_or_idx, bound)
        
    if slice_or_idx == None:
        raise Exception("Index value is None?")
        
    return range(wrap_val(slice_or_idx, None, bound), wrap_val(slice_or_idx, None, bound) + 1, 1)
    
def get_empty_2d_array(size):
    if len(size) != 2:
        raise Exception("Cannot create empty matrix with size %s" % str(size))
        
    result = []
    
    for col_i in range(size[0]):
        result.append( [""] * size[1] )
        
    return result
            
class Table:
    def __init__(self, size):
        if len(size) != 2:
            raise Exception("Cannot construct table with size: %s" % str(size))
            
        self.col_count = size[0]
        self.row_count = size[1]
        
        self.cells = get_empty_2d_array(size)
        
    def __setitem__(self, pos, values):
        if len(pos) != 2:
            raise Exception("Invalid position in table: %s" % str(pos))
            
        (col, row) = pos
        
        try:
            if isinstance(col, slice) and isinstance(row, slice):
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                        self.cells[self_col_idx][self_row_idx] = values[val_col_idx][val_row_idx]
                        
            elif isinstance(col, slice):
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    self_row_idx = wrap_val(row, None, self.row_count)
                    
                    self.cells[self_col_idx][self_row_idx] = values[val_col_idx]
                    
            elif isinstance(row, slice):
                for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                    self_col_idx = wrap_val(col, None, self.col_count)
                    
                    self.cells[self_col_idx][self_row_idx] = values[val_row_idx]
                    
            else:
                self.cells[col][row] = values

                
        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()
            
            table_size = (self.col_count, self.row_count)
            
            # Re-raise exception with line info
            raise t, Exception("[SPREADSHEET] Error setting table position %s with size %s" % (str(pos), str(table_size))), tb
        
    def __getitem__(self, pos):
        if len(pos) != 2:
            raise Exception("Invalid position in table: %s" % str(pos))
            
        (col, row) = pos
        
        try:
            if isinstance(col, slice) and isinstance(row, slice):
                
                col_range = get_range_from_slice(col, self.col_count)
                row_range = get_range_from_slice(row, self.row_count)
                
                result = get_empty_2d_array(len([idx for idx in col_range]), len([idx for idx in row_range]))
                
                for (val_col_idx, self_col_idx) in enumerate(col_range):
                    for (val_row_idx, self_row_idx) in enumerate(row_range):
                        result[val_col_idx][val_row_idx] = self.cells[self_col_idx][self_row_idx]
                        
            elif isinstance(col, slice):
                result = []
                
                for (val_col_idx, self_col_idx) in enumerate(get_range_from_slice(col, self.col_count)):
                    self_row_idx = wrap_val(row, None, self.row_count)
                    
                    result.append(self.cells[self_col_idx][self_row_idx])
                    
                return result
                    
            elif isinstance(row, slice):
                result = []
                
                for (val_row_idx, self_row_idx) in enumerate(get_range_from_slice(row, self.row_count)):
                    self_col_idx = wrap_val(col, None, self.col_count)
                    
                    result.append(self.cells[self_col_idx][self_row_idx])
                    
                return result
            else:
                return self.cells[col][row]
            
        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()
            
            table_size = (self.col_count, self.row_count)
            
            # Re-raise exception with line info
            raise t, Exception("[SPREADSHEET] Error getting table position %s with size %s" % (str(pos), str(table_size))), tb

        return result
        
    def __str__(self):
        return "\n".join([",".join(self[:, row_idx]) for row_idx in range(self.row_count)])
        
def extract_result(run, extract_value):
    if(run.results == None):
        return ""

    if(run.results.parsed == None):
        return ""

    extracted = getattr(run.results.parsed, extract_value)

    if(extracted == None):
        return ""

    return getattr(extracted, extracted._fields[0])

class Spreadsheet:
    def __init__(self):
        # Initialize member variables
        self.perf_data   = []
        self.split_names = []
        
    def add_run(self, layer, flags, results):
        # Keep track of names added (in chronological order)
        if( not(layer.split_name in self.split_names) ):
            self.split_names.append(layer.split_name)

        # Add perf run data
        self.perf_data.append(PerfRun(layer, results))

    def generate(self, file_name, extract_value):
        if len(self.perf_data) == 0:
            raise Exception("[SPREADSHEET GENERATION] No perf data found; are you sure you are actually running any layers?")

        all_disjoints = OrderedDict()
        
        for perf_run in self.perf_data:
            disjoint_str = str(perf_run.layer.test_diff_flags)
            
            if not (disjoint_str in all_disjoints):
                all_disjoints[disjoint_str] = len(all_disjoints)
        
        table = Table((len(all_disjoints) + 2, len(self.split_names) + 1))
        
        table[0,  0] = "Layer Name (Split)"
        table[-1, 0] = "Layer Flags (In Common)"

        table[1:-1, 0] = [key for key in all_disjoints]
        
        table[0,  1:] = self.split_names
        
        for (run_idx, perf_run) in enumerate(self.perf_data):
            split_idx = self.split_names.index(perf_run.layer.split_name)

            base_flags = perf_run.layer.flags - perf_run.layer.test_diff_flags
            
            base_flags_str = "=\"" + str(base_flags) + "\""
            
            if table[-1, split_idx+1] == "":
                table[-1, split_idx+1] = base_flags_str
                
            elif table[-1, split_idx+1] != base_flags_str:
                raise Exception("Conflicting base flags: \"%s\" & \"%s\" for layer \"%s\"" % (table[-1, split_idx+1], base_flags_str, perf_run.layer.split_name))
                
            disjoint_str = str(perf_run.layer.test_diff_flags)
            
            disjoint_idx = all_disjoints[disjoint_str]
            
            if table[disjoint_idx+1, split_idx+1] != "":
                raise Exception("Conflict found for %s with column %s" % (perf_run.layer.split_name, disjoint_str))
                
            table[disjoint_idx+1, split_idx+1] = str(extract_result(perf_run, extract_value))

            
        if(file_name == None):
            return

        # Output perf data in csv format
        with open(file_name, 'w') as sheet_file:
            sheet_file.write(str(table))

        # Generate algo sweep sheet if -runAllAlgo in flags
        if('runAllAlgos' in self.perf_data[0].layer.flags):
            filename, file_extension = os.path.splitext(file_name)
            fname = filename+'_runAllAlgo'+file_extension
            self.generate_sweep_algo_sheet(fname)

    def generate_sweep_algo_sheet(self, fname):
        print 'Generating run all algo sweep csv. Filename: '+fname
        algo_dict = parse_algoSweep(self.perf_data[0].results.output)
        f = open(fname, 'w')
        f.write('Layer Name, lwblasTest cmd ')
        for algo in algo_dict:
            f.write(', algo='+str(algo))
        f.write('\n')
        for (run_idx, perf_run) in enumerate(self.perf_data):
            algo_dict = parse_algoSweep(perf_run.results.output)
            f.write(str(perf_run.layer.split_name)+','+str(perf_run.layer.flags))
            for algo in algo_dict:
                f.write(', '+str(algo_dict[algo]))
            f.write('\n')
        f.close()

if __name__ == "__main__":
    table = Table((4, 4))
    
    table[:, 1] = ["apple", "tomato", "strawberry", "orange"]
    
    table[1:-1, 2] = ["john", "jacob", "jingle", "heimer"]
    
    print str(table)
