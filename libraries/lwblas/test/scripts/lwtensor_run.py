#!/usr/bin/elw python

from helpers.sys_config      import augment_lib, print_lib
from helpers.sys_config      import get_gpu_info, print_general_info, get_gpu_filter
from helpers.sys_config      import redirect_glibc_backtraces, enable_logging
from helpers.label_parser    import get_labels_from_file
from helpers.layer_parser    import get_layers_from_file, resolve_layers
from helpers.lwdnn_interface import RunCache, get_default_run, extract_list
from helpers.Spreadsheet     import Spreadsheet
from helpers.utility         import split_comma, flags_from_descs_str, OrderedDefaultDict
from helpers.lwdnnLogToTestCmd       import logToTest_generate, logToTest_checkFlagSupport, logToTest_compareExelwtionPath, stripLog

import argparse
import sys, os
import re
import copy 
import time

#*******************************************************************************
#* Argument parsing logic
#*******************************************************************************
def make_help(s, has_choices=False):

    result = s + " [default: %(default)s]"

    if(has_choices):
        result += " [choices: %(choices)s]"

    return result

arg_format = lambda prog: argparse.HelpFormatter(prog,max_help_position=100, width=100)

parser = argparse.ArgumentParser(description='lwBLAS layer tests', formatter_class=arg_format)

parser._optionals.title = "Help Options"

# General Options
gen_args   = parser.add_argument_group('General Options')
gen_args.add_argument('-dryrun', action='store_const', const=True, default=False, help=make_help("Only print test names; do not execute"))

gen_args.add_argument('-testsList_batch_size', metavar='N', dest='testsList_batch_size', default=1, help=make_help("The number of tests to write together using -testsList (1 disables this)"))

gen_args.add_argument('-global_flags', metavar='"str"', dest='global_flags_str', default=None, help=make_help("Specify flags to set for all layers (example: \"-overrides algo:0,1,2 | P: h, s\""))

gen_args.add_argument('-config', metavar='"str"', dest='filter_config_str', default=None, help=make_help("Specify filters defining current system/test configuration"))

gen_args.add_argument('-device', metavar='n', type=int, dest='device', default=0, help=make_help("Specify device index ('-d' flag)"))

gen_args.add_argument('-loop', metavar='n', type=int, dest='loop', default=1, help=make_help("Specify the exelwtion loops of the test list"))

gen_args.add_argument('-sweep_heurgen', action='store_const', const=True, default=False, help=make_help("Use heurgen backdoor to sweep across all possible values"))

gen_args.add_argument('-partition', metavar="partIndex,partCount", dest='partition_str', default=None, help=make_help("Partition layers and run only one partition (example: \"4,10\" will run partition #4 of 10 partitions of layers)"))

gen_args.add_argument('-API_log_test', action='store_const', const=True, default=False, help=make_help("Enable UID comparison for API logging and test command generation, should run without randomization"))

gen_args.add_argument('-pre_flags', metavar='"str"', dest='pre_flags_str', default="", help=make_help("Specify pre_flags to place before lwtensorTest"))

gen_args.add_argument('-lwstom_test', metavar='"str"', dest='lwstom_test', default="", help=make_help("Specify custom test through the python command line, multiple tests can be separated by comma, such as 'lwstom_test1, lwstom_test2'. The newly added option should be added to option_list in lwstomTest_layer.py"))

# Spreadsheet Options
sheet_args = parser.add_argument_group('Spreadsheet Options')
sheet_args.add_argument('-spreadsheet', metavar='"str"', dest='spreadsheet', default=None, help=make_help("Create perf spreadsheet with the given file name"))

sheet_args.add_argument('-layer_split_by_flag', metavar='"str"', dest='layer_split_by_flag_str', default=None, help=make_help("Split layers by given flag(s) (new row in \"spreadsheet\"); comma separate for multiple."))

sheet_args.add_argument('-extract', metavar='"str"', dest='extract_value', default='time', choices=extract_list, help=make_help("For perf spreadsheet, specify extraction parameter", True))

# Whitelist arguments
white_args = parser.add_argument_group('Whitelist Options')
white_args.add_argument('-whitelist_flags', metavar='"str"', dest='whitelist_flags_str', default=None, help=make_help("Only allow cases within the given flags (example: \"-whitelist_flags R: colw\""))

white_args.add_argument('-whitelist_layer_name', metavar='"str"', dest='whitelist_layer_name', default=".*", help=make_help("Only allow cases where layer name matches the given regex"))

# Caching arguments
cache_args = parser.add_argument_group('Caching Options')

cache_args.add_argument('-cache_path', metavar='"str"', dest='cache_path', default=None, help=make_help("Specify cache path; default assumes no caching. A cache stores results of calls so that another run of lwdnn_perf.py can reload the results."))

cache_args.add_argument('-cache_freq', metavar='N', dest='cache_freq', default=-1, help=make_help("Specify interval to update cache; 2 means update after every other test call. The special case of -1 will only update cache after all calls are finished."))


# Path arguments
path_args  = parser.add_argument_group('Path Options')
path_args.add_argument('-binpath', metavar='"path"', dest='bin_path', default='./../', help=make_help("Specify bin path (where binary is located)"))

path_args.add_argument('-binname', metavar='"str"', dest='bin_name', default='lwtensorTest', choices=["lwtensorTest", "coreTest"], help=make_help("Specify which binary to run", True))

path_args.add_argument('-libpath', metavar='"path"', dest='lib_path', default='./../../lib/', help=make_help("Specify lib path (where liblwtensor.so is located)"))

path_args.add_argument('-layer_file', metavar='"path"', dest='layers_path', default='./lwtensorTest.layer', help=make_help("Specify layers file (path to layer_definitions"))

path_args.add_argument('-label_file', metavar='"path"', dest='labels_path', default='./lwtensorTest.label', help=make_help("Specify labels file (path to layer_labels)"))

args = parser.parse_args()

# Print all parsed arguments
print "Printing all command line arguments"
for arg in args.__dict__:
    print ("\t%s = %s" % (arg, args.__dict__[arg]))

 
#*******************************************************************************
#* Setup LD_LIBRARY_PATH (os-agnostic)
#*******************************************************************************
# Add lib_path to OS-dependant library path
if(args.lib_path != ''):
    augment_lib(args.lib_path)

# Print current library info
print "\n\nPrinting current LIBRARY path"
print_lib()
print ""

# Redict glibc backtrace to retrieve errors from test calls
redirect_glibc_backtraces()

#*******************************************************************************
#* Enable logging if API_LOG_TEST is enabled
#*******************************************************************************

if args.API_log_test:
    enable_logging()

#*******************************************************************************
#* Initialize all default values
#******************************************************************************/
# Track results of all tests
counts = {"total": 0.0, "passed": 0.0, "waived": 0.0, "failed": 0.0}

# Initialize perf spreadsheet
perf_sheet = Spreadsheet()

# Global filters (list of filters defined by user)
filter_config = split_comma(args.filter_config_str)

#*******************************************************************************
#* GPU Detection
#*******************************************************************************
gpu = get_gpu_info(args.device, args.bin_path, args.bin_name)

if(gpu == None):
    counts["failed"] += 1
else:
    counts["passed"] += 1

counts["total"] += 1

print ""

#*******************************************************************************
#* Layer Generation & Filtration
#*******************************************************************************
# Add filter for GPU speed
filter_config.append(get_gpu_filter(gpu))

# Hold test results (to handle duplicates efficiently)
results_cache = {}

# Store all layers that aren't filtered out
filtered_layers = []

# Get all split keys
split_flag_keys = split_comma(args.layer_split_by_flag_str)

#*******************************************************************************
#* Layer Generation & Filtration
#*******************************************************************************
# Get labels
labels = get_labels_from_file(args.labels_path, filter_config)

if args.bin_name != "lwblasMgTest":
    global_device_flag = "d: %d" % args.device
else:
    global_device_flag = ""

if args.global_flags_str != None:
    global_flags_str = args.global_flags_str + " * " + global_device_flag
else:
    global_flags_str = global_device_flag

# Global flags (will take priority over layer/label definitions)
global_flags_list = flags_from_descs_str(global_flags_str, labels)

# Flag Whitelist
whitelist_flags_list = flags_from_descs_str(args.whitelist_flags_str, labels)

# Get layers
layers = get_layers_from_file(args.layers_path, whitelist_flags_list, args.whitelist_layer_name, global_flags_list, split_flag_keys, labels, args.lwstom_test)

# Partition logic (only run certain layers)
if(args.partition_str):
    partition_config = args.partition_str.split(',')
    
    if len(partition_config) != 2:
        raise Exception("Partion config is invalid: %s from \"%s\"" % (partition_config, args.partition_str))
    
    partition_index = int(partition_config[0])
    partition_count = int(partition_config[1])
    
    if partition_index < 0:
        raise Exception("Partition index cannot be negative: %d from \"%s\"" % (partition_index, args.partition_str))
        
    if partition_index >= partition_count:
        raise Exception("Partition index cannot be >= partition count: (%d >= %d) from \"%s\"" % (partition_index, partition_count, args.partition_str))

    layer_count = len(layers)
    
    # Callwlates ceil(layer_count / partition_count)
    layer_partition_count = layer_count / partition_count
    
    layer_partition_start = partition_index*layer_partition_count
    layer_partition_end   = layer_partition_start + layer_partition_count
    
    # Last partition must grab everything (including extra round off)
    if partition_index == partition_count - 1:
        layer_partition_end = len(layers)
        
    layers = layers[layer_partition_start:layer_partition_end]
    
    
#*******************************************************************************
#* Query heurgen flags
#*******************************************************************************
if(args.sweep_heurgen):
    
    query_cache = RunCache(suggested_flags=[], list_length=1, update_freq=-1, file_name=args.cache_path)
    
    layers_dict = OrderedDefaultDict(list)
    
    for layer in layers:
        layer_query_flags = copy.deepcopy(layer.flags)
        layer_query_flags["backdoor[heurgen_dbg="] = ("-1]", )

        test_name_str = "%s %s" % (args.bin_name, str(layer_query_flags))
        
        print "&&&& RUNNING %s" % (test_name_str)

        print "Running test %s : '%s/%s %s'" % (layer.test_name, args.bin_path, args.bin_name, str(layer_query_flags))

        print "Layer Details: %s" % str(layer)
        
        # Obtain run from cache; will run test if not available in cache
        test_results = query_cache.get(layer_query_flags, args.bin_path, args.bin_name, device=args.device, pre_flags_str=args.pre_flags_str)
        
        # Print output if it exists
        if(test_results.output == None):
            print "No output detected\n"
        else:
            print test_results.output
            print ""
                
        
        if(test_results.parsed != None and test_results.parsed.query != None):
            
            layers_dict[layer.base_name].append(layer_query_flags)
            
            for choice in test_results.parsed.query.choices.split(','):
                # Ignore empty backdoors
                if len(choice.trim()) == 0:
                    continue
		    
                layer_flags_copy = copy.deepcopy(layer.flags)
                
                layer_flags_copy["backdoor[heurgen_dbg="] = (choice.strip() + "]", )
                
                layers_dict[layer.base_name].append(layer_flags_copy)
        else:
            layers_dict[layer.base_name].append(layer.flags)
                
    layers = resolve_layers(layers_dict, split_flag_keys)
    

loop_num = int(args.loop)
if loop_num < 1:
    loop_num = 1
iteration = 1
while iteration <= loop_num:
    counts_loop = copy.deepcopy(counts)
    iteration += 1
    #*******************************************************************************
    #* Test spreadsheet generation (early-exit for errors)
    #*******************************************************************************
    # Create test spreadsheet just to test 
    test_sheet = Spreadsheet()
    
    for layer in layers:
        test_sheet.add_run(layer, str(layer.flags), get_default_run())
    
    # Test generation (to early detect potential errors/conflicts)
    if(args.spreadsheet != None):
        test_sheet.generate(None, args.extract_value.lower())

    #*******************************************************************************
    #* Cache initialization
    #*******************************************************************************
    cache = RunCache(suggested_flags=[layer.flags for layer in layers], list_length=int(args.testsList_batch_size), update_freq=args.cache_freq, file_name=args.cache_path)
        
    #*******************************************************************************
    #* Test Exelwtion
    #*******************************************************************************
    # Create test list file, used for cbt testing.
    vc_enable = ""
    if "VECTORCAST_ENABLE" in os.elwiron:
        vc_enable = os.elwiron["VECTORCAST_ENABLE"]
    if vc_enable == "1":
        file = open("test.list", 'w+')
        file_fail = open("Fail.txt", 'w+')

    for layer in layers:
        test_name_str = "%s %s" % (args.bin_name, str(layer.flags))
        
        if(not args.dryrun):
            print time.strftime('Current time: %Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print "&&&& RUNNING %s" % (test_name_str)
    
        print "Running test %s : '%s/%s %s'" % (layer.test_name, args.bin_path, args.bin_name, str(layer.flags))
    
        print "Layer Details: %s" % str(layer)
    
        if(args.dryrun):
            print ""
            continue
    
        # Obtain run from cache; will run test if not available in cache
        test_results = cache.get(layer.flags, args.bin_path, args.bin_name, device=args.device, pre_flags_str=args.pre_flags_str)
    
        # Print output if it exists
        if(test_results.output == None):
            print "No output detected\n"
        else:
            # Only print API log when there is an error
            if args.API_log_test:
                if test_results.error_msg:
                    print test_results.output
                    print ""
                else:
                    print stripLog(test_results.output, mode=1)
                    # print ""
            else:
                print test_results.output
                print ""
        
        # Print error if it exists
        if(test_results.error_msg):
            print "[TEST EXELWTION] Error Detected: %s" % test_results.error_msg
    
        print time.strftime('Current time: %Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    
        if vc_enable == "1":
            file.write(test_name_str+"\n")
    
        # Use return code to determine PASS/FAIL/WAIVE
        if(test_results.return_code == 0):
            print "&&&& PASSED %s" % (test_name_str)
            counts_loop["passed"] += 1
            lwrrent_test_status = "PASSED"
    
        elif(test_results.return_code == 2):
            print "&&&& WAIVED %s" % (test_name_str)
            counts_loop["waived"] += 1
            lwrrent_test_status = "WAIVED"
    
        else:
            print "&&&& FAILED %s" % (test_name_str)
            counts_loop["failed"] += 1
            lwrrent_test_status = "FAILED"
            if vc_enable == "1":
                file_fail.write(test_name_str+"\n")
        
        # 
        if  args.API_log_test and lwrrent_test_status=="PASSED":
    
            if logToTest_checkFlagSupport(layer.flags) == True:
    
                original_UID = test_results.parsed.test_UID.test_UID
                print "@@@@ Original flags: %s" %(layer.flags)
                print "@@@@ Original UID: %s" %(original_UID)
    
                generated_flags = logToTest_generate(test_results, layer.flags)
                print "@@@@ Generated flags: %s" %(generated_flags)
    
                if (generated_flags != "error") and (generated_flags != "waived"):
    
                    # don't run ref on second run, save time
                    generated_flags["T"] = ("1",)
     
                    API_rerun_results = cache.get(generated_flags, args.bin_path, args.bin_name, device=args.device, pre_flags_str=args.pre_flags_str)
                    
                    if(API_rerun_results.return_code == 0):
                        
                        generated_UID = API_rerun_results.parsed.test_UID.test_UID
                        print "@@@@ Generated UID: %s" %(generated_UID)
    
                        UID_match = (original_UID == generated_UID)
                        execPath_match = logToTest_compareExelwtionPath(test_results.output, API_rerun_results.output)
                        
                        if UID_match or execPath_match:
                            print "@@@@ API LOG TEST PASSED (UID_match=%r, ExecPath_match=%r): %s\n" % (UID_match, execPath_match, test_name_str)
                        else:
                            print "@@@@ API LOG TEST FAILED (UID AND EXECPATH MISMATCH): %s\n" % (test_name_str)
                        
                    else:
                        print "@@@@ API LOG TEST FAILED (RERUN FAIL): %s\n" % (test_name_str)
    
                elif generated_flags == "error":
                    print "@@@@ API LOG TEST FAILED (GEN FLAG ERROR): %s\n" % (test_name_str)
                    print test_results.output
                    print ""
     
                elif generated_flags == "waived":
                    print "@@@@ API LOG TEST WAIVED: %s\n" % (test_name_str)
                
            else:
                print "@@@@ API LOG TEST NOT SUPPORTED: %s\n" % (test_name_str)
    
        print ""
    
        # Add run to perf spreadsheet generator
        perf_sheet.add_run(layer, str(layer.flags), test_results)
        
        counts_loop["total"] += 1
    
    # Create for cbt testing.
    if vc_enable == "1":
        file.close()
        file_fail.close()

    # Save cache
    cache.save()
    
    # Generate perf spreadsheet if enabled by user
    if(args.spreadsheet != None):
        perf_sheet.generate(args.spreadsheet, args.extract_value.lower())
    
    #*******************************************************************************
    #* Print summaries of results
    #******************************************************************************/
    print ""
    print "RESULT"
    print "Failures      : %d" % counts_loop["failed"]
    print "Successes     : %d" % counts_loop["passed"]
    print "Waived        : %d" % counts_loop["waived"]

    print "Basic Sanity  : %4.2f%%" % ((100*counts_loop["passed"]) / (counts_loop["passed"] + counts_loop["failed"]))
    print "\n\n"
    #end of while iteration < loop_num:
