#  module serves as an interface between Python and lwdnnTest/lwblasTest

# To accomplish this, it runs lwdnnTest and parses the output to obtain result info.
#   - Result info includes elapsed time, M, N, K, algo, etc...
#   - See "def parse(output)" for a definition of all parsed data.

# The main use of this module is the function "run_flags(flags)"
#   - This will run lwdnnTest and return parsed output


import re           # For regular expressions
import subprocess   # To spawn processes (lwdnnTest)
import collections  # For named tuple
import sys          # For error handling and exiting 
try:    
    import cPickle as pickle # To load/store pickle files
except:
    import pickle
import os                # To check if file exists


from utility     import split_space
from Flags       import Flags

# Define patterns and tuples for all parsed outputs
time_pat    = re.compile(r'elapsed = (.*) sec', re.IGNORECASE)
time_tup    = ("Time", "seconds")

sol_gflops_pat  = re.compile(r'measured Gflops = (.*) ', re.IGNORECASE)
sol_gflops_tup  = ("SOL_GFlops", "sol_gflops")

median_time_pat = re.compile(r'LWCA elapsed median = (.*?) msec', re.IGNORECASE)
median_time_tup = ("Median_Time", "median_time_msec")

gflops_pat      = re.compile(r',  Gflops = (.*) ', re.IGNORECASE)
gflops_tup      = ("GFlops", "gflops")

sol_pat         = re.compile(r'measured Gflops = .* \((.*)\%\)', re.IGNORECASE)
sol_tup         = ("SOL", "sol")

gemm_pat        = re.compile(r'M.*=(.*) N.*=(.*) K.*=(.*)\n', re.IGNORECASE)
gemm_tup        = ("GemmEquivalence", "m n k")

src_pat         = re.compile(".*?(?:ImageTensor \(input\)|DGradTensor \(output\)).*?sizes=\[n=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\]")
src_tup         = ("InputDescriptor", "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4")

dest_pat        = re.compile(".*?(?:RespTensor \(output\)|DGradTensor \(output\)|DiffTensor \(input\)).*?sizes=\[n=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\]")
dest_tup        = ("OutputDescriptor", "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4")

filter_pat      = re.compile(".*?(?:FilterTensor \(input\)|WGradTensor \(output\)).*?sizes=\[k=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\]")
filter_tup      = ("FilterDescriptor", "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4")

colw_pat        = re.compile("args: Colw\s*:\s*?pad=\[(\d+),(\d+)(?:,(\d+))?\]\s*?strides=\[(\d+),(\d+)(?:,(\d+))?\]\s*?dilation=\[(\d+),(\d+)(?:,(\d+))?\]")
colw_tup        = ("ColwolutionDescriptor", "padA_0 padA_1 padA_2 strideA_0 strideA_1 strideA_2 dilationA_0 dilationA_1 dilationA_2")

pad_pat         = re.compile(r'pad=\[([0-9]+),([0-9]+)\]', re.IGNORECASE)
pad_tup         = ("Padding", "pad_h pad_w")

passed_pat      = re.compile(r'(PASSED)')
passed_tup      = ("Passed", "result")

algo_pat        = re.compile(r'Algo.*(preference|user).*([0-9]+)\n', re.IGNORECASE)
algo_tup        = ("AlgoChoice", "chosen_by choice")

query_pat       = re.compile(r'HEURGEN:.*\[(.*)\].*\[(.*)\].*"(.*)".*')
query_tup       = ("HeurgenOptions", "choosers choices name")

tune_pat        = re.compile(r'HEURGEN:.*Running Backdoor ([0-9]+) from.*\((.*)\)')
tune_tup        = ("TuningSpecs", "tuning tuners")

lwdnn_gpu_pat   = re.compile(r'\^\^\^\^\s+CSV\s+(\d+),(\d+),(.+),(\d+)')
lwdnn_gpu_tup   = ("lwDNNGPUSpecs", "sm cap clock mem")

lwblas_gpu_pat  = re.compile(r'device \d+ \(current\) : sms\s+(\d+)\s+Capabilities (\d+).(\d), SmClock (.+) Mhz, MemSize \(Mb\) (\d+), MemClock .+ Mhz, Ecc=\d+, boardGroupID=\d+')
lwblas_gpu_tup  = ("lwBLASGPUSpecs", "sm cap_major cap_minor clock mem")

status_pat      = re.compile(r'\$\$\$\$ Test on line (.*) returned status (.*)')
status_tup      = ("LwdnnTestStatus", "line status")

err_msg_pat     = re.compile(r'First error msg      : (.*)')
err_msg_tup     = ("LwdnnErrMsg", "err_msg")

processing_pat  = re.compile(r'Processing line (.*): (.*)')
processing_tup  = ("Processing", "line flags")

lwblastag_begin_pat  = re.compile(r'@@@@ BEGIN TAG no(\d+)')
lwblastag_end_pat    = re.compile(r'@@@@ END TAG no(\d+) RESULT (\w+)')
lwblastag_end_tup  = ("Tag", "testNo status")

UID_pat = re.compile(r'@@@@ Test UID             : ([0-9a-z]{32})')
UID_tup = ("LwdnnTestUID", "test_UID")

# algorithm pattern for algo sweep extraction
looping_algo_pat  = re.compile(r'#### args: looping over algos \(true\): algoCnt=([+-]?\d+),  algorithm=([+-]?\d+)')
looping_algo_tup  = ("Looping algo sweep", "algo_count algorithm")

# Concat all patterns/tuples into one
pat_list     = [time_pat, median_time_pat, sol_gflops_pat, gflops_pat, sol_pat, gemm_pat, pad_pat, passed_pat, algo_pat, query_pat, tune_pat, lwblas_gpu_pat, lwdnn_gpu_pat, src_pat, dest_pat, filter_pat, colw_pat, status_pat, err_msg_pat, processing_pat, lwblastag_end_pat, UID_pat]
tup_list     = [time_tup, median_time_tup, sol_gflops_tup, gflops_tup, sol_tup, gemm_tup, pad_tup, passed_tup, algo_tup, query_tup, tune_tup, lwblas_gpu_tup, lwdnn_gpu_tup, src_tup, dest_tup, filter_tup, colw_tup, status_tup, err_msg_tup, processing_tup, lwblastag_end_tup, UID_tup]
name_list    = ["time","median_time","sol_gflops", "gflops", "sol", "gemm", "pad", "passed", "algo", "query", "tune", "lwblas_gpu", "lwdnn_gpu", "srcDesc", "destDesc", "filterDesc", "colwDesc", "status", "err_msg", "processing", "lwblas_tag", "test_UID"]


# Create namedtuples for all parsed outputs
ParsedOutput = collections.namedtuple("ParsedOutput", " ".join(name_list))
ParsedTuples = [collections.namedtuple(*tup_opts) for tup_opts in tup_list]

for ParsedTuple in ParsedTuples:
   globals()[ParsedTuple.__name__] = ParsedTuple 

# Create namedtuple for RunResult
RunResult        = collections.namedtuple("RunResult", "flags bin_path bin_name output error_msg return_code parsed")

# Create namedtuple for RunListResult
RunListResult    = collections.namedtuple("RunListResult", "flags bin_path bin_name outputs error_msg return_code parsed")

# Create list of extractable parameters (those with 1 single value)
extract_list = [name_list[i] for i in range(len(name_list)) if len(ParsedTuples[i]._fields) == 1]

def match_to_tuple(match, TupleClass):
    # Check that match was found
    if(match):
        if(len(match.groups()) != len(TupleClass._fields)):
            raise Exception("Unable to fit groups %s into Tuple \"%s\" with fields %s" % (str(match.groups()), TupleClass.__name__, str(TupleClass._fields)))

        return TupleClass(*match.groups())

    return None

# Find the line strictly after lid in lines matching processing_pat
def get_next(lid, lines):
    for lidNext in range(lid+1, len(lines)):
        match = processing_pat.search(lines[lidNext])
        if match:
            return (lidNext, int(match.groups(1)[0]))
    return (len(lines), -1)

# Find the line strictly after lid in lines matching looping_algo_pat
def get_next_algo(lid, lines):
    for lidNext in range(lid+1, len(lines)):
        match = looping_algo_pat.search(lines[lidNext])
        if match:
            return (lidNext, int(match.groups(1)[0]), int(match.groups(1)[1]))
    return (len(lines), -1, None)
def get_algo_result(algo_output, output):
    match = time_pat.search(algo_output)
    if match:
        return match.groups(1)[0]
    elif re.compile('Test case is not supported for algorithm').search(algo_output):
        return 'not supported'
    elif re.compile('Invalid input').search(algo_output):
        return 'invalid input'
    elif re.compile('lwdaError=(\d+)').search(algo_output):
        return 'lwdaError='+str(re.compile('lwdaError=(\d+)').search(algo_output).groups(1)[0])
    else:
        print 'unrecognized algo sweep output for algo_output '
        print 'full output at for the unreconized case'
        print output
        sys.exit(0)

# Remove all duplicates lines starting with 'HEURGEN: ', keeping original ordering
# Useful when running heuristics with -T > 1, to avoid extremely large outputs
def strip_heurgen_duplicates(string):
    if string == None:
        return None
    string_split = string.split('\n')
    seen = set()
    string_nodupheurgen_split = []
    for line in string_split:
        if line.startswith('HEURGEN: '):
            if line not in seen:
                seen.add(line)
                string_nodupheurgen_split.append(line)
        else:
            string_nodupheurgen_split.append(line)
    return '\n'.join(string_nodupheurgen_split)

# Returns algo_dict = {algo#:'time sec', algo#:'not supported', algo#:'invalid input'}    
def parse_algoSweep(output):
    lines = output.split('\n')
    (lid_lwrrent, algo_count, algo) = get_next_algo(-1, lines)
    algo_dict = collections.OrderedDict()
    while lid_lwrrent < len(lines):
        (lid_next, algo_count_next, algo_next) = get_next_algo(lid_lwrrent+1, lines)
        algo_output = "\n".join(item for item in lines[lid_lwrrent:lid_next])
        algo_result = get_algo_result(algo_output, output)
        algo_dict[algo] = algo_result
        # next algo iteration
        lid_lwrrent = lid_next
        algo_count = algo_count_next
        algo = algo_next
    return algo_dict
    
# Output = complete output string
# Returns ([output1, output2, ...], [parsedoutput1, parsedoutput2, ...]) for each test of the -testsList
def parse_testsList(output):
    if(output == None):
        return (None, None)

    if '$$$$ Reading from stdin' not in output:
        print('[TEST EXELWTION] Critical error. Expected -testsList output, could not find required data')
        return ([output], [None])

    lines = output.split('\n')
    (lid_lwrrent, begin_line) = get_next(-1, lines)
    if begin_line != 1:
        print('[TEST EXELWTION] Critical error, expected to read line 1, got line {}'.format(begin_line))
        return ([output], [None])
    (lid_next, next_line) = get_next(lid_lwrrent, lines)
    parsed_result = []
    split_output = []
    expected_line = 0

    while lid_lwrrent < len(lines):
        expected_line += 1
        test_to_parse = lines[lid_lwrrent:lid_next]
        parsed_output = parse('\n'.join(test_to_parse))
        if parsed_output and parsed_output.status:
            status_line = int(parsed_output.status.line)
        else:
            print('[TEST EXELWTION] Critical error, parsing error for line {}'.format(expected_line)) 
            if not split_output: # If nothing to return (i.e., fails at first iteration), returns all the output so that we have something to print
                return ([output], [None])
            else:
                return (split_output, parsed_result)
        if begin_line != expected_line or status_line != expected_line:
            print('[TEST EXELWTION] Critical error, expected output from line {}, got output beginning line {} and ending line {}'.format(expected_line, begin_line, status_line)) 
            if not split_output: # If nothing to return (i.e., fails at first iteration), returns all the output so that we have something to print
                return ([output], [None])
            else:
                return (split_output, parsed_result)
        parsed_result.append(parsed_output)
        split_output.append(strip_heurgen_duplicates('\n'.join(test_to_parse)))
        (lid_lwrrent, begin_line) = (lid_next, next_line)
        (lid_next, next_line) = get_next(lid_lwrrent, lines)

    return (split_output, parsed_result)

def parse_testsListlwblas(output):
    if(output == None):
        return (None, None)

    lines = output.split('\n')
    parsed_result = []
    split_output = []

    testNo  = 0
    lineIdx = 0
    while True:
        beginLine = -1
        endLine   = -1

        for i in xrange(lineIdx, len(lines) - 1):
            match = lwblastag_begin_pat.search(lines[i])
            if match:
                if testNo != int(match.groups(1)[0]):
                    raise RuntimeError("callwlated test number {} doesn't match the tag no{}".format(testNo, int(match.groups(1)[0])))
                beginLine = i
                break
        else:
            break

        test_result = ''

        for i in xrange(beginLine + 1, len(lines) - 1):
            match = lwblastag_end_pat.search(lines[i])
            if match:
                test_result = match.groups(2)[0]
                endLine = i
                break
        else:
            print('cannot find END tag for no{}!'.format(testNo))
            split_output.append('\n'.join(lines[beginLine:]))
            parsed_result.append(parse(split_output[testNo]))
            return (split_output, parsed_result)

        split_output.append('\n'.join(lines[beginLine : endLine + 1]))
        parsed_result.append(parse(split_output[testNo]))

        lineIdx = endLine + 1
        testNo  = testNo + 1

    return (split_output, parsed_result)


def parse(output):
    # Don't attempt to parse None
    if(output == None):
        return None

    # Parse output to obtain matches
    matches = [pat.search(output) for pat in pat_list]

    # Place matched groups into namedtuples
    tuples  = [match_to_tuple(matches[i], ParsedTuples[i]) for i in range(len(pat_list))]
    
    return ParsedOutput(*tuples)

# Fill all values with None and return 
def get_default_run():
    # Create named tuples with None for all values
    named_tuples = [tup(*((None,)*len(tup._fields))) for tup in ParsedTuples]

    # Return None-filled RunResult
    return RunResult(None, None, None, None, None, None, ParsedOutput(*named_tuples))

# Runs given flags
def run_flags(flags, bin_path, bin_name, piped_input=None, pre_flags_str=""):
    # Initialize all outputs in case everything fails
    return_code = None
    output      = None
    error_msg   = None

    shell_pre_flags = split_space(pre_flags_str)

    shell_bin_flag  = ["%s/%s" % (bin_path, bin_name)]
    
    shell_flags     = split_space(flags)

    try:
        # Start process with requested flags (and piped output from stdout & stderr)
        process = subprocess.Popen(shell_pre_flags + shell_bin_flag + shell_flags, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Obtain output from communicate()
        output, unused_err = process.communicate(piped_input)

        # Poll for return code
        return_code = process.poll()

        # Error if return_code > 0
        if return_code:
            raise subprocess.CalledProcessError(return_code, flags, output=output)

    except Exception as e:
        # Grab output if there is any
        if('output' in e.__dict__):
            output = e.output

        # Grab return code if there is any
        if('returncode' in e.__dict__):
            return_code = e.returncode

        # Grab error (guaranteed to be given)
        error_msg = str(e)

    if 'testsList' in flags:
        (split_outputs, parsed_outputs) = parse_testsList(output)
        return RunListResult(flags, bin_path, bin_name, split_outputs, error_msg, return_code, parsed_outputs)
    elif 'file' in flags:
        (split_outputs, parsed_outputs) = parse_testsListlwblas(output)
        return RunListResult(flags, bin_path, bin_name, split_outputs, error_msg, return_code, parsed_outputs)
    else:
        return RunResult(flags, bin_path, bin_name, output, error_msg, return_code, parse(output))


def emulate_run_flags(flags, output_str):
    return_code = 0
    output      = output_str
    error_msg   = None
    
    if 'ERROR' in output_str:
        return_code = 1
        error_msg = "ERROR"
        
    return RunResult(flags, None, 'lwblasTest', output, error_msg, return_code, parse(output))
    
# Flag descriptors which are ignored by cache
flag_cache_scrubs = Flags()
flag_cache_scrubs['d'] = ('', )

def get_cache_key(flags):
    scrubbed_flags = flags - flag_cache_scrubs
    
    return tuple(split_space(scrubbed_flags))

def print_error(error_type, error_message, flags, runlist_results):
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "[{}] {}: {}".format(error_type, error_message, runlist_results.error_msg)
    if runlist_results.outputs == None:
        n_to_print = 0
    else:
        n_to_print = len(runlist_results.outputs)
    print "While testing ... (only printing first {})".format(n_to_print)
    for f in flags[:n_to_print]:
        print str(f)
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

class RunCache:
    def __init__(self, suggested_flags=[], list_length=1, update_freq=-1, file_name=None):
        # Start with empty cache
        self.cache       = {}

        # Set suggested list of tests and
        # remove duplicates
        # self.suggested_flags is an ordered dict
        self.update_suggested_flags(suggested_flags)
        self.list_length = int(list_length)

        # Set self parameters (from user inputs)
        self.update_freq = int(update_freq)
        self.file_name   = file_name

        self.run_count_since_update   = 0

        self.load()

    def get(self, flags, bin_path, bin_name, device='0', pre_flags_str=""):
        cache_key = get_cache_key(flags)
        
        if(cache_key not in self.cache):  
            self.run(flags, bin_path, bin_name, device=device, pre_flags_str=pre_flags_str)
            
        return self.cache[cache_key]
    
    def run(self, flags, bin_path, bin_name, device='0', pre_flags_str=""):

        # Use usual lwdnnTest interface
        if self.list_length == 1:
            self.cache[get_cache_key(flags)] = run_flags(flags, bin_path, bin_name, pre_flags_str=pre_flags_str)
            self.run_count_since_update += 1

        else: # Use lwdnnTest -testsList interface
            # if bin_name != "lwdnnTest":
            #     raise Exception("[TEST EXELWTION] Only lwdnnTest supports batching; set testsList_batch_size to 1 to allow exelwtion.")
                
            # Find where flags is in the list (if in it)
            key = get_cache_key(flags)
            dont_delete_first = False
            if key in self.suggested_flags:
                for (i,k) in enumerate(self.suggested_flags.iterkeys()):
                    if key == k :
                        idx = i
                        break
                flags_to_run = []
            else:
                dont_delete_first = True
                idx = 0 
                flags_to_run = [flags]

            # Add suggested_flags in a wrap-around fashion, starting from flags
            all_flags = list(self.suggested_flags.itervalues())
            all_flags = all_flags[idx:] + all_flags[:idx]
            flags_to_run.extend(all_flags[:(self.list_length - len(flags_to_run))])
            assert len(flags_to_run) <= self.list_length

            # Build tests list
            piped_input = ''
            flags_written = []
            i = 0
            for flags_2run in flags_to_run: # flags_to_run does not contain 'lwdnnTest'
                key = get_cache_key(flags_2run)
                if key not in self.cache: # For some reason, we may have ran this test before. Them we just skip it
                    # flags_2run['tag'] = ('no{}'.format(i),)
                    # i = i + 1
                    flags_str = str(flags_2run)
                    flags_written.append(flags_2run) # Need to keep track of it in that case
                    piped_input = piped_input + flags_str + ' -tagno{}'.format(i) + '\n'
                    i = i + 1
            flags_to_run = flags_written # The tests we actually run in lwdnnTest

            # Run tests list
            flags_testslist = Flags()
            if bin_name == "lwdnnTest":
                flags_testslist['testsList'] = ("1",)
                flags_testslist['returnOnWaived'] = ("1",)
                flags_testslist['returnOnFailed'] = ("1",)

            elif bin_name in ("lwblasTest", "lwblasTestLt"):
                flags_testslist['file'] = ("-",) # flags on stdin
                print "<<<< piped flags BEGIN"
                print piped_input
                print ">>>> piped flags END"

            else:
                raise Exception("[TEST EXELWTION] Only lwdnnTest, lwblasTest and lwblasTestLt support batching; set testsList_batch_size to 1 to allow exelwtion.") 

            flags_testslist['d'] = (str(device),)
            runlist_results = run_flags(flags_testslist, bin_path, bin_name, piped_input=piped_input, pre_flags_str=pre_flags_str)

            # Parse output
            error_msg = runlist_results.error_msg

            # If some tests did not run or failed, we output the lists of tests leading to that failure. This may not be a critical issue.
            if(error_msg or len(runlist_results.outputs) != len(flags_to_run)):
                print_error('TEST EXELWTION', 'Non-critical error Detected', flags_to_run, runlist_results)

            if len(runlist_results.outputs) == 0: # We couldn't parse anything, this is a critical error and we skip this test
                print_error('TEST EXELWTION', 'Critial error detected, not output parsed, skipping test with status FAILED', flags_to_run, runlist_results)
                self.cache[get_cache_key(flags)] = RunResult(flags, bin_path, None, None, 1, None) # Failed
                return

            if bin_name == "lwdnnTest":
                # Verify flags match with what -testsList runs: this should not fail under any cirlwmstance, otherwise something is really wrong in either lwdnnTest -testsList or in the parsing
                for (id, (parsed, flags_expected)) in enumerate(zip(runlist_results.parsed, flags_to_run)):
                    flags_match = False
                    flags_expected_str = str(flags_expected)
                    flags_actual_str = 'Flags not found'
                    line = 'Line not found'
                    if parsed and parsed.processing:
                        flags_actual_str = parsed.processing.flags
                        line = int(parsed.processing.line)
                        if (flags_actual_str == flags_expected_str) and (line == id+1):
                            flags_match = True
                    if not flags_match:
                        print_error('TEST EXELWTION', 'Critical error, flags do not match or line is wrong: expected (flag, line) ({}, {}), found ({}, {})'.format(flags_expected_str, id+1, flags_actual_str, line), flags_to_run, runlist_results)
                        print "Skipping tests {} with status FAILED".format(flags)
                        self.cache[get_cache_key(flags)] = RunResult(flags, bin_path, None, None, 1, None) # Failed
                        return

            # Go through files, add them to cache if result is considered valid
            first_ok = False
            for (output, parsed, flags_ran) in zip(runlist_results.outputs, runlist_results.parsed, flags_to_run):
                key_ran = get_cache_key(flags_ran)
                if bin_name == "lwdnnTest":
                    # We check if the test is good (reliable) as long as PASSED or WAIVED / FAILED with specific error codes
                    # We always accept the first one
                    if first_ok:
                        if parsed and parsed.status: # We have a status
                            if parsed.status.status != 'LWDNNBATCH_PASSED': # It's not passed
                                cases_ok = ["LWDNN_STATUS_NOT_SUPPORTED"];
                                if (not parsed.err_msg):
                                    break;
                                case_ok = False
                                for err_msg_ok in cases_ok:
                                    if err_msg_ok in parsed.err_msg.err_msg: # It's an OK (i.e. reliable status) case
                                        case_ok = True 
                                if not case_ok:
                                    break;
                        else: # We have no proper status
                            break
                    first_ok = True
                    if parsed and parsed.status:
                        if(parsed.status.status == 'LWDNNBATCH_PASSED'):
                            return_code = 0 
                        elif(parsed.status.status == 'LWDNNBATCH_WAIVED'):
                            return_code = 2
                        else:
                            return_code = 1
                    else: # If we couldn't find a status, we return Failed. 
                        return_code = 1

                else: #lwblasTest(Lt)?
                    if parsed and parsed.lwblas_tag:
                        if parsed.lwblas_tag.status == 'PASSED':
                            return_code = 0
                        elif parsed.lwblas_tag.status == 'WAIVED':
                            return_code = 2
                        else:
                            return_code = 1
                    else:
                        return_code = 1

                if return_code != 0:
                    test_result = RunResult(flags_ran, bin_path, bin_name, output, error_msg, return_code, parsed)
                else:
                    test_result = RunResult(flags_ran, bin_path, bin_name, output, None, return_code, parsed)
                # The test is good, so we 1) add it to the cache 2) remove it from the suggested list (if it was in it)
                self.cache[key_ran] = test_result
                self.run_count_since_update += 1
                if dont_delete_first:
                    dont_delete_first = False
                else:
                    del self.suggested_flags[key_ran] 


        # Update cache if within frequency
        if self.update_freq > -1 and self.run_count_since_update >= self.update_freq:
            self.save()
            self.run_count_since_update = 0

    def exists(self, flags):
        if get_cache_key(flags) in self.cache:
            return True

        return False

    def insert(self, flags, run):
        self.cache[get_cache_key(flags)] = run
        
    def save(self):
        if self.file_name:
            with open(self.file_name, "wb") as pickle_file:
                pickle.dump(self.cache, pickle_file)

    def load(self):
        if self.file_name and os.path.isfile(self.file_name):
            with open(self.file_name, "rb") as pickle_file:
                temp_cache = pickle.load(pickle_file)
                
                self.cache = dict((get_cache_key(run.flags), run._replace(parsed = parse(run.output))) for run in temp_cache.values())

    # suggested_flags = [['-n10','-r3',...],['-n25','-r5',...],...]
    def update_suggested_flags(self, suggested_flags):
        self.suggested_flags = collections.OrderedDict()
        for flags in suggested_flags:
            key = get_cache_key(flags)
            if key not in self.suggested_flags and key not in self.cache: # Not in cache (we have the result) not in the list
                self.suggested_flags[key] = flags

    def __getitem__(self, index):
        key = self.cache.keys()[index]
        return self.cache[key]
