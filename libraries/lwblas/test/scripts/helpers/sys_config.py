import sys, os
import collections # For NamedTuple

from lwdnn_interface import run_flags
from Flags           import Flags

platform = sys.platform

# Fix Python 2.x naming (incorrectly includes linux2, linux3, so on...)
if(platform.startswith('linux')):
    platform = 'linux'

# Android is basically linux style
if(platform == "android"):
    platform = 'linux'

# qnx
if(platform == "qnx"):
    platform = 'linux'
    
# Sometimes "android" returns "unknown" due to a bug in the makefile
if(platform == "unknown"):
    platform = 'linux'
    
# Get name of OS library path based on OS
os_lib_paths = {'win32': 'PATH', 'linux': 'LD_LIBRARY_PATH', 'darwin': 'DYLD_LIBRARY_PATH'}
os_lib_path = os_lib_paths[platform]

# Get separator of OS library path based on OS
os_lib_seps = {'win32': ';', 'linux': ':', 'darwin': ':'}
os_lib_sep = os_lib_seps[platform]

# Disable windows error popups
if(platform == 'win32'):
    import ctypes
        
    SEM_FAILCRITICALERRORS = 1
    SEM_NOGPFAULTERRORBOX  = 2
    SEM_NOOPENFILEERRORBOX = 0x8000
        
    ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS |
                                        SEM_NOGPFAULTERRORBOX  |
                                        SEM_NOOPENFILEERRORBOX)

def augment_lib(lib_path):
    
    previous = ""

    if(os_lib_path in os.elwiron):
        previous = os_lib_sep + os.elwiron[os_lib_path]

    # Add lib_path to OS library path
    os.elwiron[os_lib_path] = lib_path + previous

def enable_logging():
    os.elwiron["LWDNN_LOGINFO_DBG"] = "1"
    os.elwiron["LWDNN_LOGDEST_DBG"] = "stdout"


# Redirect glibc backtraces to stderr
def redirect_glibc_backtraces():

    if platform == 'linux':

        os.elwiron['LIBC_FATAL_STDERR_'] = '1'

def print_lib():
    if(os_lib_path in os.elwiron):
        print(os.elwiron[os_lib_path])
    else:
        print("%s not set" % os_lib_path)

def get_gpu_info(device, bin_path, bin_name):
    flags = Flags()
    
    parsed_gpu_field = None
    
    if bin_name == "lwdnnTest":
        # lwdnnTest flags
        flags["gpu"] = (str(device), )
        
        parsed_gpu_field = "lwdnn_gpu"
        
    else:
        # lwblasTest flags
        flags["d"] = (str(device), )
        flags["s"] = ("", )
        
        parsed_gpu_field = "lwblas_gpu"
        
    test_name_str = "%s %s" % (bin_name, str(flags))
    
    print "&&&& RUNNING %s" % test_name_str

    # Specify test being run
    print "Running test GPU : \'%s/%s %s\'\n" % (bin_path, bin_name, str(flags))

    gpu_query = run_flags(flags, bin_path, bin_name)

    passed = True

    # Print output
    if(gpu_query.output == None):
        print "No output detected\n"
        passed = False
    else:
        print gpu_query.output
        print ""

    # Detect any errors (print if so)
    if(gpu_query.error_msg != None):
        print "[GPU DETECTION] Error Detected: %s" % gpu_query.error_msg
        passed = False        

    if(gpu_query.parsed == None or getattr(gpu_query.parsed, parsed_gpu_field) == None):
        print "[GPU DETECTION] Unable to detect GPU"
        passed = False

    # If PASSED, print PASSED and return GPU info
    if(passed):
        # Print detected GPU info
        print "\nGPU DETECTED: %s\n" % str(getattr(gpu_query.parsed, parsed_gpu_field))

        print "&&&& PASSED %s" % test_name_str

        return getattr(gpu_query.parsed, parsed_gpu_field)

    # We haven't passed, print FAILED and return None
    print "&&&& FAILED %s" % test_name_str
    return None


def print_general_info(bin_path, bin_name):
    flags = Flags()
    
    if bin_name == "lwdnnTest":
        # lwdnnTest flags
        flags['g'] = ("", )
        
    else:
        # lwblasTest flags
        flags['v'] = ("", )
    
    test_name_str = "%s %s" % (bin_name, str(flags))

    print "&&&& RUNNING %s" % test_name_str

    # Specify test being run
    print "Running test general_info : \'%s/%s %s\'\n" % (bin_path, bin_name, str(flags))

    gpu_query = run_flags(flags, bin_path, bin_name)

    passed = True

    # Print output
    if(gpu_query.output == None):
        print "No output detected\n"
        passed = False
    else:
        print gpu_query.output
        print ""

    # Detect any errors (print if so)
    if(gpu_query.error_msg != None):
        print "[GPU DETECTION] Error Detected: %s" % gpu_query.error_msg
        passed = False        

    # If PASSED, print PASSED and return GPU info
    if(passed):
        print "&&&& PASSED %s" % test_name_str

        return True

    # We haven't passed, print FAILED and return None
    print "&&&& FAILED %s" % test_name_str
    return False
    

def get_gpu_filter(gpu):
    result = []
    
    if(gpu == None):
        return 'UNKNOWN'
        
    gpu_cap = None
    
    if 'lwDNN' in type(gpu).__name__:
        gpu_cap = gpu.cap
        
    if 'lwBLAS' in type(gpu).__name__:
        gpu_cap = 10 * gpu.cap_major + gpu.cap_minor
        
    if(gpu_cap == 53 or gpu_cap == 62):
        return 'SLOW'
        
    if(gpu.mem <= 3000 or gpu_cap < 35 or platform != "linux"):
        return 'MID'
    
    return 'FAST'
        
    return speed
