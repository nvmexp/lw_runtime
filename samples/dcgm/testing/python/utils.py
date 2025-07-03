## {{{ http://code.activestate.com/recipes/577479/ (r1)
from collections import namedtuple
from functools import wraps
import ctypes
import os
import sys
import fnmatch
import platform
import subprocess
import string
import stat
import getpass

import option_parser
import test_utils
from subprocess import check_output

_CacheInfo = namedtuple("CacheInfo", "hits misses maxsize lwrrsize")

def cache():
    """Memorizing cache decorator.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses maxsize, size) with
    f.cache_info().  Clear the cache and statistics with f.cache_clear().

    """

    def decorating_function(user_function,
                tuple=tuple, sorted=sorted, len=len, KeyError=KeyError):

        cache = dict()
        hits_misses = [0, 0]
        kwd_mark = object()             # separates positional and keyword args

        @wraps(user_function)
        def wrapper(*args, **kwds):
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))
            try:
                result = cache[key]
                hits_misses[0] += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                hits_misses[1] += 1
            return result

        def cache_info():
            """Report cache statistics"""
            return _CacheInfo(hits_misses[0], hits_misses[1], None, len(cache))

        def cache_clear():
            """Clear the cache and cache statistics"""
            cache.clear()
            hits_misses = [0, 0]

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function


# ----- Example ----------------------------------------------------------------

if __name__ == '__main__':

    @cache()
    def fib(n):
        if n < 2:
            return 1
        return fib(n-1) + fib(n-2)

    from random import shuffle
    inputs = list(range(30))
    shuffle(inputs)
    results = sorted(fib(n) for n in inputs)
    print(results)
    print(fib.cache_info())
        
    expected_output = '''[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 
         233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 
         46368, 75025, 121393, 196418, 317811, 514229, 832040]
         CacheInfo(hits=56, misses=30, maxsize=None, lwrrsize=30)
    '''
## end of http://code.activestate.com/recipes/577479/ }}}

def is_root():
    if is_linux():
        return os.geteuid() == 0
    else:
        return ctypes.windll.shell32.IsUserAnAdmin()

def is_real_user_root():
    """
    Effective user can be changed. But real user is always the same and can't be changed

    """
    if is_linux():
        return os.getuid() == 0
    else:
        # Windows can't change user so implementation is the same as in is_root()
        return ctypes.windll.shell32.IsUserAnAdmin()


_UserInfo = namedtuple("UserInfo", "uid, gid, name")
@cache()
def get_user_idinfo(username):
    from pwd import getpwnam
    info = getpwnam(username)
    return _UserInfo(info.pw_uid, info.pw_gid, info.pw_name)

@cache()
def get_name_by_uid(uid):
    from pwd import getpwuid
    return getpwuid(uid).pw_name

script_dir = os.path.realpath(sys.path[0])

def find_files(path = script_dir, mask = "*", skipdirs=[], relwrse=True):
    #Relwrse subdirectories?
    if relwrse:
        for root, dirnames, filenames in os.walk(path):
            if skipdirs is not None:
                [dirnames.remove(skip) for skip in skipdirs if skip in dirnames]  # don't visit directories in skipdirs list
            for filename in fnmatch.filter(filenames, mask):
                yield os.path.abspath(os.path.join(root, filename))
    else:
        #Just list files inside "path"
        filenames = os.listdir(path)
        for filename in fnmatch.filter(filenames, mask):
            yield os.path.abspath(os.path.join(path, filename))


def which(name):
    """
    Returns True if command line application is in the PATH.
    """
    if is_linux():
        return 0 == os.system('which "%s" 1> /dev/null 2> /dev/null' % name)
    else:
        # TODO on windows os.system pops (for a brief moment) cmd console
        # this function should be reimplemented so that it wouldn't happen
        return 0 == os.system('where "%s" &> NUL' % name)

"""
stores string representing current platform
expected returned values
"Linux_64bit"
"Linux_32bit"
"Windows_64bit"
"Linux_ppc64le"
"Linux_aarch64"

treats VMkernel platform as Linux
"""

lwrrent_os = platform.system()
if lwrrent_os == "VMkernel":
    lwrrent_os = "Linux" # Treat VMkernel as normal Linux.

def is_windows(os=lwrrent_os):
    return os == "Windows"
def is_linux(os=lwrrent_os):
    return os  == "Linux"

def is_lwda_supported_system():
    # LWCA is supported everywhere except in virtualization elwironments
    return is_bare_metal_system()

def is_healthmon_supported_system():
    return is_linux() and is_lwda_supported_system()

def is_esx_hypervisor_system():
    return platform.system() == "VMkernel"

def is_microsoft_hyper_v():

    try:
        dmi = check_output(["which", "dmidecode"])
    except Exception:
        pass
        return False

    if is_root():        
        if os.path.isfile(dmi.strip()):    
            systemType = check_output(["dmidecode", "-s", "system-product-name"])
            if systemType.strip() == "Virtual Machine":
                return True
        else:
            return False
    else:
        return False

# Util method to check if QEMU VM is running
# DGX-2 VM uses QEMU
def is_qemu_vm():
    """
    Returns True if QEMU VM is running on the system()
    """

    cmd = 'lshw -c system | grep QEMU | wc -l'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if int(out) > 0:
        return True
    else:
        return False

def is_bare_metal_system():
    if is_esx_hypervisor_system():
        return False
    elif is_linux() and platform.linux_distribution()[0] == "XenServer":
        return False
    elif is_linux() and is_microsoft_hyper_v():
        return False
    elif is_linux() and is_qemu_vm():
        return False
    else:
        return True

def is_64bit():
    if os.name == 'nt':
        if platform.uname()[4] == 'AMD64':
            return True
    return platform.architecture()[0] == "64bit"
     
def is_32bit():
    if os.name == 'nt':
        if platform.uname()[4] == 'x86':
            return True            
    return platform.architecture()[0] == "32bit"
    
def is_system_64bit():
    return platform.machine() in ["x86_64", "AMD64"]

# 32-bit Python on 64-bit Windows reports incorrect architecture, therefore not using platform.architecture() directly
platform_identifier = lwrrent_os + "_" + ("64bit" if is_64bit() else "32bit")
if platform.machine() ==  "ppc64le":
    platform_identifier = "Linux_ppc64le"
if platform.machine() == "aarch64":
    platform_identifier = "Linux_aarch64"
assert platform_identifier in ["Linux_32bit", "Linux_64bit", "Windows_64bit", "Linux_ppc64le", "Linux_aarch64"], "Result %s is not of expected platform" % platform_identifier

valid_file_name_characters = "-_.() " + string.ascii_letters + string.digits
def string_to_valid_file_name(s):
    """
    Replaces invalid characters from string and replaces with dot '.'

    """
    result = []
    for ch in s:
        if ch in valid_file_name_characters:
            result.append(ch)
        else:
            result.append(".")
    return string.join(result, '')

def gen_diff(left, right):
    import difflib
    s = difflib.SequenceMatcher(None, left, right)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            for k in range(i1,i2):
                l = left[k]
                r = right[k-i1+j1]
                yield (" ", l, r) 
        elif tag == "insert":
            for k in range(j1,j2):
                r = right[k]
                yield ("+", "", r) 
        elif tag == "delete":
            for k in range(i1,i2):
                l = left[k]
                yield ("-", l, "")
        elif tag == "replace":
            for k in range(i1,i2):
                l = left[k]
                r = right[k-i1+j1]
                # difflib combines blocks for some reason and returns "replace" tag
                # for lines that are the same. Let's fix that
                if l == r: 
                    yield (" ", l ,r)
                else:
                    yield ("|", l ,r)

def plural_s(val):
    """
    returns "s" if val > 1 or "" otherwise.
    Can be used in strings to have proper plural form.

    """
    if val > 1:
        return "s"
    return ""

def chunks(l, n):
    """
    returns list of list of length n.
    E.g. chunks([1, 2, 3, 4, 5], 2) returns [[1, 2],  [3, 4], [5]]
    """
    return [l[i:i+n] for i in range(0, len(l), n)]

def format_dev_sub_dev_id(pciIdPair):
    """
    pciIdPair (int pci device id, int pci sub device id or None)
    """
    if pciIdPair[1] is None:
        return "(0x%08X, None)" % pciIdPair[0]
    return "(0x%08X, 0x%08X)" % pciIdPair
        
# permission of:      other                         owner                         group
stat_everyone_read_write = stat.S_IROTH | stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP

# Exit if the current (effective) user can't create a file in the base test directory
def verify_file_permissions(user):

    # Not a complete check, but enough to verify absolute path permission issues
    try:
        filename = os.path.join(os.path.realpath(script_dir), "test.txt")
        f = open(filename, "w")
        f.write("Permission test")
        f.close()
        os.remove(filename)
    except:
        print "Please run the test framework under a less restrictive directory, with RW access to the full path."
        print "The user '%s' doesn't have sufficient permissions here." % user
        sys.exit(1)

# Exit if either the current user, or specified non-root user appear to lack sufficient
# file system permissions for the test framework
def verify_user_file_permissions():

    # Check current user
    verify_file_permissions(getpass.getuser())

    # Check non-root user, if specified
    user = option_parser.options.non_root_user
    if user:
        try:
            get_user_idinfo(user)
        except KeyError:
            print "User '%s' doesn't exist" % user
            sys.exit(1)

        with test_utils.RunAsNonRoot(reload_driver=False):
            verify_file_permissions(user)

## Util method to check if the mps server is running in the background
def is_mps_server_running():
    """
    Returns True if MPS server is running on the system
    """

    if is_linux():
        ## MPS server is only supported on Linux.
        cmd = 'ps -aux | grep lwpu-lwca | tr -s " " | cut -d " " -f 11'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if "lwpu-lwca-mps-server" in out.rstrip():
            return True
        elif "lwpu-lwca-mps-control" in out.rstrip():
            return True
        else:
            return False
    else:
        ## MPS server is not supported on Windows. Return False for Windows
        return False

def shorten_path(path, shorten_to_levels=2):
    '''
    Given a path, return a path of only the last "shorten_to_levels" levels.
    For example, shorten_path('a/b/c/d/e', 2) => "d/e"
    '''
    path = os.path.normpath(path)
    shortened_paths = path.split(os.sep)[-shorten_to_levels:]
    return os.path.join(*shortened_paths)

def create_dir(path):
    '''
    Create the full directory structure specified by path.  If the directory cannot be created 
    due to permission issues or because part of the path already exists and is not a directory
    then an OSError is raised.
    '''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def wait_for_pid_to_die(pid):
    '''This function returns once the pid no longer exists'''
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
