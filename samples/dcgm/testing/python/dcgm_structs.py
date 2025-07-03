##
# Python bindings for "dcgm_structs.h"
##

from ctypes import *
from ctypes.util import find_library
import sys
import os
import threading
import string
import json
import dcgmvalue

DCGM_MAX_STR_LENGTH                   =   256
DCGM_MAX_NUM_DEVICES                  =   16
DCGM_MAX_NUM_SWITCHES                 =   12
DCGM_LWLINK_MAX_LINKS_PER_GPU         =   12
DCGM_LWLINK_MAX_LINKS_PER_GPU_LEGACY1 =   6
DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH    =   18
DCGM_MAX_CLOCKS                       =   256
DCGM_MAX_NUM_GROUPS                   =   64
DCGM_MAX_BLOB_LENGTH                  =   4096
DCGM_MAX_VGPU_INSTANCES_PER_PGPU      =   32
DCGM_MAX_NUM_VGPU_DEVICES             =   DCGM_MAX_NUM_DEVICES * DCGM_MAX_VGPU_INSTANCES_PER_PGPU 
DCGM_VGPU_NAME_BUFFER_SIZE            =   64
DCGM_GRID_LICENSE_BUFFER_SIZE         =   128
DCGM_MAX_VGPU_TYPES_PER_PGPU          =   32
DCGM_DEVICE_UUID_BUFFER_SIZE          =   80
DCGM_MAX_FBC_SESSIONS                 =   256

#When more than one value is returned from a query, which order should it be returned in?
DCGM_ORDER_ASCENDING  = 1
DCGM_ORDER_DESCENDING = 2

DCGM_OPERATION_MODE_AUTO   = 1
DCGM_OPERATION_MODE_MANUAL = 2

DCGM_ENCODER_QUERY_H264 = 0
DCGM_ENCODER_QUERY_HEVC = 1

DCGM_FBC_SESSION_TYPE_UNKNOWN = 0   # Unknown
DCGM_FBC_SESSION_TYPE_TOSYS   = 1   # FB capture for a system buffer
DCGM_FBC_SESSION_TYPE_LWDA    = 2   # FB capture for a lwca buffer
DCGM_FBC_SESSION_TYPE_VID     = 3   # FB capture for a Vid buffer
DCGM_FBC_SESSION_TYPE_HWENC   = 4   # FB capture for a LWENC HW buffer

## C Type mappings ##
## Enums

# Return types
_dcgmReturn_t = c_uint
DCGM_ST_OK                          =  0   # Success
DCGM_ST_BADPARAM                    = -1   # A bad parameter was passed to a function
DCGM_ST_GENERIC_ERROR               = -3   # A generic, unspecified error
DCGM_ST_MEMORY                      = -4   # An out of memory error oclwred
DCGM_ST_NOT_CONFIGURED              = -5   # Setting not configured
DCGM_ST_NOT_SUPPORTED               = -6   # Feature not supported
DCGM_ST_INIT_ERROR                  = -7   # DCGM Init error
DCGM_ST_LWML_ERROR                  = -8   # When LWML returns error.
DCGM_ST_PENDING                     = -9   # Object is in pending state of something else
DCGM_ST_UNINITIALIZED               = -10  # Object is in undefined state
DCGM_ST_TIMEOUT                     = -11  # Requested operation timed out
DCGM_ST_VER_MISMATCH                = -12  # Version mismatch between received and understood API
DCGM_ST_UNKNOWN_FIELD               = -13  # Unknown field id
DCGM_ST_NO_DATA                     = -14  # No data is available
DCGM_ST_STALE_DATA                  = -15
DCGM_ST_NOT_WATCHED                 = -16  # The given field is not being updated by the cache manager
DCGM_ST_NO_PERMISSION               = -17  # We are not permissioned to perform the desired action
DCGM_ST_GPU_IS_LOST                 = -18  # GPU is no longer reachable 
DCGM_ST_RESET_REQUIRED              = -19  # GPU requires a reset 
DCGM_ST_FUNCTION_NOT_FOUND          = -20  # Unable to find function
DCGM_ST_CONNECTION_NOT_VALID        = -21  # Connection to the host engine is not valid any longer
DCGM_ST_GPU_NOT_SUPPORTED           = -22  # This GPU is not supported by DCGM
DCGM_ST_GROUP_INCOMPATIBLE          = -23  # The GPUs of the provided group are not compatible with each other for the requested operation
DCGM_ST_MAX_LIMIT                   = -24
DCGM_ST_LIBRARY_NOT_FOUND           = -25  # DCGM library could not be found
DCGM_ST_DUPLICATE_KEY               = -26  #Duplicate key passed to the function
DCGM_ST_GPU_IN_SYNC_BOOST_GROUP     = -27  #GPU is already a part of a sync boost group
DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP = -28  #GPU is a not a part of sync boost group
DCGM_ST_REQUIRES_ROOT               = -29  #This operation cannot be performed when the host engine is running as non-root
DCGM_ST_LWVS_ERROR                  = -30  #DCGM GPU Diagnostic was successfully exelwted, but reported an error.
DCGM_ST_INSUFFICIENT_SIZE           = -31  #An input argument is not large enough
DCGM_ST_FIELD_UNSUPPORTED_BY_API    = -32  #The given field ID is not supported by the API being called
DCGM_ST_MODULE_NOT_LOADED           = -33  #This request is serviced by a module of DCGM that is not lwrrently loaded
DCGM_ST_IN_USE                      = -34  #The requested operation could not be completed because the affected resource is in use
DCGM_ST_GROUP_IS_EMPTY              = -35  # The specified group is empty and this operation is not valid with an empty group
DCGM_ST_PROFILING_NOT_SUPPORTED     = -36  # Profiling is not supported for this group of GPUs or GPU
DCGM_ST_PROFILING_LIBRARY_ERROR     = -37  # The third-party Profiling module returned an unrecoverable error
DCGM_ST_PROFILING_MULTI_PASS        = -38  # The requested profiling metrics cannot be collected in a single pass
DCGM_ST_DIAG_ALREADY_RUNNING        = -39  # A diag instance is already running, cannot run a new diag until the current one finishes.
DCGM_ST_DIAG_BAD_JSON               = -40  # The DCGM GPU Diagnostic returned JSON that cannot be parsed
DCGM_ST_DIAG_BAD_LAUNCH             = -41  # Error while launching the DCGM GPU Diagnostic
DCGM_ST_DIAG_VARIANCE               = -42  # There is too much variance while training the diagnostic
DCGM_ST_DIAG_THRESHOLD_EXCEEDED     = -43  # A field value met or exceeded the error threshold.
DCGM_ST_INSUFFICIENT_DRIVER_VERSION = -44  # The installed driver version is insufficient for this API

DCGM_GROUP_DEFAULT = 0  # All the GPUs on the node are added to the group
DCGM_GROUP_EMPTY   = 1  # Creates an empty group
DCGM_GROUP_DEFAULT_LWSWITCHES = 2 # All LwSwitches of the node are added to the group

DCGM_GROUP_ALL_GPUS = 0x7fffffff
DCGM_GROUP_ALL_LWSWITCHES = 0x7ffffffe

DCGM_GROUP_MAX_ENTITIES = 64 #Maximum number of entities per entity group

DCGM_CONFIG_TARGET_STATE  = 0          # The target configuration values to be applied
DCGM_CONFIG_LWRRENT_STATE = 1          # The current configuration state

DCGM_CONFIG_POWER_CAP_INDIVIDUAL = 0 # Represents the power cap to be applied for each member of the group
DCGM_CONFIG_POWER_BUDGET_GROUP   = 1 # Represents the power budget for the entire group

DCGM_CONFIG_COMPUTEMODE_DEFAULT = 0          # Default compute mode -- multiple contexts per device
DCGM_CONFIG_COMPUTEMODE_PROHIBITED = 1       # Compute-prohibited mode -- no contexts per device
DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS = 2 #* Compute-exclusive-process mode -- only one context per device, usable from multiple threads at a time

DCGM_TOPOLOGY_BOARD = 0x1
DCGM_TOPOLOGY_SINGLE = 0x2
DCGM_TOPOLOGY_MULTIPLE = 0x4
DCGM_TOPOLOGY_HOSTBRIDGE = 0x8
DCGM_TOPOLOGY_CPU = 0x10
DCGM_TOPOLOGY_SYSTEM = 0x20
DCGM_TOPOLOGY_LWLINK1 = 0x0100
DCGM_TOPOLOGY_LWLINK2 = 0x0200
DCGM_TOPOLOGY_LWLINK3 = 0x0400
DCGM_TOPOLOGY_LWLINK4 = 0x0800
DCGM_TOPOLOGY_LWLINK5 = 0x1000
DCGM_TOPOLOGY_LWLINK6 = 0x2000

# Diagnostic per gpu tests - fixed indices for dcgmDiagResponsePerGpu_t.results[]
DCGM_MEMORY_INDEX           = 0
DCGM_DIAGNOSTIC_INDEX       = 1
DCGM_PCI_INDEX              = 2
DCGM_SM_PERF_INDEX          = 3
DCGM_TARGETED_PERF_INDEX    = 4
DCGM_TARGETED_POWER_INDEX   = 5
DCGM_MEMORY_BANDWIDTH_INDEX = 6
DCGM_PER_GPU_TEST_COUNT     = 7

# DCGM Diag Level One test indices
DCGM_SWTEST_BLACKLIST            = 0
DCGM_SWTEST_LWML_LIBRARY         = 1
DCGM_SWTEST_LWDA_MAIN_LIBRARY    = 2
DCGM_SWTEST_LWDA_RUNTIME_LIBRARY = 3
DCGM_SWTEST_PERMISSIONS          = 4
DCGM_SWTEST_PERSISTENCE_MODE     = 5
DCGM_SWTEST_ELWIRONMENT          = 6
DCGM_SWTEST_PAGE_RETIREMENT      = 7
DCGM_SWTEST_GRAPHICS_PROCESSES   = 8
DCGM_SWTEST_INFOROM              = 9

# This test is only run by itself, so it can use the 0 slot
DCGM_CONTEXT_CREATE_INDEX = 0

class DCGM_INTROSPECT_STATE(object):
    DISABLED = 0
    ENABLED = 1

# Lib loading
dcgmLib = None
libLoadLock = threading.Lock()
_dcgmLib_refcount = 0 # Incremented on each dcgmInit and decremented on dcgmShutdown


class DCGMError(Exception):
    """ Class to return error values for DCGM """
    _valClassMapping = dict()
    # List of lwrrently known error codes
    _error_code_to_string = {
        DCGM_ST_OK:                          "Success",
        DCGM_ST_BADPARAM:                    "Bad parameter passed to function",
        DCGM_ST_GENERIC_ERROR:               "Generic unspecified error",
        DCGM_ST_MEMORY:                      "Out of memory error",
        DCGM_ST_NOT_CONFIGURED:              "Setting not configured",
        DCGM_ST_NOT_SUPPORTED:               "Feature not supported",
        DCGM_ST_INIT_ERROR:                  "DCGM initialization error",
        DCGM_ST_LWML_ERROR:                  "LWML error",
        DCGM_ST_PENDING:                     "Object is in a pending state",
        DCGM_ST_UNINITIALIZED:               "Object is in an undefined state",
        DCGM_ST_TIMEOUT:                     "Timeout",
        DCGM_ST_VER_MISMATCH:                "API version mismatch",
        DCGM_ST_UNKNOWN_FIELD:               "Unknown field",
        DCGM_ST_NO_DATA:                     "No data is available",
        DCGM_ST_STALE_DATA:                  "Data is considered stale",
        DCGM_ST_NOT_WATCHED:                 "Field is not being updated",
        DCGM_ST_NO_PERMISSION:               "Not permissioned",
        DCGM_ST_GPU_IS_LOST:                 "GPU is unreachable",
        DCGM_ST_RESET_REQUIRED:              "GPU requires a reset",
        DCGM_ST_FUNCTION_NOT_FOUND:          "Unable to find function",
        DCGM_ST_CONNECTION_NOT_VALID:        "The connection to the host engine is not valid any longer",
        DCGM_ST_GPU_NOT_SUPPORTED:           "This GPU is not supported by DCGM",
        DCGM_ST_GROUP_INCOMPATIBLE:          "GPUs are incompatible with each other for the requested operation",
        DCGM_ST_MAX_LIMIT:                   "Max limit reached for the object",
        DCGM_ST_LIBRARY_NOT_FOUND:           "DCGM library could not be found",
        DCGM_ST_DUPLICATE_KEY:               "Duplicate key passed to function",
        DCGM_ST_GPU_IN_SYNC_BOOST_GROUP:     "GPU is already a part of a sync boost group",
        DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP: "GPU is not a part of the sync boost group",
        DCGM_ST_REQUIRES_ROOT:               "This operation is not supported when the host engine is running as non root",
        DCGM_ST_LWVS_ERROR:                  "DCGM GPU Diagnostic returned an error.",
        DCGM_ST_INSUFFICIENT_SIZE:           "An input argument is not large enough",
        DCGM_ST_FIELD_UNSUPPORTED_BY_API:    "The given field ID is not supported by the API being called",
        DCGM_ST_MODULE_NOT_LOADED:           "This request is serviced by a module of DCGM that is not lwrrently loaded",
        DCGM_ST_IN_USE:                      "The requested operation could not be completed because the affected resource is in use",
        DCGM_ST_GROUP_IS_EMPTY:              "The specified group is empty, and this operation is incompatible with an empty group",
        DCGM_ST_PROFILING_NOT_SUPPORTED:     "Profiling is not supported for this group of GPUs or GPU",
        DCGM_ST_PROFILING_LIBRARY_ERROR:     "The third-party Profiling module returned an unrecoverable error",
        DCGM_ST_PROFILING_MULTI_PASS:        "The requested profiling metrics cannot be collected in a single pass",
        DCGM_ST_DIAG_ALREADY_RUNNING:        "A diag instance is already running, cannot run a new diag until the current one finishes",
        DCGM_ST_DIAG_BAD_JSON:               "The GPU Diagnostic returned Json that cannot be parsed.",
        DCGM_ST_DIAG_BAD_LAUNCH:             "Error while launching the GPU Diagnostic.",
        DCGM_ST_DIAG_VARIANCE:               "The results of training DCGM GPU Diagnostic cannot be trusted because they vary too much from run to run",
        DCGM_ST_DIAG_THRESHOLD_EXCEEDED:     "A field value met or exceeded the error threshold.",
        DCGM_ST_INSUFFICIENT_DRIVER_VERSION: "The installed driver version is insufficient for this API"
    }

    def __new__(typ, value):
        """
        Maps value to a proper subclass of DCGMError.
        """
        if typ == DCGMError:
            typ = DCGMError._valClassMapping.get(value, typ)
        obj = Exception.__new__(typ)
        obj.info = None
        obj.value = value
        return obj

    def __str__(self):
        msg = None
        try:
            if self.value not in DCGMError._error_code_to_string:
                DCGMError._error_code_to_string[self.value] = str(_dcgmErrorString(self.value))
            msg = DCGMError._error_code_to_string[self.value]
        # Ensure we catch all exceptions, otherwise the error code will be hidden in a traceback
        except BaseException:
            msg = "DCGM Error with code %d" % self.value
        
        if self.info is not None:
            if msg[-1] == ".":
                msg = msg[:-1]
            msg += ": '%s'"  % self.info
        return msg

    def __eq__(self, other):
        return self.value == other.value
    
    def SetAdditionalInfo(self, msg):
        """
        Sets msg as additional information returned by the string representation of DCGMError and subclasses.
        Example output for DCGMError_Uninitialized subclass, with msg set to 'more info msg here' is 
        "DCGMError_Uninitialized: Object is in an undefined state: 'more info msg here'".
        
        Ensure that msg is a string or an object for which the __str__() method does not throw an error
        """
        self.info = msg

def dcgmExceptionClass(error_code):
    return DCGMError._valClassMapping.get(error_code)

def _extractDCGMErrorsAsClasses():
    '''
    Generates a hierarchy of classes on top of DCGMLError class.

    Each DCGM Error gets a new DCGMError subclass. This way try,except blocks can filter appropriate
    exceptions more easily.

    DCGMError is a parent class. Each DCGM_ST_* gets it's own subclass.
    e.g. DCGM_ST_UNINITIALIZED will be turned into DCGMError_Uninitialized
    '''
    this_module = sys.modules[__name__]
    dcgmErrorsNames = filter(lambda x: x.startswith("DCGM_ST_"), dir(this_module))
    for err_name in dcgmErrorsNames:
        # e.g. Turn DCGM_ST_UNINITIALIZED into DCGMError_Uninitialized
        class_name = "DCGMError_" + string.capwords(err_name.replace("DCGM_ST_", ""), "_").replace("_", "")
        err_val = getattr(this_module, err_name)
        def gen_new(val):
            def new(typ):
                obj = DCGMError.__new__(typ, val)
                return obj
            return new
        new_error_class = type(class_name, (DCGMError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        setattr(this_module, class_name, new_error_class)
        DCGMError._valClassMapping[err_val] = new_error_class
_extractDCGMErrorsAsClasses()


class struct_c_dcgmUnit_t(Structure):
    # Unit structures
    pass # opaque handle
_dcgmUnit_t = POINTER(struct_c_dcgmUnit_t)


class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" + string.join(result, ", ") + ")"

    def FieldsSizeof(self):
        size = 0
        for s,t in self._fields_:
            size = size + sizeof(t)
        return size

#JSON serializer for DCGM structures
class DcgmJSONEncoder(json.JSONEncoder):
    def default(self, o):   # pylint: disable=method-hidden
        if isinstance(o, _PrintableStructure):
            retVal = {}
            for fieldName, fieldType in o._fields_:
                subObj = getattr(o, fieldName)
                if isinstance(subObj, _PrintableStructure):
                    subObj = self.default(subObj)

                retVal[fieldName] = subObj

            return retVal
        elif isinstance(o, Array):
            retVal = []
            for i in range(len(o)):
                subVal = {}
                for fieldName, fieldType in o[i]._fields_:
                    subObj = getattr(o[i], fieldName)
                    if isinstance(subObj, _PrintableStructure):
                        subObj = self.default(subObj)

                    subVal[fieldName] = subObj

                retVal.append(subVal)
            return retVal

        #Let the parent class handle this/fail
        return json.JSONEncoder.default(self, o)

# Creates a unique version number for each struct
def make_dcgm_version(struct, ver):
    return sizeof(struct) | (ver << 24)

# Function access ##
_dcgmGetFunctionPointer_cache = dict() # function pointers are cached to prevent unnecessary libLoadLock locking
def _dcgmGetFunctionPointer(name):
    global dcgmLib

    if name in _dcgmGetFunctionPointer_cache:
        return _dcgmGetFunctionPointer_cache[name]
    
    libLoadLock.acquire()
    try:
        # ensure library was loaded
        if dcgmLib is None:
            raise DCGMError(DCGM_ST_UNINITIALIZED)
        try:
            _dcgmGetFunctionPointer_cache[name] = getattr(dcgmLib, name)
            return _dcgmGetFunctionPointer_cache[name]
        except AttributeError:
            raise DCGMError(DCGM_ST_FUNCTION_NOT_FOUND)
    finally:
        # lock is always freed
        libLoadLock.release()

# C function wrappers ##
def _LoadDcgmLibrary(libDcgmPath=''):
    '''
    Load the library if it isn't loaded already
    '''
    global dcgmLib

    if dcgmLib is None:
        # lock to ensure only one caller loads the library
        libLoadLock.acquire()

        try:
            # ensure the library still isn't loaded
            if dcgmLib is None:
                try:
                    if sys.platform[:3] == "win":
                        # cdecl calling convention
                        # load lwml.dll from %ProgramFiles%/LWPU Corporation/LWSMI/lwml.dll
                        dcgmLib = CDLL(os.path.join(os.getelw("ProgramFiles", "C:/Program Files"), "LWPU Corporation/LWSMI/dcgm.dll"))
                    else:
                        dcgmLib = CDLL(os.path.join(libDcgmPath, "libdcgm.so.1"))
                        
                except OSError as ose:
                    _dcgmCheckReturn(DCGM_ST_LIBRARY_NOT_FOUND)
                if dcgmLib is None:
                    _dcgmCheckReturn(DCGM_ST_LIBRARY_NOT_FOUND)
        finally:
            # lock is always freed
            libLoadLock.release()

def _dcgmInit(libDcgmPath=''):
    _LoadDcgmLibrary(libDcgmPath)
    # Atomically update refcount
    global _dcgmLib_refcount
    libLoadLock.acquire()
    _dcgmLib_refcount += 1
    libLoadLock.release()
    return None

def _dcgmCheckReturn(ret):
    if ret != DCGM_ST_OK:
        raise DCGMError(ret)
    return ret

def _dcgmShutdown():
    # Leave the library loaded, but shutdown the interface
    fn = _dcgmGetFunctionPointer("dcgmShutdown")
    ret = fn()
    _dcgmCheckReturn(ret)

    # Atomically update refcount
    global _dcgmLib_refcount
    libLoadLock.acquire()
    if 0 < _dcgmLib_refcount:
        _dcgmLib_refcount -= 1
    libLoadLock.release()
    return None

def _dcgmErrorString(result):
    fn = _dcgmGetFunctionPointer("dcgmErrorString")
    fn.restype = c_char_p # otherwise return is an int
    str = fn(result)
    return str

class c_dcgmConnectV2Params_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('persistAfterDisconnect', c_uint)
    ]

c_dcgmConnectV2Params_version1 = make_dcgm_version(c_dcgmConnectV2Params_v1, 1)

class c_dcgmConnectV2Params_v2(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('persistAfterDisconnect', c_uint),
        ('timeoutMs', c_uint),
        ('addressIsUnixSocket', c_uint)
    ]

c_dcgmConnectV2Params_version2 = make_dcgm_version(c_dcgmConnectV2Params_v2, 2)
c_dcgmConnectV2Params_version = c_dcgmConnectV2Params_version2

#Represents memory and proc clocks for a device
class c_dcgmClockSet_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('memClock', c_uint),         #/* Memory Clock */
        ('smClock',c_uint)          #/* SM Clock */
    ]

# Represents a entityGroupId + entityId pair to uniquely identify a given entityId inside
# a group of entities
class c_dcgmGroupEntityPair_t(_PrintableStructure):
    _fields_ = [
        ('entityGroupId', c_uint32), #Entity Group ID entity belongs to
        ('entityId', c_uint32) #Entity ID of the entity
    ]

# /**
#  * Structure to store information for DCGM group (v1, GPUs only). Use c_dcgmGroupInfo_v2 for
#  * groups containing non-GPU entities
#  */
class c_dcgmGroupInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('count', c_uint),
        ('gpuIdList', c_uint * DCGM_MAX_NUM_DEVICES),
        ('groupName', c_char * DCGM_MAX_STR_LENGTH)
    ]
c_dcgmGroupInfo_version1 = make_dcgm_version(c_dcgmGroupInfo_v1, 1)

# /**
#  * Structure to store information for DCGM group (v2)
#  */
class c_dcgmGroupInfo_v2(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('count', c_uint),
        ('groupName', c_char * DCGM_MAX_STR_LENGTH),
        ('entityList', c_dcgmGroupEntityPair_t * DCGM_GROUP_MAX_ENTITIES)
    ]
c_dcgmGroupInfo_version2 = make_dcgm_version(c_dcgmGroupInfo_v2, 2)

# /**
#  * Structure to represent error attributes
#  */
class c_dcgmErrorInfo_v1(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint),
        ('fieldId', c_ushort),
        ('status', c_int)
    ]

# /**
#  * Represents list of supported clocks for a device
#  */
class  c_dcgmDeviceSupportedClockSets_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('count', c_uint),
        ('clockSet', c_dcgmClockSet_v1 * DCGM_MAX_CLOCKS)
    ]

# /**
# * Represents accounting information for a device and pid
# */
class c_dcgmDevicePidAccountingStats_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('pid', c_uint32),
        ('gpuUtilization', c_uint32),
        ('memoryUtilization', c_uint32),
        ('maxMemoryUsage', c_uint64),
        ('startTimestamp', c_uint64),
        ('activeTimeUsec', c_uint64)
    ]

# /**
#  * Represents thermal information
#  */
class  c_dcgmDeviceThermals_v1(_PrintableStructure): 
    _fields_ = [
        ('version', c_uint),
        ('slowdownTemp', c_uint),
        ('shutdownTemp', c_uint) 
    ]

# /**
#  * Represents various power limits
#  */
class  c_dcgmDevicePowerLimits_v1(_PrintableStructure): 
    _fields_ = [
        ('version', c_uint),
        ('lwrPowerLimit', c_uint),
        ('defaultPowerLimit', c_uint),
        ('enforcedPowerLimit', c_uint),
        ('minPowerLimit', c_uint),
        ('maxPowerLimit', c_uint)
    ]

# /**
#  * Represents device identifiers
#  */
class c_dcgmDeviceIdentifiers_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('brandName', c_char * DCGM_MAX_STR_LENGTH),
        ('deviceName', c_char * DCGM_MAX_STR_LENGTH),
        ('pciBusId', c_char * DCGM_MAX_STR_LENGTH),
        ('serial', c_char * DCGM_MAX_STR_LENGTH),
        ('uuid', c_char * DCGM_MAX_STR_LENGTH),
        ('vbios', c_char * DCGM_MAX_STR_LENGTH),
        ('inforomImageVersion', c_char * DCGM_MAX_STR_LENGTH),
        ('pciDeviceId', c_uint32),
        ('pciSubSystemId', c_uint32),
        ('driverVersion', c_char * DCGM_MAX_STR_LENGTH),
        ('virtualizationMode', c_uint32)
    ]

# /**
#  * Represents memory utilization
#  */
class  c_dcgmDeviceMemoryUsage_v1(_PrintableStructure): 
    _fields_ = [
        ('version', c_uint),
        ('bar1Total', c_uint),
        ('fbTotal', c_uint),
        ('fbUsed', c_uint),
        ('fbFree', c_uint)
    ]

# /**
#  * Represents utilization values of vGPUs running on the device
#  */
class  c_dcgmDeviceVgpuUtilInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vgpuId', c_uint),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('enlwtil', c_uint),
        ('delwtil', c_uint)
    ]

# /**
#  * Utilization values for processes running within vGPU VMs using the device
#  */
class  c_dcgmDeviceVgpuProcessUtilInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vgpuId', c_uint),
        ('pid', c_uint),
        ('processName', c_char * DCGM_VGPU_NAME_BUFFER_SIZE),
        ('smUtil', c_uint),
        ('memUtil', c_uint),
        ('enlwtil', c_uint),
        ('delwtil', c_uint)
    ]

# /**
#  * Represents current encoder statistics for the given device/vGPU instance
#  */
class  c_dcgmDeviceEncStats_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('sessionCount', c_uint),
        ('averageFps', c_uint),
        ('averageLatency', c_uint)
    ]

# /**
#  * Represents information about active encoder sessions on the given vGPU instance
#  */
class  c_dcgmDeviceVgpuEncSessions_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vgpuId', c_uint),
        ('sessionId', c_uint),
        ('pid', c_uint),
        ('codecType', c_uint),
        ('hResolution', c_uint),
        ('vResolution', c_uint),
        ('averageFps', c_uint),
        ('averageLatency', c_uint)
    ]

# /**
#  * Represents current frame buffer capture sessions statistics for the given device/vGPU instance
#  */
class  c_dcgmDeviceFbcStats_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('sessionCount', c_uint),
        ('averageFps', c_uint),
        ('averageLatency', c_uint)
    ]

# /**
#  * Represents information about active FBC session on the given device/vGPU instance
#  */
class  c_dcgmDeviceFbcSessionInfo_t(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('sessionId', c_uint),
        ('pid', c_uint),
        ('vgpuId', c_uint),
        ('displayOrdinal', c_uint),
        ('sessionType', c_uint),
        ('sessionFlags', c_uint),
        ('hMaxResolution', c_uint),
        ('vMaxResolution', c_uint),
        ('hResolution', c_uint),
        ('vResolution', c_uint),
        ('averageFps', c_uint),
        ('averageLatency', c_uint)
    ]

# /**
#  * Represents all the active FBC sessions on the given device/vGPU instance
#  */
class  c_dcgmDeviceFbcSessions_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('sessionCount', c_uint),
        ('sessionInfo', c_dcgmDeviceFbcSessionInfo_t * DCGM_MAX_FBC_SESSIONS)
    ]

# /**
#  * Represents Vgpu id fields
#  */
class  c_dcgmDeviceVgpuIds_v1(_PrintableStructure): 
    _fields_ = [
        ('version', c_uint),
        ('unusedSupportedVgpuTypeCount', c_uint),
        ('unusedSupportedVgpuTypeIds', c_uint * DCGM_MAX_NUM_DEVICES),
        ('creatableVgpuTypeCount', c_uint),
        ('creatableVgpuTypeIds', c_uint * DCGM_MAX_NUM_DEVICES)
    ]

# /**
#  * Represents static info related to vGPU types supported on the device
#  */
class  c_dcgmDeviceVgpuTypeInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vgpuTypeId', c_uint),
        ('vgpuTypeName', c_char * DCGM_VGPU_NAME_BUFFER_SIZE),
        ('vgpuTypeClass', c_char * DCGM_VGPU_NAME_BUFFER_SIZE),
        ('vgpuTypeLicense', c_char * DCGM_GRID_LICENSE_BUFFER_SIZE),
        ('deviceId', c_uint),
        ('subsystemId', c_uint),
        ('numDisplayHeads', c_uint),
        ('maxInstances', c_uint),
        ('frameRateLimit', c_uint),
        ('maxResolutionX', c_uint),
        ('maxResolutionY', c_uint),
        ('fbTotal', c_uint)
    ]

# /**
#  * Represents attributes corresponding to a device
#  */
class c_dcgmDeviceAttributes_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('clockSets', c_dcgmDeviceSupportedClockSets_v1),
        ('thermalSettings', c_dcgmDeviceThermals_v1),
        ('powerLimits', c_dcgmDevicePowerLimits_v1),
        ('identifiers', c_dcgmDeviceIdentifiers_v1),
        ('memoryUsage', c_dcgmDeviceMemoryUsage_v1),
        ('unusedVgpuIds', c_dcgmDeviceVgpuIds_v1),
        ('unusedActiveVgpuInstanceCount', c_uint),
        ('unusedVgpuInstanceIds', c_uint * DCGM_MAX_NUM_DEVICES)
    ]

dcgmDeviceAttributes_version1 = make_dcgm_version(c_dcgmDeviceAttributes_v1, 1)

# /**
#  * Represents vGPU attributes corresponding to a device
#  */
class c_dcgmVgpuDeviceAttributes_v6(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('activeVgpuInstanceCount', c_uint),
        ('activeVgpuInstanceIds', c_uint * DCGM_MAX_VGPU_INSTANCES_PER_PGPU),
        ('creatableVgpuTypeCount', c_uint),
        ('creatableVgpuTypeIds', c_uint * DCGM_MAX_VGPU_TYPES_PER_PGPU),
        ('supportedVgpuTypeCount', c_uint),
        ('supportedVgpuTypeInfo', c_dcgmDeviceVgpuTypeInfo_v1 * DCGM_MAX_VGPU_TYPES_PER_PGPU),
        ('vgpuUtilInfo', c_dcgmDeviceVgpuUtilInfo_v1 * DCGM_MAX_VGPU_TYPES_PER_PGPU),
        ('gpuUtil', c_uint),
        ('memCopyUtil', c_uint),
        ('enlwtil', c_uint),
        ('delwtil', c_uint)
    ]

dcgmVgpuDeviceAttributes_version6 = make_dcgm_version(c_dcgmVgpuDeviceAttributes_v6, 1)

# /**
#  * Represents attributes specific to vGPU instance
#  */
class c_dcgmVgpuInstanceAttributes_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('vmId', c_char * DCGM_DEVICE_UUID_BUFFER_SIZE),
        ('vmName', c_char * DCGM_DEVICE_UUID_BUFFER_SIZE),
        ('vgpuTypeId', c_uint),
        ('vgpuUuid', c_char * DCGM_DEVICE_UUID_BUFFER_SIZE),
        ('vgpuDriverVersion', c_char * DCGM_DEVICE_UUID_BUFFER_SIZE),
        ('fbUsage', c_uint),
        ('licenseStatus', c_uint),
        ('frameRateLimit', c_uint)
    ]

dcgmVgpuInstanceAttributes_version1 = make_dcgm_version(c_dcgmVgpuInstanceAttributes_v1, 1)

class c_dcgmConfigPowerLimit(_PrintableStructure):
    _fields_ = [

        ('type', c_uint),
        ('val', c_uint)
    ]


class c_dcgmConfigPerfStateSettings_t(_PrintableStructure):
    _fields_ = [
        ('syncBoost', c_uint),
        ('targetClocks', c_dcgmClockSet_v1),
    ]

# Structure to represent default configuration for a device
class c_dcgmDeviceConfig_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
	    ('gpuId', c_uint),
        ('mEccMode', c_uint),
        ('mComputeMode', c_uint),
        ('mPerfState', c_dcgmConfigPerfStateSettings_t),
        ('mPowerLimit', c_dcgmConfigPowerLimit)
    ]

dcgmDeviceConfig_version1 = make_dcgm_version(c_dcgmDeviceConfig_v1, 1)

# Structure to represent default vGPU configuration for a device
class c_dcgmDeviceVgpuConfig_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
	    ('gpuId', c_uint),
        ('mEccMode', c_uint),
        ('mComputeMode', c_uint),
        ('mPerfState', c_dcgmConfigPerfStateSettings_t),
        ('mPowerLimit', c_dcgmConfigPowerLimit)
    ]

    def SetBlank(self):
        #Does not set version or gpuId
        self.mEccMode = dcgmvalue.DCGM_INT32_BLANK
        self.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
        self.mPerfState.targetClocks.memClock =  dcgmvalue.DCGM_INT32_BLANK
        self.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
        self.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
        self.mPowerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL
        self.mPowerLimit.val = dcgmvalue.DCGM_INT32_BLANK

dcgmDeviceVgpuConfig_version1 = make_dcgm_version(c_dcgmDeviceVgpuConfig_v1, 1)

# Structure to receive update on the list of metrics.
class c_dcgmPolicyUpdate_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('power', c_uint)
    ]

dcgmPolicyUpdate_version1 = make_dcgm_version(c_dcgmPolicyUpdate_v1, 1)

# Represents a Callback to receive power updates from the host engine
_dcgmRecvUpdates_t = c_void_p

# Define the structure that contains specific policy information
class c_dcgmPolicyViolation_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('notifyOnEccDbe', c_uint),
        ('notifyOnPciEvent', c_uint),
        ('notifyOnMaxRetiredPages', c_uint)
    ]

dcgmPolicyViolation_version1 = make_dcgm_version(c_dcgmPolicyViolation_v1, 1)

class c_dcgmWatchFieldValue_v1(_PrintableStructure):
    _fields_ = []

dcgmWatchFieldValue_version1 = make_dcgm_version(c_dcgmWatchFieldValue_v1, 1)

class c_dcgmUnwatchFieldValue_v1(_PrintableStructure):
    _fields_ = []

dcgmUnwatchFieldValue_version1 = make_dcgm_version(c_dcgmUnwatchFieldValue_v1, 1)

class c_dcgmUpdateAllFields_v1(_PrintableStructure):
    _fields_ = []

dcgmUpdateAllFields_version1 = make_dcgm_version(c_dcgmUpdateAllFields_v1, 1)

dcgmGetMultipleValuesForField_version1 = 1

# policy enums
DCGM_POLICY_COND_DBE = 0x1
DCGM_POLICY_COND_PCI = 0x2
DCGM_POLICY_COND_MAX_PAGES_RETIRED = 0x4
DCGM_POLICY_COND_THERMAL = 0x8
DCGM_POLICY_COND_POWER = 0x10
DCGM_POLICY_COND_LWLINK = 0x20
DCGM_POLICY_COND_XID = 0x40
DCGM_POLICY_COND_MAX = 7

DCGM_POLICY_MODE_AUTOMATED = 0
DCGM_POLICY_MODE_MANUAL = 1

DCGM_POLICY_ISOLATION_NONE = 0

DCGM_POLICY_ACTION_NONE = 0
DCGM_POLICY_ACTION_GPURESET = 1

DCGM_POLICY_VALID_NONE = 0
DCGM_POLICY_VALID_SV_SHORT = 1
DCGM_POLICY_VALID_SV_MED = 2
DCGM_POLICY_VALID_SV_LONG = 3

DCGM_POLICY_FAILURE_NONE = 0

DCGM_DIAG_LVL_ILWALID = 0
DCGM_DIAG_LVL_SHORT   = 10
DCGM_DIAG_LVL_MED     = 20
DCGM_DIAG_LVL_LONG    = 30

DCGM_DIAG_RESULT_PASS = 0
DCGM_DIAG_RESULT_SKIP = 1
DCGM_DIAG_RESULT_WARN = 2
DCGM_DIAG_RESULT_FAIL = 3
DCGM_DIAG_RESULT_NOT_RUN = 4

class c_dcgmPolicyConditionParmTypes_t(Union):
    _fields_ = [
        ('boolean', c_bool),
        ('llval', c_longlong),
    ]

class c_dcgmPolicyConditionParms_t(_PrintableStructure):
    _fields_ = [
        ('tag', c_uint),
        ('val', c_dcgmPolicyConditionParmTypes_t)
    ]

class c_dcgmPolicy_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('condition', c_uint),  # an OR'd list of DCGM_POLICY_COND_*
        ('mode', c_uint),
        ('isolation', c_uint),
        ('action', c_uint),
        ('validation', c_uint),
        ('response', c_uint),
        ('parms', c_dcgmPolicyConditionParms_t * DCGM_POLICY_COND_MAX)
    ]

dcgmPolicy_version1 = make_dcgm_version(c_dcgmPolicy_v1, 1)

class c_dcgmPolicyConditionPci_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),  # timestamp of the error
        ("counter", c_uint)         # value of the PCIe replay counter
    ]
    
class c_dcgmPolicyConditionDbe_t(_PrintableStructure):
    LOCATIONS = {
        'L1': 0,
        'L2': 1,
        'DEVICE': 2,
        'REGISTER': 3,
        'TEXTURE': 4
    }
    
    _fields_ = [
        ("timestamp", c_longlong),  # timestamp of the error 
        ("location", c_int),        # location of the error (one of self.LOCATIONS)
        ("numerrors", c_uint)       # number of errors       
    ]
    
class c_dcgmPolicyConditionMpr_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),  # timestamp of the error             
        ("sbepages", c_uint),       # number of pending pages due to SBE 
        ("dbepages", c_uint)        # number of pending pages due to DBE 
    ]

class c_dcgmPolicyConditionThermal_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),      # timestamp of the error
        ("thermalViolation", c_uint)    # Temperature reached that violated policy
    ]
    
class c_dcgmPolicyConditionPower_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),      # timestamp of the error
        ("powerViolation", c_uint)      # Power value reached that violated policyy
    ]

class c_dcgmPolicyConditionLwlink_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),      # timestamp of the error
        ("fieldId", c_ushort),          # FieldId of the lwlink error counter
        ("counter", c_uint)      # Error value reached that violated policyy
    ]
class c_dcgmPolicyConditionXID_t(_PrintableStructure):
    _fields_ = [
        ("timestamp", c_longlong),      # timestamp of the error
        ("errnum", c_uint)              # XID error number
    ]
class c_dcgmPolicyCallbackResponse_v1(_PrintableStructure):
    class Value(Union):
        # implement more of the fields when a test requires them
        _fields_ = [
            ("dbe", c_dcgmPolicyConditionDbe_t),            #  ECC DBE return structure                   
            ("pci", c_dcgmPolicyConditionPci_t),            #  PCI replay error return structure          
            ("mpr", c_dcgmPolicyConditionMpr_t),            #  Max retired pages limit return structure   
            ("thermal", c_dcgmPolicyConditionThermal_t),    #  Thermal policy violations return structure 
            ("power", c_dcgmPolicyConditionPower_t),        #  Power policy violations return structure   
            ("lwlink", c_dcgmPolicyConditionLwlink_t),      # Lwlink policy violations return structure..
            ("xid", c_dcgmPolicyConditionXID_t)             # XID policy violations return structure
        ]
    
    _fields_ = [
        ("version", c_uint),
        ("condition", c_int),   # an OR'ed list of DCGM_POLICY_COND_*
        ("val", Value)
    ]

class c_dcgmFieldValue_v1_value(Union):
    _fields_ = [
        ('i64', c_int64),
        ('dbl', c_double),
        ('str', c_char * DCGM_MAX_STR_LENGTH),
        ('blob', c_byte * DCGM_MAX_BLOB_LENGTH)
    ]

# This structure is used to represent value for the field to be queried.
class c_dcgmFieldValue_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('fieldId', c_ushort),
        ('fieldType', c_short),
        ('status', c_int),
        ('ts', c_int64),
        ('value', c_dcgmFieldValue_v1_value)
    ]

dcgmFieldValue_version1 = make_dcgm_version(c_dcgmFieldValue_v1, 1)

# This structure is used to represent value for the field to be queried (version 2)
class c_dcgmFieldValue_v2(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('entityGroupId', c_uint),
        ('entityId', c_uint),
        ('fieldId', c_ushort),
        ('fieldType', c_short),
        ('status', c_int),
        ('unused', c_uint),
        ('ts', c_int64),
        ('value', c_dcgmFieldValue_v1_value)
    ]

dcgmFieldValue_version2 = make_dcgm_version(c_dcgmFieldValue_v2, 2)


#Field value flags used by dcgm_agent.dcgmEntitiesGetLatestValues()
DCGM_FV_FLAG_LIVE_DATA = 0x00000001

DCGM_HEALTH_WATCH_PCIE      = 0x1
DCGM_HEALTH_WATCH_LWLINK    = 0x2
DCGM_HEALTH_WATCH_PMU       = 0x4
DCGM_HEALTH_WATCH_MLW       = 0x8
DCGM_HEALTH_WATCH_MEM       = 0x10
DCGM_HEALTH_WATCH_SM        = 0x20
DCGM_HEALTH_WATCH_INFOROM   = 0x40
DCGM_HEALTH_WATCH_THERMAL   = 0x80
DCGM_HEALTH_WATCH_POWER     = 0x100
DCGM_HEALTH_WATCH_DRIVER    = 0x200
DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL = 0x400
DCGM_HEALTH_WATCH_LWSWITCH_FATAL = 0x800
DCGM_HEALTH_WATCH_ALL       = 0xFFFFFFFF
DCGM_HEALTH_WATCH_COUNT_V1     = 10
DCGM_HEALTH_WATCH_COUNT_V2     = 12

DCGM_HEALTH_RESULT_PASS = 0
DCGM_HEALTH_RESULT_WARN = 10
DCGM_HEALTH_RESULT_FAIL = 20

class c_dcgmDiagErrorDetail_t(_PrintableStructure):
    _fields_ = [
        ('msg', c_char * 1024),
        ('code', c_uint)
    ]

class c_dcgmHealthResponseGpuSystem_v1(_PrintableStructure):
    _fields_ = [
        ('system', c_uint),
        ('health', c_uint),
        ('errorString', c_char * 1024)
    ]

class c_dcgmHealthResponseGpu_t(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint),
        ('overallHealth', c_uint),
        ('incidentCount', c_uint),
        ('systems', c_dcgmHealthResponseGpuSystem_v1 * DCGM_HEALTH_WATCH_COUNT_V1)
    ]

class c_dcgmHealthResponse_v1(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint),
        ('overallHealth', c_uint),
        ('gpuCount', c_uint),
        ('gpu', c_dcgmHealthResponseGpu_t * DCGM_MAX_NUM_DEVICES)
    ]

dcgmHealthResponse_version1 = make_dcgm_version(c_dcgmHealthResponse_v1, 1)

class c_dcgmHealthResponseEntity_v1(_PrintableStructure):
    _fields_ = [
        ('entityGroupId', c_uint32),
        ('entityId', c_uint32),
        ('overallHealth', c_uint32),
        ('incidentCount', c_uint),
        ('systems', c_dcgmHealthResponseGpuSystem_v1 * DCGM_HEALTH_WATCH_COUNT_V2)
    ]

class c_dcgmHealthResponse_v2(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint32),
        ('overallHealth', c_uint32),
        ('entityCount', c_uint32),
        ('entities', c_dcgmHealthResponseEntity_v1 * DCGM_GROUP_MAX_ENTITIES)
    ]

dcgmHealthResponse_version2 = make_dcgm_version(c_dcgmHealthResponse_v2, 2)

class c_dcgmHealthResponseGpuSystem_v2(_PrintableStructure):
    _fields_ = [
        ('system', c_uint),
        ('health', c_uint),
        ('errors', c_dcgmDiagErrorDetail_t * 4),
        ('errorCount', c_uint),
    ]

class c_dcgmHealthResponseEntity_v2(_PrintableStructure):
    _fields_ = [
        ('entityGroupId', c_uint32),
        ('entityId', c_uint32),
        ('overallHealth', c_uint32),
        ('incidentCount', c_uint),
        ('systems', c_dcgmHealthResponseGpuSystem_v2 * DCGM_HEALTH_WATCH_COUNT_V2)
    ]

class c_dcgmHealthResponse_v3(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('version', c_uint32),
        ('overallHealth', c_uint32),
        ('entityCount', c_uint32),
        ('entities', c_dcgmHealthResponseEntity_v2 * DCGM_GROUP_MAX_ENTITIES)
    ]

dcgmHealthResponse_version3 = make_dcgm_version(c_dcgmHealthResponse_v3, 3)


#Pid info structs
class c_dcgmStatSummaryInt64_t(_PrintableStructure):
    _fields_ = [
        ('milwalue', c_int64),
        ('maxValue', c_int64),
        ('average', c_int64)
    ]

class c_dcgmStatSummaryInt32_t(_PrintableStructure):
    _fields_ = [
        ('milwalue', c_int32),
        ('maxValue', c_int32),
        ('average', c_int32)
    ]

class c_dcgmStatSummaryFp64_t(_PrintableStructure):
    _fields_ = [
        ('milwalue', c_double),
        ('maxValue', c_double),
        ('average', c_double)
    ]

class c_dcgmProcessUtilInfo_t(_PrintableStructure):
    _fields_ = [
        ('pid', c_uint),
        ('smUtil', c_double),
        ('memUtil', c_double)
    ]

class c_dcgmHealthResponseInfo_t(_PrintableStructure):
    _fields_ = [
        ('system', c_uint),
        ('health', c_uint)
    ]


DCGM_MAX_PID_INFO_NUM = 16
class c_dcgmPidSingleInfo_t(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint32),
        ('energyConsumed', c_int64),
        ('pcieRxBandwidth', c_dcgmStatSummaryInt64_t),
        ('pcieTxBandwidth', c_dcgmStatSummaryInt64_t),
        ('pcieReplays', c_int64),
        ('startTime', c_int64),
        ('endTime', c_int64),
        ('processUtilization', c_dcgmProcessUtilInfo_t),
        ('smUtilization', c_dcgmStatSummaryInt32_t),
        ('memoryUtilization', c_dcgmStatSummaryInt32_t),
        ('eccSingleBit', c_uint32),
        ('eccDoubleBit', c_uint32),
        ('memoryClock', c_dcgmStatSummaryInt32_t),
        ('smClock', c_dcgmStatSummaryInt32_t),
        ('numXidCriticalErrors', c_int32),
        ('xidCriticalErrorsTs', c_int64 * 10),
        ('numOtherComputePids', c_int32),
        ('otherComputePids', c_uint32 * DCGM_MAX_PID_INFO_NUM),
        ('numOtherGraphicsPids', c_int32),
        ('otherGraphicsPids', c_uint32 * DCGM_MAX_PID_INFO_NUM),
        ('maxGpuMemoryUsed', c_int64),
        ('powerViolationTime', c_int64),
        ('thermalViolationTime', c_int64),
        ('reliabilityViolationTime', c_int64),
        ('boardLimitViolationTime', c_int64),
        ('lowUtilizationTime', c_int64),
        ('syncBoostTime', c_int64),
        ('overallHealth', c_uint),
        ('incidentCount', c_uint),
        ('systems', c_dcgmHealthResponseInfo_t * DCGM_HEALTH_WATCH_COUNT_V1)
    ]

class c_dcgmPidInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('pid', c_uint32),
        ('unused', c_uint32),
        ('numGpus', c_int32),
        ('summary', c_dcgmPidSingleInfo_t),
        ('gpus', c_dcgmPidSingleInfo_t * 16)
    ]

dcgmPidInfo_version1 = make_dcgm_version(c_dcgmPidInfo_v1, 1)

class c_dcgmRunningProcess_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('pid', c_uint32),
        ('memoryUsed', c_uint64)
    ]

dcgmRunningProcess_version1 = make_dcgm_version(c_dcgmRunningProcess_v1, 1)



class c_dcgmGpuUsageInfo_t(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint32),
        ('energyConsumed', c_int64),
        ('powerUsage', c_dcgmStatSummaryFp64_t),
        ('pcieRxBandwidth', c_dcgmStatSummaryInt64_t),
        ('pcieTxBandwidth', c_dcgmStatSummaryInt64_t),
        ('pcieReplays', c_int64),
        ('startTime', c_int64),
        ('endTime', c_int64),
        ('smUtilization', c_dcgmStatSummaryInt32_t),
        ('memoryUtilization', c_dcgmStatSummaryInt32_t),
        ('eccSingleBit', c_uint32),
        ('eccDoubleBit', c_uint32),
        ('memoryClock', c_dcgmStatSummaryInt32_t),
        ('smClock', c_dcgmStatSummaryInt32_t),
        ('numXidCriticalErrors', c_int32),
        ('xidCriticalErrorsTs', c_int64 * 10),
        ('numComputePids', c_int32),
        ('computePids', c_dcgmProcessUtilInfo_t * DCGM_MAX_PID_INFO_NUM ),
        ('numGraphicsPids', c_int32),
        ('graphicsPids', c_dcgmProcessUtilInfo_t * DCGM_MAX_PID_INFO_NUM ),
        ('maxGpuMemoryUsed', c_int64),
        ('powerViolationTime', c_int64),
        ('thermalViolationTime', c_int64),
        ('reliabilityViolationTime', c_int64),
        ('boardLimitViolationTime', c_int64),
        ('lowUtilizationTime', c_int64),
        ('syncBoostTime', c_int64),
        ('overallHealth', c_uint),
        ('incidentCount', c_uint),
        ('systems', c_dcgmHealthResponseInfo_t * DCGM_HEALTH_WATCH_COUNT_V1)
    ]

class c_dcgmJobInfo_v2(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('numGpus', c_int32),
        ('summary', c_dcgmGpuUsageInfo_t),
        ('gpus', c_dcgmGpuUsageInfo_t * 16)
    ]

dcgmJobInfo_version2 = make_dcgm_version(c_dcgmJobInfo_v2, 2)

class c_dcgmDiagTestResult_v1(_PrintableStructure):
    _fields_ = [
        ('result', c_uint),
        ('warning', c_char * 1024),
        ('info', c_char * 1024)
    ]

class c_dcgmDiagTestResult_v2(_PrintableStructure):
    _fields_ = [
        ('result', c_uint),
        ('error', c_dcgmDiagErrorDetail_t),
        ('info', c_char * 1024)
    ]

class c_dcgmDiagResponsePerGpu_v1(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint),
        ('hwDiagnosticReturn', c_uint),
        ('results', c_dcgmDiagTestResult_v1 * DCGM_PER_GPU_TEST_COUNT)
    ]

class c_dcgmDiagResponsePerGpu_v2(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint),
        ('hwDiagnosticReturn', c_uint),
        ('results', c_dcgmDiagTestResult_v2 * DCGM_PER_GPU_TEST_COUNT)
    ]

class c_dcgmDiagResponse_v3(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('gpuCount', c_uint),
        ('blacklist', c_uint),
        ('lwmlLibrary', c_uint),
        ('lwdaMainLibrary', c_uint),
        ('lwdaRuntimeLibrary', c_uint),
        ('permissions', c_uint),
        ('persistenceMode', c_uint),
        ('environment', c_uint),
        ('pageRetirement', c_uint),
        ('inforom', c_uint),
        ('graphicsProcesses', c_uint),
        ('perGpuResponses', c_dcgmDiagResponsePerGpu_v1 * DCGM_MAX_NUM_DEVICES),
        ('systemError', c_char * 1024)
    ]

dcgmDiagResponse_version3 = make_dcgm_version(c_dcgmDiagResponse_v3, 3)

DCGM_SWTEST_COUNT = 10
LEVEL_ONE_MAX_RESULTS = 16

class c_dcgmDiagResponse_v4(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('gpuCount', c_uint),
        ('levelOneTestCount', c_uint),
        ('levelOneResults', c_dcgmDiagTestResult_v1 * LEVEL_ONE_MAX_RESULTS),
        ('perGpuResponses', c_dcgmDiagResponsePerGpu_v1 * DCGM_MAX_NUM_DEVICES),
        ('systemError', c_char * 1024),
        ('trainingMsg', c_char * 1024),
    ]

dcgmDiagResponse_version4 = make_dcgm_version(c_dcgmDiagResponse_v4, 4)

class c_dcgmDiagResponse_v5(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('gpuCount', c_uint),
        ('levelOneTestCount', c_uint),
        ('levelOneResults', c_dcgmDiagTestResult_v2 * LEVEL_ONE_MAX_RESULTS),
        ('perGpuResponses', c_dcgmDiagResponsePerGpu_v2 * DCGM_MAX_NUM_DEVICES),
        ('systemError',     c_dcgmDiagErrorDetail_t),
        ('trainingMsg',     c_char * 1024)
    ]

dcgmDiagResponse_version5 = make_dcgm_version(c_dcgmDiagResponse_v5, 5)

DCGM_AFFINITY_BITMASK_ARRAY_SIZE = 8

class c_dcgmDeviceTopologyPath_t(_PrintableStructure):
    _fields_ = [
        ('gpuId', c_uint32),
        ('path', c_uint32),
        ('localLwLinkIds', c_uint32)
    ]

class c_dcgmDeviceTopology_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('cpuAffinityMask', c_ulong * DCGM_AFFINITY_BITMASK_ARRAY_SIZE),
        ('numGpus', c_uint32),
        ('gpuPaths', c_dcgmDeviceTopologyPath_t * (DCGM_MAX_NUM_DEVICES - 1))
    ]

dcgmDeviceTopology_version1 = make_dcgm_version(c_dcgmDeviceTopology_v1, 1)

class c_dcgmGroupTopology_v1(_PrintableStructure):
    _fields_ = [ 
        ('version', c_uint32),
        ('groupCpuAffinityMask', c_ulong * DCGM_AFFINITY_BITMASK_ARRAY_SIZE),
        ('numaOptimalFlag', c_uint32),
        ('slowestPath', c_uint32)
    ]   

dcgmGroupTopology_version1 = make_dcgm_version(c_dcgmGroupTopology_v1, 1) 



# Maximum number of field groups that can exist
DCGM_MAX_NUM_FIELD_GROUPS = 64

# Maximum number of field IDs that can be in a single field group
DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP = 128

class c_dcgmFieldGroupInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('numFieldIds', c_uint32),
        ('fieldGroupId', c_void_p),
        ('fieldGroupName', c_char * DCGM_MAX_STR_LENGTH),
        ('fieldIds', c_uint16 * DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    ]

dcgmFieldGroupInfo_version1 = make_dcgm_version(c_dcgmFieldGroupInfo_v1, 1)


class c_dcgmAllFieldGroup_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('numFieldGroups', c_uint32),
        ('fieldGroups', c_dcgmFieldGroupInfo_v1 * DCGM_MAX_NUM_FIELD_GROUPS)
    ]

dcgmAllFieldGroup_version1 = make_dcgm_version(c_dcgmAllFieldGroup_v1, 1)


class DCGM_INTROSPECT_LVL(object):
    '''
    Identifies a level to retrieve field introspection info for
    '''
    INVALID = 0
    FIELD = 1
    FIELD_GROUP = 2
    ALL_FIELDS = 3
    
class c_dcgmIntrospectContext_v1(_PrintableStructure):
    '''
    Identifies the retrieval context for introspection API calls.
    '''
    _fields_ = [
        ('version', c_uint32),
        ('introspectLvl', c_int),         # one of DCGM_INTROSPECT_LVL_?
        ('fieldGroupId', c_void_p)        # Only needed if \ref introspectLvl is FIELD_GROUP
    ]
    
dcgmIntrospectContext_version1 = make_dcgm_version(c_dcgmIntrospectContext_v1, 1)

class c_dcgmIntrospectMemory_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('bytesUsed', c_longlong)  # The total number of bytes being used to store all of the fields being watched
    ]

dcgmIntrospectMemory_version1 = make_dcgm_version(c_dcgmIntrospectMemory_v1, 1)

class c_dcgmIntrospectFieldsExecTime_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),               # version number (dcgmIntrospectFieldsExecTime_version)                              
        ('meanUpdateFreqUsec', c_longlong),  # the mean update frequency of all fields                                                                                      
        ('recentUpdateUsec', c_double),      # the sum of every field's most recent exelwtion time after they                                            
                                             # have been normalized to \ref meanUpdateFreqUsec.           
                                             # This is roughly how long it takes to update fields every \ref meanUpdateFreqUsec                                                                              
                                                                
        ('totalEverUpdateUsec', c_longlong), # The total amount of time, ever, that has been spent updating all the fields                                
    ]
                                                                                                                                    
dcgmIntrospectFieldsExecTime_version1 = make_dcgm_version(c_dcgmIntrospectFieldsExecTime_v1, 1)

class c_dcgmIntrospectFullFieldsExecTime_v1(_PrintableStructure):
    '''
    Full introspection info for field exelwtion time
    '''
    _fields_ = [
        ('version', c_uint32),
        ('aggregateInfo', c_dcgmIntrospectFieldsExecTime_v1),   # info that includes global and device scope
        ('hasGlobalInfo', c_int),                               # 0 means \ref globalInfo is populated, !0 means it's not
        ('globalInfo', c_dcgmIntrospectFieldsExecTime_v1),      # info that only includes global field scope
        ('gpuInfoCount', c_uint),                               # count of how many entries in \ref gpuInfo are populated
        ('gpuIdsForGpuInfo', c_uint * DCGM_MAX_NUM_DEVICES),    # the GPU ID at a given index identifies which gpu
                                                                # the corresponding entry in \ref gpuInfo is from
        ('gpuInfo', c_dcgmIntrospectFieldsExecTime_v1 * DCGM_MAX_NUM_DEVICES),  # info that is separated by the
                                                                                # GPU ID that the watches were for
    ]

dcgmIntrospectFullFieldsExecTime_version1 = make_dcgm_version(c_dcgmIntrospectFullFieldsExecTime_v1, 1)

class c_dcgmIntrospectFullMemory_v1(_PrintableStructure):
    '''
    Full introspection info for field memory
    '''
    _fields_ = [
        ('version', c_uint32),
        ('aggregateInfo', c_dcgmIntrospectMemory_v1),   # info that includes global and device scope
        ('hasGlobalInfo', c_int),                       # 0 means \ref globalInfo is populated, !0 means it's not
        ('globalInfo', c_dcgmIntrospectMemory_v1),      # info that only includes global field scope
        ('gpuInfoCount', c_uint),                               # count of how many entries in \ref gpuInfo are populated
        ('gpuIdsForGpuInfo', c_uint * DCGM_MAX_NUM_DEVICES),    # the GPU ID at a given index identifies which gpu
                                                                # the corresponding entry in \ref gpuInfo is from
        ('gpuInfo', c_dcgmIntrospectMemory_v1 * DCGM_MAX_NUM_DEVICES),  # info that is separated by the
                                                                        # GPU ID that the watches were for
    ]

dcgmIntrospectFullMemory_version1 = make_dcgm_version(c_dcgmIntrospectFullMemory_v1, 1)

class c_dcgmIntrospectCpuUtil_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),  #!< version number (dcgmIntrospectCpuUtil_version)                     
        ('total', c_double),    #!< fraction of device's CPU resources that were used                
        ('kernel', c_double),   #!< fraction of device's CPU resources that were used in kernel mode 
        ('user', c_double),     #!< fraction of device's CPU resources that were used in user mode   
    ]

dcgmIntrospectCpuUtil_version1 = make_dcgm_version(c_dcgmIntrospectCpuUtil_v1, 1)

DCGM_MAX_CONFIG_FILE_LEN = 10000
DCGM_MAX_TEST_NAMES = 20
DCGM_MAX_TEST_NAMES_LEN = 50
DCGM_MAX_TEST_PARMS = 100
DCGM_MAX_TEST_PARMS_LEN = 100
DCGM_GPU_LIST_LEN = 50
DCGM_FILE_LEN = 30
DCGM_PATH_LEN = 128
DCGM_THROTTLE_MASK_LEN = 50

# Flags options for running the GPU diagnostic
DCGM_RUN_FLAGS_VERBOSE     = 0x0001
DCGM_RUN_FLAGS_STATSONFAIL = 0x0002
DCGM_RUN_FLAGS_TRAIN       = 0x0004
DCGM_RUN_FLAGS_FORCE_TRAIN = 0x0008
DCGM_RUN_FLAGS_FAIL_EARLY  = 0x0010 # Enable fail early checks for the Targeted Stress, Targeted Power, SM Stress, and Diagnostic tests

class c_dcgmRunDiag_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('flags', c_uint), # flags specifying binary options for running it. Lwrrently verbose and stats on fail
        ('debugLevel', c_uint), # 0-5 for the debug level the GPU diagnostic will use for logging
        ('groupId', c_void_p), # group of GPUs to verify. Cannot be specified together with gpuList.
        ('validate', c_uint), # 0-3 for which tests to run. Optional.
        ('testNames', c_char * DCGM_MAX_TEST_NAMES * DCGM_MAX_TEST_NAMES_LEN), # Specifed list of test names. Optional.
        ('testParms', c_char * DCGM_MAX_TEST_PARMS * DCGM_MAX_TEST_PARMS_LEN), # Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
        ('gpuList', c_char * DCGM_GPU_LIST_LEN), # Comma-separated list of gpus. Cannot be specified with the groupId.
        ('debugLogFile', c_char * DCGM_FILE_LEN), # Alternate name for the debug log file that should be used
        ('statsPath', c_char * DCGM_PATH_LEN), # Path that the plugin's statistics files should be written to
    ]

dcgmRunDiag_version1 = make_dcgm_version(c_dcgmRunDiag_v1, 1)

class c_dcgmRunDiag_v2(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('flags', c_uint), # flags specifying binary options for running it. Lwrrently verbose and stats on fail
        ('debugLevel', c_uint), # 0-5 for the debug level the GPU diagnostic will use for logging
        ('groupId', c_void_p), # group of GPUs to verify. Cannot be specified together with gpuList.
        ('validate', c_uint), # 0-3 for which tests to run. Optional.
        ('testNames', c_char * DCGM_MAX_TEST_NAMES * DCGM_MAX_TEST_NAMES_LEN), # Specifed list of test names. Optional.
        ('testParms', c_char * DCGM_MAX_TEST_PARMS * DCGM_MAX_TEST_PARMS_LEN), # Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
        ('gpuList', c_char * DCGM_GPU_LIST_LEN), # Comma-separated list of gpus. Cannot be specified with the groupId.
        ('debugLogFile', c_char * DCGM_FILE_LEN), # Alternate name for the debug log file that should be used
        ('statsPath', c_char * DCGM_PATH_LEN), # Path that the plugin's statistics files should be written to
        ('configFileContents', c_char * DCGM_MAX_CONFIG_FILE_LEN), # Contents of lwvs config file (likely yaml)
    ]

dcgmRunDiag_version2 = make_dcgm_version(c_dcgmRunDiag_v2, 2)

class c_dcgmRunDiag_v3(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('flags', c_uint), # flags specifying binary options for running it. Lwrrently verbose and stats on fail
        ('debugLevel', c_uint), # 0-5 for the debug level the GPU diagnostic will use for logging
        ('groupId', c_void_p), # group of GPUs to verify. Cannot be specified together with gpuList.
        ('validate', c_uint), # 0-3 for which tests to run. Optional.
        ('testNames', c_char * DCGM_MAX_TEST_NAMES * DCGM_MAX_TEST_NAMES_LEN), # Specifed list of test names. Optional.
        ('testParms', c_char * DCGM_MAX_TEST_PARMS * DCGM_MAX_TEST_PARMS_LEN), # Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
        ('gpuList', c_char * DCGM_GPU_LIST_LEN), # Comma-separated list of gpus. Cannot be specified with the groupId.
        ('debugLogFile', c_char * DCGM_FILE_LEN), # Alternate name for the debug log file that should be used
        ('statsPath', c_char * DCGM_PATH_LEN), # Path that the plugin's statistics files should be written to
        ('configFileContents', c_char * DCGM_MAX_CONFIG_FILE_LEN), # Contents of lwvs config file (likely yaml)
        ('throttleMask', c_char * DCGM_THROTTLE_MASK_LEN), # Throttle reasons to ignore as either integer mask or csv list of reasons
    ]

dcgmRunDiag_version3 = make_dcgm_version(c_dcgmRunDiag_v3, 3)

class c_dcgmRunDiag_v4(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('flags', c_uint), # flags specifying binary options for running it. Lwrrently verbose and stats on fail
        ('debugLevel', c_uint), # 0-5 for the debug level the GPU diagnostic will use for logging
        ('groupId', c_void_p), # group of GPUs to verify. Cannot be specified together with gpuList.
        ('validate', c_uint), # 0-3 for which tests to run. Optional.
        ('testNames', c_char * DCGM_MAX_TEST_NAMES * DCGM_MAX_TEST_NAMES_LEN), # Specifed list of test names. Optional.
        ('testParms', c_char * DCGM_MAX_TEST_PARMS * DCGM_MAX_TEST_PARMS_LEN), # Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
        ('gpuList', c_char * DCGM_GPU_LIST_LEN), # Comma-separated list of gpus. Cannot be specified with the groupId.
        ('debugLogFile', c_char * DCGM_FILE_LEN), # Alternate name for the debug log file that should be used
        ('statsPath', c_char * DCGM_PATH_LEN), # Path that the plugin's statistics files should be written to
        ('configFileContents', c_char * DCGM_MAX_CONFIG_FILE_LEN), # Contents of lwvs config file (likely yaml)
        ('throttleMask', c_char * DCGM_THROTTLE_MASK_LEN), # Throttle reasons to ignore as either integer mask or csv list of reasons
        ('pluginPath', c_char * DCGM_PATH_LEN), # Custom path to the diagnostic plugins
        ('trainingValues', c_uint), # Number of iterations for training.
        ('trainingVariance', c_uint), # Acceptable training variance as a percentage of the value. (0-100)
        ('trainingTolerance',c_uint), # Acceptable training tolerance as a percentage of the value. (0-100)
        ('goldelwaluesFile', c_char * DCGM_PATH_LEN), # The path where the golden values should be recorded
    ]

dcgmRunDiag_version4 = make_dcgm_version(c_dcgmRunDiag_v4, 4)

class c_dcgmRunDiag_v5(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('flags', c_uint), # flags specifying binary options for running it. Lwrrently verbose and stats on fail
        ('debugLevel', c_uint), # 0-5 for the debug level the GPU diagnostic will use for logging
        ('groupId', c_void_p), # group of GPUs to verify. Cannot be specified together with gpuList.
        ('validate', c_uint), # 0-3 for which tests to run. Optional.
        ('testNames', c_char * DCGM_MAX_TEST_NAMES * DCGM_MAX_TEST_NAMES_LEN), # Specifed list of test names. Optional.
        ('testParms', c_char * DCGM_MAX_TEST_PARMS * DCGM_MAX_TEST_PARMS_LEN), # Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
        ('gpuList', c_char * DCGM_GPU_LIST_LEN), # Comma-separated list of gpus. Cannot be specified with the groupId.
        ('debugLogFile', c_char * DCGM_PATH_LEN), # Alternate name for the debug log file that should be used
        ('statsPath', c_char * DCGM_PATH_LEN), # Path that the plugin's statistics files should be written to
        ('configFileContents', c_char * DCGM_MAX_CONFIG_FILE_LEN), # Contents of lwvs config file (likely yaml)
        ('throttleMask', c_char * DCGM_THROTTLE_MASK_LEN), # Throttle reasons to ignore as either integer mask or csv list of reasons
        ('pluginPath', c_char * DCGM_PATH_LEN), # Custom path to the diagnostic plugins
        ('trainingValues', c_uint), # Number of iterations for training.
        ('trainingVariance', c_uint), # Acceptable training variance as a percentage of the value. (0-100)
        ('trainingTolerance',c_uint), # Acceptable training tolerance as a percentage of the value. (0-100)
        ('goldelwaluesFile', c_char * DCGM_PATH_LEN), # The path where the golden values should be recorded
        ('failCheckInterval', c_uint), # How often the fail early checks should occur when DCGM_RUN_FLAGS_FAIL_EARLY is set.
    ]

dcgmRunDiag_version5 = make_dcgm_version(c_dcgmRunDiag_v5, 5)

# Latest c_dcgmRunDiag class
c_dcgmRunDiag_t = c_dcgmRunDiag_v5

# Latest version for dcgmRunDiag_t
dcgmRunDiag_version = dcgmRunDiag_version5


#Flags for dcgmGetEntityGroupEntities's flags parameter
DCGM_GEGE_FLAG_ONLY_SUPPORTED = 0x00000001 #Only return entities that are supported by DCGM.

#Identifies a GPU LWLink error type returned by DCGM_FI_DEV_GPU_LWLINK_ERRORS
DCGM_GPU_LWLINK_ERROR_RECOVERY_REQUIRED = 1 # LWLink link recovery error oclwrred
DCGM_GPU_LWLINK_ERROR_FATAL             = 2 # LWLink link fatal error oclwrred

# Topology hints for dcgmSelectGpusByTopology()
DCGM_TOPO_HINT_F_NONE = 0x00000000 # No hints specified
DCGM_TOPO_HINT_F_IGNOREHEALTH = 0x00000001 # Ignore the health of the GPUs when picking GPUs for job exelwtion.
                                           # By default, only healthy GPUs are considered.

class c_dcgmTopoSchedHint_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint), # version of this message
        ('inputGpuIds', c_uint64), # bitmask of the GPU ids to choose from
        ('numGpus', c_uint32), # the number of GPUs that DCGM should chooose
        ('hintFlags', c_uint64), # Hints to ignore certain factors for the scheduling hint
    ]

dcgmTopoSchedHint_version1 = make_dcgm_version(c_dcgmTopoSchedHint_v1, 1)


#DCGM LwLink link states used by c_dcgmLwLinkGpuLinkStatus_v1 & 2 and c_dcgmLwLinkLwSwitchLinkStatus_t's linkState field
DcgmLwLinkLinkStateNotSupported = 0 # LwLink is unsupported by this GPU (Default for GPUs)
DcgmLwLinkLinkStateDisabled     = 1 # LwLink is supported for this link but this link is disabled (Default for LwSwitches)
DcgmLwLinkLinkStateDown         = 2 # This LwLink link is down (inactive)
DcgmLwLinkLinkStateUp           = 3 # This LwLink link is up (active)


# State of LwLink links for a GPU 
class c_dcgmLwLinkGpuLinkStatus_v1(_PrintableStructure):
    _fields_ = [
        ('entityId', c_uint32),   # Entity ID of the GPU (gpuId)
        ('linkState', c_uint32 * DCGM_LWLINK_MAX_LINKS_PER_GPU_LEGACY1),  #Link state of each link of this GPU
    ]

# State of LwLink links for a GPU 
class c_dcgmLwLinkGpuLinkStatus_v2(_PrintableStructure):
    _fields_ = [
        ('entityId', c_uint32),   # Entity ID of the GPU (gpuId)
        ('linkState', c_uint32 * DCGM_LWLINK_MAX_LINKS_PER_GPU),  #Link state of each link of this GPU
    ]

#State of LwLink links for a LwSwitch
class c_dcgmLwLinkLwSwitchLinkStatus_t(_PrintableStructure):
    _fields_ = [
        ('entityId', c_uint32), # Entity ID of the LwSwitch (physicalId)
        ('linkState', c_uint32 * DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH) #Link state of each link of this LwSwitch
    ]

class c_dcgmLwLinkStatus_v1(_PrintableStructure):
    '''
    LwSwitch link status for all GPUs and LwSwitches in the system
    '''
    _fields_ = [
        ('version', c_uint32),       # version of this message. Should be dcgmLwLinkStatus_version1
        ('numGpus', c_uint32),       # Number of GPUs populated in gpus[]
        ('gpus', c_dcgmLwLinkGpuLinkStatus_v1 * DCGM_MAX_NUM_DEVICES),  #Per-GPU LwLink link statuses
        ('numLwSwitches', c_uint32), # Number of LwSwitches populated in lwSwitches[]
        ('lwSwitches', c_dcgmLwLinkLwSwitchLinkStatus_t * DCGM_MAX_NUM_SWITCHES) #Per-LwSwitch LwLink link statuses
    ]

dcgmLwLinkStatus_version1 = make_dcgm_version(c_dcgmLwLinkStatus_v1, 1)

class c_dcgmLwLinkStatus_v2(_PrintableStructure):
    '''
    LwSwitch link status for all GPUs and LwSwitches in the system
    '''
    _fields_ = [
        ('version', c_uint32),       # version of this message. Should be dcgmLwLinkStatus_version1
        ('numGpus', c_uint32),       # Number of GPUs populated in gpus[]
        ('gpus', c_dcgmLwLinkGpuLinkStatus_v2 * DCGM_MAX_NUM_DEVICES),  #Per-GPU LwLink link statuses
        ('numLwSwitches', c_uint32), # Number of LwSwitches populated in lwSwitches[]
        ('lwSwitches', c_dcgmLwLinkLwSwitchLinkStatus_t * DCGM_MAX_NUM_SWITCHES) #Per-LwSwitch LwLink link statuses
    ]

dcgmLwLinkStatus_version2 = make_dcgm_version(c_dcgmLwLinkStatus_v2, 2)

# Bitmask values for dcgmGetFieldIdSummary
DCGM_SUMMARY_MIN      = 0x00000001
DCGM_SUMMARY_MAX      = 0x00000002
DCGM_SUMMARY_AVG      = 0x00000004
DCGM_SUMMARY_SUM      = 0x00000008
DCGM_SUMMARY_COUNT    = 0x00000010
DCGM_SUMMARY_INTEGRAL = 0x00000020
DCGM_SUMMARY_DIFF     = 0x00000040
DCGM_SUMMARY_SIZE     = 7

class c_dcgmSummaryResponse_t(_PrintableStructure):
    class ResponseValue(Union):
        _fields_ = [
            ('i64', c_int64),
            ('dbl', c_double),
        ]

    _fields_ = [
        ('fieldType', c_uint),
        ('summaryCount', c_uint),
        ('values', ResponseValue * DCGM_SUMMARY_SIZE),
    ]

class c_dcgmFieldSummaryRequest_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('fieldId', c_ushort),
        ('entityGroupType', c_uint32),
        ('entityId', c_uint),
        ('summaryTypeMask', c_uint32),
        ('startTime', c_uint64),
        ('endTime', c_uint64),
        ('response', c_dcgmSummaryResponse_t),
    ]

dcgmFieldSummaryRequest_version1 = make_dcgm_version(c_dcgmFieldSummaryRequest_v1, 1)

# Module IDs
DcgmModuleIdCore           = 0 # Core DCGM
DcgmModuleIdLwSwitch       = 1 # LwSwitch Module
DcgmModuleIdVGPU           = 2 # VGPU Module
DcgmModuleIdIntrospect     = 3 # Introspection Module
DcgmModuleIdHealth         = 4 # Health Module
DcgmModuleIdPolicy         = 5 # Policy Module
DcgmModuleIdConfig         = 6 # Config Module
DcgmModuleIdDiag           = 7 # GPU Diagnostic Module
DcgmModuleIdProfiling      = 8 # Profiling Module
DcgmModuleIdCount          = 9 # 1 greater than largest ID above

# Module Status
DcgmModuleStatusNotLoaded   = 0 # Module has not been loaded yet
DcgmModuleStatusBlacklisted = 1 # Module has been blacklisted from being loaded
DcgmModuleStatusFailed      = 2 # Loading the module failed
DcgmModuleStatusLoaded      = 3 # Module has been loaded

DCGM_MODULE_STATUSES_CAPACITY = 16

class c_dcgmModuleGetStatusesModule_t(_PrintableStructure):
    _fields_ = [
        ('id', c_uint32),     #One of DcgmModuleId*
        ('status', c_uint32), #One of DcgmModuleStatus*
    ]

class c_dcgmModuleGetStatuses_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('numStatuses', c_uint32),
        ('statuses', c_dcgmModuleGetStatusesModule_t * DCGM_MODULE_STATUSES_CAPACITY),
    ]

dcgmModuleGetStatuses_version1 = make_dcgm_version(c_dcgmModuleGetStatuses_v1, 1)


DCGM_PROF_MAX_NUM_GROUPS          = 10 # Maximum number of metric ID groups that can exist in DCGM
DCGM_PROF_MAX_FIELD_IDS_PER_GROUP = 8  # Maximum number of field IDs that can be in a single DCGM profiling metric group

class c_dcgmProfMetricGroupInfo_t(_PrintableStructure):
    _fields_ = [
        ('majorId', c_ushort),
        ('minorId', c_ushort),
        ('numFieldIds', c_uint32),
        ('fieldIds', c_ushort * DCGM_PROF_MAX_FIELD_IDS_PER_GROUP),
    ]

class c_dcgmProfGetMetricGroups_v2(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('unused', c_uint32),
        ('groupId', c_void_p),
        ('numMetricGroups', c_uint32),
        ('unused1', c_uint32),
        ('metricGroups', c_dcgmProfMetricGroupInfo_t * DCGM_PROF_MAX_NUM_GROUPS),
    ]

dcgmProfGetMetricGroups_version1 = make_dcgm_version(c_dcgmProfGetMetricGroups_v2, 2)

class c_dcgmProfWatchFields_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('groupId', c_void_p),
        ('numFieldIds', c_uint32),
        ('fieldIds', c_ushort * 16),
        ('updateFreq', c_int64),
        ('maxKeepAge', c_double),
        ('maxKeepSamples', c_int32),
        ('flags', c_uint32),
    ]

dcgmProfWatchFields_version1 = make_dcgm_version(c_dcgmProfWatchFields_v1, 1)

class c_dcgmProfUnwatchFields_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('groupId', c_void_p),
        ('flags', c_uint32),
    ]

dcgmProfUnwatchFields_version1 = make_dcgm_version(c_dcgmProfUnwatchFields_v1, 1)


class c_dcgmVersionInfo_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint32),
        ('changelist', c_char * DCGM_MAX_STR_LENGTH),
        ('platform', c_char * DCGM_MAX_STR_LENGTH),
        ('branch', c_char * DCGM_MAX_STR_LENGTH),
        ('driverVersion', c_char * DCGM_MAX_STR_LENGTH),
        ('buildDate', c_char * DCGM_MAX_STR_LENGTH),
    ]


dcgmVersionInfo_version1 = make_dcgm_version(c_dcgmVersionInfo_v1, 1)
