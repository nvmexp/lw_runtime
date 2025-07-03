import ctypes
import dcgm_structs
import test_utils

## Device structures
class struct_c_LWdevice(ctypes.Structure):
    pass # opaque handle
c_LWdevice = ctypes.POINTER(struct_c_LWdevice)

# constants
LWDA_SUCCESS = 0
LW_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50

_lwdaLib = None
def _loadLwda():
    global _lwdaLib
    if _lwdaLib is None:
        _lwdaLib = ctypes.CDLL("liblwda.so.1")
        lwInitFn = getattr(_lwdaLib, "lwInit")
        assert LWDA_SUCCESS == lwInitFn(ctypes.c_uint(0))

def _unloadLwda():
    global _lwdaLib
    _lwdaLib = None

def lwDeviceGetCount():
    global _lwdaLib
    _loadLwda()
    lwDeviceGetCountFn = getattr(_lwdaLib, "lwDeviceGetCount")
    c_count = ctypes.c_uint(0)
    assert LWDA_SUCCESS == lwDeviceGetCountFn(ctypes.byref(c_count))
    _unloadLwda()
    return c_count.value

def lwDeviceGet(idx):
    global _lwdaLib
    _loadLwda()
    lwDeviceGetFn = getattr(_lwdaLib, "lwDeviceGet")
    c_dev = c_LWdevice()
    assert LWDA_SUCCESS == lwDeviceGetFn(ctypes.byref(c_dev), ctypes.c_uint(idx))
    _unloadLwda()
    return c_dev

def lwDeviceGetBusId(c_dev):
    global _lwdaLib
    _loadLwda()
    lwDeviceGetAttributeFn = getattr(_lwdaLib, "lwDeviceGetAttribute")
    c_domain = ctypes.c_uint()
    c_bus = ctypes.c_uint()
    c_device = ctypes.c_uint()
    assert LWDA_SUCCESS == lwDeviceGetAttributeFn(ctypes.byref(c_domain),
        ctypes.c_uint(LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID), c_dev)
    assert LWDA_SUCCESS == lwDeviceGetAttributeFn(ctypes.byref(c_bus),
        ctypes.c_uint(LW_DEVICE_ATTRIBUTE_PCI_BUS_ID), c_dev)
    assert LWDA_SUCCESS == lwDeviceGetAttributeFn(ctypes.byref(c_device),
        ctypes.c_uint(LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID), c_dev)
    _unloadLwda()
    return "%04x:%02x:%02x.0" % (c_domain.value, c_bus.value, c_device.value)

