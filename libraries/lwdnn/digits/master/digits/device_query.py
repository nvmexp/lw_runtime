#!/usr/bin/elw python2
# Copyright (c) 2014-2017, LWPU CORPORATION.  All rights reserved.
from __future__ import absolute_import

import argparse
import ctypes
import platform


class c_lwdaDeviceProp(ctypes.Structure):
    """
    Passed to lwdart.lwdaGetDeviceProperties()
    """
    _fields_ = [
        ('name', ctypes.c_char * 256),
        ('totalGlobalMem', ctypes.c_size_t),
        ('sharedMemPerBlock', ctypes.c_size_t),
        ('regsPerBlock', ctypes.c_int),
        ('warpSize', ctypes.c_int),
        ('memPitch', ctypes.c_size_t),
        ('maxThreadsPerBlock', ctypes.c_int),
        ('maxThreadsDim', ctypes.c_int * 3),
        ('maxGridSize', ctypes.c_int * 3),
        ('clockRate', ctypes.c_int),
        ('totalConstMem', ctypes.c_size_t),
        ('major', ctypes.c_int),
        ('minor', ctypes.c_int),
        ('textureAlignment', ctypes.c_size_t),
        ('texturePitchAlignment', ctypes.c_size_t),
        ('deviceOverlap', ctypes.c_int),
        ('multiProcessorCount', ctypes.c_int),
        ('kernelExecTimeoutEnabled', ctypes.c_int),
        ('integrated', ctypes.c_int),
        ('canMapHostMemory', ctypes.c_int),
        ('computeMode', ctypes.c_int),
        ('maxTexture1D', ctypes.c_int),
        ('maxTexture1DMipmap', ctypes.c_int),
        ('maxTexture1DLinear', ctypes.c_int),
        ('maxTexture2D', ctypes.c_int * 2),
        ('maxTexture2DMipmap', ctypes.c_int * 2),
        ('maxTexture2DLinear', ctypes.c_int * 3),
        ('maxTexture2DGather', ctypes.c_int * 2),
        ('maxTexture3D', ctypes.c_int * 3),
        ('maxTexture3DAlt', ctypes.c_int * 3),
        ('maxTextureLwbemap', ctypes.c_int),
        ('maxTexture1DLayered', ctypes.c_int * 2),
        ('maxTexture2DLayered', ctypes.c_int * 3),
        ('maxTextureLwbemapLayered', ctypes.c_int * 2),
        ('maxSurface1D', ctypes.c_int),
        ('maxSurface2D', ctypes.c_int * 2),
        ('maxSurface3D', ctypes.c_int * 3),
        ('maxSurface1DLayered', ctypes.c_int * 2),
        ('maxSurface2DLayered', ctypes.c_int * 3),
        ('maxSurfaceLwbemap', ctypes.c_int),
        ('maxSurfaceLwbemapLayered', ctypes.c_int * 2),
        ('surfaceAlignment', ctypes.c_size_t),
        ('conlwrrentKernels', ctypes.c_int),
        ('ECCEnabled', ctypes.c_int),
        ('pciBusID', ctypes.c_int),
        ('pciDeviceID', ctypes.c_int),
        ('pciDomainID', ctypes.c_int),
        ('tccDriver', ctypes.c_int),
        ('asyncEngineCount', ctypes.c_int),
        ('unifiedAddressing', ctypes.c_int),
        ('memoryClockRate', ctypes.c_int),
        ('memoryBusWidth', ctypes.c_int),
        ('l2CacheSize', ctypes.c_int),
        ('maxThreadsPerMultiProcessor', ctypes.c_int),
        ('streamPrioritiesSupported', ctypes.c_int),
        ('globalL1CacheSupported', ctypes.c_int),
        ('localL1CacheSupported', ctypes.c_int),
        ('sharedMemPerMultiprocessor', ctypes.c_size_t),
        ('regsPerMultiprocessor', ctypes.c_int),
        ('managedMemSupported', ctypes.c_int),
        ('isMultiGpuBoard', ctypes.c_int),
        ('multiGpuBoardGroupID', ctypes.c_int),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_int * 128),
        # added later with lwdart.lwdaDeviceGetPCIBusId
        # (needed by LWML)
        ('pciBusID_str', ctypes.c_char * 16),
    ]


class struct_c_lwmlDevice_t(ctypes.Structure):
    """
    Handle to a device in LWML
    """
    pass  # opaque handle
c_lwmlDevice_t = ctypes.POINTER(struct_c_lwmlDevice_t)


class c_lwmlMemory_t(ctypes.Structure):
    """
    Passed to lwml.lwmlDeviceGetMemoryInfo()
    """
    _fields_ = [
        ('total', ctypes.c_ulonglong),
        ('free', ctypes.c_ulonglong),
        ('used', ctypes.c_ulonglong),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_ulonglong * 8),
    ]


class c_lwmlUtilization_t(ctypes.Structure):
    """
    Passed to lwml.lwmlDeviceGetUtilizationRates()
    """
    _fields_ = [
        ('gpu', ctypes.c_uint),
        ('memory', ctypes.c_uint),
        # Extra space for new fields in future toolkits
        ('__future_buffer', ctypes.c_uint * 8),
    ]


def get_library(name):
    """
    Returns a ctypes.CDLL or None
    """
    try:
        if platform.system() == 'Windows':
            return ctypes.windll.LoadLibrary(name)
        else:
            return ctypes.cdll.LoadLibrary(name)
    except OSError:
        pass
    return None


def get_lwdart():
    """
    Return the ctypes.DLL object for lwdart or None
    """
    if platform.system() == 'Windows':
        arch = platform.architecture()[0]
        for ver in range(90, 50, -5):
            lwdart = get_library('lwdart%s_%d.dll' % (arch[:2], ver))
            if lwdart is not None:
                return lwdart
    elif platform.system() == 'Darwin':
        for major in xrange(9, 5, -1):
            for minor in (5, 0):
                lwdart = get_library('liblwdart.%d.%d.dylib' % (major, minor))
                if lwdart is not None:
                    return lwdart
        return get_library('liblwdart.dylib')
    else:
        for major in xrange(9, 5, -1):
            for minor in (5, 0):
                lwdart = get_library('liblwdart.so.%d.%d' % (major, minor))
                if lwdart is not None:
                    return lwdart
        return get_library('liblwdart.so')
    return None


def get_lwml():
    """
    Return the ctypes.DLL object for lwdart or None
    """
    if platform.system() == 'Windows':
        return get_library('lwml.dll')
    else:
        for name in (
                'liblwidia-ml.so.1',
                'liblwidia-ml.so',
                'lwml.so'):
            lwml = get_library(name)
            if lwml is not None:
                return lwml
    return None

devices = None


def get_devices(force_reload=False):
    """
    Returns a list of c_lwdaDeviceProp's
    Prints an error and returns None if something goes wrong

    Keyword arguments:
    force_reload -- if False, return the previously loaded list of devices
    """
    global devices
    if not force_reload and devices is not None:
        # Only query LWCA once
        return devices
    devices = []

    lwdart = get_lwdart()
    if lwdart is None:
        return []

    # check LWCA version
    lwda_version = ctypes.c_int()
    rc = lwdart.lwdaRuntimeGetVersion(ctypes.byref(lwda_version))
    if rc != 0:
        print 'lwdaRuntimeGetVersion() failed with error #%s' % rc
        return []
    if lwda_version.value < 6050:
        print 'ERROR: Lwca version must be >= 6.5, not "%s"' % lwda_version.value
        return []

    # get number of devices
    num_devices = ctypes.c_int()
    rc = lwdart.lwdaGetDeviceCount(ctypes.byref(num_devices))
    if rc != 0:
        print 'lwdaGetDeviceCount() failed with error #%s' % rc
        return []

    # query devices
    for x in xrange(num_devices.value):
        properties = c_lwdaDeviceProp()
        rc = lwdart.lwdaGetDeviceProperties(ctypes.byref(properties), x)
        if rc == 0:
            pciBusID_str = ' ' * 16
            # also save the string representation of the PCI bus ID
            rc = lwdart.lwdaDeviceGetPCIBusId(ctypes.c_char_p(pciBusID_str), 16, x)
            if rc == 0:
                properties.pciBusID_str = pciBusID_str
            devices.append(properties)
        else:
            print 'lwdaGetDeviceProperties() failed with error #%s' % rc
        del properties
    return devices


def get_device(device_id):
    """
    Returns a c_lwdaDeviceProp
    """
    return get_devices()[int(device_id)]


def get_lwml_info(device_id):
    """
    Gets info from LWML for the given device
    Returns a dict of dicts from different LWML functions
    """
    device = get_device(device_id)
    if device is None:
        return None

    lwml = get_lwml()
    if lwml is None:
        return None

    rc = lwml.lwmlInit()
    if rc != 0:
        raise RuntimeError('lwmlInit() failed with error #%s' % rc)

    try:
        # get device handle
        handle = c_lwmlDevice_t()
        rc = lwml.lwmlDeviceGetHandleByPciBusId(ctypes.c_char_p(device.pciBusID_str), ctypes.byref(handle))
        if rc != 0:
            raise RuntimeError('lwmlDeviceGetHandleByPciBusId() failed with error #%s' % rc)

        # Grab info for this device from LWML
        info = {}

        memory = c_lwmlMemory_t()
        rc = lwml.lwmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        if rc == 0:
            info['memory'] = {
                'total': memory.total,
                'used': memory.used,
                'free': memory.free,
            }

        utilization = c_lwmlUtilization_t()
        rc = lwml.lwmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization))
        if rc == 0:
            info['utilization'] = {
                'gpu': utilization.gpu,
                'memory': utilization.memory,  # redundant
            }

        temperature = ctypes.c_int()
        rc = lwml.lwmlDeviceGetTemperature(handle, 0, ctypes.byref(temperature))
        if rc == 0:
            info['temperature'] = temperature.value

        return info
    finally:
        rc = lwml.lwmlShutdown()
        if rc != 0:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIGITS Device Query')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if not len(get_devices()):
        print 'No devices found.'

    for i, device in enumerate(get_devices()):
        print 'Device #%d:' % i
        print '>>> LWCA attributes:'
        for name, t in device._fields_:
            if name in ['__future_buffer']:
                continue
            if not args.verbose and name not in [
                    'name', 'totalGlobalMem', 'clockRate', 'major', 'minor', ]:
                continue

            if 'c_int_Array' in t.__name__:
                val = ','.join(str(v) for v in getattr(device, name))
            else:
                val = getattr(device, name)

            print '  %-28s %s' % (name, val)

        info = get_lwml_info(i)
        if info is not None:
            print '>>> LWML attributes:'
            lwml_fmt = '  %-28s %s'
            if 'memory' in info:
                print lwml_fmt % ('Total memory',
                                  '%s MB' % (info['memory']['total'] / 2**20,))
                print lwml_fmt % ('Used memory',
                                  '%s MB' % (info['memory']['used'] / 2**20,))
                if args.verbose:
                    print lwml_fmt % ('Free memory',
                                      '%s MB' % (info['memory']['free'] / 2**20,))
            if 'utilization' in info:
                print lwml_fmt % ('Memory utilization',
                                  '%s%%' % info['utilization']['memory'])
                print lwml_fmt % ('GPU utilization',
                                  '%s%%' % info['utilization']['gpu'])
            if 'temperature' in info:
                print lwml_fmt % ('Temperature',
                                  '%s C' % info['temperature'])
        print
