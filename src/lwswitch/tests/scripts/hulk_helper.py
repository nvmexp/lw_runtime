#! /usr/bin/elw python

# _LWRM_COPYRIGHTBEGIN
#
# Copyright 2019 by LWPU Corporation. All rights reserved. All
# information contained herein is proprietary and confidential to LWPU
# Corporation. Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# _LWRM_COPYRIGHTEND

# The target of this script is to provide helper functions to load hulk license
# by retrieving the hulk license details from config file for a given GPU.
# It works with Python 2/Python 3 on Windows/Linux

import os
import platform
import subprocess
import threading
import time
import logging
import json
import ast                      # for json parsing. Colwerts str to json

def print_lwr_time():
    logging.info(time.strftime("[INFO] current time is %H:%M:%S %Z, %b %d, %Y"))

def stream_pipe(pipe, line_handler):
    while True:
        line = pipe.readline()
        # Prevent early version of python to test the 'bytes' type since it
        # is not implemented in some early versions:
        if type(line) is bytes:
            line = line.decode("utf-8", "replace")

        if line == "":
            break;

        line_handler(line)

def exelwte_process(processPath):
    """
    Execute a test file
    """
    stdout = []
    stderr = []
    print_lwr_time()
    logging.info("Exelwting " + colwert_list_to_string(processPath))
    process = subprocess.Popen(processPath, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Because there is not a way to poll on the file objects in Windows,
    # consume each one in a separate thread.
    def handle_stdout(line):
        line = line.rstrip()
        stdout.append(line)
        logging.info(line)
        if line.startswith('^^^^'):
            print_lwr_time()

    def handle_stderr(line):
        line = line.rstrip()
        stderr.append(line)
        logging.error(line)

    consume_stdout = threading.Thread(group=None, target=lambda : stream_pipe(process.stdout, handle_stdout))
    consume_stderr = threading.Thread(group=None, target=lambda : stream_pipe(process.stderr, handle_stderr))
    consume_stdout.start()
    consume_stderr.start()
    consume_stdout.join()
    consume_stderr.join()
    # Wait for process to terminate, so file handles are freed
    process.communicate()
    return stdout, stderr

def escape_chars(text, chars_to_escape):
    """
    Prefix each character listed in chars_to_escape with escape character and
    return the updated string
    """
    for character in chars_to_escape:
        text = text.replace(character, '\\' + character)
    return text

def colwert_list_to_string(list_param):
    str_out = ' '.join(str(e) for e in list_param)
    return str_out

def load_module(module_name, maybe_sudo=[]):
    """
    Load driver module
    """
    if (platform.system() == "Linux"):
        logging.info(" -> Loading driver module : " + module_name)
        exelwte_process(maybe_sudo + ["modprobe", str(module_name)])

def unload_module(module_name, maybe_sudo=[]):
    """
    Unload driver module
    """
    if (platform.system() == "Linux"):
        logging.info(" -> Unloading driver module : " + module_name)
        exelwte_process(maybe_sudo + ["rmmod", str(module_name)])

def service_stop(service_name, maybe_sudo=[]):
    """
    Stop service
    """
    if (platform.system() == "Linux"):
        logging.info(" -> Stopping service : " + service_name)
        exelwte_process(maybe_sudo + ["systemctl", "stop", str(service_name)])

def service_disable(service_name, maybe_sudo=[]):
    """
    Disable service
    """
    if (platform.system() == "Linux"):
        logging.info(" -> Disabling service : " + service_name)
        exelwte_process(maybe_sudo + ["systemctl", "disable", str(service_name)])

def reset_gpu(maybe_sudo=[]):
    """
    Enable persistence mode so that the tests don't hit RM
    adapter init/teardown, as that's very slow.
    """
    if (platform.system() == "Linux"):
        exelwte_process(maybe_sudo + ["lwpu-smi", "-r"])

def enable_persistence_mode(maybe_sudo=[]):
    """
    Enable persistence mode so that the tests don't hit RM
    adapter init/teardown, as that's very slow.
    """
    if (platform.system() == "Linux"):
        exelwte_process(maybe_sudo + ["lwpu-smi", "-pm", "1"])

def disable_persistence_mode(maybe_sudo=[]):
    """
    Disable persistence mode so that file handles are closed
    """
    if (platform.system() == "Linux"):
        exelwte_process(maybe_sudo + ["lwpu-smi", "-pm", "0"])

def get_uuid(gpu_index):
    """
    Extract and return the uuid info.
    """
    logging.info("Retrieving UUID for gpu index: " + str(gpu_index))
    command = ['lwpu-smi','--query-gpu=uuid','--format=csv,noheader', '--index=' + str(gpu_index)]
    stdout, stderr = exelwte_process(command)

    if stdout is not None:
        uuid = stdout[0].strip()
        logging.info("UUID = " + uuid)
        return uuid

    return None

def get_serial(gpu_index):
    """
    Extract and return the serial info.
    """
    logging.info("Retrieving serial for gpu index: " + str(gpu_index))
    command = ['lwpu-smi', '--query-gpu=serial','--format=csv,noheader', '--index=' + str(gpu_index)]
    stdout, stderr = exelwte_process(command)

    if stdout is not None:
        serial = stdout[0].strip()
        logging.info("Serial = " + serial)
        return serial

    return None

def get_gpu_bus_id(gpu_index):
    """
    Extract and return the gpu_bus_id info.
    """
    logging.info("Retrieving PCI bus id for gpu index: " + str(gpu_index))
    command = ['lwpu-smi', '--query-gpu=gpu_bus_id','--format=csv,noheader', '--index=' + str(gpu_index)]
    stdout, stderr = exelwte_process(command)

    if stdout is not None:
        #
        # lwpu-smi returns domain with 8 digits length.
        # Trim starting 4 zeroes, to match the domain in /procfs.
        #
        pci_bdf = stdout[0].strip()
        pci_bdf = pci_bdf[4:]
        logging.info("pci_bdf = " + pci_bdf)
        return pci_bdf

    return None

def get_binary_size(binary_file_path):
    """
    Returns the file size.
    """
    binary_size = os.path.getsize(str(binary_file_path))
    logging.info("Binary Size =" + str(binary_size));
    return binary_size


def unload_lw_kernel_module(maybe_sudo=[]):
    """
    Unload lwpu kernel modules.
    """
    if (platform.system() == "Linux"):
        disable_persistence_mode()

        # Unload all kernel modules
        logging.info("Unloading Kernel Modules:")

        unload_module("lwidia_drm")

        unload_module("lwidia_modeset")

        unload_module("lwidia_vgpu_vfio")

        unload_module("lwidia_uvm")

        unload_module("lwpu")

def load_lw_kernel_module(maybe_sudo=[]):
    """
    Load lwpu kernel modules.
    """
    if (platform.system() == "Linux"):
        # Load only compute kernel modules
        logging.info("Loading Kernel Modules:")

        load_module("lwpu")

        load_module("lwidia_uvm")

def pre_test_system_setup(maybe_sudo=[]):
    """
    Perform pre-test launch system setup:
    - Disable Persistence mode
    - Stop LWPU Services
    - Disable LWPU Services
    - Perform GPU and LWSwitch reset to ensure links are reset.
    """
    if (platform.system() == "Linux"):
        logging.info("Pre-test system setup:")

        # Disable persistence mode
        disable_persistence_mode()

        # Stop all LWPU services
        service_stop("lwpu-fabricmanager")
        service_stop("lwsm")
        service_stop("dcgm")

        # Disable all LWPU services
        service_disable("lwpu-fabricmanager")
        service_disable("lwsm")
        service_disable("dcgm")

        # Reset GPU and LWSwitch to reset links.
        reset_gpu()
        unload_lw_kernel_module()

def load_driver_with_hulk_license(hulk_file_path, pci_bdf, maybe_sudo=[]):
    """
    Load lwpu kernel module with hulk license
    """
    if (platform.system() == "Linux"):
        # Unload all kernel modules
        unload_lw_kernel_module()

        # Retrive Hulk binary size
        hulk_binary_size = get_binary_size(hulk_file_path)
        if hulk_binary_size == 0:
            return False

        # Load lwpu module, with hulk binary size set using RMHulkCertSize.
        logging.info(" -> Loading HULK license: " + hulk_file_path + " on PCI " + pci_bdf + " with size " + str(hulk_binary_size) + " bytes")
        cmd = ["modprobe", "lwpu", "LWreg_RegistryDwords=\"RMHulkCertSize=" + str(hulk_binary_size) + "\""]
        exelwte_process(maybe_sudo + cmd)

        # Store Hulk in registry in procfs
        pci_bdf_escaped = escape_chars(pci_bdf, ":")
        procfs_gpu_registry_path = "/proc/driver/lwpu/gpus/" + pci_bdf_escaped + "/registry"
        logging.info(" -> Store Hulk in registry via procfs : " + procfs_gpu_registry_path)
        cmd = ["bash", "-c", "echo -n \"RMHulkCert=\" | cat - " + str(hulk_file_path) + " > " + str(procfs_gpu_registry_path)]
        exelwte_process(maybe_sudo + cmd)

        # Disable persistence mode until bug 2506113 is fixed as DRAM ECC Injection causes hang.
        # enable_persistence_mode()
        return True
    else:
        # On windows we ignore this step
        return False


def find_hulk_filename_for_gpu(hulk_config_file, requested_uuid, requested_serial, test_name):
    """
    Search the json config to find the hulk license matching to either
    requested_uuid or to requested_serial.

    If match is found, return that hulk license file path.
    Else, return None.
    """
    with open(hulk_config_file) as f:
        # Read file data
        json_raw_data = f.read()

        # Colwert string to JSON object using AST
        json_object = ast.literal_eval(json_raw_data)

        # Check if hulk for specific uuid or serial exists
        for hulk_tuple_entry in json_object["hulk_licenses"]:
            serial = hulk_tuple_entry["serial"]
            uuid = hulk_tuple_entry["uuid"]
            use_for_tests = hulk_tuple_entry["use_for_tests"]

            # If Hulk license is found, return its absolute path.
            if (((uuid == requested_uuid) or (serial == requested_serial)) and (test_name in use_for_tests)):
                logging.info("Found Hulk tuple matching serial value and UUID: ({}) , ({}) , ({})".format(hulk_tuple_entry, serial, uuid))
                hulk_file_path_rel = str(hulk_tuple_entry["hulk_path"])
                hulk_file_path_abs = os.path.dirname(os.path.realpath(__file__)) + str("/") + hulk_file_path_rel

                return hulk_file_path_abs

    logging.info("No Hulk license tuple found in config file for serial value and UUID: ({}) , ({}) ".format(requested_serial, requested_uuid))
    return None

def load_hulk_license_if_available(test_name, gpu_index):
    uuid = get_uuid(gpu_index)
    serial = get_serial(gpu_index)
    pci_bdf = get_gpu_bus_id(gpu_index)

    logging.info("Test name ({}). Starting load_hulk_license_if_available()".format(test_name))

    hulk_config_path = os.path.dirname(os.path.realpath(__file__)) + "/srt_dvs_config/hulk_license_map.json"
    hulk_config_filename = find_hulk_filename_for_gpu(hulk_config_path, uuid, serial, test_name)

    if ((hulk_config_filename != None) and (hulk_config_filename)):
        hulk_loaded = load_driver_with_hulk_license(hulk_config_filename, pci_bdf)
        logging.info("hulk_loaded = " + str(hulk_loaded))
        return hulk_loaded

    logging.info("Hulk license loading failed. No Hulk file found.")
    return False

def init_hulk_helper():
    # Initialise the logger
    logFilePath = os.getcwd() + "/srt_dvs_tests_hulk_helper.log"
    logging.basicConfig(filename=logFilePath, filemode='w', level=logging.INFO)
    logging.info("Init hulk helper")

