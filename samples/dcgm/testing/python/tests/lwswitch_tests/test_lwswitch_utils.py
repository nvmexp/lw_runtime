import apps
import logger
import subprocess
import test_utils
import time

# test timeouts
FABRIC_RESET_TIMEOUT_SECS = 300
FABRIC_MANAGER_APP_TIMEOUT_SECS = 900
FABRIC_MANAGER_WAIT_TIME_SECS = 30.0
FABRIC_MANAGER_NON_FATAL_ERROR_POLLING_INTERVAL = 60.0

# fabric manager arguments
FABRIC_MANAGER_ARGS = "-l -g --log-level 4 --log-rotate --log-filename /var/log/fabricmanager.log"

# p2p_bandwidth app argument lists
P2P_TEST_LIST = ["-l"]
MEMCPY_DTOD_WRITE_CE_BANDWIDTH = ["-t", "Memcpy_DtoD_Write_CE_Bandwidth"]
MEMCPY_DTOD_READ_CE_BANDWIDTH = ["-t", "Memcpy_DtoD_Read_CE_Bandwidth"]

def is_lwidia_docker_running():
    """
    Return True if lwpu-docker service is running on the system
    """
    cmd = 'systemctl status lwpu-docker'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if "running" in out.rstrip():
        return True
    else:
        return False

def start_lwidia_docker():
    """
    Stop lwpu-docker service
    """
    cmd = 'systemctl start lwpu-docker'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err

def stop_lwidia_docker():
    """
    Stop lwpu-docker service
    """
    cmd = 'systemctl stop lwpu-docker'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err

def get_lwswitch_pci_bdf():
    """ Get lwswitch PCI BDFs """
    bdf = []
    try:
        lsPciOutput = subprocess.check_output("lspci | grep -i lwpu | grep -i bridge", shell=True)
    except subprocess.CalledProcessError as e:
        logger.error(e.message)
        return bdf

    lwswitches = lsPciOutput.split('\n')
    for i in range(len(lwswitches)):
        dev = lwswitches[i].split()
        if len(dev) > 0:
            bdf.append(dev[0])

    return bdf

def get_gpu_pci_bdf():
    """ Get GPU PCI BDFs """
    bdf = []
    try:
        lsPciOutput = subprocess.check_output("lspci | grep -i lwpu | grep -i '3d controller'", shell=True)
    except subprocess.CalledProcessError as e:
        logger.error(e.message)
        return bdf

    gpus = lsPciOutput.split('\n')
    for i in range(len(gpus)):
        dev = gpus[i].split()
        if len(dev) > 0:
            bdf.append(dev[0])

    return bdf

def fabric_reset():
    """
    Return true if fabric is reset successfully with lwpu-smi -r
    """
    # fabric manager has to be stopped before fabric reset
    if test_utils.is_hostengine_running():
        logger.info("Cannot reset fabric, lw-hostengine is running.")
        return False

    # lwpu-docker service has to be stopped before fabric reset
    lwidia_docker_running = is_lwidia_docker_running()

    if lwidia_docker_running:
        logger.info("Stop lwpu-docker service.")
        stop_lwidia_docker()

    # reset fabric with lwpu-smi
    if (0 == apps.LwidiaSmiApp(["-r"]).run(timeout=FABRIC_RESET_TIMEOUT_SECS)):
        logger.info("Reset fabric successfully.")
        ret = True
    else:
        ret = False

    # retart lwpu-docker service if it was running before reset
    if lwidia_docker_running:
        logger.info("Start lwpu-docker service.")
        start_lwidia_docker()

    return ret

def is_dgx_2_full_topology():
    """
    Return true if detect all lwswitches and GPUs on two base boards or one base board
    """
    switch_bdf = get_lwswitch_pci_bdf()
    gpu_bdf = get_gpu_pci_bdf()

    if len(switch_bdf) == 12 and len(gpu_bdf) == 16:
        return True
    elif len(switch_bdf) == 6 and len(gpu_bdf) == 8:
        return True
    else:
        return False
