import shlex
import time
import apps
import logger
import subprocess
import test_utils
import test_lwswitch_utils

def test_lwswitch_fabric_reset():
    """
    Verifies that fabric can pass traffic successfully after fabric reset
    """

    # fabric manager has to be stopped before fabric reset
    if test_utils.is_hostengine_running():
        test_utils.skip_test("Skipping, fabric manager is already running.")

    # reset fabric with lwpu-smi
    assert True == test_lwswitch_utils.fabric_reset(), \
        "Failed to reset fabric"

    # start fabric manager
    fm = apps.LwHostEngineApp(shlex.split(test_lwswitch_utils.FABRIC_MANAGER_ARGS))
    fm.start(timeout=test_lwswitch_utils.FABRIC_MANAGER_APP_TIMEOUT_SECS)
    logger.info("Start fabric manager pid: %s" % fm.getpid())

    # wait for fabric manager finish programming the fabric
    test_utils.wait_for_fabric_manager_ready()

    # Make sure traffic can pass successfully after reset
    test_utils.run_p2p_bandwidth_app(test_lwswitch_utils.MEMCPY_DTOD_WRITE_CE_BANDWIDTH)

    # stop fabric manager
    logger.info("Stop fabric manager")
    fm.terminate()
    fm.validate()

def inject_fatal_error(lwswitch_bdf_str):
    """
    Return True if injecting lwswitch fatal error with lwpex2 successfully
    """
    # clear dmesg brefore injecting error
    cmd = 'dmesg -c'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # inject fatal error
    logger.info("Inject fatal error to LWSwitch %s" % lwswitch_bdf_str)
    args = "--bdf " + lwswitch_bdf_str + " -r W:0:856710:FFFF -i 1"
    test_utils.run_lwpex2_app(shlex.split(args))

    # wait for the error to be in dmesg
    time.sleep(1)

    # look for fatal error in dmesg after injecting error
    cmd = 'dmesg > /tmp/dmesg.txt; grep "Fatal" /tmp/dmesg.txt | wc -l'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if int(out) > 0:
        return True
    else:
        logger.info("Fatal error not found")
        return False

def test_lwswitch_error_fatal_reset():
    """
    Verifies that fatal error can be recovered by fabric reset
    """

    # fabric manager has to be stopped before this test
    if test_utils.is_hostengine_running():
        test_utils.skip_test("Skipping, fabric manager is already running.")

    switch_bdf = test_lwswitch_utils.get_lwswitch_pci_bdf()
    if len(switch_bdf) == 0:
        logger.info("No LWSwitch found")
        return

    # start fabric manager
    fm = apps.LwHostEngineApp(shlex.split(test_lwswitch_utils.FABRIC_MANAGER_ARGS))
    fm.start(timeout=test_lwswitch_utils.FABRIC_MANAGER_APP_TIMEOUT_SECS)
    logger.info("Start fabric manager pid: %s" % fm.getpid())

    # wait for fabric manager finish programming the fabric
    test_utils.wait_for_fabric_manager_ready()

    # Inject fatal error to the first switch
    ret = inject_fatal_error(switch_bdf[0])

    # fabric manager has to be stopped before fabric reset
    logger.info("Stop fabric manager")
    fm.terminate()
    fm.validate()

    assert True == ret, "Failed to inject switch fatal error."

    # wait for a few second for fabric manager to finish

    time.sleep(3)

    # reset fabric with lwpu-smi
    assert True == test_lwswitch_utils.fabric_reset(), \
        "Failed to reset fabric"

    # start fabric manager again
    fm = apps.LwHostEngineApp(shlex.split(test_lwswitch_utils.FABRIC_MANAGER_ARGS))
    fm.start(timeout=test_lwswitch_utils.FABRIC_MANAGER_APP_TIMEOUT_SECS)
    logger.info("Start fabric manager pid: %s" % fm.getpid())

    # wait for fabric manager finish programming the fabric
    test_utils.wait_for_fabric_manager_ready()

    # Generate some traffic for the lwswitches
    test_utils.run_p2p_bandwidth_app(test_lwswitch_utils.MEMCPY_DTOD_WRITE_CE_BANDWIDTH)

    # stop fabric manager
    logger.info("Stop fabric manager")
    fm.terminate()
    fm.validate()


