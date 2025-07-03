import pydcgm
import dcgm_field_helpers
import dcgm_fields
import dcgm_structs
import shlex
import time
import logger
import subprocess
import test_utils
import test_lwswitch_utils

def test_lwswitch_traffic_p2p():
    """
    Verifies that fabric can pass p2p read and write traffic successfully
    """

    # TX_0 and RX_0 on port 0
    lwSwitchBandwidth0FieldIds = []
    for i in range(dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00,
                   dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P00 + 1, 1):
        lwSwitchBandwidth0FieldIds.append(i)

    # TX_1 and RX_1 on port 0
    lwSwitchBandwidth1FieldIds = []
    for i in range(dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00,
                   dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P00 + 1, 1):
        lwSwitchBandwidth1FieldIds.append(i)

    dcgmHandle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")

    groupName = "test_lwswitches"
    allLwSwitchesGroup = pydcgm.DcgmGroup(dcgmHandle, groupName=groupName,
                                          groupType=dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES)

    fgName = "test_lwswitches_bandwidth0"
    lwSwitchBandwidth0FieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, name=fgName,
                                                         fieldIds=lwSwitchBandwidth0FieldIds)

    fgName = "test_lwswitches_bandwidth1"
    lwSwitchBandwidth1FieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, name=fgName,
                                                         fieldIds=lwSwitchBandwidth1FieldIds)

    updateFreq = int(20 / 2.0) * 1000000
    maxKeepAge = 600.0
    maxKeepSamples = 0

    lwSwitchBandwidth0Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
        dcgmHandle.handle, allLwSwitchesGroup.GetId(),
        lwSwitchBandwidth0FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
        updateFreq, maxKeepAge, maxKeepSamples, 0)
    lwSwitchBandwidth1Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
        dcgmHandle.handle, allLwSwitchesGroup.GetId(),
        lwSwitchBandwidth1FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
        updateFreq, maxKeepAge, maxKeepSamples, 0)

    # wait for FM reports and populates stats
    time.sleep(30)

    # read the counters before sending traffic
    lwSwitchBandwidth0Watcher.GetMore()
    lwSwitchBandwidth1Watcher.GetMore()

    for entityGroupId in lwSwitchBandwidth0Watcher.values.keys():
        for entityId in lwSwitchBandwidth0Watcher.values[entityGroupId]:
            bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00
            bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00

            counter0TxBefore = lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            bandwidth0FieldId += 1
            counter0RxBefore = lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            counter1TxBefore = lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value
            bandwidth1FieldId += 1
            counter1RxBefore = lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value

    # Generate write traffic for the lwswitches
    test_utils.run_p2p_bandwidth_app(test_lwswitch_utils.MEMCPY_DTOD_WRITE_CE_BANDWIDTH)

    # Generate read traffic for the lwswitches
    test_utils.run_p2p_bandwidth_app(test_lwswitch_utils.MEMCPY_DTOD_READ_CE_BANDWIDTH)

    # read the counters again after sending traffic
    lwSwitchBandwidth0Watcher.GetMore()
    lwSwitchBandwidth1Watcher.GetMore()

    for entityGroupId in lwSwitchBandwidth0Watcher.values.keys():
        for entityId in lwSwitchBandwidth0Watcher.values[entityGroupId]:
            bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00
            bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00

            counter0TxAfter = lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            bandwidth0FieldId += 1
            counter0RxAfter = lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[
                -1].value
            counter1TxAfter = lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value
            bandwidth1FieldId += 1
            counter1RxAfter = lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[
                -1].value

    assert counter0TxAfter > counter0TxBefore, "Counter0Tx did not increase"
    assert counter0RxAfter > counter0RxBefore, "counter0Rx did not increase"
    assert counter1TxAfter > counter1TxBefore, "Counter1Tx did not increase"
    assert counter1RxAfter > counter1RxBefore, "counter1Rx did not increase"

def inject_non_fatal_error(lwswitch_bdf_str):
    """
    Return True if injecting lwswitch fatal error with lwpex2 successfully
    """
    # clear dmesg brefore injecting error
    cmd = 'dmesg -c'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # inject non fatal error
    logger.info("Inject non fatal error to LWSwitch %s" % lwswitch_bdf_str)
    args = "--bdf " + lwswitch_bdf_str + " -r W:0:850040:F0094 -i 1"
    test_utils.run_lwpex2_app(shlex.split(args))

    # Wait for lwlink retraining
    # For non-fatal error, the polling interval is lwrrently 30 sec
    time.sleep(test_lwswitch_utils.FABRIC_MANAGER_NON_FATAL_ERROR_POLLING_INTERVAL)
    
    # look for fatal error in dmesg after injecting error
    cmd = 'dmesg > /tmp/dmesg.txt; grep "Non-fatal" /tmp/dmesg.txt | wc -l'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if int(out) > 0:
        return True
    else:
        logger.info("Non fatal error not found")
        return False

def test_lwswitch_error_non_fatal_recovery():
    """
    Verifies that non fatal error can be recovered
    """

    switch_bdf = test_lwswitch_utils.get_lwswitch_pci_bdf()
    if len(switch_bdf) == 0:
        logger.info("No LWSwitch found")
        return

    # Inject non fatal error to the first switch
    assert True == inject_non_fatal_error(switch_bdf[0]), \
        "Failed to inject switch non fatal error."

    # Make sure traffic can pass successfully after recovery
    test_utils.run_p2p_bandwidth_app(test_lwswitch_utils.MEMCPY_DTOD_WRITE_CE_BANDWIDTH)

