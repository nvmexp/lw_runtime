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

def is_dgx_2():
    p = subprocess.Popen("lshw -c system | grep product: | grep DGX-2 | wc -l",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if int(out) != 1:
        logger.info("lwswitch-audit test only supported on DGX-2")
        return False

    switch_bdf = test_lwswitch_utils.get_lwswitch_pci_bdf()
    if len(switch_bdf) == 12:
        return True
    else:
        logger.info("lwswitch-audit test only supported on DGX-2 with 12 LWSwitch")
        return False


def validate_request_entry_change():
    #Ilwalidate request entry
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --req 0:5:3:0:0", 
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q -s 0 -d 3",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert int(out) == 5
    
    #fix request entry 
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --req 0:5:3:1:4",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q -s 0 -d 3",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert int(out) == 6

    logger.info("Reading request entries correctly")

def validate_response_entry_change():
    #Ilwalidate response entry
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --res 0:5:7:0:0",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q -s 1 -d 0",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert int(out) == -1
    
    #fix response entry 
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --res 0:5:7:1:16",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q -s 1 -d 0",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert int(out) == 6
    logger.info("Reading response entries correctly")

def test_lwswitch_num_connections():
    if is_dgx_2() == False:
        return

    #read all connections
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --csv",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    exit_code = p.returncode
   
    lwlinks = [] 
    for line in out.splitlines():
        lwlinks.append(line.split(",")[1:17])

    lwlinks = lwlinks[1:17]
    if len(lwlinks) != 16:
        logger.info("lwswitch-audit test only supported on DGX-2 with 12 LWSwitch and 16 GPUs")
        return

    #Check all links are correct
    for i in range(0,16):
        for j in range(0, 16):
            if i == j:
                assert lwlinks[i][j] == 'X'
            else:
                assert int(lwlinks[i][j]) == 6
    logger.info("Number of links between all GPUs correct")

    validate_request_entry_change()
    validate_response_entry_change()

def test_lwswitch_router_link_ids():
    if is_dgx_2() == False:
        return

    #Invalid Requestor Link ID simulated
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --rlid 0:5:132 | grep '\[Error\]:Requestor Link ID' | wc -l",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    assert int(out) == 1

    #Valid Requestor Link ID simulated
    p = subprocess.Popen("./apps/lwswitch-audit/lwswitch-audit -q --rlid 0:5:1| grep '\[Error\]:Requestor Link ID' | wc -l",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    assert int(out) == 0
    logger.info("Reading Router Link IDs correctly")

