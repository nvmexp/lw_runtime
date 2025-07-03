import sys
sys.path.insert(0, '../')
import dcgm_structs
import dcgm_fields
import dcgm_agent
import dcgmvalue
from threading import Thread
from time import sleep

## Look at __name__ == "__main__" for entry point to the script

class RunDCGM():
    
    def __init__(self, ip, opMode):
        self.ip = ip
        self.opMode = opMode
    
    def __enter__(self):
        dcgm_structs._dcgmInit()
        self.handle = dcgm_agent.dcgmInit()
        return self.handle
        
    def __exit__(self, eType, value, traceback):
        dcgm_agent.dcgmShutdown()


## Helper method to colwert enum to system name
def helper_colwert_system_enum_to_sytem_name(system):
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_PCIE):
        return "PCIe"

    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_LWLINK):
        return "LwLink"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_PMU):
        return "PMU"

    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_MLW):
        return "MLW"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_MEM):
        return "MEM"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_SM):
        return "SM"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_INFOROM):
        return "Inforom"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_THERMAL):
        return "Thermal"
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_POWER):
        return "Power"   
    
    if system & (1 << dcgm_structs.DCGM_HEALTH_WATCH_DRIVER):
        return "Driver"
    
    
## helper method to colwert helath return to a string for display purpose        
def colwert_overall_health_to_string(health):
    if health == dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        return "Pass"
    elif health == dcgm_structs.DCGM_HEALTH_RESULT_WARN:
        return "Warn"
    elif  health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL:
        return "Fail"
    else :
        return "N/A"
    
## Worker function
def agent_worker_function(dcgmHandle, groupId):
    NUM_ITERATIONS = 5
    count = 0 
    
    groupId = groupId
    
    ## Add the health watches
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_ALL
    dcgm_agent.dcgmHealthSet(dcgmHandle, groupId, newSystems)

    while True:
        dcgm_agent.dcgmUpdateAllFields(dcgmHandle, 1)

        try:
            ## Ilwoke Health checks
            group_health = dcgm_agent.dcgmHealthCheck(dcgmHandle, groupId)
            print "Overall Health for the group: %s" % colwert_overall_health_to_string(group_health.overallHealth)
            
            for index in range (0, group_health.gpuCount):
                print "GPU ID : %d" % group_health.gpu[index].gpuId
 
                for incident in range (0, group_health.gpu[index].incidentCount):
                    print "system tested     : %d" % group_health.gpu[index].systems[incident].system
                    print "system health     : %s" % colwert_overall_health_to_string(group_health.gpu[index].systems[incident].health)
                    print "system health err : %s" % group_health.gpu[index].systems[incident].errorString
                    print "\n"

        except dcgm_structs.DCGMError as e:
            errorCode = e.value
            print "dcgmEngineHelathCheck returned error: %d" % errorCode
            sys.exc_clear()
        
        count = count + 1
        
        if count == NUM_ITERATIONS:
            break

        sleep(2)
        
        
## Main
def main():
    
    ## Initilaize the DCGM Engine as manual operation mode. This implies that it's exelwtion is 
    ## controlled by the monitoring agent. The user has to periodically call APIs such as 
    ## dcgmEnginePolicyTrigger and dcgmEngineUpdateAllFields which tells DCGM to wake up and 
    ## perform data collection and operations needed for policy management.
    with RunDCGM('127.0.0.1', dcgm_structs.DCGM_OPERATION_MODE_MANUAL) as handle:
    
        ## Create a default group. (Default group is comprised of all the GPUs on the node)
        ## Let's call the group as "all_gpus_group". The method returns an opaque handle (groupId) to
        ## identify the newly created group.
        groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "all_gpus_group")
        
        ## Ilwoke method to get information on the newly created group
        groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
        
        ## Create reference to DCGM status handler which can be used to get the statuses for multiple 
        ## operations on one or more devices present in the group
        status_handle = dcgm_agent.dcgmStatusCreate()
    
        ## The worker function can be exelwted as a separate thread or as part of the main thread.
        ## Exelwted as a separate thread here
        thread = Thread(target = agent_worker_function, args = (handle, groupId))
        thread.start()
    
        ##########################################
        # Any other useful work can be placed here
        ##########################################
        
        thread.join()
        print "Worker thread completed"
        
        ## Destroy the group
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except dcgm_structs.DCGMError as e:
            print >>sys.stderr, "Failed to remove the test group, error: %s" % e 
            sys.exit(1)
    
        ## Destroy the status handle
        try:
            dcgm_agent.dcgmStatusDestroy(status_handle)
        except dcgm_structs.DCGMError as e:
            print >>sys.stderr, "Failed to remove status handler, error: %s" % e   
            sys.exit(1)
    
if __name__ == '__main__':
    main()
    