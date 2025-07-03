Shared Fabric Mode Basic Flow Verification and Street Test Utility.

Note: All the dependent libs are copied under the directory to remove dependency with Perforce Source Tree. Also, the utility is not using unix-build as QA and Field Support requires to compile and run this test.

1) In Bare metal system, start FM in one of the following shared fabric mode in /usr/share/lwpu/lwswitch/fabricmanager.cfg
   * FABRIC_MODE=1      /* Shared LWSwitch multitenancy mode */
   * FBARIC_MODE=2      /* vGPU based multitenancy mode */

2) Restart Fabric Manager service after that or Reboot the system.

3) For vGPU mode, please enable SR-IOV functionality for all GPUs using the following script 
   * sudo ./sriov-manage -e ALL

4) To compile the test code, you will need FM Shared LWSwitch API headers files installed
   a) They will be copied to the required place for .RUN based installation
   b) For Debian/RPM, install the corresponding "devel-" package
   c) Then build the test by running "make". The output exelwtable is "shared_fabric_test"
 
5) The test source has to be modified to fix the partition GPU BDF information
   a) Once the test is built, just list the current partition information by running "./shared_fabric_test -l"
   b) The partition information will be dumped into a "partition_list.json" file
   c) This file has all the partition ID, GPU Physical IDs, and PCI BDF Information. Do the following to change the same
        1) Open platformModelDelta.cpp and modify the table "paritionIdToGpuBdf"
        2) Edit the list of partition information accordingly.
        3) For Two Delta base board, total 35 partitions and for single base board it will be 15
        4) Once you list the 16 GPU (for Two Delta) or 8 GPU (for single Delta) partitions, the rest will be easy as the partitions follow sequential physical Ids.
   d) The default number of stress test iteration is 50 (Change this MAX_PARTITION_ACTIVATION_COUNT definition in platformModelDelta.h file)
   e) After all the above changes, rebuild the test app by doing "make" again
   
6) Then do normal partition activation/deactivation or partition activation deactivation stress test.

7) If something goes wrong, the machine requires a reboot or complete SBR

  
    
