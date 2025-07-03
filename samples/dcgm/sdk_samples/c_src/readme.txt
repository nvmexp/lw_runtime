DCGM SDK Samples
----------------

-----------------------------------------------
Sample: Group Configuration 
Folder: 0_configuration_sample

This sample goes through the process of creating a group, adding GPUs to it and then getting, setting and enforcing a configuration
on that group. Some error handling through status handles is also shown.

Key concepts:
- Querying for GPUs on system
- Group creation
- Managing group configurations
- Status handles


-----------------------------------------------
Sample: Group Health, Watches and Diagnostics
Folder: 1_healthAndDiagnostics_sample

This sample demonstrates the process of creating a group and managing health watches for that group. Demonstrates setting watches, 
querying them for information and also running group diagnostics.

Key concepts:
- Group creation
- Managing group health watches
- Managing group field watches
- Running group diagnostics
- DCGM fields and field collections


-----------------------------------------------
Sample: Process statistics
Folder: 2_processStatistics_sample

This sample goes through the process of enabling process watches on a group, running a process and viewing the statistics of the 
group while the process ran.

Key concepts:
- Using the default group (All GPUs)
- Managing process watches


-----------------------------------------------
Sample: Group Policy
Folder: 3_policy_sample

This sample demonstrates the process of creating a group and getting/setting the policies of the GPUs in that group. Policy 
registration is also shown.

Key concepts:
- Group Creation
- Managing group policies
- Registering for policy violation callbacks
- Status handles
