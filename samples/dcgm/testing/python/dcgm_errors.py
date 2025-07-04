import ctypes
import dcgm_structs

DCGM_FR_OK                                  = 0  # No error
DCGM_FR_UNKNOWN                             = 1  # Unknown error code
DCGM_FR_UNRECOGNIZED                        = 2  # Unrecognized error code
DCGM_FR_PCI_REPLAY_RATE                     = 3  # Unacceptable rate of PCI errors
DCGM_FR_VOLATILE_DBE_DETECTED               = 4  # Uncorrectable volatile double bit error
DCGM_FR_VOLATILE_SBE_DETECTED               = 5  # Unacceptable rate of volatile single bit errors
DCGM_FR_PENDING_PAGE_RETIREMENTS            = 6  # Pending page retirements detected
DCGM_FR_RETIRED_PAGES_LIMIT                 = 7  # Unacceptable total page retirements detected
DCGM_FR_RETIRED_PAGES_DBE_LIMIT             = 8  # Unacceptable total page retirements due to uncorrectable errors
DCGM_FR_CORRUPT_INFOROM                     = 9  # Corrupt inforom found
DCGM_FR_CLOCK_THROTTLE_THERMAL              = 10 # Clocks being throttled due to overheating
DCGM_FR_POWER_UNREADABLE                    = 11 # Cannot get a reading for power from LWML
DCGM_FR_CLOCK_THROTTLE_POWER                = 12 # Clock being throttled due to power restrictions
DCGM_FR_LWLINK_ERROR_THRESHOLD              = 13 # Unacceptable rate of LWLink errors
DCGM_FR_LWLINK_DOWN                         = 14 # LWLink is down
DCGM_FR_LWSWITCH_FATAL_ERROR                = 15 # Fatal errors on the LWSwitch
DCGM_FR_LWSWITCH_NON_FATAL_ERROR            = 16 # Non-fatal errors on the LWSwitch
DCGM_FR_LWSWITCH_DOWN                       = 17 # LWSwitch is down
DCGM_FR_NO_ACCESS_TO_FILE                   = 18 # Cannot access a file
DCGM_FR_LWML_API                            = 19 # Error oclwrred on an LWML API
DCGM_FR_DEVICE_COUNT_MISMATCH               = 20 # Disagreement in GPU count between /dev and LWML
DCGM_FR_BAD_PARAMETER                       = 21 # Bad parameter passed to API
DCGM_FR_CANNOT_OPEN_LIB                     = 22 # Cannot open a library that must be accessed
DCGM_FR_BLACKLISTED_DRIVER                  = 23 # A blacklisted driver (nouveau) is active
DCGM_FR_LWML_LIB_BAD                        = 24 # The LWML library is missing expected functions
DCGM_FR_GRAPHICS_PROCESSES                  = 25 # Graphics processes are active on this GPU
DCGM_FR_HOSTENGINE_CONN                     = 26 # Unstable connection to lw-hostengine (daemonized DCGM)
DCGM_FR_FIELD_QUERY                         = 27 # Error querying a field from DCGM
DCGM_FR_BAD_LWDA_ELW                        = 28 # The environment has variables that hurt LWCA
DCGM_FR_PERSISTENCE_MODE                    = 29 # Persistence mode is disabled
DCGM_FR_LOW_BANDWIDTH                       = 30 # The bandwidth is unacceptably low
DCGM_FR_HIGH_LATENCY                        = 31 # Latency is too high
DCGM_FR_CANNOT_GET_FIELD_TAG                = 32 # Cannot find a tag for a field
DCGM_FR_FIELD_VIOLATION                     = 33 # The value for the specified error field is above 0
DCGM_FR_FIELD_THRESHOLD                     = 34 # The value for the specified field is above the threshold
DCGM_FR_FIELD_VIOLATION_DBL                 = 35 # The value for the specified error field is above 0
DCGM_FR_FIELD_THRESHOLD_DBL                 = 36 # The value for the specified field is above the threshold
DCGM_FR_UNSUPPORTED_FIELD_TYPE              = 37 # Field type cannot be supported
DCGM_FR_FIELD_THRESHOLD_TS                  = 38 # The value for the specified field is above the threshold
DCGM_FR_FIELD_THRESHOLD_TS_DBL              = 39 # The value for the specified field is above the threshold
DCGM_FR_THERMAL_VIOLATIONS                  = 40 # Thermal violations detected
DCGM_FR_THERMAL_VIOLATIONS_TS               = 41 # Thermal violations detected with a timestamp
DCGM_FR_TEMP_VIOLATION                      = 42 # Temperature is too high
DCGM_FR_THROTTLING_VIOLATION                = 43 # Non-benign clock throttling is oclwrring
DCGM_FR_INTERNAL                            = 44 # An internal error was detected
DCGM_FR_PCIE_GENERATION                     = 45 # PCIe generation is too low
DCGM_FR_PCIE_WIDTH                          = 46 # PCIe width is too low
DCGM_FR_ABORTED                             = 47 # Test was aborted by a user signal
DCGM_FR_TEST_DISABLED                       = 48 # This test is disabled for this GPU
DCGM_FR_CANNOT_GET_STAT                     = 49 # Cannot get telemetry for a needed value
DCGM_FR_STRESS_LEVEL                        = 50 # Stress level is too low (bad performance)
DCGM_FR_LWDA_API                            = 51 # Error calling the specified LWCA API
DCGM_FR_FAULTY_MEMORY                       = 52 # Faulty memory detected on this GPU
DCGM_FR_CANNOT_SET_WATCHES                  = 53 # Unable to set field watches in DCGM
DCGM_FR_LWDA_UNBOUND                        = 54 # LWCA context is no longer bound
DCGM_FR_ECC_DISABLED                        = 55 # ECC memory is disabled right now
DCGM_FR_MEMORY_ALLOC                        = 56 # Cannot allocate memory
DCGM_FR_LWDA_DBE                            = 57 # LWCA detected unrecovable double-bit error
DCGM_FR_MEMORY_MISMATCH                     = 58 # Memory error detected
DCGM_FR_LWDA_DEVICE                         = 59 # No LWCA device discoverable for existing GPU
DCGM_FR_ECC_UNSUPPORTED                     = 60 # ECC memory is unsupported by this SKU
DCGM_FR_ECC_PENDING                         = 61 # ECC memory is in a pending state
DCGM_FR_MEMORY_BANDWIDTH                    = 62 # Memory bandwidth is too low
DCGM_FR_TARGET_POWER                        = 63 # Cannot hit the target power draw
DCGM_FR_API_FAIL                            = 64 # The specified API call failed
DCGM_FR_API_FAIL_GPU                        = 65 # The specified API call failed for the specified GPU
DCGM_FR_LWDA_CONTEXT                        = 66 # Cannot create a LWCA context on this GPU
DCGM_FR_DCGM_API                            = 67 # DCGM API failure
DCGM_FR_CONLWRRENT_GPUS                     = 68 # Need multiple GPUs to run this test
DCGM_FR_TOO_MANY_ERRORS                     = 69 # More errors than fit in the return struct
DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD          = 70 # More than 100 CRC errors are happening per second
DCGM_FR_LWLINK_ERROR_CRITICAL               = 71 # LWLink error for a field that should always be 0
DCGM_FR_ENFORCED_POWER_LIMIT                = 72 # The enforced power limit is too low to hit the target
DCGM_FR_MEMORY_ALLOC_HOST                   = 73 # Cannot allocate memory on the host
DCGM_FR_GPU_OP_MODE                         = 74 # Bad GPU operating mode for running plugin
DCGM_FR_NO_MEMORY_CLOCKS                    = 75 # No memory clocks with the needed MHz were found
DCGM_FR_NO_GRAPHICS_CLOCKS                  = 76 # No graphics clocks with the needed MHz were found
DCGM_FR_HAD_TO_RESTORE_STATE                = 77 # Note that we had to restore a GPU's state
DCGM_FR_L1TAG_UNSUPPORTED                   = 78 # L1TAG test is unsupported by this SKU
DCGM_FR_L1TAG_MISCOMPARE                    = 79 # L1TAG test failed on a miscompare
DCGM_FR_ERROR_SENTINEL                      = 80 # MUST BE THE LAST ERROR CODE

# Standard message for running a field diagnostic 
TRIAGE_RUN_FIELD_DIAG_MSG = "Run a field diagnostic on the GPU."

# Define DCGM error priorities
DCGM_ERROR_MONITOR     = 0 # Can perform workload, but needs to be monitored.
DCGM_ERROR_ISOLATE     = 1 # Cannot perform workload. GPU should be isolated.
DCGM_ERROR_UNKNOWN     = 2 # This error code is not recognized


# Messages for the error codes. All messages must be defined in the ERROR_CODE_MSG <msg> format
# where <msg> is the actual message.

DCGM_FR_OK_MSG                        = "The operation completed successfully."
DCGM_FR_UNKNOWN_MSG                   = "Unknown error."
DCGM_FR_UNRECOGNIZED_MSG              = "Unrecognized error code."
# replay limit, gpu id, replay errors detected
DCGM_FR_PCI_REPLAY_RATE_MSG           =  "Detected more than %u PCIe replays per minute for GPU %u : %d"
# dbes deteced, gpu id
DCGM_FR_VOLATILE_DBE_DETECTED_MSG     =  "Detected %d volatile double-bit ECC error(s) in GPU %u."
# sbe limit, gpu id, sbes detected
DCGM_FR_VOLATILE_SBE_DETECTED_MSG     =  "More than %u single-bit ECC error(s) detected in GPU %u Volatile SBEs: %lld"
# gpu id
DCGM_FR_PENDING_PAGE_RETIREMENTS_MSG  =  "A pending retired page has been detected in GPU %u."
# retired pages detected, gpud id
DCGM_FR_RETIRED_PAGES_LIMIT_MSG       =  "%u or more retired pages have been detected in GPU %u. "
# retired pages due to dbes detected, gpu id
DCGM_FR_RETIRED_PAGES_DBE_LIMIT_MSG   = "An excess of %u retired pages due to DBEs have been detected and" \
                                                    " more than one page has been retired due to DBEs in the past" \
                                                    " week in GPU %u."
# gpu id
DCGM_FR_CORRUPT_INFOROM_MSG          =  "A corrupt InfoROM has been detected in GPU %u."
# gpu id
DCGM_FR_CLOCK_THROTTLE_THERMAL_MSG   =  "Detected clock throttling due to thermal violation in GPU %u."
# gpu id
DCGM_FR_POWER_UNREADABLE_MSG         =  "Cannot reliably read the power usage for GPU %u."
# gpu id
DCGM_FR_CLOCK_THROTTLE_POWER_MSG     =  "Detected clock throttling due to power violation in GPU %u."
# lwlink errors detected, lwlink id, error threshold
DCGM_FR_LWLINK_ERROR_THRESHOLD_MSG   =  "Detected %ld LwLink errors on LwLink %u which exceeds threshold of %u"
# gpu id, lwlink id
DCGM_FR_LWLINK_DOWN_MSG              =  "GPU %u's LwLink link %d is lwrrently down"
# lwswitch id, lwlink id
DCGM_FR_LWSWITCH_FATAL_ERROR_MSG     =  "Detected fatal errors on LwSwitch %u link %u"
# lwswitch id, lwlink id
DCGM_FR_LWSWITCH_NON_FATAL_ERROR_MSG =  "Detected nonfatal errors on LwSwitch %u link %u"
# lwswitch id, lwlink port
DCGM_FR_LWSWITCH_DOWN_MSG            =  "LwSwitch physical ID %u's LwLink port %d is lwrrently down."
# file path, error detail
DCGM_FR_NO_ACCESS_TO_FILE_MSG         = "File %s could not be accessed directly: %s"
# purpose for communicating with LWML, LWML error as string, LWML error
DCGM_FR_LWML_API_MSG                  = "Error calling LWML API %s: %s"
DCGM_FR_DEVICE_COUNT_MISMATCH_MSG     = "The number of devices LWML returns is different than the number "\
                                                "of devices in /dev."
# function name
DCGM_FR_BAD_PARAMETER_MSG             = "Bad parameter to function %s cannot be processed"
# library name, error returned from dlopen
DCGM_FR_CANNOT_OPEN_LIB_MSG           = "Cannot open library %s: '%s'"
# the name of the blacklisted driver
DCGM_FR_BLACKLISTED_DRIVER_MSG        = "Found blacklisted driver: %s"
# the name of the function that wasn't found
DCGM_FR_LWML_LIB_BAD_MSG              = "Cannot get pointer to %s from liblwidia-ml.so"
DCGM_FR_GRAPHICS_PROCESSES_MSG        = "LWVS has detected graphics processes running on at least one "\
                                                "GPU. This may cause some tests to fail."
# error message from the API call
DCGM_FR_HOSTENGINE_CONN_MSG           = "Could not connect to the host engine: '%s'"
# field name, gpu id
DCGM_FR_FIELD_QUERY_MSG               = "Could not query field %s for GPU %u"
# environment variable name
DCGM_FR_BAD_LWDA_ELW_MSG              = "Found LWCA performance-limiting environment variable '%s'."
# gpu id
DCGM_FR_PERSISTENCE_MODE_MSG          = "Persistence mode for GPU %u is lwrrently disabled. The DCGM "\
                                                "diagnostic requires peristence mode to be enabled."
DCGM_FR_LOW_BANDWIDTH_MSG             = "Bandwidth of GPU %u in direction %s of %.2f did not exceed "\
                                                "minimum required bandwidth of %.2f."
DCGM_FR_HIGH_LATENCY_MSG              = "Latency type %s of GPU %u value %.2f exceeded maximum allowed "\
                                                "latency of %.2f."
DCGM_FR_CANNOT_GET_FIELD_TAG_MSG      = "Unable to get field information for field id %hu"
DCGM_FR_FIELD_VIOLATION_MSG           = "Detected %ld %s for GPU %u"
DCGM_FR_FIELD_THRESHOLD_MSG           = "Detected %ld %s for GPU %u which is above the threshold %ld"
DCGM_FR_FIELD_VIOLATION_DBL_MSG       = "Detected %.1f %s for GPU %u"
DCGM_FR_FIELD_THRESHOLD_DBL_MSG       = "Detected %.1f %s for GPU %u which is above the threshold %.1f"
DCGM_FR_UNSUPPORTED_FIELD_TYPE_MSG    = "Field %s is not supported by this API because it is neither an "\
                                                "int64 nor a double type."
DCGM_FR_FIELD_THRESHOLD_TS_MSG        = "%s met or exceeded the threshold of %lu per second: %lu at "\
                                                "%.1f seconds into the test."
DCGM_FR_FIELD_THRESHOLD_TS_DBL_MSG    = "%s met or exceeded the threshold of %.1f per second: %.1f at "\
                                                "%.1f seconds into the test."
DCGM_FR_THERMAL_VIOLATIONS_MSG        = "There were thermal violations totaling %lu seconds for GPU %u"
DCGM_FR_THERMAL_VIOLATIONS_TS_MSG     = "Thermal violations totaling %lu samples started at %.1f seconds "\
                                                "into the test for GPU %u"
DCGM_FR_TEMP_VIOLATION_MSG            = "Temperature %lld of GPU %u exceeded user-specified maximum "\
                                                "allowed temperature %lld"
DCGM_FR_THROTTLING_VIOLATION_MSG      = "Clocks are being throttling for GPU %u because of clock "\
                                                "throttling starting %.1f seconds into the test. %s"
DCGM_FR_INTERNAL_MSG                  = "There was an internal error during the test: '%s'"
DCGM_FR_PCIE_GENERATION_MSG           = "GPU %u is running at PCI link generation %d, which is below "\
                                                "the minimum allowed link generation of %d (parameter '%s')"
DCGM_FR_PCIE_WIDTH_MSG                = "GPU %u is running at PCI link width %dX, which is below the "\
                                                "minimum allowed link generation of %d (parameter '%s')"
DCGM_FR_ABORTED_MSG                   = "Test was aborted early due to user signal"
DCGM_FR_TEST_DISABLED_MSG             = "The %s test is skipped for this GPU."
DCGM_FR_CANNOT_GET_STAT_MSG           = "Unable to generate / collect stat %s for GPU %u"
DCGM_FR_STRESS_LEVEL_MSG              = "Max stress level of %.1f did not reach desired stress level of "\
                                                "%.1f for GPU %u"
DCGM_FR_LWDA_API_MSG                  = "Error using LWCA API %s"
DCGM_FR_FAULTY_MEMORY_MSG             = "Found %d faulty memory elements on GPU %u"
DCGM_FR_CANNOT_SET_WATCHES_MSG        = "Unable to add field watches to DCGM: %s"
DCGM_FR_LWDA_UNBOUND_MSG              = "Lwca GPU %d is no longer bound to a LWCA context...Aborting"
DCGM_FR_ECC_DISABLED_MSG              = "Skipping test %s because ECC is not enabled on GPU %u"
DCGM_FR_MEMORY_ALLOC_MSG              = "Couldn't allocate at least %.1f%% of GPU memory on GPU %u"
DCGM_FR_LWDA_DBE_MSG                  = "LWCA APIs have indicated that a double-bit ECC error has "\
                                                "oclwred on GPU %u."
DCGM_FR_MEMORY_MISMATCH_MSG           = "A memory mismatch was detected on GPU %u, but no error was "\
                                                "reported by LWCA or LWML."
DCGM_FR_LWDA_DEVICE_MSG               = "Unable to find a corresponding LWCA device for GPU %u: '%s'"
DCGM_FR_ECC_UNSUPPORTED_MSG           = "This card does not support ECC Memory. Skipping test."
DCGM_FR_ECC_PENDING_MSG               = "ECC memory for GPU %u is in a pending state."
DCGM_FR_MEMORY_BANDWIDTH_MSG          = "GPU %u only achieved a memory bandwidth of %.2f GB/s, failing "\
                                                "to meet %.2f GB/s for test %d"
DCGM_FR_TARGET_POWER_MSG              = "Max power of %.1f did not reach desired power minimum %s of "\
                                                "%.1f for GPU %u"
DCGM_FR_API_FAIL_MSG                  = "API call %s failed: '%s'"
DCGM_FR_API_FAIL_GPU_MSG              = "API call %s failed for GPU %u: '%s'"
DCGM_FR_LWDA_CONTEXT_MSG              = "GPU %u failed to create a LWCA context: %s"
DCGM_FR_DCGM_API_MSG                  = "Error using DCGM API %s"
DCGM_FR_CONLWRRENT_GPUS_MSG           = "Unable to run conlwrrent pair bandwidth test without 2 or more "\
                                        "gpus. Skipping"
DCGM_FR_TOO_MANY_ERRORS_MSG           = "This API can only return up to four errors per system. "\
                                        "Additional errors were found for this system that couldn't be "\
                                        "communicated."
DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD_MSG = "%.1f %s LwLink errors found oclwring per second on GPU %u, "\
                                        "exceeding the limit of 100 per second."
DCGM_FR_LWLINK_ERROR_CRITICAL_MSG      = "Detected %ld %s LwLink errors on GPU %u's LWLink (should be 0)"
DCGM_FR_ENFORCED_POWER_LIMIT_MSG       = "Enforced power limit on GPU %u set to %.1f, which is too low to "\
                                         "attempt to achieve target power %.1f"
DCGM_FR_MEMORY_ALLOC_HOST_MSG          = "Cannot allocate %zu bytes on the host"
DCGM_FR_GPU_OP_MODE_MSG                = "Skipping plugin due to a GPU being in GPU Operating Mode: LOW_DP."
DCGM_FR_NO_MEMORY_CLOCKS_MSG           = "No memory clocks <= %u MHZ were found in %u supported memory clocks."
DCGM_FR_NO_GRAPHICS_CLOCKS_MSG         = "No graphics clocks <= %u MHZ were found in %u supported graphics clocks for memory clock %u MHZ."
DCGM_FR_HAD_TO_RESTORE_STATE_MSG       = "Had to restore GPU state on LWML GPU(s): %s"
DCGM_FR_L1TAG_UNSUPPORTED_MSG          = "This card does not support the L1 cache test. Skipping test."
DCGM_FR_L1TAG_MISCOMPARE_MSG           = "The L1 cache test failed with a miscompare."


# Suggestions for next steps for the corresponding error message
DCGM_FR_OK_NEXT                       = "N/A"
DCGM_FR_UNKNOWN_NEXT                  = ""
DCGM_FR_UNRECOGNIZED_NEXT             = ""
DCGM_FR_PCI_REPLAY_RATE_NEXT          = "Reconnect PCIe card. Run system side PCIE diagnostic utilities "\
                                                "to verify hops off the GPU board. If issue is on the board, run "\
                                                "the field diagnostic."
DCGM_FR_VOLATILE_DBE_DETECTED_NEXT    = "Drain the GPU and reset it or reboot the node."
DCGM_FR_VOLATILE_SBE_DETECTED_NEXT    = "Monitor - this GPU can still perform workload."
DCGM_FR_PENDING_PAGE_RETIREMENTS_NEXT = "If volatile double bit errors exist, drain the GPU and reset it "\
                                                "or reboot the node. Otherwise, monitor - GPU can still perform "\
                                                "workload."
DCGM_FR_RETIRED_PAGES_LIMIT_NEXT      = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_RETIRED_PAGES_DBE_LIMIT_NEXT  = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_CORRUPT_INFOROM_NEXT          = "Flash the InfoROM to clear this corruption."
DCGM_FR_CLOCK_THROTTLE_THERMAL_NEXT   = "Check the cooling on this machine."
DCGM_FR_POWER_UNREADABLE_NEXT         = ""
DCGM_FR_CLOCK_THROTTLE_POWER_NEXT     = "Monitor the power conditions. This GPU can still perform workload."
DCGM_FR_LWLINK_ERROR_THRESHOLD_NEXT   = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_LWLINK_DOWN_NEXT              = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_LWSWITCH_FATAL_ERROR_NEXT     = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_LWSWITCH_NON_FATAL_ERROR_NEXT = "Monitor the LWSwitch. It can still perform workload."
DCGM_FR_LWSWITCH_DOWN_NEXT            = ""
DCGM_FR_NO_ACCESS_TO_FILE_NEXT        = "Check relevant permissions, access, and existence of the file."
DCGM_FR_LWML_API_NEXT                 = "Check the error condition and ensure that appropriate libraries "\
                                                "are present and accessible."
DCGM_FR_DEVICE_COUNT_MISMATCH_NEXT    = "Check for the presence of cgroups, operating system blocks, and "\
                                                "or unsupported / older cards"
DCGM_FR_BAD_PARAMETER_NEXT            = ""
DCGM_FR_CANNOT_OPEN_LIB_NEXT          = "Check for the existence of the library and set LD_LIBRARY_PATH "\
                                                "if needed."
DCGM_FR_BLACKLISTED_DRIVER_NEXT       = "Please load the appropriate driver."
DCGM_FR_LWML_LIB_BAD_NEXT             = "Make sure that the required version of liblwidia-ml.so "\
                                                "is present and accessible on the system."
DCGM_FR_GRAPHICS_PROCESSES_NEXT       = "Stop the graphics processes or run this diagnostic on a server "\
                                                "that is not being used for display purposes."
DCGM_FR_HOSTENGINE_CONN_NEXT          = "If hostengine is run separately, please ensure that it is up "\
                                                "and responsive."
DCGM_FR_FIELD_QUERY_NEXT              = ""
DCGM_FR_BAD_LWDA_ELW_NEXT             = "Please unset this environment variable to address test failures."
DCGM_FR_PERSISTENCE_MODE_NEXT         = "Enable persistence mode by running \"lwpu-smi -i <gpuId> -pm "\
                                                "1 \" as root."
DCGM_FR_LOW_BANDWIDTH_NEXT            = "Verify that your minimum bandwidth setting is appropriate for "\
                                                "all topological consequences."
DCGM_FR_HIGH_LATENCY_NEXT             = ""
DCGM_FR_CANNOT_GET_FIELD_TAG_NEXT     = ""
DCGM_FR_FIELD_VIOLATION_NEXT          = ""
DCGM_FR_FIELD_THRESHOLD_NEXT          = ""
DCGM_FR_FIELD_VIOLATION_DBL_NEXT      = ""
DCGM_FR_FIELD_THRESHOLD_DBL_NEXT      = ""
DCGM_FR_UNSUPPORTED_FIELD_TYPE_NEXT   = ""
DCGM_FR_FIELD_THRESHOLD_TS_NEXT       = ""
DCGM_FR_FIELD_THRESHOLD_TS_DBL_NEXT   = ""
DCGM_FR_THERMAL_VIOLATIONS_NEXT       = ""
DCGM_FR_THERMAL_VIOLATIONS_TS_NEXT    = ""
DCGM_FR_TEMP_VIOLATION_NEXT           = "Verify that the user-specified temperature maximum is set "\
                                                "correctly. If it is, check the cooling for this GPU and node."
DCGM_FR_THROTTLING_VIOLATION_NEXT     = ""
DCGM_FR_INTERNAL_NEXT                 = ""
DCGM_FR_PCIE_GENERATION_NEXT          = ""
DCGM_FR_PCIE_WIDTH_NEXT               = ""
DCGM_FR_ABORTED_NEXT                  = ""
DCGM_FR_TEST_DISABLED_NEXT            = ""
DCGM_FR_CANNOT_GET_STAT_NEXT          = "If running a standalone lw-hostengine, verify that it is up "\
                                                "and responsive."
DCGM_FR_STRESS_LEVEL_NEXT             = ""
DCGM_FR_LWDA_API_NEXT                 = ""
DCGM_FR_FAULTY_MEMORY_NEXT            = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_CANNOT_SET_WATCHES_NEXT       = ""
DCGM_FR_LWDA_UNBOUND_NEXT             = ""
DCGM_FR_ECC_DISABLED_NEXT             = "Enable ECC memory by running \"lwpu-smi -i <gpuId> -e 1\" "\
                                                "to enable. This may require a GPU reset or reboot to take effect."
DCGM_FR_MEMORY_ALLOC_NEXT             = ""
DCGM_FR_LWDA_DBE_NEXT                 = TRIAGE_RUN_FIELD_DIAG_MSG 
DCGM_FR_MEMORY_MISMATCH_NEXT          = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_LWDA_DEVICE_NEXT              = ""
DCGM_FR_ECC_UNSUPPORTED_NEXT          = ""
DCGM_FR_ECC_PENDING_NEXT              = "Please reboot to activate it."
DCGM_FR_MEMORY_BANDWIDTH_NEXT         = ""
DCGM_FR_TARGET_POWER_NEXT             = ""
DCGM_FR_API_FAIL_NEXT                 = ""
DCGM_FR_API_FAIL_GPU_NEXT             = ""
DCGM_FR_LWDA_CONTEXT_NEXT             = "Please make sure the correct driver version is installed and "\
                                                "verify that no conflicting libraries are present."
DCGM_FR_DCGM_API_NEXT                 = ""
DCGM_FR_CONLWRRENT_GPUS_NEXT          = ""
DCGM_FR_TOO_MANY_ERRORS_NEXT          = ""
DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD_NEXT = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_LWLINK_ERROR_CRITICAL_NEXT    = TRIAGE_RUN_FIELD_DIAG_MSG
DCGM_FR_ENFORCED_POWER_LIMIT_NEXT     = "If this enforced power limit is necessary, then this test "\
                                        "cannot be run. If it is unnecessary, then raise the enforced "\
                                        "power limit setting to be able to run this test."
DCGM_FR_MEMORY_ALLOC_HOST_NEXT        = "Manually kill processes or restart your machine."
DCGM_FR_GPU_OP_MODE_NEXT              = "Fix by running lwpu-smi as root with: lwpu-smi --gom=0 -i "\
                                        "<gpu index>"
DCGM_FR_NO_MEMORY_CLOCKS_NEXT         = ""
DCGM_FR_NO_GRAPHICS_CLOCKS_NEXT       = ""
DCGM_FR_HAD_TO_RESTORE_STATE_NEXT     = ""
DCGM_FR_L1TAG_UNSUPPORTED_NEXT        = ""
DCGM_FR_L1TAG_MISCOMPARE_NEXT         = TRIAGE_RUN_FIELD_DIAG_MSG

def dcgmErrorGetPriorityByCode(code):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmErrorGetPriorityByCode")
    ret = fn(code)
    return ret

def dcgmErrorGetFormatMsgByCode(code):
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmErrorGetFormatMsgByCode")
    fn.restype = ctypes.c_char_p
    ret = fn(code)
    return ret
