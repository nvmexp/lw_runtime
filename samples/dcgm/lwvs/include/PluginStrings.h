#ifndef PLUGINSTRINGS_H
#define PLUGINSTRINGS_H

/*****************************************************************************/
/********************* NOTE **************************************************/
/* All parameters added here need to be added to ParameterValidator.cpp or you
 * will get an error if you attempt to use them. *****************************/
/*****************************************************************************/

/*****************************************************************************/
/* Parameters common for all tests */
#define PS_PLUGIN_NAME "name"
#define PS_LOGFILE "logfile"
#define PS_LOGFILE_TYPE "logfile_type"
#define PS_RUN_IF_GOM_ENABLED "run_if_gom_enabled" //Should this plugin run if GOM mode is enabled

/******************************************************************************
 * SOFTWARE PLUGIN
 *****************************************************************************/
#define SW_STR_DO_TEST "do_test"
#define SW_STR_REQUIRE_PERSISTENCE "require_persistence_mode"
#define SW_PLUGIN_LF_NAME "software"

/******************************************************************************
 * BUSGRIND PLUGIN
 *****************************************************************************/
#define BG_PLUGIN_NAME "PCIe"
#define BG_PLUGIN_WL_NAME "pcie" //Plugin name on the whitelist (must be lowercase of BG_PLUGIN_NAME)
#define BG_PLUGIN_LF_NAME "pcie" //Logfile name for text/json/binary

/* Public parameters - we expect users to change these */
#define BG_STR_TEST_PINNED              "test_pinned"
#define BG_STR_TEST_UNPINNED            "test_unpinned"
#define BG_STR_TEST_P2P_ON              "test_p2p_on"
#define BG_STR_TEST_P2P_OFF             "test_p2p_off"
#define BG_STR_LWSWITCH_NON_FATAL_CHECK "check_non_fatal"

/* Private parameters */
#define BG_STR_IS_ALLOWED "is_allowed"   /* Is the busgrind plugin allowed to run? */

/* Private sub-test parameters. These apply to some sub tests and not others */
#define BG_STR_INTS_PER_COPY    "num_ints_per_copy"
#define BG_STR_ITERATIONS       "iterations"

/* Public sub-test parameters. These apply to some sub tests and not others */
#define BG_STR_MIN_BANDWIDTH    "min_bandwidth" /* Bandwidth below this in GB/s is
                                                   considered a failure for a given
                                                   sub test */
#define BG_STR_MAX_LATENCY "max_latency" /* Latency above this in microseconds is
                                            considered a failure for a given
                                            sub test */
#define BG_STR_MIN_PCI_GEN "min_pci_generation" /* Minimum PCI generation allowed.
                                                   PCI generation below this will
                                                   cause a sub test failure */
#define BG_STR_MIN_PCI_WIDTH "min_pci_width" /* Minimum PCI width allowed. 16x = 16 etc
                                                PCI width below this will cause a
                                                sub test failure */

#define BG_STR_MAX_PCIE_REPLAYS "max_pcie_replays" /* Maximum PCIe replays allowed per device
                                                      while the plugin runs. If more replays
                                                      occur than this threshold, this plugin
                                                      will fail */
#define BG_STR_MAX_MEMORY_CLOCK "max_memory_clock" /* Maximum memory clock in MHZ to use when locking
                                                      application clocks to max while busgrind
                                                      runs. */
#define BG_STR_MAX_GRAPHICS_CLOCK "max_graphics_clock" /* Maximum graphics clock in MHZ to use when
                                                          locking application clocks to max while
                                                          busgrind runs */

#define BG_STR_CRC_ERROR_THRESHOLD "lwlink_crc_error_threshold" /* threshold at which CRC errors should cause a
                                                                   failure */

/* Sub tests tags */
#define BG_SUBTEST_H2D_D2H_SINGLE_PINNED "h2d_d2h_single_pinned"
#define BG_SUBTEST_H2D_D2H_SINGLE_UNPINNED "h2d_d2h_single_unpinned"

#define BG_SUBTEST_H2D_D2H_CONLWRRENT_PINNED "h2d_d2h_conlwrrent_pinned"
#define BG_SUBTEST_H2D_D2H_CONLWRRENT_UNPINNED "h2d_d2h_conlwrrent_unpinned"

#define BG_SUBTEST_H2D_D2H_LATENCY_PINNED "h2d_d2h_latency_pinned"
#define BG_SUBTEST_H2D_D2H_LATENCY_UNPINNED "h2d_d2h_latency_unpinned"

#define BG_SUBTEST_P2P_BW_P2P_ENABLED "p2p_bw_p2p_enabled"
#define BG_SUBTEST_P2P_BW_P2P_DISABLED "p2p_bw_p2p_disabled"

#define BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_ENABLED "p2p_bw_conlwrrent_p2p_enabled"
#define BG_SUBTEST_P2P_BW_CONLWRRENT_P2P_DISABLED "p2p_bw_conlwrrent_p2p_disabled"

#define BG_SUBTEST_1D_EXCH_BW_P2P_ENABLED "1d_exch_bw_p2p_enabled"
#define BG_SUBTEST_1D_EXCH_BW_P2P_DISABLED "1d_exch_bw_p2p_disabled"

#define BG_SUBTEST_P2P_LATENCY_P2P_ENABLED "p2p_latency_p2p_enabled"
#define BG_SUBTEST_P2P_LATENCY_P2P_DISABLED "p2p_latency_p2p_disabled"

/******************************************************************************
 * CONSTANT POWER PLUGIN
 *****************************************************************************/
#define CP_PLUGIN_NAME "Targeted Power"
#define CP_PLUGIN_WL_NAME "targeted power" //Plugin name on the whitelist (must be lowercase of CP_PLUGIN_NAME)
#define CP_PLUGIN_LF_NAME "targeted_power" //Logfile name for text/json/binary

/* Public parameters - we expect users to change these */
#define CP_STR_TEST_DURATION    "test_duration"
#define CP_STR_TARGET_POWER     "target_power"
#define CP_STR_TEMPERATURE_MAX  "temperature_max"
#define CP_STR_FAIL_ON_CLOCK_DROP "fail_on_clock_drop"

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define CP_STR_USE_DGEMM            "use_dgemm"
#define CP_STR_LWDA_STREAMS_PER_GPU "lwda_streams_per_gpu"
#define CP_STR_READJUST_INTERVAL    "readjust_interval"
#define CP_STR_PRINT_INTERVAL       "print_interval"
#define CP_STR_TARGET_POWER_MIN_RATIO "target_power_min_ratio"
#define CP_STR_TARGET_POWER_MAX_RATIO "target_power_max_ratio"
#define CP_STR_MOV_AVG_PERIODS      "moving_average_periods"
#define CP_STR_TARGET_MOVAVG_MIN_RATIO "target_movavg_min_ratio"
#define CP_STR_TARGET_MOVAVG_MAX_RATIO "target_movavg_max_ratio"
#define CP_STR_ENFORCED_POWER_LIMIT    "enforced_power_limit"

#define CP_STR_MAX_MEMORY_CLOCK "max_memory_clock" /* Maximum memory clock in MHZ to use when locking
                                                      application clocks to max while targeted power
                                                      runs. */
#define CP_STR_MAX_GRAPHICS_CLOCK "max_graphics_clock" /* Maximum graphics clock in MHZ to use when
                                                          locking application clocks to max while
                                                          targeted power runs */
#define CP_STR_OPS_PER_REQUEUE "ops_per_requeue" /* How many matrix multiplication operations should
                                                    we queue every time the stream is idle. Setting
                                                    this higher overcomes the kernel launch latency */
#define CP_STR_STARTING_MATRIX_DIM "starting_matrix_dim" /* Starting dimension N in NxN for our matrix
                                                            to start ramping up from. Setting this higher
                                                            decreases the ramp-up time needed to hit
                                                            our power target. */
#define CP_STR_IS_ALLOWED "is_allowed"   /* Is the targeted power plugin allowed to run? */
#define CP_STR_SBE_ERROR_THRESHOLD   "max_sbe_errors" // Threshold beyond which sbe's are treated as errors

/******************************************************************************
 * CONSTANT PERF PLUGIN
 *****************************************************************************/
#define CPERF_PLUGIN_NAME "Targeted Stress"
#define CPERF_PLUGIN_WL_NAME "targeted stress" //Plugin name on the whitelist (Must be lowercase of CPERF_PLUGIN_NAME)
#define CPERF_PLUGIN_LF_NAME "targeted_stress" //Logfile name for text/json/binary

/* Public parameters - we expect users to change these */
#define CPERF_STR_TEST_DURATION    "test_duration"
#define CPERF_STR_TARGET_PERF      "target_stress"
#define CPERF_STR_TARGET_PERF_MIN_RATIO "target_perf_min_ratio"
#define CPERF_STR_TEMPERATURE_MAX  "temperature_max"

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define CPERF_STR_IS_ALLOWED "is_allowed"   /* Is the targeted stress plugin allowed to run? */
#define CPERF_STR_USE_DGEMM            "use_dgemm"
#define CPERF_STR_LWDA_STREAMS_PER_GPU "lwda_streams_per_gpu"
#define CPERF_STR_LWDA_OPS_PER_STREAM  "ops_per_stream_queue"

#define CPERF_STR_MAX_PCIE_REPLAYS "max_pcie_replays" /* Maximum PCIe replays allowed per device
                                                         while the plugin runs. If more replays
                                                         occur than this threshold, this plugin
                                                         will fail */

#define CPERF_STR_MAX_MEMORY_CLOCK "max_memory_clock" /* Maximum memory clock in MHZ to use when locking
                                                         application clocks to max while targeted perf
                                                         runs. */
#define CPERF_STR_MAX_GRAPHICS_CLOCK "max_graphics_clock" /* Maximum graphics clock in MHZ to use when
                                                             locking application clocks to max while
                                                             targeted perf runs */
#define CPERF_STR_SBE_ERROR_THRESHOLD   "max_sbe_errors" // Threshold beyond which sbe's are treated as errors


/******************************************************************************
 * EUD PLUGIN
 *****************************************************************************/
/*#define EUD_PLUGIN_LF_NAME "diagnostic"
#define EUD_PLUGIN_WL_NAME "diagnostic" */

/* Test parameters */
/*#define EUD_PLUGIN_RUN_MODE "run_mode"
#define EUD_PLUGIN_SINGLE_TEST "single_test" */
/* Private test parameters */
/* #define EUD_STR_IS_ALLOWED "is_allowed"   // Is the EUD plugin allowed to run?
#define EUD_LOG_FILE_PATH "eud_log_file_path" */

/******************************************************************************
 * MEMORY PLUGIN
 *****************************************************************************/
#define MEMORY_PLUGIN_LF_NAME "memory"
#define MEMORY_PLUGIN_WL_NAME "memory"

#define MEMORY_STR_IS_ALLOWED "is_allowed" /* Is the memory plugin allowed to run? */

// Parameters controlling the cache subtest
#define MEMORY_SUBTEST_L1TAG              "gpu_memory_cache"
#define MEMORY_L1TAG_STR_IS_ALLOWED       "is_allowed"    /* Is the l1tag subtest allowed to run? */
#define MEMORY_L1TAG_STR_TEST_DURATION    "test_duration"
#define MEMORY_L1TAG_STR_TEST_LOOPS       "test_loops"
#define MEMORY_L1TAG_STR_INNER_ITERATIONS "inner_iterations"
#define MEMORY_L1TAG_STR_ERROR_LOG_LEN    "log_len"
#define MEMORY_L1TAG_STR_DUMP_MISCOMPARES "dump_miscompares"

/******************************************************************************
 * HARDWARE PLUGIN
 *****************************************************************************/
#define HARDWARE_PLUGIN_LF_NAME "hardware"

/******************************************************************************
 * CONSTANT PERF PLUGIN
 *****************************************************************************/
#define SMPERF_PLUGIN_NAME "SM Stress"
#define SMPERF_PLUGIN_WL_NAME "sm stress" //Plugin name on the whitelist (Must be lowercase of CPERF_PLUGIN_NAME)
#define SMPERF_PLUGIN_LF_NAME "sm_stress" //Logfile name for text/json/binary

/* Public parameters - we expect users to change these */
#define SMPERF_STR_TEST_DURATION    "test_duration"
#define SMPERF_STR_TARGET_PERF      "target_stress"
#define SMPERF_STR_TARGET_PERF_MIN_RATIO "target_perf_min_ratio"
#define SMPERF_STR_TEMPERATURE_MAX  "temperature_max"
#define SMPERF_STR_SBE_ERROR_THRESHOLD   "max_sbe_errors" // Threshold beyond which sbe's are treated as errors

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */
#define SMPERF_STR_IS_ALLOWED           "is_allowed" /* Is the sm stress plugin allowed to run? */
#define SMPERF_STR_USE_DGEMM            "use_dgemm"
//#define SM_PERF_STR_LWDA_STREAMS_PER_GPU "lwda_streams_per_gpu"
//#define SM_PERF_STR_LWDA_OPS_PER_STREAM  "ops_per_stream_queue"

#define SMPERF_STR_MAX_MEMORY_CLOCK "max_memory_clock" /* Maximum memory clock in MHZ to
                                                          use when locking application
                                                          clocks to max while smperf runs. */
#define SMPERF_STR_MAX_GRAPHICS_CLOCK "max_graphics_clock" /* Maximum graphics clock in MHZ
                                                          to use when locking application
                                                          clocks to max while smperf runs */

/****************************************************************************
 * GPU BURN PLUGIN
 ***************************************************************************/
#define GPUBURN_PLUGIN_NAME "Diagnostic"
#define GPUBURN_PLUGIN_WL_NAME "diagnostic"
#define GPUBURN_PLUGIN_LF_NAME "diagnostic"

/* Public parameters - we expect users to change these */
#define GPUBURN_STR_SBE_ERROR_THRESHOLD "max_sbe_errors" // Threshold beyond which sbe's are treated as errors
#define GPUBURN_STR_TEST_DURATION   "test_duration"    // Length of the test
#define GPUBURN_STR_USE_DOUBLES     "use_doubles"      // Use doubles instead of floating point
#define GPUBURN_STR_TEMPERATURE_MAX "temperature_max"  // Max temperature allowed during test
#define GPUBURN_STR_IS_ALLOWED          "is_allowed"   // Is this plugin allowed to run?

/****************************************************************************
 * CONTEXT CREATE PLUGIN
 ***************************************************************************/
#define CTXCREATE_PLUGIN_NAME    "Context Create"
#define CTXCREATE_PLUGIN_WL_NAME "context create"
#define CTXCREATE_PLUGIN_LF_NAME "context_create"

/* Private parameters */
#define CTXCREATE_IS_ALLOWED         "is_allowed"       // Is this plugin allowed to run
#define CTXCREATE_IGNORE_EXCLUSIVE   "ignore_exclusive" // Attempt the test even if exlusive mode is set

/****************************************************************************
 * MEMORY BANDWIDTH PLUGIN
 ***************************************************************************/
#define MEMBW_PLUGIN_NAME "Memory Bandwidth"
#define MEMBW_PLUGIN_WL_NAME "memory bandwidth"
#define MEMBW_PLUGIN_LF_NAME "memory_bandwidth"

#define MEMBW_STR_MINIMUM_BANDWIDTH "minimum_bandwidth" // minimum bandwidth in MB / s

#define MEMBW_STR_IS_ALLOWED "is_allowed"    /* Is the memory bandwidth plugin allowed to run? */
#define MEMBW_STR_SBE_ERROR_THRESHOLD   "max_sbe_errors" // Threshold beyond which sbe's are treated as errors

/*****************************************************************************
 * PER PLUGIN ERROR DEFINITIONS AND THEIR BITMASKS
 *****************************************************************************/

/******************************************************************************
 * SM PERF PLUGIN
 *****************************************************************************/
#define SMPERF_ERR_GENERIC                  0x0000100000000000ULL

/******************************************************************************
 * GPUBURN PLUGIN
 *****************************************************************************/

#define GPUBURN_ERR_GENERIC                 0x0001000000000000ULL

/******************************************************************************
 * BUSGRIND PLUGIN
 *****************************************************************************/
#define BG_ERR_PEER_ACCESS_DENIED           0x0000080000000000ULL
#define BG_ERR_LWDA_ALLOC_FAIL              0x0000040000000000ULL
#define BG_ERR_LWDA_GENERAL_FAIL            0x0000020000000000ULL
#define BG_ERR_LWLINK_ERROR                 0x0000010000000000ULL
#define BG_ERR_PCIE_REPLAY_ERROR            0x0000008000000000ULL
#define BG_ERR_LWDA_SYNC_FAIL               0x0000004000000000ULL
#define BG_ERR_BW_TOO_LOW                   0x0000002000000000ULL
#define BG_ERR_LATENCY_TOO_HIGH             0x0000001000000000ULL

/* Old constants that have been consolidated */
#define BG_ERR_LWDA_EVENT_FAIL              BG_ERR_LWDA_GENERAL_FAIL
#define BG_ERR_LWDA_STREAM_FAIL             BG_ERR_LWDA_GENERAL_FAIL

/******************************************************************************
 * CONSTANT POWER PLUGIN
 *****************************************************************************/
#define CP_ERR_BIT_ERROR                    0x0000000100000000ULL
#define CP_ERR_WORKER_BEGIN_FAIL            0x0000000080000000ULL
#define CP_ERR_LWDA_MEMCPY_FAIL             0x0000000040000000ULL
#define CP_ERR_LWDA_ALLOC_FAIL              0x0000000020000000ULL
#define CP_ERR_LWDA_STREAM_FAIL             0x0000000010000000ULL
#define CP_ERR_LWML_FAIL                    0x0000000008000000ULL
#define CP_ERR_LWBLAS_FAIL                  0x0000000004000000ULL
#define CP_ERR_GPU_POWER_TOO_LOW            0x0000000002000000ULL
#define CP_ERR_GPU_POWER_TOO_HIGH           0x0000000001000000ULL
#define CP_ERR_GPU_TEMP_TOO_HIGH            0x0000000000800000ULL
#define CP_ERR_GPU_POWER_VIOLATION_COUNTERS 0x0000000000400000ULL
#define CP_ERR_GPU_THERM_VIOLATION_COUNTERS 0x0000000000200000ULL
#define CP_ERR_GPU_CLOCKS_DROPPED           0x0000000000100000ULL

/******************************************************************************
 * CONSTANT STRESS PLUGIN
 *****************************************************************************/
#define CPERF_ERR_PCIE_REPLAY_ERROR               0x0000000000040000ULL
#define CPERF_ERR_BIT_ERROR                       0x0000000000020000ULL
#define CPERF_ERR_WORKER_BEGIN_FAIL               0x0000000000010000ULL
#define CPERF_ERR_LWDA_MEMCPY_FAIL                0x0000000000008000ULL
#define CPERF_ERR_LWDA_ALLOC_FAIL                 0x0000000000004000ULL
#define CPERF_ERR_LWDA_STREAM_FAIL                0x0000000000002000ULL
#define CPERF_ERR_LWML_FAIL                       0x0000000000001000ULL
#define CPERF_ERR_LWBLAS_FAIL                     0x0000000000000800ULL
#define CPERF_ERR_GPU_PERF_TOO_LOW                0x0000000000000400ULL
#define CPERF_ERR_GPU_TEMP_TOO_HIGH               0x0000000000000200ULL
#define CPERF_ERR_GPU_THERM_VIOLATION_COUNTERS    0x0000000000000100ULL

/******************************************************************************
 * MEMORY PLUGIN
 *****************************************************************************/
#define MEM_ERR_LWDA_ALLOC_FAIL             0x0000000000000020ULL
#define MEM_ERR_DBE_FAIL                    0x0000000000000010ULL

/******************************************************************************
 * SOFTWARE PLUGIN
 *****************************************************************************/
#define SW_ERR_FAIL                         0x0000000000000001ULL

#endif //PLUGINSTRINGS_H
