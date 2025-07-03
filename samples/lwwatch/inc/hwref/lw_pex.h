/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// lw_pex.h
//
//*****************************************************

#ifndef _LW_PEX_H_
#define _LW_PEX_H_

// PCI Express Cap Space
#define PCIE_CAP_ID    0x10

// PCI Express Extended Cap Space
#define PCIE_AERR_CAP_ID     0x1
#define PCIE_VC_CAP_ID       0x2

#define LW_XVE_DEV_CTRL                                        0x00000004 /* RW-4R */
#define LW_XVE_DEV_CTRL_CMD_IO_SPACE                                   0:0 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_IO_SPACE_DISABLED                   0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_IO_SPACE_ENABLED                    0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE                               1:1 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE_DISABLED               0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE_ENABLED                0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE__PROD                  0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_BUS_MASTER                                 2:2 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_BUS_MASTER_DISABLED                 0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_BUS_MASTER_ENABLED                  0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_BUS_MASTER__PROD                    0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_SPECIAL_CYCLE                              3:3 /* C--VF */
#define LW_XVE_DEV_CTRL_CMD_SPECIAL_CYCLE_DISABLED              0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_CMD_MEM_WRITE_AND_ILWALIDATE                   4:4 /* C--VF */
#define LW_XVE_DEV_CTRL_CMD_MEM_WRITE_AND_ILWALIDATE_DISABLED   0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_CMD_VGA_PALETTE_SNOOP                          5:5 /* C--VF */
#define LW_XVE_DEV_CTRL_CMD_VGA_PALETTE_SNOOP_DISABLED          0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_CMD_PERR                                       6:6 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_PERR_DISABLED                       0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_PERR_ENABLED                        0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_IDSEL_STEP                                 7:7 /* C--VF */
#define LW_XVE_DEV_CTRL_CMD_IDSEL_STEP_DISABLED                 0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_CMD_SERR                                       8:8 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_SERR_DISABLED                       0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_SERR_ENABLED                        0x00000001 /* RW--V */
#define LW_XVE_DEV_CTRL_CMD_FAST_BACK2BACK                             9:9 /* C--VF */
#define LW_XVE_DEV_CTRL_CMD_FAST_BACK2BACK_DISABLED             0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_CMD_INTERRUPT_DISABLE                        10:10 /* RWIVF */
#define LW_XVE_DEV_CTRL_CMD_INTERRUPT_DISABLE_INIT              0x00000000 /* RWI-V */
#define LW_XVE_DEV_CTRL_CMD_INTERRUPT_DISABLE__PROD             0x00000000 /* RW--V */
#define LW_XVE_DEV_CTRL_STAT_INTERRUPT                               19:19 /* C--VF */
#define LW_XVE_DEV_CTRL_STAT_INTERRUPT_NOT_PENDING              0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_STAT_INTERRUPT_PENDING                  0x00000001 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_CAPLIST                                 20:20 /* C--VF */
#define LW_XVE_DEV_CTRL_STAT_CAPLIST_NOT_PRESENT                0x00000000 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_CAPLIST_PRESENT                    0x00000001 /* C---V */
#define LW_XVE_DEV_CTRL_STAT_66MHZ                                   21:21 /* C--VF */
#define LW_XVE_DEV_CTRL_STAT_66MHZ_INCAPABLE                    0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_STAT_66MHZ_CAPABLE                      0x00000001 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_FAST_BACK2BACK                          23:23 /* C--VF */
#define LW_XVE_DEV_CTRL_STAT_FAST_BACK2BACK_INCAPABLE           0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_STAT_FAST_BACK2BACK_CAPABLE             0x00000001 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR                        24:24 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR_ACTIVE            0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR_NOT_ACTIVE        0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR_CLEAR             0x00000001 /* -W--C */
#define LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING                           26:25 /* C--VF */
#define LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_FAST                 0x00000000 /* C---V */
#define LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_MEDIUM               0x00000001 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_SLOW                 0x00000002 /* ----V */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT                   27:27 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT_NO           0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT_YES          0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT_CLEAR        0x00000001 /* -W--C */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT                   28:28 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT_NO           0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT_YES          0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT_CLEAR        0x00000001 /* -W--C */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT                   29:29 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT_NO           0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT_YES          0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT_CLEAR        0x00000001 /* -W--C */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR                           30:30 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR_NOT_ACTIVE           0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR_ACTIVE               0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR_CLEAR                0x00000001 /* -W--C */
#define LW_XVE_DEV_CTRL_STAT_DETECTED_PERR                           31:31 /* RWIVF */
#define LW_XVE_DEV_CTRL_STAT_DETECTED_PERR_NOT_ACTIVE           0x00000000 /* R-I-V */
#define LW_XVE_DEV_CTRL_STAT_DETECTED_PERR_ACTIVE               0x00000001 /* R---V */
#define LW_XVE_DEV_CTRL_STAT_DETECTED_PERR_CLEAR                0x00000001 /* -W--C */


#define LW_XVE_MISC_1                                        0x0000000C /* RW-4R */
#define LW_XVE_MISC_1_CACHE_LINE_SIZE                               7:0 /* RWIVF */
#define LW_XVE_MISC_1_CACHE_LINE_SIZE_INIT                   0x00000000 /* RWI-V */
#define LW_XVE_MISC_1_CACHE_LINE_SIZE__PROD                  0x00000000 /* RW--V */
#define LW_XVE_MISC_1_MASTER_LATENCY_TIMER                        15:11 /* C--VF */
#define LW_XVE_MISC_1_MASTER_LATENCY_TIMER_0_CLOCKS          0x00000000 /* C---V */
#define LW_XVE_MISC_1_HEADER_TYPE                                 23:16 /* C--VF */
#define LW_XVE_MISC_1_HEADER_TYPE_SINGLEFUNC                 0x00000000 /* C---V */
#define LW_XVE_MISC_1_HEADER_TYPE_MULTIFUNC                  0x00000080 /* ----V */


// PCI Express Capability
#define LW_XVE_PCI_EXPRESS_CAPABILITY                               0x00000078 /* R--4R */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_CAPABILITY_ID                   7:0 /* C--VF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_CAPABILITY_ID_INIT       0x00000010 /* C---V */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR            15:8 /* C--VF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR_INIT 0x00000000 /* C---V */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_VERSION                            19:16 /* C--VF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_VERSION_INIT                  0x00000001 /* C---V */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_DEVICE_PORT_TYPE                   23:20 /* C--VF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_DEVICE_PORT_TYPE_INIT         0x00000000 /* C---V */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_SLOT_IMPLEMENTED                   24:24 /* R-IVF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_SLOT_IMPLEMENTED_INIT         0x00000000 /* R-I-V */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_INTERRUPT_MESSAGE_NUMBER           29:25 /* C--VF */
#define LW_XVE_PCI_EXPRESS_CAPABILITY_INTERRUPT_MESSAGE_NUMBER_INIT 0x00000000 /* C---V */

#define LW_XVR_PCI_EXPRESS_CAPABILITY_DEVICE_PORT_TYPE                   23:20 /* C--VF */



#define LW_XVE_DEVICE_CAPABILITY                                        0x0000007C /* R--4R */
#define LW_XVE_DEVICE_CAPABILITY_MAX_PAYLOAD_SIZE                              2:0 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_MAX_PAYLOAD_SIZE_INIT                  0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_PHANTOM_FUNCTIONS_SUPPORTED                   4:3 /* C--VF */
#define LW_XVE_DEVICE_CAPABILITY_PHANTOM_FUNCTIONS_SUPPORTED_INIT       0x00000000 /* C---V */
#define LW_XVE_DEVICE_CAPABILITY_EXTENDED_TAG_FIELD_SIZE                       5:5 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_EXTENDED_TAG_FIELD_SIZE_INIT           0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_ENDPOINT_L0S_ACCEPTABLE_LATENCY               8:6 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_ENDPOINT_L0S_ACCEPTABLE_LATENCY_INIT   0x00000003 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_ENDPOINT_L1_ACCEPTABLE_LATENCY               11:9 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_ENDPOINT_L1_ACCEPTABLE_LATENCY_INIT    0x00000002 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_ATTENTION_BUTTON_PRESENT                    12:12 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_ATTENTION_BUTTON_PRESENT_INIT          0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_ATTENTION_INDICATOR_PRESENT                 13:13 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_ATTENTION_INDICATOR_PRESENT_INIT   0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_POWER_INDICATOR_PRESENT                     14:14 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_POWER_INDICATOR_PRESENT_INIT           0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_CAPTURED_SLOT_POWER_LIMIT_VALUE             25:18 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_CAPTURED_SLOT_POWER_LIMIT_VALUE_INIT   0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CAPABILITY_CAPTURED_SLOT_POWER_LIMIT_SCALE             27:26 /* R-IVF */
#define LW_XVE_DEVICE_CAPABILITY_CAPTURED_SLOT_POWER_LIMIT_SCALE_INIT   0x00000000 /* R-I-V */




#define LW_XVE_DEVICE_CONTROL_STATUS                                        0x00000080 /* RWI4R */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_REPORTING_ENABLE                   0:0 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_REPORTING_ENABLE_INIT       0x00000000 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_REPORTING_ENABLE__PROD      0x00000000 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_REPORTING_ENABLE              1:1 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_REPORTING_ENABLE_INIT  0x00000000 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_REPORTING_ENABLE__PROD 0x00000000 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_REPORTING_ENABLE                  2:2 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_REPORTING_ENABLE_INIT      0x00000000 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_REPORTING_ENABLE__PROD     0x00000000 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQ_REPORTING_ENABLE                   3:3 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQ_REPORTING_ENABLE_INIT       0x00000000 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQ_REPORTING_ENABLE__PROD      0x00000000 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_RELAXED_ORDERING                       4:4 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_RELAXED_ORDERING_INIT           0x00000001 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_RELAXED_ORDERING__PROD          0x00000001 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_MAX_PAYLOAD_SIZE                              7:5 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_MAX_PAYLOAD_SIZE_INIT                  0x00000000 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_EXTENDED_TAG_FIELD_ENABLE                     8:8 /* R-IVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_EXTENDED_TAG_FIELD_ENABLE_INIT         0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_PHANTOM_FUNCTIONS_ENABLE                      9:9 /* R-IVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_PHANTOM_FUNCTIONS_ENABLE_INIT          0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_AUXILLARY_POWER_PM_ENABLE                   10:10 /* R-IVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_AUXILLARY_POWER_PM_ENABLE_INIT         0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_NO_SNOOP                             11:11 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_NO_SNOOP_INIT                   0x00000001 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_NO_SNOOP__PROD                  0x00000001 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_MAX_READ_REQUEST_SIZE                       14:12 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_MAX_READ_REQUEST_SIZE_INIT             0x00000002 /* RWI-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_MAX_READ_REQUEST_SIZE__PROD            0x00000002 /* RW--V */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_DETECTED                         16:16 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_DETECTED_INIT               0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_DETECTED__PROD              0x00000000 /* R---V */
#define LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_DETECTED_CLEAR              0x00000001 /* -W--C */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_DETECTED                    17:17 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_DETECTED_INIT          0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_DETECTED__PROD         0x00000000 /* R---V */
#define LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_DETECTED_CLEAR         0x00000001 /* -W--C */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_DETECTED                        18:18 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_DETECTED_INIT              0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_DETECTED__PROD             0x00000000 /* R---V */
#define LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_DETECTED_CLEAR             0x00000001 /* -W--C */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQUEST_DETECTED                     19:19 /* RWIVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQUEST_DETECTED_INIT           0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQUEST_DETECTED__PROD          0x00000000 /* R---V */
#define LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQUEST_DETECTED_CLEAR      0x00000001 /* -W--C */
#define LW_XVE_DEVICE_CONTROL_STATUS_AUX_POWER_DETECTED                          20:20 /* R-IVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_AUX_POWER_DETECTED_INIT                0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_AUX_POWER_DETECTED__PROD               0x00000000 /* R---V */
#define LW_XVE_DEVICE_CONTROL_STATUS_TRANSACTIONS_PENDING                        21:21 /* R-IVF */
#define LW_XVE_DEVICE_CONTROL_STATUS_TRANSACTIONS_PENDING_INIT              0x00000000 /* R-I-V */
#define LW_XVE_DEVICE_CONTROL_STATUS_TRANSACTIONS_PENDING__PROD             0x00000000 /* R---V */


#define LW_XVE_LINK_CAPABILITIES                                     0x00000084 /* R--4R */
#define LW_XVE_LINK_CAPABILITIES_MAX_LINK_SPEED                             3:0 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_MAX_LINK_SPEED_INIT                 0x00000001 /* R-I-V */
#define LW_XVE_LINK_CAPABILITIES_MAX_LINK_WIDTH                             9:4 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_MAX_LINK_WIDTH_INIT                 0x00000010 /* R-I-V */
#define LW_XVE_LINK_CAPABILITIES_ACTIVE_STATE_LINK_PM_SUPPORT             11:10 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_ACTIVE_STATE_LINK_PM_SUPPORT_INIT   0x00000003 /* R-I-V */
#define LW_XVE_LINK_CAPABILITIES_L0S_EXIT_LATENCY                         14:12 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_L0S_EXIT_LATENCY_INIT               0x00000004 /* R-I-V */
#define LW_XVE_LINK_CAPABILITIES_L1_EXIT_LATENCY                          17:15 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_L1_EXIT_LATENCY_INIT                0x00000002 /* R-I-V */
#define LW_XVE_LINK_CAPABILITIES_PORT_NUMBER                              31:24 /* R-IVF */
#define LW_XVE_LINK_CAPABILITIES_PORT_NUMBER_INIT                    0x00000000 /* R-I-V */


#define LW_XVE_LINK_CONTROL_STATUS                                        0x00000088 /* RWI4R */
#define LW_XVE_LINK_CONTROL_STATUS_ACTIVE_STATE_LINK_PM_CONTROL                  1:0 /* RWIVF */
#define LW_XVE_LINK_CONTROL_STATUS_ACTIVE_STATE_LINK_PM_CONTROL_INIT      0x00000000 /* RWI-V */
#define LW_XVE_LINK_CONTROL_STATUS_ACTIVE_STATE_LINK_PM_CONTROL__PROD     0x00000000 /* RW--V */
#define LW_XVE_LINK_CONTROL_STATUS_READ_COMPLETION_BOUNDARY                      3:3 /* RWIVF */
#define LW_XVE_LINK_CONTROL_STATUS_READ_COMPLETION_BOUNDARY_INIT          0x00000001 /* RWI-V */
#define LW_XVE_LINK_CONTROL_STATUS_READ_COMPLETION_BOUNDARY__PROD         0x00000001 /* RW--V */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_DISABLE                                  4:4 /* C--VF */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_DISABLE_INIT              0x00000000 /* C---V */
#define LW_XVE_LINK_CONTROL_STATUS_RETRAIN_LINK                                  5:5 /* C--VF */
#define LW_XVE_LINK_CONTROL_STATUS_RETRAIN_LINK_INIT                      0x00000000 /* C---V */
#define LW_XVE_LINK_CONTROL_STATUS_COMMON_CLOCK_CONFIGURATION                    6:6 /* RWIVF */
#define LW_XVE_LINK_CONTROL_STATUS_COMMON_CLOCK_CONFIGURATION_INIT        0x00000000 /* RWI-V */
#define LW_XVE_LINK_CONTROL_STATUS_COMMON_CLOCK_CONFIGURATION__PROD       0x00000000 /* RW--V */
#define LW_XVE_LINK_CONTROL_STATUS_EXTENDED_SYNCH                                7:7 /* RWIVF */
#define LW_XVE_LINK_CONTROL_STATUS_EXTENDED_SYNCH_INIT                    0x00000000 /* RWI-V */
#define LW_XVE_LINK_CONTROL_STATUS_EXTENDED_SYNCH__PROD                   0x00000000 /* RW--V */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED                                  19:16 /* R-IVF */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED_INIT                        0x00000001 /* R-I-V */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_SPEED__PROD                       0x00000001 /* R---V */
#define LW_XVE_LINK_CONTROL_STATUS_NEGOTIATED_LINK_WIDTH                       25:20 /* R-IVF */
#define LW_XVE_LINK_CONTROL_STATUS_NEGOTIATED_LINK_WIDTH_INIT             0x00000010 /* R-I-V */
#define LW_XVE_LINK_CONTROL_STATUS_TRAINING_ERROR                              26:26 /* R-IVF */
#define LW_XVE_LINK_CONTROL_STATUS_TRAINING_ERROR_INIT                    0x00000000 /* R-I-V */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_TRAINING                               27:27 /* R-IVF */
#define LW_XVE_LINK_CONTROL_STATUS_LINK_TRAINING_INIT                     0x00000000 /* R-I-V */
#define LW_XVE_LINK_CONTROL_STATUS_SLOT_CLOCK_CONFIGURATON                     28:28 /* R-IVF */
#define LW_XVE_LINK_CONTROL_STATUS_SLOT_CLOCK_CONFIGURATON_INIT       0x00000001 /* R-I-V */
#define LW_XVE_LINK_CONTROL_STATUS_SLOT_CLOCK_CONFIGURATON__PROD      0x00000001 /* R---V */

#define LW_XVE_SLOT_CAPABILITIES                                   0x0000008C /* C--4R */
#define LW_XVE_SLOT_CAPABILITIES_ATTENTION_BUTTON_PRESENT                 0:0 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_ATTENTION_BUTTON_PRESENT_INIT     0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_POWER_CONTROLLER_PRESENT                 1:1 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_POWER_CONTROLLER_PRESENT_INIT     0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_MRL_SENSOR_PRESENT                       2:2 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_MRL_SENSOR_PRESENT_INIT           0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_ATTENTION_INDICATOR_PRESENT              3:3 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_ATTENTION_INDICATOR_PRESENT_INIT  0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_POWER_INDICATOR_PRESENT                  4:4 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_POWER_INDICATOR_PRESENT_INIT      0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_HOT_PLUG_SURPRISE                        5:5 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_HOT_PLUG_SURPRISE_INIT            0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_HOT_PLUG_CAPABLE                         6:6 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_HOT_PLUG_CAPABLE_INIT             0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_SLOT_POWER_LIMIT_VALUE                  14:7 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_SLOT_POWER_LIMIT_VALUE_INIT       0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_SLOT_POWER_LIMIT_SCALE                 16:15 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_SLOT_POWER_LIMIT_SCALE_INIT       0x00000000 /* C---V */
#define LW_XVE_SLOT_CAPABILITIES_PHYSICAL_SLOT_NUMBER                   31:19 /* C--VF */
#define LW_XVE_SLOT_CAPABILITIES_PHYSICAL_SLOT_NUMBER_INIT         0x00000000 /* C---V */

// created by Edu Patrice
#define LW_XVE_ROOT_CONTROL_SERR_CORRECTABLE_ERROR_ENABLE                 0:0 /* C--VF */
#define LW_XVE_ROOT_CONTROL_SERR_CORRECTABLE_ERROR_ENABLE_INIT     0x00000000 /* C---V */
#define LW_XVE_ROOT_CONTROL_SERR_NON_FATAL_ERROR_ENABLE                   1:1 /* C--VF */
#define LW_XVE_ROOT_CONTROL_SERR_NON_FATAL_ERROR_ENABLE_INIT       0x00000000 /* C---V */
#define LW_XVE_ROOT_CONTROL_SERR_FATAL_ERROR_ENABLE                       2:2 /* C--VF */
#define LW_XVE_ROOT_CONTROL_SERR_FATAL_ERROR_ENABLE_INIT                0x00000000 /* C---V */
#define LW_XVE_ROOT_CONTROL_PME_INTERRUPT_ENABLE                          3:3 /* C--VF */
#define LW_XVE_ROOT_CONTROL_PME_INTERRUPT_ENABLE_INIT              0x00000000 /* C---V */




// VC

#define LW_XVE_VCCAP_HDR_NXT_PWR_BUDGET                  0x00000128 /* C---V */
#define LW_XVE_VCCAP_PVC1                                0x00000104 /* R-I4R */
#define LW_XVE_VCCAP_PVC1_EVC                                   2:0 /* R-IVF */
#define LW_XVE_VCCAP_PVC1_EVC_CNT                        0x00000000 /* R-I-V */
#define LW_XVE_VCCAP_PVC1_LPVC                                  6:4 /* R-IVF */
#define LW_XVE_VCCAP_PVC1_LPVC_CNT                       0x00000000 /* R-I-V */
#define LW_XVE_VCCAP_PVC1_REF                                   9:8 /* C--VF */
#define LW_XVE_VCCAP_PVC1_REF_100                        0x00000000 /* C---V */
#define LW_XVE_VCCAP_PVC1_PATS                                11:10 /* C--VF */
#define LW_XVE_VCCAP_PVC1_PATS_0                         0x00000000 /* C---V */


#define LW_XVR_VCCAP_PVC2                                0x00000108 /* C--4R */

#define LW_XVR_VCCAP_PVC2_ARB                                   7:0 /* C--VF */
#define LW_XVR_VCCAP_PVC2_ARB_WRR32                      0x00000000 /* C---V */
#define LW_XVR_VCCAP_PVC2_OFF                                 31:24 /* C--VF */
#define LW_XVR_VCCAP_PVC2_OFF_03                         0x00000000 /* C---V */

#define LW_XVR_VCCAP_PCSR                                0x0000010C /* RW-4R */

#define LW_XVR_VCCAP_PCSR_LOAD                                  0:0 /* RWIVF */
#define LW_XVR_VCCAP_PCSR_LOAD_0                         0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_PCSR_SEL                                   3:1 /* RWIVF */
#define LW_XVR_VCCAP_PCSR_SEL_1                          0x00000001 /* RWI-V */
#define LW_XVR_VCCAP_PCSR_STAT                                16:16 /* R--VF */

#define LW_XVR_VCCAP_VCR0                                0x00000110 /* R--4R */

#define LW_XVR_VCCAP_VCR0_UNUSED0                              14:0 /* C--VF */
#define LW_XVR_VCCAP_VCR0_UNUSED0_0                      0x00000000 /* C---V */
#define LW_XVR_VCCAP_VCR0_REJECT_SNOOP                        15:15 /* R--VF */
#define LW_XVR_VCCAP_VCR0_UNUSED1                             31:16 /* C--VF */
#define LW_XVR_VCCAP_VCR0_UNUSED1_0                      0x00000000 /* C---V */


#define LW_XVR_VCCAP_CTRL0                               0x00000114 /* RW-4R */

#define LW_XVR_VCCAP_CTRL0_TC0                                  0:0 /* C--VF */
#define LW_XVR_VCCAP_CTRL0_TC0_Y                         0x00000001 /* C---V */
#define LW_XVR_VCCAP_CTRL0_MAP                                  7:1 /* RWIVF */
#define LW_XVR_VCCAP_CTRL0_MAP_ALL                       0x0000007F /* RWI-V */
#define LW_XVR_VCCAP_CTRL0_LOAD                               16:16 /* RWIVF */
#define LW_XVR_VCCAP_CTRL0_LOAD_0                        0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_CTRL0_SEL                                19:17 /* RWIVF */
#define LW_XVR_VCCAP_CTRL0_SEL_0                         0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_CTRL0_VCID                               26:24 /* C--VF */
#define LW_XVR_VCCAP_CTRL0_VCID_0                        0x00000000 /* C---V */
#define LW_XVR_VCCAP_CTRL0_VCEN                               31:31 /* C--VF */
#define LW_XVR_VCCAP_CTRL0_VCEN_1                        0x00000001 /* C---V */

#define LW_XVR_VCCAP_STAT0                               0x00000118 /* R--4R */

#define LW_XVR_VCCAP_STAT0_UNUSED_LSB                          15:0 /* C--VF */
#define LW_XVR_VCCAP_STAT0_UNUSED_LSB_0                  0x00000000 /* C---V */
#define LW_XVR_VCCAP_STAT0_PATS                               16:16 /* C--VF */
#define LW_XVR_VCCAP_STAT0_PATS_0                        0x00000000 /* C---V */
#define LW_XVR_VCCAP_STAT0_VNP                                17:17 /* R--VF */
#define LW_XVR_VCCAP_STAT0_UNUSED                             31:18 /* C--VF */
#define LW_XVR_VCCAP_STAT0_UNUSED_0                      0x00000000 /* C---V */

#define LW_XVR_VCCAP_VCR1                                0x0000011C /* R--4R */

#define LW_XVR_VCCAP_VCR1_UNUSED0                              14:0 /* C--VF */
#define LW_XVR_VCCAP_VCR1_UNUSED0_0                      0x00000000 /* C---V */
#define LW_XVR_VCCAP_VCR1_REJECT_SNOOP                        15:15 /* R--VF */
#define LW_XVR_VCCAP_VCR1_UNUSED1                             31:16 /* C--VF */
#define LW_XVR_VCCAP_VCR1_UNUSED1_0                      0x00000000 /* C---V */

#define LW_XVR_VCCAP_CTRL1                               0x00000120 /* RW-4R */

#define LW_XVR_VCCAP_CTRL1_TC0                                  0:0 /* C--VF */
#define LW_XVR_VCCAP_CTRL1_TC0_DISABLE                   0x00000000 /* C---V */
#define LW_XVR_VCCAP_CTRL1_MAP                                  7:1 /* RWIVF */
#define LW_XVR_VCCAP_CTRL1_MAP_NONE                      0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_CTRL1_LOAD                               16:16 /* RWIVF */
#define LW_XVR_VCCAP_CTRL1_LOAD_0                        0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_CTRL1_SEL                                19:17 /* RWIVF */
#define LW_XVR_VCCAP_CTRL1_SEL_0                         0x00000000 /* RWI-V */
#define LW_XVR_VCCAP_CTRL1_VCID                               26:24 /* RWIVF */
#define LW_XVR_VCCAP_CTRL1_VCID_0                        0x00000001 /* RWI-V */
#define LW_XVR_VCCAP_CTRL1_VCEN                               31:31 /* RWIVF */
#define LW_XVR_VCCAP_CTRL1_VCEN_1                        0x00000000 /* RWI-V */

#define LW_XVR_VCCAP_STAT1                               0x00000124 /* R--4R */

#define LW_XVR_VCCAP_STAT1_UNUSED_LSB                          15:0 /* C--VF */
#define LW_XVR_VCCAP_STAT1_UNUSED_LSB_0                  0x00000000 /* C---V */
#define LW_XVR_VCCAP_STAT1_PATS                               16:16 /* C--VF */
#define LW_XVR_VCCAP_STAT1_PATS_0                        0x00000000 /* C---V */
#define LW_XVR_VCCAP_STAT1_VNP                                17:17 /* R--VF */
#define LW_XVR_VCCAP_STAT1_UNUSED                             31:18 /* C--VF */
#define LW_XVR_VCCAP_STAT1_UNUSED_0                      0x00000000 /* C---V */


// AERR
#define LW_XVR_ERPTCAP_UCERR                             0x00000164 /* RW-4R */

#define LW_XVR_ERPTCAP_UCERR_TRAINING_ERR                       0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_TRAINING_ERR_FALSE                 0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_TRAINING_ERR_TRUE                  0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_TRAINING_ERR_CLEAR                 0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_DLINK_PROTO_ERR                    4:4 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_DLINK_PROTO_ERR_FALSE              0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_DLINK_PROTO_ERR_TRUE               0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_DLINK_PROTO_ERR_CLEAR              0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_POS_TLP                          12:12 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_POS_TLP_FALSE                      0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_POS_TLP_TRUE                       0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_POS_TLP_CLEAR                      0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_FC_PROTO_ERR                     13:13 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_FC_PROTO_ERR_DEFAULT               0x0 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_COMP_TO                          14:14 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_COMP_TO_FALSE                      0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_COMP_TO_TRUE                       0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_COMP_TO_CLEAR                      0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_COMP_ABORT                       15:15 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_COMP_ABORT_FALSE                   0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_COMP_ABORT_TRUE                    0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_COMP_ABORT_CLEAR                   0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_UNEXP_COMP                       16:16 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_UNEXP_COMP_FALSE                   0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_UNEXP_COMP_TRUE                    0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_UNEXP_COMP_CLEAR                   0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_RCV_OVFL                         17:17 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_RCV_OVFL_DEFAULT                   0x0 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_MF_TLP                           18:18 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MF_TLP_FALSE                       0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_MF_TLP_TRUE                        0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_MF_TLP_CLEAR                       0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_ECRC_ERR                         19:19 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_ECRC_ERR_FALSE                     0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_ECRC_ERR_TRUE                      0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_ECRC_ERR_CLEAR                     0x1 /* -W--C */

#define LW_XVR_ERPTCAP_UCERR_UNSUP_REQ_ERR                    20:20 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_UNSUP_REQ_ERR_FALSE                0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_UCERR_UNSUP_REQ_ERR_TRUE                 0x1 /* R---V */
#define LW_XVR_ERPTCAP_UCERR_UNSUP_REQ_ERR_CLEAR                0x1 /* -W--C */


//===============================================================================
//
//.TITLE
//Uncorrectable Error Mask Register
//-----------------------------------
#define LW_XVR_ERPTCAP_UCERR_MK                          0x00000168 /* RW-4R */

#define LW_XVR_ERPTCAP_UCERR_MK_TRAINING_ERR                    0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_TRAINING_ERR_NOT_MASKED         0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_TRAINING_ERR_MASKED             0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_DLINK_PROTO_ERR                 4:4 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_DLINK_PROTO_ERR_NOT_MASKED      0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_DLINK_PROTO_ERR_MASKED          0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_POS_TLP                       12:12 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_POS_TLP_NOT_MASKED              0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_POS_TLP_MASKED                  0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_FC_PROTO_ERR                  13:13 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_MK_FC_PROTO_ERR_DEFAULT            0x0 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_MK_COMP_TO                       14:14 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_COMP_TO_NOT_MASKED              0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_COMP_TO_MASKED                  0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_COMP_ABORT                    15:15 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_COMP_ABORT_NOT_MASKED           0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_COMP_ABORT_MASKED               0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_UNEXP_COMP                    16:16 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_UNEXP_COMP_NOT_MASKED           0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_UNEXP_COMP_MASKED               0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_RCV_OVFL                      17:17 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_MK_RCV_OVFL_DEFAULT                0x0 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_MK_MF_TLP                        18:18 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_MF_TLP_NOT_MASKED               0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_MF_TLP_MASKED                   0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_ECRC_ERR                      19:19 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_ECRC_ERR_NOT_MASKED             0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_ECRC_ERR_MASKED                 0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_MK_UNSUP_REQ_ERR                 20:20 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_MK_UNSUP_REQ_ERR_NOT_MASKED        0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_MK_UNSUP_REQ_ERR_MASKED            0x1 /* RW--V */

//===============================================================================
//.TITLE
//Uncorrectable Error Severity Register
//-------------------------------------

#define LW_XVR_ERPTCAP_UCERR_SEVR                        0x0000016C /* RW-4R */

#define LW_XVR_ERPTCAP_UCERR_SEVR_TRAINING_ERR                  0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_TRAINING_ERR_NON_FATAL        0x0 /* RW--V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_TRAINING_ERR_FATAL            0x1 /* RWI-V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_DLINK_PROTO_ERR               4:4 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_DLINK_PROTO_ERR_NON_FATAL     0x0 /* RW--V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_DLINK_PROTO_ERR_FATAL         0x1 /* RWI-V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_POS_TLP                     12:12 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_POS_TLP_NON_FATAL             0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_POS_TLP_FATAL                 0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_FC_PROTO_ERR                13:13 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_FC_PROTO_ERR_NON_FATAL        0x0 /* ----V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_FC_PROTO_ERR_FATAL            0x1 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_TO                     14:14 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_TO_NON_FATAL             0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_TO_FATAL                 0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_ABORT                  15:15 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_ABORT_NON_FATAL          0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_COMP_ABORT_FATAL              0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_UNEXP_COMP                  16:16 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_UNEXP_COMP_NON_FATAL          0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_UNEXP_COMP_FATAL              0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_RCV_OVFL                    17:17 /* C--VF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_RCV_OVFL_NON_FATAL            0x0 /* ----V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_RCV_OVFL_FATAL                0x1 /* C---V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_MF_TLP                      18:18 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_MF_TLP_NON_FATAL              0x0 /* RW--V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_MF_TLP_FATAL                  0x1 /* RWI-V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_ECRC_ERR                    19:19 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_ECRC_ERR_NON_FATAL            0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_ECRC_ERR_FATAL                0x1 /* RW--V */

#define LW_XVR_ERPTCAP_UCERR_SEVR_UNSUP_REQ_ERR               20:20 /* RWCVF */
#define LW_XVR_ERPTCAP_UCERR_SEVR_UNSUP_REQ_ERR_NON_FATAL       0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_UCERR_SEVR_UNSUP_REQ_ERR_FATAL           0x1 /* RW--V */

//===============================================================================

//-----------------------------------------

#define LW_XVR_ERPTCAP_CERR                              0x00000170 /* RW-4R */

#define LW_XVR_ERPTCAP_CERR_RCV_ERR                             0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_RCV_ERR_FALSE                       0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_CERR_RCV_ERR_TRUE                        0x1 /* R---V */
#define LW_XVR_ERPTCAP_CERR_RCV_ERR_CLEAR                       0x1 /* -W--C */

#define LW_XVR_ERPTCAP_CERR_BAD_TLP                             6:6 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_BAD_TLP_FALSE                       0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_CERR_BAD_TLP_TRUE                        0x1 /* R---V */
#define LW_XVR_ERPTCAP_CERR_BAD_TLP_CLEAR                       0x1 /* -W--C */

#define LW_XVR_ERPTCAP_CERR_BAD_DLLP                            7:7 /* RWIVF */
#define LW_XVR_ERPTCAP_CERR_BAD_DLLP_FALSE                      0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_CERR_BAD_DLLP_TRUE                       0x1 /* R---V */
#define LW_XVR_ERPTCAP_CERR_BAD_DLLP_CLEAR                      0x1 /* -W--C */

#define LW_XVR_ERPTCAP_CERR_RPLY_RLOV                           8:8 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_RPLY_RLOV_FALSE                     0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_CERR_RPLY_RLOV_TRUE                      0x1 /* R---V */
#define LW_XVR_ERPTCAP_CERR_RPLY_RLOV_CLEAR                     0x1 /* -W--C */

#define LW_XVR_ERPTCAP_CERR_RPLY_TO                           12:12 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_RPLY_TO_FALSE                       0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_CERR_RPLY_TO_TRUE                        0x1 /* R---V */
#define LW_XVR_ERPTCAP_CERR_RPLY_TO_CLEAR                       0x1 /* -W--C */

//===============================================================================
//===============================================================================
//.TITLE
//Correctable Error Mask Register
//---------------------------------

#define LW_XVR_ERPTCAP_CERR_MK                           0x00000174 /* RW-4R */

#define LW_XVR_ERPTCAP_CERR_MK_RCV_ERR                          0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_MK_RCV_ERR_NOT_MASKED               0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_CERR_MK_RCV_ERR_MASKED                   0x1 /* RW--V */

#define LW_XVR_ERPTCAP_CERR_MK_BAD_TLP                          6:6 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_MK_BAD_TLP_NOT_MASKED               0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_CERR_MK_BAD_TLP_MASKED                   0x1 /* RW--V */

#define LW_XVR_ERPTCAP_CERR_MK_BAD_DLLP                         7:7 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_MK_BAD_DLLP_NOT_MASKED              0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_CERR_MK_BAD_DLLP_MASKED                  0x1 /* RW--V */

#define LW_XVR_ERPTCAP_CERR_MK_RPLY_RLOV                        8:8 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_MK_RPLY_RLOV_NOT_MASKED             0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_CERR_MK_RPLY_RLOV_MASKED                 0x1 /* RW--V */

#define LW_XVR_ERPTCAP_CERR_MK_RPLY_TO                        12:12 /* RWCVF */
#define LW_XVR_ERPTCAP_CERR_MK_RPLY_TO_NOT_MASKED               0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_CERR_MK_RPLY_TO_MASKED                   0x1 /* RW--V */
//===============================================================================
//.TITLE
//Advanced Error Capablities and Control Register
//-----------------------------------------------

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL                  0x00000178 /* R--4R */

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ERR_PTR                 4:0 /* R--VF */

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_CAP            5:5 /* C--VF */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_CAP_TRUE       0x1 /* C---V */

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_EN             6:6 /* RWCVF */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_EN_FALSE       0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_EN_TRUE        0x1 /* RW--V */

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_CAP            7:7 /* C--VF */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_CAP_TRUE       0x1 /* C---V */

#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_EN             8:8 /* RWCVF */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_EN_FALSE       0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_EN_TRUE        0x1 /* RW--V */

//===============================================================================
//===============================================================================
//.TITLE
//Header Log Register
//-------------------
/*
This register is 16 bytes.  The header is captured such that the fields of the
header read by software in the same way the headers are presented in the spec,
when the register is read using dword accesses.  Therefore, byte 0 of the
header is located in byte 3 of the Header Log register, byte 1 of the header
is in the byte 2 of the header Log register and so forth.
*/
/*
 31           24 23           16 15            8 7             0
.-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-.
|    Byte 0     |    Byte 1     |    Byte 2     |    Byte 3     |
`-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-'
|    Byte 4     |    Byte 5     |    Byte 6     |    Byte 7     |
`-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-'
|    Byte 8     |    Byte 9     |    Byte 10    |    Byte 11    |
`-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-'
|    Byte 12    |    Byte 13    |    Byte 14    |    Byte 15    |
`-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-'
*/

#define LW_XVR_ERPTCAP_HDR_LOG_DW0                       0x0000017C /* R--4R */
#define LW_XVR_ERPTCAP_HDR_LOG_DW0_0                           31:0 /* R--VF */

#define LW_XVR_ERPTCAP_HDR_LOG_DW1                       0x00000180 /* R--4R */
#define LW_XVR_ERPTCAP_HDR_LOG_DW1_1                           31:0 /* R--VF */

#define LW_XVR_ERPTCAP_HDR_LOG_DW2                       0x00000184 /* R--4R */
#define LW_XVR_ERPTCAP_HDR_LOG_DW2_2                           31:0 /* R--VF */

#define LW_XVR_ERPTCAP_HDR_LOG_DW3                       0x00000188 /* R--4R */
#define LW_XVR_ERPTCAP_HDR_LOG_DW3_3                           31:0 /* R--VF */
//===============================================================================
/*
.TITLE

Root Error Command Register
---------------------------

When set the bit enables the generation of an interrupt when the corresponding
error is reported by any of the devices in the hierarchy associated with this
Root Port.
*/

#define LW_XVR_ERPTCAP_ERR_CMD                           0x0000018C /* RW-4R */

#define LW_XVR_ERPTCAP_ERR_CMD_COR_ERR_RPT_EN                   0:0 /* RWIVF */
#define LW_XVR_ERPTCAP_ERR_CMD_COR_ERR_RPT_EN_FALSE             0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_ERR_CMD_COR_ERR_RPT_EN_TRUE              0x1 /* RW--V */

#define LW_XVR_ERPTCAP_ERR_CMD_NONFATAL_ERR_RPT_EN              1:1 /* RWIVF */
#define LW_XVR_ERPTCAP_ERR_CMD_NONFATAL_ERR_RPT_EN_FALSE        0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_ERR_CMD_NONFATAL_ERR_RPT_EN_TRUE         0x1 /* RW--V */

#define LW_XVR_ERPTCAP_ERR_CMD_FATAL_ERR_RPT_EN                 2:2 /* RWIVF */
#define LW_XVR_ERPTCAP_ERR_CMD_FATAL_ERR_RPT_EN_FALSE           0x0 /* RWI-V */
#define LW_XVR_ERPTCAP_ERR_CMD_FATAL_ERR_RPT_EN_TRUE            0x1 /* RW--V */

//===============================================================================
//-------------------------------------------------------------------------------

#define LW_XVR_ERPTCAP_ERR_STS                           0x00000190 /* RW-4R */

#define LW_XVR_ERPTCAP_ERR_STS_COR_RCVD                         0:0 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_COR_RCVD_FALSE                   0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_COR_RCVD_TRUE                    0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_COR_RCVD_CLEAR                   0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_MULT_COR_RCVD                    1:1 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_COR_RCVD_FALSE              0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_COR_RCVD_TRUE               0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_COR_RCVD_CLEAR              0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_UNCOR_RCVD                       2:2 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_UNCOR_RCVD_FALSE                 0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_UNCOR_RCVD_TRUE                  0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_UNCOR_RCVD_CLEAR                 0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_MULT_UNCOR_RCVD                  3:3 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_UNCOR_RCVD_FALSE            0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_UNCOR_RCVD_TRUE             0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_MULT_UNCOR_RCVD_CLEAR            0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_FIRST_FATAL_RCVD                 4:4 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_FIRST_FATAL_RCVD_FALSE           0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_FIRST_FATAL_RCVD_TRUE            0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_FIRST_FATAL_RCVD_CLEAR           0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_NONFATAL_RCVD                    5:5 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_NONFATAL_RCVD_FALSE              0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_NONFATAL_RCVD_TRUE               0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_NONFATAL_RCVD_CLEAR              0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_FATAL_RCVD                       6:6 /* RWCVF */
#define LW_XVR_ERPTCAP_ERR_STS_FATAL_RCVD_FALSE                 0x0 /* R-I-V */
#define LW_XVR_ERPTCAP_ERR_STS_FATAL_RCVD_TRUE                  0x1 /* R---V */
#define LW_XVR_ERPTCAP_ERR_STS_FATAL_RCVD_CLEAR                 0x1 /* -W--C */

#define LW_XVR_ERPTCAP_ERR_STS_ADV_ERR_INTR_MSG_NUM           31:27 /* C--VF */
#define LW_XVR_ERPTCAP_ERR_STS_ADV_ERR_INTR_MSG_NUM_ONE         0x1 /* C---V */
//===============================================================================

/*Error Source Identification Register
------------------------------------

This register identifies the source (Requestor ID) of first correctable and
uncorrectable (non-fatal/fatal) errors reported in the Root Error Status
register.
*/
#define LW_XVR_ERPTCAP_ERR_ID                            0x00000194 /* R--4R */

#define LW_XVR_ERPTCAP_ERR_ID_ERR_COR                          15:0 /* R--VF */
#define LW_XVR_ERPTCAP_ERR_ID_ERR_COR_DEFAULT                0x0000 /* R---V */

#define LW_XVR_ERPTCAP_ERR_ID_ERR_UNCOR                       31:16 /* R--VF */
#define LW_XVR_ERPTCAP_ERR_ID_ERR_UNCOR_DEFAULT              0x0000 /* R---V */

//===============================================================================

#endif // _LW_PEX_H_

