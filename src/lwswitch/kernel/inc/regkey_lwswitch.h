/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _REGKEY_LWSWITCH_H_
#define _REGKEY_LWSWITCH_H_

#include "g_lwconfig.h"
#include "export_lwswitch.h"

/*
 * LW_SWITCH_REGKEY_TXTRAIN_OPTIMIZATION_ALGORITHM - Select TXTRAIN optimization algorithm
 *
 * LWLink3.0 Allows for multiple optimization algorithms A0-A7
 * Documentation on details about each algorithm can be found in
 * the IAS section "4.4.3.3. Optimization Algorithms"
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL                                     "TxTrainControl"

#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_NOP                                 0x00000000
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_FOM_FORMAT                          2:0
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_FOM_FORMAT_NOP                      0x00000000
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_FOM_FORMAT_FOMA                     0x00000001
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_FOM_FORMAT_FOMB                     0x00000002
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM              10:3
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_NOP          0x00000000
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A0           0x00000001
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A1           0x00000002
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A2           0x00000004
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A3           0x00000008
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A4           0x00000010
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A5           0x00000020
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A6           0x00000040
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_A7           0x00000080
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM                15:11
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_NOP            0x00000000
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_B0             0x00000001
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_B1             0x00000002
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_B2             0x00000004
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_B3             0x00000008
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_MANTISSA         19:16
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_MANTISSA_NOP     0x00000000
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_EXPONENT         23:20
#define LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_EXPONENT_NOP     0x00000000

/*
 * LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT - Toggle external fabric management.
 *
 * Switch driver lwrrently uses lwlink core driver APIs which internally trigger
 * link initialization and training. However, lwlink core driver now exposes a
 * set of APIs for managing lwlink fabric externally (from user mode).
 *
 * When the regkey is enabled, switch driver will skip use of APIs which trigger
 * link initialization and training. In that case, link training needs to be
 * triggered externally.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT    "ExternalFabricMgmt"

#define LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT_DISABLE   0x0
#define LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT_ENABLE    0x1

/*
 * LW_SWITCH_REGKEY_CROSSBAR_DBI - Enable/disable crossbar DBI
 * DBI - Data bus ilwersion provides some small power savings.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_CROSSBAR_DBI           "CrossbarDBI"

#define LW_SWITCH_REGKEY_CROSSBAR_DBI_DISABLE   0x0
#define LW_SWITCH_REGKEY_CROSSBAR_DBI_ENABLE    0x1

/*
 * LW_SWITCH_REGKEY_LINK_DBI - Enable/disable link DBI
 * DBI - Data bus ilwersion provides some small power savings.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_LINK_DBI               "LinkDBI"

#define LW_SWITCH_REGKEY_LINK_DBI_DISABLE       0x0
#define LW_SWITCH_REGKEY_LINK_DBI_ENABLE        0x1

/*
 * LW_SWITCH_REGKEY_AC_COUPLING_MASK
 *
 * Value is a bitmask of which links are AC coupled and should be
 * configured with SETACMODE.
 * All links default to DC coupled.
 *
 * Mask  contains links  0-31
 * Mask2 contains links 32-63
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_AC_COUPLED_MASK     "ACCoupledMask"
#define LW_SWITCH_REGKEY_AC_COUPLED_MASK2    "ACCoupledMask2"

/*
 * LW_SWITCH_REGKEY_SWAP_CLK_OVERRIDE
 *
 * Value is a bitmask applied directly to _SWAP_CLK field.
 * bit 0: select source for RXCLK_0P/N - ports 0-7
 * bit 1: select source for RXCLK_1P/N - ports 16-17
 * bit 2: select source for RXCLK_2P/N - ports 8-15
 * bit 3: unconnected
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_SWAP_CLK_OVERRIDE  "SwapClkOverride"

#define LW_SWITCH_REGKEY_SWAP_CLK_OVERRIDE_FIELD    3:0

/*
 * LW_SWITCH_REGKEY_ENABLE_LINK_MASK - Mask of links to enable
 *
 * By default, all links are enabled
 *
 * [0]=1 - Enable link 0
 *  :
 * [31]=1 - Enable link 31
 *
 * Mask  contains links  0-31
 * Mask2 contains links 32-63
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_ENABLE_LINK_MASK      "LinkEnableMask"
#define LW_SWITCH_REGKEY_ENABLE_LINK_MASK2     "LinkEnableMask2"

/*
 * LW_SWITCH_REGKEY_BANDWIDTH_SHAPER
 *
 * Selects among various transaction fairness modes affecting bandwidth
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_BANDWIDTH_SHAPER   "BandwidthShaper"

#define LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_PROD              0x0
#define LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_XSD               0x1
#define LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_BUCKET_BW         0x2
#define LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_BUCKET_TX_FAIR    0x3

/*
 * LW_SWITCH_REGKEY_SSG_CONTROL
 *
 * Internal use only (supported only on MODS)
 * Allows SSG interface to tweak internal behavior for testing & debugging
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_SSG_CONTROL                            "SSGControl"
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_UPHY_INIT      0:0
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_UPHY_INIT_NO   (0x00000000)
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_UPHY_INIT_YES  (0x00000001)
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_DLPL_INIT      1:1
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_DLPL_INIT_NO   (0x00000000)
#define LW_SWITCH_REGKEY_SSG_CONTROL_BREAK_AFTER_DLPL_INIT_YES  (0x00000001)

/*
 * LW_SWITCH_REGKEY_SKIP_BUFFER_READY
 *
 * Used to optionally skip the initialization of LWLTLC_TX_CTRL_BUFFER_READY,
 * LWLTLC_RX_CTRL_BUFFER_READY, and NPORT_CTRL_BUFFER_READY registers.
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY                      "SkipBufferReady"
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_TLC                  0:0
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_TLC_NO               (0x00000000)
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_TLC_YES              (0x00000001)
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_NPORT                1:1
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_NPORT_NO             (0x00000000)
#define LW_SWITCH_REGKEY_SKIP_BUFFER_READY_NPORT_YES            (0x00000001)

/*
 * LW_SWITCH_REGKEY_SOE_DISABLE - Disables init and usage of SOE by the kernel driver
 *
 * The LWSwitch driver relies on SOE for some features, but can operate
 * without it, with reduced functionality.
 *
 * When the regkey is set to YES, the Lwswitch driver disregards SOE and will not
 * bootstrap it with the driver payload image. All interactions between
 * the driver and SOE are disabled.
 *
 * Driver unload doesn't idle already bootstrapped SOE. Hence it is
 * recommended to reset device in order disable SOE completely. The pre-OS image
 * will still be running even though SOE is disabled through the driver.
 *
 * If set to NO, the SOE will function as normal.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_SOE_DISABLE            "SoeDisable"
#define LW_SWITCH_REGKEY_SOE_DISABLE_NO         0x0
#define LW_SWITCH_REGKEY_SOE_DISABLE_YES        0x1

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
/*
 * LW_SWITCH_REGKEY_SOE_BOOT_CORE - Selects SOE core between falcon and riscv
 *
 * TODO : Remove this regkey when soe-ls10 falcon branch is removed.
 * Tracked in bug 3495816.
 *
 * Public: Available in release drivers
 *
 */
#else
/*
 * LW_SWITCH_REGKEY_SOE_BOOT_CORE - Selects SOE core between falcon and riscv
 *
 * Public: Available in release drivers
 */
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#define LW_SWITCH_REGKEY_SOE_BOOT_CORE          "SoeBootCore"
#define LW_SWITCH_REGKEY_SOE_BOOT_CORE_FALCON   0x0
#define LW_SWITCH_REGKEY_SOE_BOOT_CORE_RISCV    0x1
#define LW_SWITCH_REGKEY_SOE_BOOT_CORE_DEFAULT  0x2

/*
 * LW_SWITCH_REGKEY_ENABLE_PM
 *
 * Used to optionally send the ENABLE_PM command to MINION on link training
 * and DISABLE_PM on link teardown.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_ENABLE_PM                              "EnablePM"
#define LW_SWITCH_REGKEY_ENABLE_PM_NO                           0x0
#define LW_SWITCH_REGKEY_ENABLE_PM_YES                          0x1

/*
 * LW_SWITCH_REGKEY_MINION_SET_UCODE*
 *
 * The following regkeys are used to override MINION image in the driver.
 *
 * The ucode image is overriden from .js file given along the regkey -lwswitch_set_minion_ucode.
 *
 * Private: Internal use only
 */

/*
 * Overrides MINION image data with g_os_ucode_data_lwswitch_minion it fetches from js file.
 */
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_DATA                     "MinionSetUCodeData"

/*
 * Overrides MINION header with g_os_ucode_header_lwswitch_minion it fetches from js file.
 */
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_HDR                      "MinionSetUCodeHdr"

/*
 * Overrides MINION ucode data size with g_os_ucode_data_lwswitch_minion_size it fetches from js file.
 */
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_DATA_SIZE                "MinionSetUCodeDataSize"

/*
 * Overrides MINION ucode data size with g_os_ucode_data_lwswitch_minion_size it fetches from js file.
 */
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_HDR_SIZE                "MinionSetUCodeHdrSize"

/*
 * LW_SWITCH_REGKEY_CHIPLIB_FORCED_LINK_CONFIG_MASK
 *
 * Internal use only
 * This notifies the driver that we are using a chiplib forced link config
 * to initialize and train the links.
 * Mask  contains links  0-31
 * Mask2 contains links 32-63
 *
 * This is intended for sim platforms only where MINION is not available
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_CHIPLIB_FORCED_LINK_CONFIG_MASK             "ChiplibForcedLinkConfigMask"
#define LW_SWITCH_REGKEY_CHIPLIB_FORCED_LINK_CONFIG_MASK2            "ChiplibForcedLinkConfigMask2"

/*
 * Initiates DMA selftest on SOE during init. Default is disable.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_SOE_DMA_SELFTEST                       "SoeDmaSelfTest"
#define LW_SWITCH_REGKEY_SOE_DMA_SELFTEST_DISABLE                0x00
#define LW_SWITCH_REGKEY_SOE_DMA_SELFTEST_ENABLE                 0x01

/*
 * Enables the storing and retrieving of training seed data from minion
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_MINION_CACHE_SEEDS                     "MinionCacheSeeds"
#define LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE              0x00
#define LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_ENABLE               0x01

/*
 * Disables logging of latency counters
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING                 "LatencyCounterLogging"
#define LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING_DISABLE         0x00
#define LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING_ENABLE          0x01

/*
 * Knob to change LWLink link speed
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_SPEED_CONTROL                          "SpeedControl"
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED                    4:0
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_DEFAULT            0x00
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_16G                0x01
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_20G                0x03
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_25G                0x08
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_25_78125G          0x08
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_32G                0x0E
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_40G                0x0F
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_50G                0x10
#define LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_53_12500G          0x11

/*
 * Enable/Disable periodic flush to inforom. Default is disabled.
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING           "InforomBbxPeriodicFlush"
#define LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING_DISABLE   0x00
#define LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING_ENABLE    0x01

/*
 * The rate at which the lifetime data about the LWSwitch is written into the BBX object in seconds.
 * This is gated by LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_PERIODICITY          "InforomBbxWritePeriodicity"
#define LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_PERIODICITY_DEFAULT  600 // 600 seconds (10 min)

/*
 * The minimum duration the driver must run before writing to the BlackBox Recorder (BBX) object
 * on driver exit (in seconds).
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_MIN_DURATION             "InforomBbxWriteMinDuration"
#define LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_MIN_DURATION_DEFAULT     30 // 30 seconds

/*
 * Change ATO timer value
 *
 * Public: Available in release drivers
 */
#define LW_SWITCH_REGKEY_ATO_CONTROL                            "ATOControl"
#define LW_SWITCH_REGKEY_ATO_CONTROL_DEFAULT                    0x0
#define LW_SWITCH_REGKEY_ATO_CONTROL_TIMEOUT                    19:0
#define LW_SWITCH_REGKEY_ATO_CONTROL_TIMEOUT_DEFAULT            0x00
#define LW_SWITCH_REGKEY_ATO_CONTROL_DISABLE                    20:20
#define LW_SWITCH_REGKEY_ATO_CONTROL_DISABLE_FALSE              0x00
#define LW_SWITCH_REGKEY_ATO_CONTROL_DISABLE_TRUE               0x01

/*
 * Change STO timer value
 *
 * Public: Available in release drivers
 */
#define LW_SWITCH_REGKEY_STO_CONTROL                            "STOControl"
#define LW_SWITCH_REGKEY_STO_CONTROL_DEFAULT                    0x0
#define LW_SWITCH_REGKEY_STO_CONTROL_TIMEOUT                    19:0
#define LW_SWITCH_REGKEY_STO_CONTROL_TIMEOUT_DEFAULT            0x00
#define LW_SWITCH_REGKEY_STO_CONTROL_DISABLE                    20:20
#define LW_SWITCH_REGKEY_STO_CONTROL_DISABLE_FALSE              0x00
#define LW_SWITCH_REGKEY_STO_CONTROL_DISABLE_TRUE               0x01

/*
 * LW_SWITCH_REGKEY_MINION_DISABLE - Disables init and usage of MINION by the kernel driver
 *
 * The LWSwitch driver relies on MINION for some features, but can operate
 * without it and is required for Bug 2848340.
 *
 * When the regkey is set to YES, the Lwswitch driver disregards MINION and will not
 * bootstrap it. All interactions between the driver and MINION are disabled.
 *
 * If set to NO, the MINION will function as normal.
 *
 * Private: Internal use only
 */

#define LW_SWITCH_REGKEY_MINION_DISABLE            "MinionDisable"
#define LW_SWITCH_REGKEY_MINION_DISABLE_NO         0x0
#define LW_SWITCH_REGKEY_MINION_DISABLE_YES        0x1

/*
 * LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET - Selects the core on which Minion will run
 *
 * When the regkey is set to FALCON, the Lwswitch driver will run MINION on Falcon core.
 *
 * If set to RISCV, the MINION will run on RISCV core in Non-Manifest Mode.
 * If set to RISCV_MANIFEST, the MINION will run on RISCV core in Manifest Mode.
 *
 * In the default option, RISCV_BCR_CTRL register will be used to get the default core.
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET                "MinionSetUcodeTarget"
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_DEFAULT        0x0
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_FALCON         0x1
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_RISCV          0x2
#define LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_RISCV_MANIFEST 0x3

/*
 * LW_SWITCH_REGKEY_MINION_SET_SIMMODE - Selects simmode settings to send to MINION
 *
 * Regkey is set to either SLOW, MEDIUM or FAST depending on the environment and timing
 * needed by MINION to setup alarms during the training sequence
 *
 * In the default option, no SIMMODE is selected
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_MINION_SET_SIMMODE          "MinionSetSimmode"
#define LW_SWITCH_REGKEY_MINION_SET_SIMMODE_DEFAULT     0x0
#define LW_SWITCH_REGKEY_MINION_SET_SIMMODE_FAST        0x1
#define LW_SWITCH_REGKEY_MINION_SET_SIMMODE_MEDIUM      0x2
#define LW_SWITCH_REGKEY_MINION_SET_SIMMODE_SLOW        0x3

/*
 * LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS - Selects SMF settings to send to MINION
 *
 * Regkey is set to either SLOW, MEDIUM or FAST depending on the environment and timing
 * needed by MINION to setup alarms during the training sequence
 *
 * In the default option, no SMF settings are selected
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS        "MinionSmfSettings"
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_DEFAULT        0x0
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_FAST           0x1
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_MEDIUM         0x2
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_SLOW           0x3
#define LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_MEDIUM_SERIAL  0x4

/*
 * LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES - Selects uphy tables to send to MINION
 *
 * Regkey is set to either SHORT or FAST depending on the environment and timing
 * needed by MINION to setup alarms during the training sequence
 *
 * In the default option, no UPHY table is selected
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES        "MinionSelectUphyTables"
#define LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_DEFAULT     0x0
#define LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_SHORT       0x1
#define LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_FAST        0x2

/*
 * LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS - Programs the L1_RECAL fields
 *
 * Regkey is used to program the the following:
 *
 * MIN_RECAL_TIME_MANTISSA
 * MIN_RECAL_TIME_EXPONENT
 * MAX_RECAL_PERIOD_MANTISSA
 * MAX_RECAL_PERIOD_EXPONENT
 *
 * In the default option, no L1_RECAL fields are programmed
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS                               "LinkRecalSettings"
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP                           0x0
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_MIN_RECAL_TIME_MANTISSA       3:0
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_MIN_RECAL_TIME_EXPONENT       7:4
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_MAX_RECAL_PERIOD_MANTISSA    11:8
#define LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_MAX_RECAL_PERIOD_EXPONENT    15:12

/*
 * Used to disable private internal-use only regkeys from release build drivers
 */

#define LW_SWITCH_REGKEY_PRIVATE                1
#define LW_SWITCH_REGKEY_PUBLIC                 0

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
#define LW_SWITCH_REGKEY_PRIVATE_ALLOWED       1
#else
#define LW_SWITCH_REGKEY_PRIVATE_ALLOWED       0
#endif  //defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
/*
 * LW_SWITCH_REGKEY_LINK_TRAINING_SELECT - Select the Link training to be done
 *
 * For LS10, links can be trained via non-ALI or ALI training. This regkey will
 * allow for overriding System Defaults and can force either training method
 * when desired.
 */
#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT           "LinkTrainingMode"
#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_DEFAULT   0x0
#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_NON_ALI   0x1
#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_ALI       0x2
#else
/*
 * LW_SWITCH_REGKEY_LINK_TRAINING_SELECT - Select the Link training to be done
 *
 * This regkey will
 * allow for overriding System Defaults and can force either training method
 * when desired.
 */

#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT           "LinkTrainingMode"
#define LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_DEFAULT   0x0
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
/*
 * LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL - Enable access to all I2C Ports/Devices
 *
 * Private: Internal use only
 */
#define LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL                "I2cAccessControl"
#define LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_DEFAULT        0x0
#define LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_ENABLE         0x1
#define LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_DISABLE        0x0

/*
 * LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT - Configure the CRC bit error rate for the short interrupt
 * 
 * Public: Available in release drivers
 */
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT                  "CRCBitErrorRateShort"
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_OFF              0x0
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_MAN    2:0
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_EXP    3:3
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_MAN    6:4
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_EXP    11:8

/*
 * LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG - Configure the CRC bit error rate for the long interrupt
 * 
 * Public: Available in release drivers
 */
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG                       "CRCBitErrorRateLong"
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_OFF                   0x000
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_BUG_3365481_CASE_1    0x803
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_BUG_3365481_CASE_2    0x703
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_BUG_3365481_CASE_5    0x34D
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_BUG_3365481_CASE_6    0x00F
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_MAN         2:0
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_EXP         3:3
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_MAN         6:4
#define LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_EXP         12:8

#endif //_REGKEY_LWSWITCH_H_
