/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrlcb33.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW_CONF_COMPUTE control commands and parameters */

#define LW_CONF_COMPUTE_CTRL_CMD(cat,idx)                   LWXXXX_CTRL_CMD(0xCB33, LWCB33_CTRL_##cat, idx)

#define LWCB33_CTRL_RESERVED          (0x00)
#define LWCB33_CTRL_CONF_COMPUTE      (0x01)

/*
 * LW_CONF_COMPUTE_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible return values:
 *   LW_OK
 */
#define LW_CONF_COMPUTE_CTRL_CMD_NULL (0xcb330000) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES
 *   This control call returns overall system and gpu capabilities
 *
 *   Final operating environment depends on a lot of factors:
 *      APM: Ampere Protected Memory
 *      HCC: Hopper Confidential Compute
 *   ---------------------------------------------------------------------------
 *   SrNo   CPU TEE   GPU TEE   GPU Platform  GPU mode    SW Status   System Elw
 *   ---------------------------------------------------------------------------
 *     1    AMD SEV   APM/HCC   Silicon       Production  Production  Production
 *     2    AMD SEV   APM/HCC   Silicon       Production  Development Simulation
 *     3    <Any>     APM/HCC   <Any>         Debug       <Any>       Simulation
 *     4    Non SEV   APM/HCC   <Any>         <Any>       <Any>       Simulation
 *     5    <Any>     APM/HCC   FMOD/EMU/RTL  <Any>       <Any>       Simulation
 *   ---------------------------------------------------------------------------
 *
 *   Prameters:
 *      cpuCapability: [OUT]
 *          This indicates if cpu is capable of AMD SEV
 *      gpusCapability: [OUT]
 *          This indicates if all gpus in the system support APM/HCC.
 *          This field doesn't mean APM/HCC is enabled.
 *      environment: [OUT]
 *          System environment can be production or simulation
 *      ccFeature: [OUT]
 *          Specifies if all gpus in the system have APM/HCC feature enabled
 *          CC feature can be enabled/disabled using this control call:
 *          LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_CC_FEATURE
 *      devMode: [OUT]
 *          Dev mode is used for debugging/profiling
 *          Dev mode is set at system level and implies that all GPUs in the
 *          system have this mode enabled/disabled
 *          Dev mode can be enabled/disabled using this control call:
 *          LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_DEV_MODE
 *
 *      cpuCapability, gpusCapability & environment are determined by the
 *       driver and cannot be modified later on
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES (0xcb330101) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x1" */

#define LW_CONF_COMPUTE_SYSTEM_CPU_CAPABILITY_NONE       0
#define LW_CONF_COMPUTE_SYSTEM_CPU_CAPABILITY_AMD_SEV    1

#define LW_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_NONE      0
#define LW_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_APM       1
#define LW_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_HCC       2

#define LW_CONF_COMPUTE_SYSTEM_ELWIRONMENT_UNAVAILABLE   0
#define LW_CONF_COMPUTE_SYSTEM_ELWIRONMENT_SIM           1
#define LW_CONF_COMPUTE_SYSTEM_ELWIRONMENT_PROD          2

#define LW_CONF_COMPUTE_SYSTEM_FEATURE_DISABLED          0
#define LW_CONF_COMPUTE_SYSTEM_FEATURE_APM_ENABLED       1
#define LW_CONF_COMPUTE_SYSTEM_FEATURE_HCC_ENABLED       2

#define LW_CONF_COMPUTE_SYSTEM_DEV_MODE_DISABLED         0
#define LW_CONF_COMPUTE_SYSTEM_DEV_MODE_ENABLED          1

typedef struct LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES_PARAMS {
    LwU8 cpuCapability;
    LwU8 gpusCapability;
    LwU8 environment;
    LwU8 ccFeature;
    LwU8 devMode;
} LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_CC_FEATURE
 *   This control call can be used to enable/disable APM/HCC feature.
 *   cc feature that gets enabled/disabled depends on system capability,
 *   user cannot choose between APM/HCC.
 *   This control call requires system reboot to take effect.
 *   This is a PRIVILEGED control call and can be set via tools like lwpu-smi
 *
 *      bEnable: [IN]
 *          LW_TRUE:  enable feature on all gpus
 *          LW_FALSE: disable feature on all gpus
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_CC_FEATURE (0xcb330102) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x2" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_CC_FEATURE_PARAMS {
    LwBool bEnable;
} LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_CC_FEATURE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_DEV_MODE
 *   This control call can be used to enable/disable dev mode for all gpus.
 *   This control call requires system reboot to take effect.
 *   This is a PRIVILEGED control call and can be set via tools like lwpu-smi
 *
 *      bEnable: [IN]
 *          LW_TRUE:  enable dev mode on all gpus
 *          LW_FALSE: disable dev mode on all gpus
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_DEV_MODE (0xcb330103) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x3" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_DEV_MODE_PARAMS {
    LwBool bEnable;
} LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_DEV_MODE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_GPUS_STATE
 *   This control call can be used to determine if all GPUs are ready to accept
 *   work form clients.
 *
 *      bAcceptClientRequest: [OUT]
 *          LW_TRUE: all gpus accepting client work requests
 *          LW_FALSE: all gpus blocking client work requests
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_GPUS_STATE (0xcb330104) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x4" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_GPUS_STATE_PARAMS {
    LwBool bAcceptClientRequest;
} LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_GPUS_STATE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_GPUS_STATE
 *   This control call can be used to set gpu state to accept client requests
 *   or to block client requests
 *   This is a PRIVILEGED control call and can be set via admin tools
 *
 *      bAcceptClientRequest:[IN]
 *          LW_TRUE: set all gpus state to accept client work requests
 *          LW_FALSE: set all gpus state to block client work requests
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_GPUS_STATE (0xcb330105) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x5" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_GPUS_STATE_PARAMS {
    LwBool bAcceptClientRequest;
} LW_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_GPUS_STATE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_VIDMEM_SIZE
 *   This control call returns protected and unprotected vidmem size
 *
 *      hSubDevice: [IN]
 *          subdevice handle for the gpu whose vidmem size is requested
 *      protectedMemSizeInKb: [OUT]
 *          total protected memory size in kB
 *      unprotectedMemSizeInKb: [OUT]
 *          total unprotected memory size in kB
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_VIDMEM_SIZE (0xcb330106) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x6" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_VIDMEM_SIZE_PARAMS {
    LwHandle hSubDevice;
    LW_DECLARE_ALIGNED(LwU64 protectedMemSizeInKb, 8);
    LW_DECLARE_ALIGNED(LwU64 unprotectedMemSizeInKb, 8);
} LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_VIDMEM_SIZE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_GPU_SET_VIDMEM_SIZE
 *   This control call updates protected and unprotected vidmem size.
 *   All memory is protected if APM/HCC is enabled. User can override
 *   unprotectedMemSizeInKb and that will adjust protectedMemSizeInKb accordingly.
 *   This is a PRIVILEGED control call and can be set via tools like lwpu-smi.
 *   Vidmem size can be updated after driver load and before any client FB
 *   allocations are made.
 *
 *      hSubDevice: [IN]
 *          subdevice handle for the gpu whose vidmem size is requested
 *      protectedMemSizeInKb: [OUT]
 *          total protected memory size in kB
 *      unprotectedMemSizeInKb: [IN/OUT]
 *          total unprotected memory size in kB
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GPU_SET_VIDMEM_SIZE (0xcb330107) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x7" */

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GPU_SET_VIDMEM_SIZE_PARAMS {
    LwHandle hSubDevice;
    LW_DECLARE_ALIGNED(LwU64 protectedMemSizeInKb, 8);
    LW_DECLARE_ALIGNED(LwU64 unprotectedMemSizeInKb, 8);
} LW_CONF_COMPUTE_CTRL_CMD_GPU_SET_VIDMEM_SIZE_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_GET_NUM_SUPPORTED_CC_SELWRE_CHANNELS
 *   This control call returns the max number of AES capable channels SEC2 and CE support.
 *
 *      hSubDevice: [IN]
 *          subdevice handle for the GPU queried
 *      numSupportedSec2CCSelwreChannels: [OUT]
 *          Max number of AES capable channels SEC2 supports
 *      numSupportedCeCCSelwreChannels: [OUT]
 *          Max number of channels CE supports with encrypt/decrypt
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GET_NUM_SUPPORTED_CC_SELWRE_CHANNELS (0xcb330108) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x8" */

#define LW_CONF_COMPUTE_CTRL_CMD_GET_NUM_SUPPORTED_CC_SELWRE_CHANNELS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GET_NUM_SUPPORTED_CC_SELWRE_CHANNELS_PARAMS {
    LwHandle hSubDevice;
    LwU32    numSupportedSec2CCSelwreChannels;
    LwU32    numSupportedCeCCSelwreChannels;
} LW_CONF_COMPUTE_CTRL_CMD_GET_NUM_SUPPORTED_CC_SELWRE_CHANNELS_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE
 *   This control call returns the GPU session certificate for the specified GPU.
 *   The certificate size is the maximum of the certificate size of APM and CC.
 *
 *      hSubDevice: [IN]
 *          Subdevice handle for the GPU queried
 *      certChain: [OUT]
 *          Certificate chain for the GPU queried
 *      certChainSize: [OUT]
 *          Actual size of certChain data
 *      attestationCertChain: [OUT]
 *          Attestation certificate chain for the GPU queried
 *      attestationCertChainSize: [OUT]
 *          Actual size of attestationCertChain data
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE    (0xcb330109) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0x9" */

#define LW_CONF_COMPUTE_CERT_CHAIN_MAX_SIZE             0x1000
#define LW_CONF_COMPUTE_ATTESTATION_CERT_CHAIN_MAX_SIZE 0x1000

#define LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE_PARAMS {
    LwHandle hSubDevice;
    LwU8     certChain[LW_CONF_COMPUTE_CERT_CHAIN_MAX_SIZE];
    LwU32    certChainSize;
    LwU8     attestationCertChain[LW_CONF_COMPUTE_ATTESTATION_CERT_CHAIN_MAX_SIZE];
    LwU32    attestationCertChainSize;
} LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE_PARAMS;

 /*
 * LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION
 *   This control call returns the GPU attestation report for the specified GPU.
 *   The attestation report size is the maximum of the attestation report size of APM and CC.
 *
 *      hSubDevice: [IN]
 *          Subdevice handle for the GPU queried
 *      nonce: [IN]
 *          spdm supports 32 bytes on nonce
 *      attestationReport: [OUT]
 *          Attestation report of the GPU queried
 *      attestationReportSize: [OUT]
 *          Actual size of the report
 *      isCecAttestationReportPresent : [OUT]
 *          Indicates if the next 2 feilds are valid
 *      cecAttestationReport: [OUT]
 *          Cec attestation report for the gpu queried
 *      cecAttestationReportSize: [OUT]
 *          Actual size of the cec attestation report
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION_REPORT (0xcb33010a) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0xA" */

#define LW_CONF_COMPUTE_GPU_ATTESTATION_REPORT_MAX_SIZE     0x2000
#define LW_CONF_COMPUTE_GPU_CEC_ATTESTATION_REPORT_MAX_SIZE 0x1000
#define LW_CONF_COMPUTE_NONCE_SIZE                          0x20

#define LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION_REPORT_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION_REPORT_PARAMS {
    LwHandle hSubDevice;
    LwU8     nonce[LW_CONF_COMPUTE_NONCE_SIZE];
    LwU8     attestationReport[LW_CONF_COMPUTE_GPU_ATTESTATION_REPORT_MAX_SIZE];
    LwU32    attestationReportSize;
    LwBool   isCecAttestationReportPresent;
    LwU8     cecAttestationReport[LW_CONF_COMPUTE_GPU_CEC_ATTESTATION_REPORT_MAX_SIZE];
    LwU32    cecAttestationReportSize;
} LW_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION_REPORT_PARAMS;

/*
 * LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SELWRE_CHANNELS
 *   This control call returns the max number of Conf Compute capable channels SEC2 and CE support.
 *
 *      hSubDevice: [IN]
 *          subdevice handle for the GPU queried
 *      maxSec2Channels: [OUT]
 *          Max number of conf compute capable channels SEC2 supports
 *      maxCeChannels: [OUT]
 *          Max number of channels CE supports with encrypt/decrypt
 *
 * Possible return values:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_CLIENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SELWRE_CHANNELS (0xcb33010b) /* finn: Evaluated from "(FINN_LW_CONFIDENTIAL_COMPUTE_CONF_COMPUTE_INTERFACE_ID << 8) | 0xB" */

#define LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SELWRE_CHANNELS_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SELWRE_CHANNELS_PARAMS {
    LwHandle hSubDevice;
    LwU32    maxSec2Channels;
    LwU32    maxCeChannels;
} LW_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SELWRE_CHANNELS_PARAMS;

/* _ctrlcb33_h_ */
