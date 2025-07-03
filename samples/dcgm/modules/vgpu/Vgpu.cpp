/*
 * File:   Vgpu.cpp
 */

#include <sstream>
#include <iostream>
#include <string.h>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "Vgpu.h"
#include "CommandOutputController.h"

using namespace std;

const etblDCGMEngineInternal *t_pEtblDcgm = NULL;

/**************************************************************************/

/* Get vGPU Config */
char VGPU_DISPLAY[] =
        "+-------------------------------+------------------------+------------------------------------------------------+\n"
        "| <HEADER1                                                                                                   >  |\n"
        "| <HEADER2                    > | TARGET CONFIG          | CURRENT CONFIG                                       |\n"
        "+===============================+========================+======================================================+\n"
        "| ECC Mode                      |  <TRG_ECC           >  |  <LWR_ECC                                         >  |\n"
        "| Power Limit                   |  <TRG_PWR           >  |  <LWR_PWR                                         >  |\n"
        "| Compute Mode                  |  <TRG_COMP_MODE     >  |  <LWR_COMP_MODE                                   >  |\n"
        "+-------------------------------+-------------------------------------------------------------------------------+\n"
        "|                               |                                                                               |\n"
        "| Attribute                     | Value                                                                         |\n"
        "+===============================+===============================================================================+\n"
        "| Device Name                   |  <ATTRIB_DEV_NAME                                                          >  |\n"
        "| Brand Name                    |  <ATTRIB_BRAND_NAME                                                        >  |\n"
        "| Driver Version                |  <ATTRIB_DRV_VRS                                                           >  |\n"
        "| UUID                          |  <ATTRIB_UUID                                                              >  |\n"
        "| Serial                        |  <ATTRIB_SERIAL                                                            >  |\n"
        "| VBIOS Version                 |  <ATTRIB_VBIOS_VRS                                                         >  |\n"
        "| INFOROM Image Version         |  <ATTRIB_IFRM_IMG_VRS                                                      >  |\n"
        "| Operating Mode                |  <ATTRIB_OPR_MODE                                                          >  |\n"
        "| BAR1 Total                    |  <ATTRIB_BAR1                                                              >  |\n"
        "| FB Total                      |  <ATTRIB_FB                                                                >  |\n"
        "| FB Free                       |  <ATTRIB_FB_FREE                                                           >  |\n"
        "| FB Used                       |  <ATTRIB_FB_USED                                                           >  |\n"
        "| PCI Bus ID                    |  <ATTRIB_PCI_BUS_ID                                                        >  |\n"
        "| PCI Device ID                 |  <ATTRIB_PCI_DEV_ID                                                        >  |\n"
        "| PCI SubSystem ID              |  <ATTRIB_PCI_SUBSYS_ID                                                     >  |\n"
        "| Supported vGPUs Type Count    |  <ATTRIB_SUPPORTED_VGPU_TYPE_COUNT                                         >  |\n"
        "| Creatable vGPUs Type Count    |  <ATTRIB_CRT_VGPU_TYPE_COUNT                                               >  |\n"
        "| Active vGPUs Instance Count   |  <ATTRIB_ACT_VGPU_INSTANCE_COUNT                                           >  |\n"
        "| GPU Utilization               |  <ATTRIB_GPU_UTIL                                                          >  |\n"
        "| Memory Utilization            |  <ATTRIB_MEM_UTIL                                                          >  |\n"
        "| Encoder Utilization           |  <ATTRIB_ENC_UTIL                                                          >  |\n"
        "| Decoder Utilization           |  <ATTRIB_DEC_UTIL                                                          >  |\n"
        "+-------------------------------+-------------------------------------------------------------------------------+\n";

char VGPU_CRT_TYPE_ID_HEADER[] =
        "|Creatable vGPU Type Ids        |  <ATTRIB_CRT_VGPU_TYPE_IDS                                                 >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_CRT_TYPE_ID_LIST[] =
        "|                               |  <ATTRIB_CRT_VGPU_TYPE_ID_LIST                                             >  |\n";

char VGPU_CRT_TYPE_ID_FOOTER[] =
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_ACT_INSTANCE_ID_HEADER[] =
        "|Active vGPU Instance Ids       |  <ATTRIB_ACT_VGPU_INSTANCE_IDS                                             >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_ACT_INSTANCE_ID_LIST[] =
        "|                               |  <ATTRIB_ACT_VGPU_INSTANCE_ID_LIST                                         >  |\n";

char VGPU_ACT_INSTANCE_ID_FOOTER[] =
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_ACT_INSTANCE_METRICS_HEADER[] =
        "|Active vGPU Instance Metrics   |  <ATTRIB_ACT_VGPU_INSTANCE_METRICS                                         >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_ACT_INSTANCE_METRICS_LIST[] =
        "|vGPU Instance VM ID            |  <ATTRIB_VGPU_INSTANCE_VM_ID                                               >  |\n"
        "|vGPU Instance VM Name          |  <ATTRIB_VGPU_INSTANCE_VM_NAME                                             >  |\n"
        "|vGPU Instance Type ID          |  <ATTRIB_VGPU_INSTANCE_TYPE_ID                                             >  |\n"
        "|vGPU Instance UUID             |  <ATTRIB_VGPU_INSTANCE_UUID                                                >  |\n"
        "|vGPU Instance Driver Version   |  <ATTRIB_VGPU_INSTANCE_DRIVER_VERSION                                      >  |\n"
        "|vGPU Instance FB Usage         |  <ATTRIB_VGPU_INSTANCE_FB_USAGE                                            >  |\n"
        "|vGPU Instance License Status   |  <ATTRIB_VGPU_INSTANCE_LICENSE_STATUS                                      >  |\n"
        "|vGPU Instance Frame Rate Limit |  <ATTRIB_VGPU_INSTANCE_FRAME_RATE_LIMIT                                    >  |\n"
        "|GPU Utilization For vGPU       |  <ATTRIB_VGPU_SM_UTIL                                                      >  |\n"
        "|Memory Utilization For vGPU    |  <ATTRIB_VGPU_MEMORY_UTIL                                                  >  |\n"
        "|Encoder Utilization For vGPU   |  <ATTRIB_VGPU_ENCODER_UTIL                                                 >  |\n"
        "|Decoder Utilization For vGPU   |  <ATTRIB_VGPU_DECODER_UTIL                                                 >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_ACT_INSTANCE_METRICS_FOOTER[] =
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_SUPPORTED_TYPE_INFO_HEADER[] =
        "|Supported vGPUs Info           |  <ATTRIB_SUPPORTED_TYPE_INFO                                               >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_SUPPORTED_TYPE_INFO_LIST[] =
        "|vGPU Type ID                   |  <ATTRIB_SUPPORTED_TYPE_INFO_TYPE_ID                                       >  |\n"
        "|Name                           |  <ATTRIB_SUPPORTED_TYPE_INFO_NAME                                          >  |\n"
        "|Class                          |  <ATTRIB_SUPPORTED_TYPE_INFO_CLASS                                         >  |\n"
        "|License                        |  <ATTRIB_SUPPORTED_TYPE_INFO_LICENSE                                       >  |\n"
        "|Device ID                      |  <ATTRIB_SUPPORTED_TYPE_INFO_DEV_ID                                        >  |\n"
        "|SubSystem ID                   |  <ATTRIB_SUPPORTED_TYPE_INFO_SUBSYS_ID                                     >  |\n"
        "|Display Heads                  |  <ATTRIB_SUPPORTED_TYPE_INFO_NUM_HEADS                                     >  |\n"
        "|Max Instances                  |  <ATTRIB_SUPPORTED_TYPE_INFO_MAX_INSTANCES                                 >  |\n"
        "|Frame Rate Limit               |  <ATTRIB_SUPPORTED_TYPE_INFO_FRL                                           >  |\n"
        "|Max Resolution in X            |  <ATTRIB_SUPPORTED_TYPE_INFO_XDIM                                          >  |\n"
        "|Max Resolution in Y            |  <ATTRIB_SUPPORTED_TYPE_INFO_YDIM                                          >  |\n"
        "|Fb Total                       |  <ATTRIB_SUPPORTED_TYPE_INFO_FB_TOTAL                                      >  |\n"
        "+---------------------------------------------------------------------------------------------------------------+\n";

char VGPU_SUPPORTED_TYPE_INFO_FOOTER[] =
        "+===============================================================================================================+\n";

#define HEADER_TOP_TAG "<HEADER1"
#define HEADER_BOTTOM_TAG "<HEADER2"

#define VGPU_CONFIG_LWRR_ECC_MODE_TAG "<LWR_ECC"
#define VGPU_CONFIG_LWRR_PWR_LIM_TAG "<LWR_PWR"
#define VGPU_CONFIG_LWRR_COMPUTE_MODE_TAG "<LWR_COMP_MODE"
#define VGPU_CONFIG_TRG_ECC_MODE_TAG "<TRG_ECC"
#define VGPU_CONFIG_TRG_PWR_LIM_TAG "<TRG_PWR"
#define VGPU_CONFIG_TRG_COMPUTE_MODE_TAG "<TRG_COMP_MODE"
#define VGPU_ATTRIB_BAR1_TOTAL_TAG "<ATTRIB_BAR1"
#define VGPU_ATTRIB_DEV_NAME_TAG "<ATTRIB_DEV_NAME"
#define VGPU_ATTRIB_BRAND_NAME_TAG "<ATTRIB_BRAND_NAME"
#define VGPU_ATTRIB_DRV_VRS_TAG "<ATTRIB_DRV_VRS"
#define VGPU_ATTRIB_UUID_TAG "<ATTRIB_UUID"
#define VGPU_ATTRIB_SERIAL_TAG "<ATTRIB_SERIAL"
#define VGPU_ATTRIB_VBIOS_VRS_TAG "<ATTRIB_VBIOS_VRS"
#define VGPU_ATTRIB_IFRM_IMG_VRS_TAG "<ATTRIB_IFRM_IMG_VRS"
#define VGPU_ATTRIB_OPR_MODE_TAG "<ATTRIB_OPR_MODE"
#define VGPU_ATTRIB_FB_TOTAL_TAG "<ATTRIB_FB"
#define VGPU_ATTRIB_FB_FREE_TAG "<ATTRIB_FB_FREE"
#define VGPU_ATTRIB_FB_USED_TAG "<ATTRIB_FB_USED"
#define VGPU_ATTRIB_PCI_BUS_ID_TAG "<ATTRIB_PCI_BUS_ID"
#define VGPU_ATTRIB_PCI_DEV_ID_TAG "<ATTRIB_PCI_DEV_ID"
#define VGPU_ATTRIB_PCI_SUBSYS_ID_TAG "<ATTRIB_PCI_SUBSYS_ID"
#define VGPU_ATTRIB_SUPPORTED_VGPU_TYPE_COUNT_TAG "<ATTRIB_SUPPORTED_VGPU_TYPE_COUNT"
#define VGPU_ATTRIB_CRT_VGPU_TYPE_COUNT_TAG "<ATTRIB_CRT_VGPU_TYPE_COUNT"
#define VGPU_ATTRIB_CRT_VGPU_TYPE_IDS_TAG "<ATTRIB_CRT_VGPU_TYPE_IDS"
#define VGPU_ATTRIB_CRT_VGPU_TYPE_ID_LIST_TAG "<ATTRIB_CRT_VGPU_TYPE_ID_LIST"
#define VGPU_ATTRIB_ACT_VGPU_INSTANCE_COUNT_TAG "<ATTRIB_ACT_VGPU_INSTANCE_COUNT"
#define VGPU_ATTRIB_GPU_UTIL_TAG "<ATTRIB_GPU_UTIL"
#define VGPU_ATTRIB_MEM_UTIL_TAG "<ATTRIB_MEM_UTIL"
#define VGPU_ATTRIB_ENC_UTIL_TAG "<ATTRIB_ENC_UTIL"
#define VGPU_ATTRIB_DEC_UTIL_TAG "<ATTRIB_DEC_UTIL"
#define VGPU_ATTRIB_ACT_VGPU_INSTANCE_IDS_TAG "<ATTRIB_ACT_VGPU_INSTANCE_IDS"
#define VGPU_ATTRIB_ACT_VGPU_INSTANCE_ID_LIST_TAG "<ATTRIB_ACT_VGPU_INSTANCE_ID_LIST"
#define VGPU_ATTRIB_ACT_VGPU_INSTANCE_METRICS_TAG "<ATTRIB_ACT_VGPU_INSTANCE_METRICS"
#define VGPU_ATTRIB_VGPU_INSTANCE_VM_ID_TAG "<ATTRIB_VGPU_INSTANCE_VM_ID"
#define VGPU_ATTRIB_VGPU_INSTANCE_VM_NAME_TAG "<ATTRIB_VGPU_INSTANCE_VM_NAME"
#define VGPU_ATTRIB_VGPU_INSTANCE_TYPE_ID_TAG "<ATTRIB_VGPU_INSTANCE_TYPE_ID"
#define VGPU_ATTRIB_VGPU_INSTANCE_UUID_TAG "<ATTRIB_VGPU_INSTANCE_UUID"
#define VGPU_ATTRIB_VGPU_INSTANCE_DRIVER_VERSION_TAG "<ATTRIB_VGPU_INSTANCE_DRIVER_VERSION"
#define VGPU_ATTRIB_VGPU_INSTANCE_FB_USAGE_TAG "<ATTRIB_VGPU_INSTANCE_FB_USAGE"
#define VGPU_ATTRIB_VGPU_INSTANCE_LICENSE_STATUS_TAG "<ATTRIB_VGPU_INSTANCE_LICENSE_STATUS"
#define VGPU_ATTRIB_VGPU_INSTANCE_FRAME_RATE_LIMIT_TAG "<ATTRIB_VGPU_INSTANCE_FRAME_RATE_LIMIT"
#define VGPU_ATTRIB_VGPU_SM_UTIL_TAG "<ATTRIB_VGPU_SM_UTIL"
#define VGPU_ATTRIB_VGPU_MEMORY_UTIL_TAG "<ATTRIB_VGPU_MEMORY_UTIL"
#define VGPU_ATTRIB_VGPU_ENCODER_UTIL_TAG "<ATTRIB_VGPU_ENCODER_UTIL"
#define VGPU_ATTRIB_VGPU_DECODER_UTIL_TAG "<ATTRIB_VGPU_DECODER_UTIL"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_TAG "<ATTRIB_SUPPORTED_TYPE_INFO"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_TYPE_ID_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_TYPE_ID"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_NAME_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_NAME"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_CLASS_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_CLASS"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_LICENSE_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_LICENSE"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_DEV_ID_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_DEV_ID"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_SUBSYS_ID_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_SUBSYS_ID"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_NUM_HEADS_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_NUM_HEADS"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_MAX_INSTANCES_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_MAX_INSTANCES"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_XDIM_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_XDIM"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_YDIM_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_YDIM"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_FB_TOTAL_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_FB_TOTAL"
#define VGPU_ATTRIB_SUPPORTED_TYPE_INFO_FRL_TAG "<ATTRIB_SUPPORTED_TYPE_INFO_FRL"

/*****************************************************************************/
Vgpu::Vgpu() {
}


Vgpu::~Vgpu() {

}

/*****************************************************************************/
int Vgpu::RunGetVgpuConfig(dcgmHandle_t pLwcmHandle, bool verbose)
{
    dcgmGroupInfo_t stLwcmGroupInfo;
    dcgmStatus_t stHandle = 0;
    dcgmVgpuConfig_t *pLwcmLwrrentConfig = NULL;
    dcgmVgpuConfig_t *pLwcmTargetConfig = NULL;
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t result;
    dcgmReturn_t targetResult;
    dcgmDeviceAttributes_t stDeviceAttributes;
    dcgmVgpuDeviceAttributes_t stVgpuDeviceAttributes;
    dcgmVgpuInstanceAttributes_t stVgpuInstanceAttributes;
    CommandOutputController cmdView = CommandOutputController();
    GPUErrorOutputController gpuErrView;
    unsigned int i;
    stringstream ss;

    // fetch export table
    result = dcgmInternalGetExportTable((const void**)&t_pEtblDcgm, &ETID_DCGMEngineInternal);

    if (result != DCGM_ST_OK){
        std::cout << "Error: get the export table. Return: " << errorString(result) << std::endl;
        return result;
    }

    stDeviceAttributes.version = dcgmDeviceAttributes_version;
    stVgpuDeviceAttributes.version = dcgmVgpuDeviceAttributes_version;
    stVgpuInstanceAttributes.version = dcgmVgpuInstanceAttributes_version;

    /* Add config watches for the newly created group */
    result = dcgmUpdateAllFields(pLwcmHandle, 1);
    if (DCGM_ST_OK != result) {
        std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
        PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
        return result;
    }

    stLwcmGroupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(pLwcmHandle, mGroupId, &stLwcmGroupInfo);
    if (DCGM_ST_OK != result) {
        std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
        cout << "Error: Unable to get group information. Return: "<< error << endl;
        PRINT_ERROR("%u,%d","Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Create Status handler */
    result = dcgmStatusCreate(&stHandle);
    if (DCGM_ST_OK != result) {
        cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    pLwcmLwrrentConfig = new dcgmVgpuConfig_t[stLwcmGroupInfo.count];
    for (i = 0; i < stLwcmGroupInfo.count; i++) {
        pLwcmLwrrentConfig[i].version = dcgmVgpuConfig_version;
    }

    pLwcmTargetConfig = new dcgmVgpuConfig_t[stLwcmGroupInfo.count];
    for (i = 0; i < stLwcmGroupInfo.count; i++) {
        pLwcmTargetConfig[i].version = dcgmVgpuConfig_version;
    }


    result = DCGM_CALL_ETBL(t_pEtblDcgm, fpdcgmVgpuConfigGet, (pLwcmHandle,  mGroupId, DCGM_CONFIG_LWRRENT_STATE,
                            stLwcmGroupInfo.count, pLwcmLwrrentConfig, stHandle));

    targetResult = DCGM_CALL_ETBL(t_pEtblDcgm, fpdcgmVgpuConfigGet, (pLwcmHandle,  mGroupId, DCGM_CONFIG_TARGET_STATE,
                                  stLwcmGroupInfo.count, pLwcmTargetConfig, stHandle));

    // Populate information in displayInfo for each GPU and print
    cmdView.setDisplayStencil(VGPU_DISPLAY);

    for (i = 0; i < stLwcmGroupInfo.count; i++) {

        ss.str("");
        if (verbose) {
            ss << "GPU ID: " << pLwcmLwrrentConfig[i].gpuId;
            cmdView.addDisplayParameter(HEADER_TOP_TAG, ss.str());
            // Get device name
            dcgmGetDeviceAttributes(pLwcmHandle, pLwcmLwrrentConfig[i].gpuId, &stDeviceAttributes);
            DCGM_CALL_ETBL(t_pEtblDcgm, fpdcgmGetVgpuDeviceAttributes, (pLwcmHandle, pLwcmLwrrentConfig[i].gpuId, &stVgpuDeviceAttributes));
            cmdView.addDisplayParameter(HEADER_BOTTOM_TAG, stDeviceAttributes.identifiers.deviceName);
        } else {
            ss << "Group of " << stLwcmGroupInfo.count << " GPUs";
            cmdView.addDisplayParameter(HEADER_BOTTOM_TAG, ss.str());
            cmdView.addDisplayParameter(HEADER_TOP_TAG, stLwcmGroupInfo.groupName);
        }

        // Current Configurations
        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmLwrrentConfig, &dcgmVgpuConfig_t::computeMode, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_COMPUTE_MODE_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_COMPUTE_MODE_TAG, HelperDisplayComputeMode(pLwcmLwrrentConfig[i].computeMode));
        }

        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmLwrrentConfig, &dcgmVgpuConfig_t::eccMode, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_ECC_MODE_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_ECC_MODE_TAG, HelperDisplayBool(pLwcmLwrrentConfig[i].eccMode));
        }

        if (!verbose && !HelperCheckIfAllTheSamePowerLim(pLwcmLwrrentConfig, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_PWR_LIM_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_LWRR_PWR_LIM_TAG, pLwcmLwrrentConfig[i].powerLimit.val);
        }

        // Target Configurations
        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmTargetConfig, &dcgmVgpuConfig_t::computeMode, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_COMPUTE_MODE_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_COMPUTE_MODE_TAG, HelperDisplayComputeMode(pLwcmTargetConfig[i].computeMode));
        }

        if (!verbose && !HelperCheckIfAllTheSameMode(pLwcmTargetConfig, &dcgmVgpuConfig_t::eccMode, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_ECC_MODE_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_ECC_MODE_TAG, HelperDisplayBool(pLwcmTargetConfig[i].eccMode));
        }

        if (!verbose && !HelperCheckIfAllTheSamePowerLim(pLwcmTargetConfig, stLwcmGroupInfo.count)){
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_PWR_LIM_TAG, "****");
        } else {
            cmdView.addDisplayParameter(VGPU_CONFIG_TRG_PWR_LIM_TAG, pLwcmTargetConfig[i].powerLimit.val);
        }

        // Attributes

        cmdView.addDisplayParameter(VGPU_ATTRIB_DEV_NAME_TAG, stDeviceAttributes.identifiers.deviceName);

        cmdView.addDisplayParameter(VGPU_ATTRIB_BRAND_NAME_TAG, stDeviceAttributes.identifiers.brandName);

        cmdView.addDisplayParameter(VGPU_ATTRIB_DRV_VRS_TAG, stDeviceAttributes.identifiers.driverVersion);

        cmdView.addDisplayParameter(VGPU_ATTRIB_VBIOS_VRS_TAG, stDeviceAttributes.identifiers.vbios);

        cmdView.addDisplayParameter(VGPU_ATTRIB_IFRM_IMG_VRS_TAG, stDeviceAttributes.identifiers.inforomImageVersion);

        cmdView.addDisplayParameter(VGPU_ATTRIB_UUID_TAG, stDeviceAttributes.identifiers.uuid);

        cmdView.addDisplayParameter(VGPU_ATTRIB_SERIAL_TAG, stDeviceAttributes.identifiers.serial);

        cmdView.addDisplayParameter(VGPU_ATTRIB_OPR_MODE_TAG, stDeviceAttributes.identifiers.virtualizationMode);

        cmdView.addDisplayParameter(VGPU_ATTRIB_PCI_BUS_ID_TAG, stDeviceAttributes.identifiers.pciBusId);

        cmdView.addDisplayParameter(VGPU_ATTRIB_PCI_DEV_ID_TAG, stDeviceAttributes.identifiers.pciDeviceId);

        cmdView.addDisplayParameter(VGPU_ATTRIB_PCI_SUBSYS_ID_TAG, stDeviceAttributes.identifiers.pciSubSystemId);

        cmdView.addDisplayParameter(VGPU_ATTRIB_BAR1_TOTAL_TAG, stDeviceAttributes.memoryUsage.bar1Total);

        cmdView.addDisplayParameter(VGPU_ATTRIB_FB_TOTAL_TAG, stDeviceAttributes.memoryUsage.fbTotal);

        cmdView.addDisplayParameter(VGPU_ATTRIB_FB_USED_TAG, stDeviceAttributes.memoryUsage.fbFree);

        cmdView.addDisplayParameter(VGPU_ATTRIB_FB_FREE_TAG, stDeviceAttributes.memoryUsage.fbUsed);

        cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_VGPU_TYPE_COUNT_TAG, stVgpuDeviceAttributes.supportedVgpuTypeCount);

        cmdView.addDisplayParameter(VGPU_ATTRIB_CRT_VGPU_TYPE_COUNT_TAG, stVgpuDeviceAttributes.creatableVgpuTypeCount);

        cmdView.addDisplayParameter(VGPU_ATTRIB_ACT_VGPU_INSTANCE_COUNT_TAG, stVgpuDeviceAttributes.activeVgpuInstanceCount);

        cmdView.addDisplayParameter(VGPU_ATTRIB_GPU_UTIL_TAG, stVgpuDeviceAttributes.gpuUtil);

        cmdView.addDisplayParameter(VGPU_ATTRIB_MEM_UTIL_TAG, stVgpuDeviceAttributes.memCopyUtil);

        cmdView.addDisplayParameter(VGPU_ATTRIB_ENC_UTIL_TAG, stVgpuDeviceAttributes.enlwtil);

        cmdView.addDisplayParameter(VGPU_ATTRIB_DEC_UTIL_TAG, stVgpuDeviceAttributes.delwtil);

        cmdView.display();

        cmdView.setDisplayStencil(VGPU_CRT_TYPE_ID_HEADER);
        cmdView.addDisplayParameter(VGPU_ATTRIB_CRT_VGPU_TYPE_IDS_TAG, " ");
        cmdView.display();

        for (unsigned int k = 0; k < stVgpuDeviceAttributes.creatableVgpuTypeCount; k++) {
            cmdView.setDisplayStencil(VGPU_CRT_TYPE_ID_LIST);
            cmdView.addDisplayParameter(VGPU_ATTRIB_CRT_VGPU_TYPE_ID_LIST_TAG, stVgpuDeviceAttributes.creatableVgpuTypeIds[k]);
            cmdView.display();
        }

        std::cout << VGPU_CRT_TYPE_ID_FOOTER;

        cmdView.setDisplayStencil(VGPU_ACT_INSTANCE_ID_HEADER);
        cmdView.addDisplayParameter(VGPU_ATTRIB_ACT_VGPU_INSTANCE_IDS_TAG, " ");
        cmdView.display();

        for (unsigned int k = 0; k < stVgpuDeviceAttributes.activeVgpuInstanceCount; k++) {
            cmdView.setDisplayStencil(VGPU_ACT_INSTANCE_ID_LIST);
            cmdView.addDisplayParameter(VGPU_ATTRIB_ACT_VGPU_INSTANCE_ID_LIST_TAG, stVgpuDeviceAttributes.activeVgpuInstanceIds[k]);
            cmdView.display();
        }

        std::cout << VGPU_ACT_INSTANCE_ID_FOOTER;

        cmdView.setDisplayStencil(VGPU_SUPPORTED_TYPE_INFO_HEADER);
        cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_TAG, " ");
        cmdView.display();

        for (unsigned int k = 0; k < stVgpuDeviceAttributes.supportedVgpuTypeCount; k++) {
            cmdView.setDisplayStencil(VGPU_SUPPORTED_TYPE_INFO_LIST);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_TYPE_ID_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].vgpuTypeInfo.vgpuTypeId);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_NAME_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].vgpuTypeName);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_CLASS_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].vgpuTypeClass);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_LICENSE_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].vgpuTypeLicense);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_DEV_ID_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].deviceId);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_SUBSYS_ID_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].subsystemId);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_NUM_HEADS_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].numDisplayHeads);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_MAX_INSTANCES_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].maxInstances);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_XDIM_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].maxResolutionX);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_YDIM_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].maxResolutionY);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_FB_TOTAL_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].fbTotal);
            cmdView.addDisplayParameter(VGPU_ATTRIB_SUPPORTED_TYPE_INFO_FRL_TAG, stVgpuDeviceAttributes.supportedVgpuTypeInfo[k].frameRateLimit);
            cmdView.display();
        }

        std::cout << VGPU_SUPPORTED_TYPE_INFO_FOOTER;

        cmdView.setDisplayStencil(VGPU_ACT_INSTANCE_METRICS_HEADER);
        cmdView.addDisplayParameter(VGPU_ATTRIB_ACT_VGPU_INSTANCE_METRICS_TAG, " ");
        cmdView.display();

        for (unsigned int k = 0; k < stVgpuDeviceAttributes.activeVgpuInstanceCount; k++) {
            DCGM_CALL_ETBL(t_pEtblDcgm, fpdcgmGetVgpuInstanceAttributes, (pLwcmHandle, stVgpuDeviceAttributes.activeVgpuInstanceIds[k], &stVgpuInstanceAttributes));
            cmdView.setDisplayStencil(VGPU_ACT_INSTANCE_METRICS_LIST);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_VM_ID_TAG, stVgpuInstanceAttributes.vmId);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_VM_NAME_TAG, stVgpuInstanceAttributes.vmName);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_TYPE_ID_TAG, stVgpuInstanceAttributes.vgpuTypeId);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_UUID_TAG, stVgpuInstanceAttributes.vgpuUuid);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_DRIVER_VERSION_TAG, stVgpuInstanceAttributes.vgpuDriverVersion);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_FB_USAGE_TAG, stVgpuInstanceAttributes.fbUsage);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_LICENSE_STATUS_TAG, stVgpuInstanceAttributes.licenseStatus);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_INSTANCE_FRAME_RATE_LIMIT_TAG, stVgpuInstanceAttributes.frameRateLimit);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_SM_UTIL_TAG, stVgpuDeviceAttributes.vgpuUtilInfo[k].smUtil);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_MEMORY_UTIL_TAG, stVgpuDeviceAttributes.vgpuUtilInfo[k].memUtil);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_ENCODER_UTIL_TAG, stVgpuDeviceAttributes.vgpuUtilInfo[k].enlwtil);
            cmdView.addDisplayParameter(VGPU_ATTRIB_VGPU_DECODER_UTIL_TAG, stVgpuDeviceAttributes.vgpuUtilInfo[k].delwtil);
            cmdView.display();
        }

        std::cout << VGPU_ACT_INSTANCE_METRICS_FOOTER;

        cmdView.setDisplayStencil(VGPU_DISPLAY);

        if (!verbose) break; // only need one output in this case
    }

    if (!verbose){
        std::cout << "**** Non-homogenous settings across group. Use with –v flag to see details.\n";
    }

    /**
     * Check for errors (if any)
     */
    if (DCGM_ST_OK != result) {
        cout << "\nUnable to get some of the configuration properties. Return: " << errorString(result) << endl;
        /* Look at status to get individual errors */
        gpuErrView.addError(stHandle);
        gpuErrView.display();
        ret = DCGM_ST_GENERIC_ERROR;
        goto cleanup_local;
    }

    cleanup_local:
    /* Destroy Status message */
    if (stHandle) {
        result = dcgmStatusDestroy(stHandle);
        if (DCGM_ST_OK != result) {
            cout << "Unable to destroy status handler. Return: " << result << endl;
        }
    }

    if (pLwcmLwrrentConfig) {
        delete [] pLwcmLwrrentConfig;
    }

    if (pLwcmTargetConfig) {
        delete [] pLwcmTargetConfig;
    }

    return ret;
}

/*****************************************************************************/
int Vgpu::RunSetVgpuConfig(dcgmHandle_t pLwcmHandle)
{
   dcgmReturn_t ret = DCGM_ST_OK;
   dcgmReturn_t result;
   dcgmStatus_t stHandle = 0;
   GPUErrorOutputController gpuErrView;

    // fetch export table
    result = dcgmInternalGetExportTable((const void**)&t_pEtblDcgm, &ETID_DCGMEngineInternal);

    if (result != DCGM_ST_OK){
        std::cout << "Error: get the export table. Return: " << errorString(result) << std::endl;
        return result;
    }

   /* Add config watches for the newly created group */
   result = dcgmUpdateAllFields(pLwcmHandle, 1);
   if (DCGM_ST_OK != result) {
       std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
       PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
       return result;
   }

   /* Create Status handler */
   result = dcgmStatusCreate(&stHandle);
   if (DCGM_ST_OK != result) {
       cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
       ret = DCGM_ST_GENERIC_ERROR;
       goto cleanup_local;
   }

   mConfigVal.version = dcgmVgpuConfig_version;

   result = DCGM_CALL_ETBL(t_pEtblDcgm, fpdcgmVgpuConfigSet, (pLwcmHandle,  mGroupId, &mConfigVal, stHandle));
   if (DCGM_ST_OK != result){
       std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
       cout << "Error: Unable to set some of the configuration properties. Return: "<< error << endl;

       gpuErrView.addError(stHandle);
       gpuErrView.display();

       PRINT_ERROR("%u, %d","Error: Unable to set configuration on group %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);

       ret = result;
       goto cleanup_local;
   } else {
       cout << "Configuration successfully set.\n";
   }

   cleanup_local:
   /* Destroy Status message */
   if (stHandle) {
       result = dcgmStatusDestroy(stHandle);
       if (DCGM_ST_OK != result) {
           cout << "Error: Unable to destroy status handler. Return: " <<  errorString(result) << endl;
       }
   }

   return ret;
}

/*****************************************************************************/
int Vgpu::RunEnforceVgpuConfig(dcgmHandle_t pLwcmHandle)
{
   dcgmGroupInfo_t stLwcmGroupInfo;
   dcgmStatus_t stHandle = 0;
   dcgmReturn_t ret = DCGM_ST_OK;
   dcgmReturn_t result;
   GPUErrorOutputController gpuErrView;
   /* Add config watches for the newly created group */
   result = dcgmUpdateAllFields(pLwcmHandle, 1);
   if (DCGM_ST_OK != result) {
       std::cout << "Error: Unable to update fields. Return: " << errorString(result) << endl;
       PRINT_ERROR("%d","Error: UpdateAllFields. Return: %d", result);
       return result;
   }

   stLwcmGroupInfo.version = dcgmGroupInfo_version;
   result = dcgmGroupGetInfo(pLwcmHandle, mGroupId, &stLwcmGroupInfo);
   if (DCGM_ST_OK != result) {
       std::string error = (result == DCGM_ST_NOT_CONFIGURED)? "The Group is not found" : errorString(result);
       cout << "Error: Unable to get group information. Return: "<< error << endl;
       PRINT_ERROR("%u,%d","Error: GroupGetInfo for GroupId: %u. Return: %d", (unsigned int)(uintptr_t)mGroupId, result);
       return DCGM_ST_GENERIC_ERROR;
   }

   /* Create Status handler */
   result = dcgmStatusCreate(&stHandle);
   if (DCGM_ST_OK != result) {
       cout << "Error: Unable to create status handler. Return:" << errorString(result) << endl;
       ret = DCGM_ST_GENERIC_ERROR;
       goto cleanup_local;
   }

   result = dcgmConfigEnforce(pLwcmHandle, mGroupId, stHandle);

   /**
    * Check for errors (if any)
    */
   if (DCGM_ST_OK != result) {
       cout << " Error: Unable to enforce some of the configuration properties. Return: " << errorString(result) << endl;

       // Add this to override not very informative error messages within the status handle. BUG ->
       gpuErrView.addErrorStringOverride(DCGM_FI_UNKNOWN, DCGM_ST_NOT_CONFIGURED, "Unknown - Target configuration not specified.");

       gpuErrView.addError(stHandle);
       gpuErrView.display();

       ret = DCGM_ST_GENERIC_ERROR;
       goto cleanup_local;
   } else {
       cout << "Configuration successfully enforced.\n";
   }

   cleanup_local:
   /* Destroy Status message */
   if (stHandle) {
       result = dcgmStatusDestroy(stHandle);
       if (DCGM_ST_OK != result) {
           cout << "Error: Unable to destroy status handler. Return: " << errorString(result) << endl;
       }
   }

   return ret;
}

/*****************************************************************************/
template <typename TMember>
bool Vgpu::HelperCheckIfAllTheSameMode(dcgmVgpuConfig_t *configs, TMember member, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].*member != configs[i].*member){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
bool Vgpu::HelperCheckIfAllTheSamePowerLim(dcgmVgpuConfig_t *configs, unsigned int numGpus){

    for (unsigned int i = 1; i < numGpus; i++){
        if (configs[0].powerLimit.val != configs[i].powerLimit.val){
            return false;
        }
    }
    return true;
}

/*****************************************************************************/
int Vgpu::SetArgs(unsigned int groupId, dcgmVgpuConfig_t* pConfigVal)
{
    mGroupId = (dcgmGpuGrp_t)(intptr_t)groupId;

    if (NULL != pConfigVal) {
        mConfigVal = *pConfigVal;
    }

    return 0;
}

/*****************************************************************************/
std::string Vgpu::HelperDisplayComputeMode(unsigned int val){
    std:stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        if (DCGM_CONFIG_COMPUTEMODE_DEFAULT == val)
            ss << "Unrestricted";
        else if (DCGM_CONFIG_COMPUTEMODE_PROHIBITED == val)
            ss << "Prohibited";
        else if (DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS == val)
            ss << "E. Process";
        else
            ss << "Unknown"; /* This should never happen */
    }

    return ss.str();
}

/****************************************************************************/
std::string Vgpu::HelperDisplayBool(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        if (0 == val){
            ss << "Disabled";
        } else if (1 == val){
            ss << "Enabled";
        } else {
            ss << "Error";
        }
    }

    return ss.str();
}


/*****************************************************************************
 *****************************************************************************
 * Set Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
SetVgpuConfig::SetVgpuConfig(std::string hostname, Vgpu &obj) {
    mHostName = hostname;
    vgpuObj = obj;
}

/*****************************************************************************/
SetVgpuConfig::~SetVgpuConfig() {
}

/*****************************************************************************/
int SetVgpuConfig::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return vgpuObj.RunSetVgpuConfig(mLwcmHandle);
}


/*****************************************************************************
 *****************************************************************************
 * Get Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
GetVgpuConfig::GetVgpuConfig(std::string hostname, Vgpu &obj, bool verbose) {
    mHostName = hostname;
    vgpuObj = obj;
    this->verbose = verbose;
}

/*****************************************************************************/
GetVgpuConfig::~GetVgpuConfig() {

}

/*****************************************************************************/
int GetVgpuConfig::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    // Setting verbose to true by default, because we always want to see all
    // fields for each GPU, and they might differ.
    verbose = 1;
    return vgpuObj.RunGetVgpuConfig(mLwcmHandle, verbose);
}


/*****************************************************************************
 *****************************************************************************
 * Enforce Configuration Ilwoker
 *****************************************************************************
 *****************************************************************************/

EnforceVgpuConfig::EnforceVgpuConfig(std::string hostname, Vgpu &obj) {
    mHostName = hostname;
    vgpuObj = obj;
}

EnforceVgpuConfig::~EnforceVgpuConfig() {

}

int EnforceVgpuConfig::Execute() {
    dcgmReturn_t connection = (dcgmReturn_t) Command::Execute();
    if (connection != DCGM_ST_OK){
        std::cout << "Error: Unable to connect to host engine. " << errorString(connection) << "." <<std::endl;
        return DCGM_ST_BADPARAM;
    }
    return vgpuObj.RunEnforceVgpuConfig(mLwcmHandle);
}
