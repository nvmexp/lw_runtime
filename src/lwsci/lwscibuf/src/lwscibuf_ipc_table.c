/*
 * lwscibuf_ipc_table.c
 *
 * LwSciBuf S/W Unit implementation for IPC Table and IPC Routes.
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_ipc_table.h"

#include <string.h>
#include "lwscilog.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscibuf_utils.h"
#include "lwscibuf_ipc_table_priv.h"

#include <unistd.h>


/**
 * Need this macro as the size of header is declared as 1
 */
#define ADJUSTED_SIZEOF(x) (sizeof(x) - sizeof(uint8_t))

/******************************************************
 *    All Static Function Definitions                 *
 ******************************************************/

#if (LWSCI_DEBUG == 0)
#define LwSciBufPrintIpcRoute(ipcRoute)
#else
#define MAX_LINE_SIZE_FOR_IPC 100
#define MAX_STR_SIZE_FOR_IPC 20

static void LwSciBufPrintIpcRoute(
    const LwSciBufIpcRoute* ipcRoute)
{
    size_t endpointIdx = 0;
    size_t len = 0;
    char line[MAX_LINE_SIZE_FOR_IPC] = "\0";
    char str[MAX_STR_SIZE_FOR_IPC] = "\0";
    LWSCI_FNENTRY("");

    if (NULL == ipcRoute) {
        LWSCI_ERR_STR(" Cannot print NULL IPC Route.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    for (endpointIdx = 0;
         endpointIdx < ipcRoute->endpointCount;
         endpointIdx++) {
        snprintf(str, MAX_STR_SIZE_FOR_IPC, " %"PRIX32" %"PRIX32" %"PRIX64"",
            ipcRoute->ipcEndpointList[endpointIdx].topoId.SocId,
            ipcRoute->ipcEndpointList[endpointIdx].topoId.VmId,
            ipcRoute->ipcEndpointList[endpointIdx].vuId);
        if ((sizeof(line) - 1U) > (len + strlen(str))) {
            (void)strcat(line, str);
            len += strlen(str);
        } else {
            /* flush the line to the output */
            line[len] = (char)0;
            LWSCI_INFO(" %s\n", line);

            /* reset line buffer */
            (void)memset(line, 0x0, sizeof(line));
            len = strlen(str);
            LwSciCommonMemcpyS(line, sizeof(line) - 1, str, len);
        }
    }
    LWSCI_INFO(" %s NULL \n", line);

ret:
    LWSCI_FNEXIT("");
}
#endif

static LwSciError getTopoId(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufIpcTopoId* topoId)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

/* LwSciIpcEndpointGetTopoId() is not defined for safety builds */
#if (LW_IS_SAFETY == 0)
        err = LwSciIpcEndpointGetTopoId(ipcEndpoint, &topoId->topoId);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciIpcEndpointGetTopoId failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
#else
        topoId->topoId.SocId = LWSCIIPC_SELF_SOCID;
        topoId->topoId.VmId = LWSCIIPC_SELF_VMID;
#endif

/* LwSciIpcEndpointGetVuid() throws error on x86 */
#if !defined(__x86_64__)
        if (topoId->topoId.SocId == LWSCIIPC_SELF_SOCID) {
            err = LwSciIpcEndpointGetVuid(ipcEndpoint, &topoId->vuId);
            if (LwSciError_Success != err) {
                LWSCI_ERR_STR("LwSciIpcEndpointGetTopoId failed");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else { /* C2C case */
            /* LwSciIpcEndpointGetVuid() returns _NotSupported
               for C2C based ipcEndpoints, so replacing with a mock */
            topoId->vuId = ((ipcEndpoint << 32) & 0xFFFFFFFF00000000) |
                (getpid() & 0x00000000FFFFFFFF);
        }
#else
        /* pid will be unique per process while LwSciIpcEndpoint will be
         * unique in the given process. By using the callwlation below, we can
         * ensure that vuId is unique for inter-thread and inter-process
         * boundary. Remove this when LwSciIpc starts supporting GetVuId() in
         * x86.
         */
        topoId->vuId = ((ipcEndpoint << 32) & 0xFFFFFFFF00000000) |
            (getpid() & 0x00000000FFFFFFFF);
#endif

ret:
    LWSCI_FNEXIT("");
    return err;
}

static inline LwSciError LwSciBufIpcTableAttrDataExportSize(
    size_t *exportSize,
    const LwSciBufIpcTableAttrData* data)
{
    LwSciError sciErr = LwSciError_Success;
    uint8_t status = OP_FAIL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs: data: %p\n", data);

    u64Add((sizeof(data->key) + sizeof(data->len)), data->len, exportSize, &status);
    if (OP_SUCCESS != status) {
        sciErr = LwSciError_Overflow;
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: ExportSize : %"PRIu64"\n", exportSize);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIpcRouteExportInternal(
    const LwSciBufIpcRoute* ipcRoute,
    void* descBuf,
    size_t bufSize,
    size_t* copiedSize,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError sciErr = LwSciError_Success;
    size_t copySize = 0;
    size_t endpointCount = 0;
    uint8_t arithmeticStatus = OP_FAIL;
    void* descBufOffset = descBuf;
    LwSciBufIpcTopoId tmpTopoId = {};

    LWSCI_FNENTRY("");

    *copiedSize = 0;
    copySize = LwSciBufIpcRouteExportSize(ipcRoute, ipcEndpoint);


    if (bufSize < copySize) {
        LWSCI_ERR_STR("Not enough buffer to export the Ipc Route. \n");
        LWSCI_ERR_ULONG("Buffer size: , ", bufSize);
        LWSCI_ERR_ULONG("Required Size: \n", copySize);
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - CopySize Ptr: %p descBuf: %p bufSize: %"PRIu64
               "ipcRoute \n", copiedSize, descBuf, bufSize);
    LwSciBufPrintIpcRoute(ipcRoute);

    if (0U != ipcEndpoint) {
        u64Add(ipcRoute->endpointCount, 1U, &endpointCount, &arithmeticStatus);
        if (OP_SUCCESS != arithmeticStatus) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr = getTopoId(ipcEndpoint, &tmpTopoId);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("getTopoId() failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        endpointCount = ipcRoute->endpointCount;
    }

    LwSciCommonMemcpyS(descBufOffset, sizeof(uint64_t), &endpointCount,
                      sizeof(uint64_t));

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descBufOffset = (uint8_t *)descBufOffset + sizeof(uint64_t);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

    if (0U != endpointCount) {
        if (0U != ipcEndpoint) {
            if (NULL != ipcRoute->ipcEndpointList) {
                LwSciCommonMemcpyS(descBufOffset,
                    copySize - sizeof(uint64_t) - sizeof(LwSciBufIpcTopoId),
                    ipcRoute->ipcEndpointList,
                    copySize - sizeof(uint64_t) - sizeof(LwSciBufIpcTopoId));

                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),
                    "LwSciBuf-ADV-MISRAC2012-017")
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
                    "LwSciBuf-ADV-MISRAC2012-014")
                descBufOffset = (uint8_t *)descBufOffset + (copySize -
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
                            sizeof(uint64_t) - sizeof(LwSciBufIpcTopoId));
            }

            LwSciCommonMemcpyS(descBufOffset, sizeof(LwSciBufIpcTopoId),
                &tmpTopoId, sizeof(LwSciBufIpcTopoId));
        } else {
            LwSciCommonMemcpyS(descBufOffset, copySize - sizeof(uint64_t),
                ipcRoute->ipcEndpointList, copySize - sizeof(uint64_t));
        }
    }
    *copiedSize = copySize;

    if (0U != ipcEndpoint) {
        size_t tmpIndex = 0U;
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),
            "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
            "LwSciBuf-ADV-MISRAC2012-014")
        LwSciBufIpcTopoId* tmpTopoIdList =
            (LwSciBufIpcTopoId*)((uint8_t*)descBuf + sizeof(uint64_t));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

        /* Check if we are crossing the communication boundary. If we are
         * crossing the boundary then all the endpoints which are within that
         * boundary are tagged with the boundary information. This helps other
         * peer importing the attribute list to identify the communication
         * boundary.
         */
        if (LWSCIIPC_SELF_VMID != tmpTopoId.topoId.VmId) {
            /* We are crossing VM boundary. Check all endpoints for which
             * VmId == LWSCIIPC_SELF_VMID (implying that the endpoint has not
             * crossed VM boundary and it will cross the VM boundary via export)
             * and replace LWSCIIPC_SELF_VMID with actual VmId.
             */
            for (tmpIndex = 0U; tmpIndex < endpointCount; tmpIndex++) {
                if (LWSCIIPC_SELF_VMID ==
                    tmpTopoIdList[tmpIndex].topoId.VmId) {
                    tmpTopoIdList[tmpIndex].topoId.VmId =
                        tmpTopoId.topoId.VmId;
                }
            }
        }

        if (LWSCIIPC_SELF_SOCID != tmpTopoId.topoId.SocId) {
            /* We are crossing SoC boundary (and VM boundary). Check all
             * endpoints for which SocId == LWSCIIPC_SELF_SOCID (implying that
             * the endpoint has not crossed SoC boundary and it will cross SoC
             * boundary via export) and replace LWSCIIPC_SELF_SOCID with the
             * actual SoCId. Also, check endpoints for which
             * VmId == LWSCIIPC_SELF_VMID (implying that the endpoint has not
             * crossed VM boundary and it will cross the VM boundary by virtue
             * of crossing SoC boundary via export) and replace
             * LWSCIIPC_SELF_VMID with actual VmId.
             */
            for (tmpIndex = 0U; tmpIndex < endpointCount; tmpIndex++) {
                if (LWSCIIPC_SELF_SOCID ==
                    tmpTopoIdList[tmpIndex].topoId.SocId) {
                    tmpTopoIdList[tmpIndex].topoId.SocId =
                        tmpTopoId.topoId.SocId;
                }
            }
        }
    }

    LWSCI_INFO("Outputs - Copied Size %"PRIu64"\n", *copiedSize);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIpcTableAttrDataExport(
    const LwSciBufIpcTableAttrData* data,
    void* descBuf,
    size_t bufSize,
    size_t* copiedSize)
{
    LwSciError sciErr = LwSciError_Success;
    size_t copySize = 0;
    void* descBufOffset = descBuf;

    LWSCI_FNENTRY("");

    *copiedSize = 0;
    sciErr = LwSciBufIpcTableAttrDataExportSize(&copySize, data);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (bufSize < copySize) {
        LWSCI_ERR_STR("Not enough buffer to export the Ipc Table AttrData. \n");
        LWSCI_ERR_ULONG("Buffer size: , ", bufSize);
        LWSCI_ERR_ULONG("Required Size: \n", copySize);
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - Data: %p CopySize Ptr: %p descBuf: %p bufSize: %"
               PRIu64"\n", data, copiedSize, descBuf, bufSize);

    LwSciCommonMemcpyS(descBuf, sizeof(data->key), &data->key,
        sizeof(data->key));
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descBufOffset = (uint8_t *)descBufOffset + sizeof(data->key);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    LwSciCommonMemcpyS(descBufOffset, sizeof(data->len), &data->len,
        sizeof(data->len));
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descBufOffset = (uint8_t *)descBufOffset + sizeof(data->len);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    LwSciCommonMemcpyS(descBufOffset, data->len, data->value, data->len);
    *copiedSize = copySize;

    LWSCI_INFO("Outputs - Copied Size %"PRIu64"\n", *copiedSize);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIpcTableAttrDataImport(
    const void* descBuf,
    size_t bufSize,
    size_t* importedLen,
    LwSciBufIpcTableAttrData** data)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTableAttrData *tableData = NULL;
    size_t lenImported = 0;
    uint8_t addStatus = OP_FAIL;
    const void* descBufOffset = descBuf;

    LWSCI_FNENTRY("");
    if (0U == bufSize) {
        LWSCI_ERR_STR("Invalid inputs for importing IpcTable AttrData.\n");
        LWSCI_ERR_ULONG("Length \n", bufSize);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - Desc %p Length %"PRIu64
              " Data ptr %p ImportedLen ptr: %p\n", descBuf, bufSize, data,
              importedLen);


    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    tableData = LwSciCommonCalloc(1, sizeof(*tableData));
    if (NULL == tableData) {
        LWSCI_ERR_STR("Failed to allocate memory to import Table Data.\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciCommonMemcpyS(&tableData->key, sizeof(tableData->key), descBufOffset,
        sizeof(tableData->key));
    lenImported = sizeof(tableData->key);
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descBufOffset = (const uint8_t *)descBufOffset + sizeof(tableData->key);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

    LwSciCommonMemcpyS(&tableData->len, sizeof(tableData->len), descBufOffset,
        sizeof(tableData->len));
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descBufOffset = (const uint8_t *)descBufOffset + sizeof(tableData->len);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    lenImported += sizeof(tableData->len);

    tableData->value = LwSciCommonCalloc(1, tableData->len);
    if (NULL == tableData->value) {
        LWSCI_ERR_STR("Failed to allocate memory to import Table data value.\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_table_data;
    }

    LwSciCommonMemcpyS(tableData->value, tableData->len, descBufOffset,
        tableData->len);
    u64Add(lenImported, tableData->len, &lenImported, &addStatus);
    if (OP_SUCCESS != addStatus) {
        sciErr = LwSciError_Overflow;
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_table_data_value;
    }

    *data = tableData;

    LWSCI_INFO("Outputs - Imported Attr Data %p \n", *data);

    *importedLen = lenImported;
    LWSCI_INFO("Imported Buffer Length: %"PRIu64"\n", *importedLen);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_table_data_value:
    LwSciCommonFree(tableData->value);
free_table_data:
    LwSciCommonFree(tableData);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIpcRouteImportInternal(
    LwSciIpcEndpoint ipcEndpoint,
    const void* desc,
    size_t bufLen,
    size_t* lenImported,
    bool routeOnlyDesc,
    LwSciBufIpcRoute** ipcRoute)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcRoute* importedIpcRoute = NULL;
    size_t importedEndpointCount = 0;
    size_t importedLen = 0, len =0;
    uint8_t status = OP_FAIL;
    const void* descOffset = desc;
    LwSciBufIpcTopoId tmpTopoId = {};

    LWSCI_FNENTRY("");
    if (bufLen < sizeof(uint64_t)) {
        LWSCI_ERR_STR("Invalid input parameters to import IPC Route. \n");
        LWSCI_ERR_ULONG("Length \n", bufLen);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs- Desc ptr: %p, Length : %"PRIu64", IpcRoute: %p \n",
                   desc, bufLen, ipcRoute);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    importedIpcRoute = LwSciCommonCalloc(1, sizeof(*importedIpcRoute));
    if (NULL == importedIpcRoute) {
        LWSCI_ERR_STR("Failed to allocate for importing IPC route. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    importedLen = sizeof(uint64_t);
    LwSciCommonMemcpyS(&importedEndpointCount, sizeof(importedEndpointCount),
        descOffset, importedLen);
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    descOffset = (const uint8_t *)descOffset + importedLen;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))

    if (routeOnlyDesc) {
        size_t descEndpointCount = 0;

        descEndpointCount =
            (bufLen - sizeof(uint64_t))/sizeof(LwSciBufIpcTopoId);
        if (importedEndpointCount != descEndpointCount) {
            LWSCI_ERR_STR("Invalid descriptor to import IPC route. \n");
            sciErr = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_route;
        }
    }

    importedIpcRoute->endpointCount = importedEndpointCount;
    if (0U != ipcEndpoint) {
        u64Add(importedIpcRoute->endpointCount, 1, &importedIpcRoute->endpointCount, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_route;
        }

        sciErr = getTopoId(ipcEndpoint, &tmpTopoId);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("getTopoId() failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_route;
        }
    }

    if (0U != importedIpcRoute->endpointCount) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        importedIpcRoute->ipcEndpointList =
            LwSciCommonCalloc(importedIpcRoute->endpointCount,
                    sizeof(LwSciBufIpcTopoId));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == importedIpcRoute->ipcEndpointList) {
            LWSCI_ERR_STR("Failed to allocate IPC Endpoint list while importing\n");
            sciErr = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_route;
        }

        len = (importedEndpointCount * sizeof(LwSciBufIpcTopoId));
        LwSciCommonMemcpyS(importedIpcRoute->ipcEndpointList, len, descOffset, len);
        u64Add(importedLen, len, &importedLen, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_endpoint_list;
        }
    } else {
        importedIpcRoute->ipcEndpointList = NULL;
    }

    if (0U != ipcEndpoint) {
        size_t tmpIndex = 0U;

        LwSciCommonMemcpyS(&importedIpcRoute->ipcEndpointList[
            importedEndpointCount], sizeof(tmpTopoId), &tmpTopoId,
            sizeof(tmpTopoId));

        /* Check if we are crossing SoC boundary. If we are importing by
         * crossing the SoC boundary then we need colwert all the ipc endpoints
         * for which SoId was changed from LWSCIIPC_SELF_SOCID to actual SocId
         * back to LWSCIIPC_SELF_SOCID.
         */
        if (LWSCIIPC_SELF_SOCID != tmpTopoId.topoId.SocId) {
            for (tmpIndex = 0U; tmpIndex < importedIpcRoute->endpointCount;
                tmpIndex++) {
                if (tmpTopoId.topoId.SocId ==
                    importedIpcRoute->ipcEndpointList[tmpIndex].topoId.SocId) {
                    importedIpcRoute->ipcEndpointList[tmpIndex].topoId.SocId =
                    LWSCIIPC_SELF_SOCID;
                }
            }
        }

        /* Check if we are crossing VM boundary. If we are importing by
         * crossing the VM boundary then we need colwert all the ipc endpoints
         * for which VmId was changed from LWSCIIPC_SELF_VMID to actual VmId
         * back to LWSCIIPC_SELF_VMID.
         */
        if (LWSCIIPC_SELF_VMID != tmpTopoId.topoId.VmId) {
            for (tmpIndex = 0U; tmpIndex < importedIpcRoute->endpointCount;
                tmpIndex++) {
                if ((tmpTopoId.topoId.VmId ==
                    importedIpcRoute->ipcEndpointList[tmpIndex].topoId.VmId) &&
                    (LWSCIIPC_SELF_SOCID ==
                    importedIpcRoute->ipcEndpointList[tmpIndex].topoId.SocId)) {
                    importedIpcRoute->ipcEndpointList[tmpIndex].topoId.VmId =
                    LWSCIIPC_SELF_VMID;
                }
            }
        }
    }

    *ipcRoute = importedIpcRoute;

    LWSCI_INFO("Output - Imported IPC Route\n");
    LwSciBufPrintIpcRoute(*ipcRoute);

    if (NULL != lenImported) {
        *lenImported = importedLen;
        LWSCI_INFO("Buffer Len after import %"PRIu64"\n", *lenImported);
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_ipc_endpoint_list:
    LwSciCommonFree(importedIpcRoute->ipcEndpointList);
free_ipc_route:
    LwSciCommonFree(importedIpcRoute);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIpcTableAddIpcRoute(
    LwSciBufIpcRoute** dstIpcRoute,
    const LwSciBufIpcRoute* srcIpcRoute)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcRoute* ipcRoute = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs - DestIpcRoutePtr: %p SrcIpcRoute: \n",
         dstIpcRoute);
    LwSciBufPrintIpcRoute(srcIpcRoute);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    ipcRoute = LwSciCommonCalloc(1, sizeof(*ipcRoute));
    if (NULL == ipcRoute) {
        LWSCI_ERR_STR("Failed to allocate memory for ipc route. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((NULL == srcIpcRoute) || (0U == srcIpcRoute->endpointCount) ||
        (NULL == srcIpcRoute->ipcEndpointList)) {
        ipcRoute->ipcEndpointList = NULL;
        ipcRoute->endpointCount = 0U;
    } else {

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        ipcRoute->ipcEndpointList = LwSciCommonCalloc(srcIpcRoute->endpointCount,
                sizeof(LwSciBufIpcTopoId));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == ipcRoute->ipcEndpointList) {
            LWSCI_ERR_STR("Failed to allocate memory for ipc endpointlist. \n");
            sciErr = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_route;
        }

        ipcRoute->endpointCount = srcIpcRoute->endpointCount;
        LwSciCommonMemcpyS(ipcRoute->ipcEndpointList,
            (sizeof(LwSciBufIpcTopoId) * ipcRoute->endpointCount),
            srcIpcRoute->ipcEndpointList,
            (sizeof(LwSciBufIpcTopoId) * ipcRoute->endpointCount));
    }

    *dstIpcRoute = ipcRoute;

    LWSCI_INFO("Outputs - IPCEndpointCount: %"PRIu64", EndpointList: %p \n",
        ipcRoute->endpointCount, ipcRoute->ipcEndpointList);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_route:
    LwSciCommonFree(ipcRoute);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

/******************************************************
 *    All IpcTable related Public Functions           *
 ******************************************************/
size_t LwSciBufIpcRouteExportSize(
    const LwSciBufIpcRoute* ipcRoute,
    LwSciIpcEndpoint ipcEndpoint)
{
    size_t exportSize = 0;
    uint8_t mulStatus = OP_FAIL;
    uint8_t addStatus = OP_FAIL;
    size_t tmpSize = 0U;
    const LwSciBufIpcRoute* tmpIpcRoute;
    /* Exporting NULL IPC route is allowed.
     * Because when attrlist is created, there is no
     * known ipc route.
     */
    LwSciBufIpcRoute nullIpcRoute = { NULL, 0 };

    LWSCI_FNENTRY("");
    if (NULL == ipcRoute) {
        LWSCI_INFO("Input: Null ipc route.\n");
        tmpIpcRoute = &nullIpcRoute;
    } else {
        tmpIpcRoute = ipcRoute;
    }

    LWSCI_INFO("Inputs: Ipcroute: \n");
    LwSciBufPrintIpcRoute(tmpIpcRoute);

    /*
     * Note: sizeof is used with uint64_t instead of endpointcount
     * because endpointcount is of size_t type which is not compatible
     * type for trasporting across SoCs.
     */
    u64Mul(sizeof(LwSciBufIpcTopoId), tmpIpcRoute->endpointCount, &tmpSize, &mulStatus);
    u64Add(sizeof(uint64_t), tmpSize, &exportSize, &addStatus);

    if (OP_SUCCESS != (mulStatus & addStatus)) {
        LWSCI_ERR_STR("Buffer overflow");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0U != ipcEndpoint) {
        u64Add(sizeof(LwSciBufIpcTopoId), exportSize, &exportSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            LWSCI_ERR_STR("Buffer overflow");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    LWSCI_INFO("Output: ExportSize : %"PRIu64"\n", exportSize);

ret:
    LWSCI_FNEXIT("");
    return (exportSize);
}

size_t LwSciBufIpcTableExportSize(
    const LwSciBufIpcTable* const ipcTable,
    LwSciIpcEndpoint ipcEndpoint)
{
    size_t exportSize = 0, lwrrIdx = 0;
    const LwSciBufIpcTableEntry* ipcTableEntry;
    size_t tmpSize = 0U;
    uint8_t addStatus = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;

    LWSCI_FNENTRY("");
    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Cannot compute size of null ipc table.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs: ipcTable: %p ipcEndpoint %"PRIu64"\n",
            ipcTable, ipcEndpoint);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    exportSize = ADJUSTED_SIZEOF(LwSciBufIpcTableExportHeader);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (lwrrIdx = 0; lwrrIdx < ipcTable->validEntryCount; lwrrIdx++) {
        ipcTableEntry = &(ipcTable->ipcTableEntryArr[lwrrIdx]);

        if (0U == ipcTableEntry->entryExportSize) {
            /* nothing to export */
            continue;
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        u64Add(ADJUSTED_SIZEOF(LwSciBufIpcTableEntryExportHeader), ipcTableEntry->entryExportSize, &tmpSize, &addStatus);
        u64Add(exportSize, tmpSize, &exportSize, &addStatus2);
        if (OP_SUCCESS != (addStatus & addStatus2)) {
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (0U != ipcEndpoint) {
            u64Add(exportSize, sizeof(LwSciBufIpcTopoId), &exportSize,
                &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("Buffer overflow");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }

    LWSCI_INFO("Output: ExportSize : %"PRIu64"\n", exportSize);
ret:
    LWSCI_FNEXIT("");
    return (exportSize);
}

LwSciError LwSciBufCreateIpcTable(
    size_t entryCount,
    LwSciBufIpcTable** outIpcTable)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTableEntry* ipcTableEntryArr = NULL;
    LwSciBufIpcTable* ipcTable = NULL;
    uint64_t index = 0U;

    LWSCI_FNENTRY("");

    if ((0U == entryCount) || (NULL == outIpcTable)) {
        LWSCI_ERR_STR("Bad input parameters for IPC table creation. \n");
        LWSCI_ERR_ULONG("Entry count: \n", entryCount);
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - Entry count: %"PRIu64", IPC Table: %p\n",
        entryCount, outIpcTable);

    *outIpcTable = NULL;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    ipcTableEntryArr = LwSciCommonCalloc(entryCount, sizeof(*ipcTableEntryArr));
    if (NULL == ipcTableEntryArr) {
        LWSCI_ERR_STR("Failed to allocate ipcTable Entry array. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    ipcTable = LwSciCommonCalloc(1, sizeof(*ipcTable));
    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Failed to allocate ipcTable. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto entry_free;
    }

    ipcTable->ipcTableEntryArr = ipcTableEntryArr;
    ipcTable->allocEntryCount = entryCount;
    ipcTable->validEntryCount = 0;

    for (index = 0U; index < entryCount; index++) {
        lwListInit(&ipcTable->ipcTableEntryArr[index].ipcAttrEntryHead);
    }

    *outIpcTable = ipcTable;

    LWSCI_INFO("Outputs - Ipc Entry Arr: %p, ipcTable: %p \n",
         ipcTable->ipcTableEntryArr, ipcTable);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

entry_free:
    LwSciCommonFree(ipcTableEntryArr);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static void LwSciBufIpcAddAttributeToTableParamCheck(
    LwSciBufIpcTable* const ipcTable,
    uint64_t index,
    size_t len,
    const void* value)
{
    if ((NULL == ipcTable) || (0U == len) || (NULL == value) ||
        (ipcTable->allocEntryCount <= index)) {
        LWSCI_ERR_STR("Bad input parameters for adding IPC table entry. \n");
        LWSCI_ERR_ULONG("Value Len: \n", len);
        LwSciCommonPanic();
    }

}

LwSciError LwSciBufIpcAddRouteToTable(
    LwSciBufIpcTable* ipcTable,
    const LwSciBufIpcRoute* ipcRoute,
    uint64_t index)
{
    LwSciError err = LwSciError_Success;
    LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    LwSciBufIpcRoute nullIpcRoute = {};
    const LwSciBufIpcRoute* tmpIpcRoute = NULL;

    LWSCI_FNENTRY("");

    if ((NULL == ipcTable) || (ipcTable->allocEntryCount <= index)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufIpcAddRouteToTable");
        LwSciCommonPanic();
    }

    (void)memset(&nullIpcRoute, 0x0, sizeof(nullIpcRoute));

    if (NULL == ipcRoute) {
        tmpIpcRoute = &nullIpcRoute;
    } else {
        tmpIpcRoute = ipcRoute;
    }

    ipcTableEntry = &(ipcTable->ipcTableEntryArr[index]);
    if (NULL != ipcTableEntry->ipcRoute) {
        /* The route is already set at given index. This should not happen
         * since we set route at given index in the table only once.
         */
        LwSciCommonPanic();
    }

    err = LwSciBufIpcTableAddIpcRoute(&ipcTableEntry->ipcRoute, tmpIpcRoute);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to add new IPC Route into IPC table.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    ipcTableEntry->entryExportSize =
        LwSciBufIpcRouteExportSize(tmpIpcRoute, 0U);
    ipcTable->validEntryCount++;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufIpcAddAttributeToTable(
    LwSciBufIpcTable* ipcTable,
    uint64_t index,
    uint32_t attrKey,
    uint64_t len,
    const void* value)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    LwSciBufIpcTableAttrData *data = NULL;
    uint8_t status = OP_FAIL;
    size_t tmpExportSize = 0U;

    LWSCI_FNENTRY("");

    LwSciBufIpcAddAttributeToTableParamCheck(ipcTable, index, len, value);

    LWSCI_INFO("Inputs - IpcTable: %p, index: %"PRIu64", AttrKey: %"PRIu32
                ", Len: %"PRIu64", Value: %p",
               ipcTable, index, attrKey, len, value);

    ipcTableEntry = &(ipcTable->ipcTableEntryArr[index]);
    if (NULL == ipcTableEntry->ipcRoute) {
        /* This should not happen since we should have added IPC routes in the
         * table before adding attributes.
         */
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    lwListForEachEntry(data, &(ipcTableEntry->ipcAttrEntryHead),
        listEntry) {
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        if (data->key == attrKey) {
            /* The attribute is already set for the IPC route and caller
             * is trying to set the same attribute again. This should not
             * happen since IPC route contains the attributes for
             * unreconciled list to which the IPC route belongs and the
             * unreconciled list wont have the same attribute set twice
             * (since we dont allow setting duplicate attributes).
             */
            LwSciCommonPanic();
        }
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    data = LwSciCommonCalloc(1, sizeof(*data));
    if (NULL == data) {
        LWSCI_ERR_STR("Failed to allocate IPC table entry for the attr-key.\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    data->key = attrKey;
    data->value = LwSciCommonCalloc(1, len);
    if (NULL == data->value) {
        LWSCI_ERR_STR("Failed to allocate AttrKey value for IPC table.\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_data;
    }
    LwSciCommonMemcpyS(data->value, len, value, len);
    data->len = len;

    sciErr = LwSciBufIpcTableAttrDataExportSize(&tmpExportSize, data);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_data_value;
    }
    u64Add(ipcTableEntry->entryExportSize, tmpExportSize,
        &ipcTableEntry->entryExportSize, &status);
    if (OP_SUCCESS != status) {
        sciErr = LwSciError_Overflow;
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_data_value;
    }
    lwListAppend(&data->listEntry, &ipcTableEntry->ipcAttrEntryHead);

    LWSCI_INFO("Data: %p for attrKey: %"PRIu32"\n", data, attrKey);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_data_value:
    LwSciCommonFree(data->value);

free_data:
    LwSciCommonFree(data);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufInitIpcTableIter(
    const LwSciBufIpcTable* inputIpcTable,
    LwSciBufIpcTableIter** outIpcIter)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTableIter* ipcIter = NULL;

    LWSCI_FNENTRY("");
    if ((NULL == inputIpcTable) || (NULL == outIpcIter)) {
        /* NULL ipcEndpoint is acceptable and can be used
         * to iterate over all entries of IPC table.
         */
        LWSCI_ERR_STR("Invalid inputs for Initializing IPC Table iterator. \n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - IPC Table : %p IPC Endpoint: %p Table Iterator: %p\n",
          inputIpcTable, outIpcIter);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    ipcIter = LwSciCommonCalloc(1, sizeof(*ipcIter));
    if (NULL == ipcIter) {
        LWSCI_ERR_STR("Failed to allocate ipc Iterator. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    ipcIter->ipcTable = inputIpcTable;
    ipcIter->ipcRoute = NULL;
    ipcIter->lwrrMatchEntryIdx = LWSCIBUF_ILWALID_IPCTABLE_IDX;
    *outIpcIter = ipcIter;

    LWSCI_INFO("Outputs - Initialized IpcIter : %p\n", *outIpcIter);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufIpcIterLwrrGetAttrKey(
    const LwSciBufIpcTableIter* ipcIter,
    uint32_t attrKey,
    size_t* len,
    const void** value)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufIpcTable* ipcTable = NULL;
    const LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    const LwSciBufIpcTableAttrData *data = NULL;
    bool matchFound = false;

    LWSCI_FNENTRY("");

    if ((NULL == ipcIter) || (NULL == len) || (NULL == value)) {
        LWSCI_ERR_STR("Invalid input to get AttrKey for Current Iterator index.\n");
        LwSciCommonPanic();
    }

    if (LWSCIBUF_ILWALID_IPCTABLE_IDX == ipcIter->lwrrMatchEntryIdx) {
        LWSCI_ERR_STR("Iterator is at invalid index. \n");
        sciErr = LwSciError_IlwalidOperation;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *len = 0;
    *value = NULL;
    ipcTable = ipcIter->ipcTable;
    ipcTableEntry = &(ipcTable->ipcTableEntryArr[ipcIter->lwrrMatchEntryIdx]);
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    lwListForEachEntry(data, &(ipcTableEntry->ipcAttrEntryHead), listEntry) {
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        if (data->key == attrKey) {
            matchFound = true;
            break;
        }
    }
    if (matchFound) {
        *len = data->len;
        *value = data->value;
    }
    LWSCI_INFO("Outputs- Length: %"PRIu64" Value: %p\n", *len, *value);
    if (NULL != *value) {
        LWSCI_INFO("Test actual value of key: %d",*(const uint64_t *)*value);
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

bool LwSciBufIpcTableIterNext(
    LwSciBufIpcTableIter* ipcIter)
{
    const LwSciBufIpcTable* ipcTable = NULL;
    const LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    bool foundNext = false;

    LWSCI_FNENTRY("");

    if (NULL == ipcIter) {
        LWSCI_ERR_STR("Cannot iterate NULL Iterator. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - ipcIter %p", ipcIter);

    if (LWSCIBUF_ILWALID_IPCTABLE_IDX == ipcIter->lwrrMatchEntryIdx) {
        ipcIter->lwrrMatchEntryIdx = 0U;
    } else {
        ipcIter->lwrrMatchEntryIdx++;
    }

    ipcTable = ipcIter->ipcTable;

    if (ipcIter->lwrrMatchEntryIdx >= ipcTable->validEntryCount) {
        foundNext = false;
        ipcIter->lwrrMatchEntryIdx = LWSCIBUF_ILWALID_IPCTABLE_IDX;
        goto ret;
    }

    ipcTableEntry =
        &(ipcTable->ipcTableEntryArr[ipcIter->lwrrMatchEntryIdx]);

    ipcIter->ipcRoute = ipcTableEntry->ipcRoute;
    foundNext = true;

    LWSCI_INFO("Outputs - foundNext entry %s", (foundNext == true)?"Yes":"No");

ret:
    LWSCI_FNEXIT("");
    return (foundNext);
}

#if (LW_IS_SAFETY == 0)
void LwSciBufPrintIpcTable(
    const LwSciBufIpcTable* ipcTable)
{
    size_t lwrrIdx = 0;
    const LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    const LwSciBufIpcTableAttrData* data = NULL;

    LWSCI_FNENTRY("");
    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Cannot print NULL ipc Table. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("\n IPC Table (Entry Count- Allocated: %"PRIu64
               ", Valid: %"PRIu64") \n",
         ipcTable->allocEntryCount, ipcTable->validEntryCount);

    for (lwrrIdx = 0; lwrrIdx < ipcTable->validEntryCount; lwrrIdx++) {
        ipcTableEntry = &ipcTable->ipcTableEntryArr[lwrrIdx];
        LWSCI_INFO("   Entry %"PRIu64" EntryExportSize: %"PRIu64"\n",
             lwrrIdx, ipcTableEntry->entryExportSize);
        LWSCI_INFO("   IPC Route: \n");
        LwSciBufPrintIpcRoute(ipcTableEntry->ipcRoute);
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
        lwListForEachEntry(data, &(ipcTableEntry->ipcAttrEntryHead),
            listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWSCI_INFO("   Key: %"PRIu64" Len: %"PRIu64" Value: %"PRIX64"\n",
                 data->key, data->len, *(uint64_t *)data->value);
        }
        LWSCI_INFO("\n\n\n");
    }

ret:
    LWSCI_FNEXIT("");
}
#endif

static void LwSciBufIpcTableExportParamCheck(
    const LwSciBufIpcTable* ipcTable,
    void** desc,
    const size_t* len)
{
    if ((NULL == ipcTable) || (NULL == desc) || (NULL == len)) {
        LWSCI_ERR_STR("Invalid inputs for exporting Ipc Table. \n");
        LwSciCommonPanic();
    }

    if (0U == ipcTable->validEntryCount) {
        LWSCI_ERR_STR("Cannot export IpcTable with zero entries.\n");
        LwSciCommonPanic();
    }
}


LwSciError LwSciBufIpcTableExport(
    const LwSciBufIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    void** desc,
    size_t* len)
{
    LwSciError sciErr = LwSciError_Success;
    size_t allocSize = 0;
    size_t lwrrIdx = 0;
    size_t remSize = 0;
    size_t copySize = 0;
    uint8_t* exportDesc = NULL;
    const LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    LwSciBufIpcTableExportHeader* hdr = NULL;
    LwSciBufIpcTableEntryExportHeader* entryHdr = NULL;
    const LwSciBufIpcTableAttrData* data = NULL;
    const LwSciBufIpcRoute* ipcRoute = NULL;
    uint8_t status = OP_FAIL;
    uint8_t status2 = OP_FAIL;

    LWSCI_FNENTRY("");

    LwSciBufIpcTableExportParamCheck(ipcTable, desc, len);

    *desc = NULL;
    *len = 0;
    allocSize = LwSciBufIpcTableExportSize(ipcTable, ipcEndpoint);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (ADJUSTED_SIZEOF(LwSciBufIpcTableExportHeader) == allocSize) {
        LWSCI_INFO("No need to export Ipc Table with Zero entries.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    exportDesc = LwSciCommonCalloc(1, allocSize);
    if (NULL == exportDesc) {
        LWSCI_ERR_STR("Failed to allocate buffer for exporting Ipc Table. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    hdr = (LwSciBufIpcTableExportHeader *)(void *)exportDesc;
    hdr->entryCount = 0;
    hdr->ipcEndpointSize = sizeof(LwSciBufIpcTopoId);
    hdr->totalSize = allocSize;
    entryHdr = hdr->entryStart;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    u64Sub(allocSize, ADJUSTED_SIZEOF(*hdr), &remSize, &status);
    if (OP_SUCCESS != status) {
        sciErr = LwSciError_Overflow;
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_export_desc;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (lwrrIdx = 0; lwrrIdx < ipcTable->validEntryCount; lwrrIdx++) {
        ipcTableEntry = &(ipcTable->ipcTableEntryArr[lwrrIdx]);

        if (0U == ipcTableEntry->entryExportSize) {
            /* nothing to export */
            continue;
        }

        ipcRoute = ipcTableEntry->ipcRoute;

        /* There is an entry found that needs to be exported. */
        entryHdr->entrySize = 0;
        entryHdr->keyCount = 0;

        u64Add(hdr->entryCount, 1, &hdr->entryCount, &status);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        u64Sub(remSize, ADJUSTED_SIZEOF(*entryHdr), &remSize, &status2);
        if (OP_SUCCESS != (status & status2)) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_export_desc;
        }
        exportDesc = entryHdr->desc;

        sciErr = LwSciBufIpcRouteExportInternal(ipcRoute, exportDesc, remSize,
                    &copySize, ipcEndpoint);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_ULONG("Unable to export Ipc Route for entry ", lwrrIdx);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_export_desc;
        }

        u64Sub(remSize, copySize, &remSize, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_export_desc;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_4), "LwSciBuf-ADV-MISRAC2012-013")
        exportDesc = (uint8_t*)((uintptr_t)exportDesc + copySize);
        entryHdr->entrySize += copySize;

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        lwListForEachEntry(data, &(ipcTableEntry->ipcAttrEntryHead),
            listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 15_4))
            sciErr = LwSciBufIpcTableAttrDataExport(data, exportDesc, remSize,
                                                    &copySize);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_UINT("Unable to export Attrdata for AttrKey ", data->key);
                LWSCI_ERR_ULONG("of entry \n", lwrrIdx);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_export_desc;
            }

            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_4), "LwSciBuf-ADV-MISRAC2012-013")
            exportDesc = (uint8_t*)((uintptr_t)exportDesc + copySize);

            u64Sub(remSize, copySize, &remSize, &status);
            u64Add(entryHdr->entrySize, copySize, &entryHdr->entrySize, &status2);
            if (OP_SUCCESS != (status & status2)) {
                sciErr = LwSciError_Overflow;
                LWSCI_ERR_STR("Buffer overflow\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_export_desc;
            }
            entryHdr->keyCount++;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        entryHdr = (LwSciBufIpcTableEntryExportHeader *) (void *)exportDesc;
    }

    *desc = hdr;
    *len = allocSize;

    LWSCI_INFO("Output - Export Desc buffer %p Desc Length %"PRIu64"\n",
         *desc, *len);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_export_desc:
    LwSciCommonFree(exportDesc);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufIpcTableImport(
    const void* desc,
    size_t len,
    LwSciBufIpcTable** ipcTable,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTable* newTable = NULL;
    LwSciBufIpcTableEntry* ipcTableEntry = NULL;
    const LwSciBufIpcTableExportHeader* hdr = NULL;
    const LwSciBufIpcTableEntryExportHeader* entryHdr = NULL;
    LwSciBufIpcTableAttrData* data = NULL;
    size_t lwrrIdx = 0;
    size_t lenAfterImports = 0;
    size_t ipcRouteLen = 0;
    size_t ipcDataLen = 0;
    size_t keyIdx = 0;
    uint8_t status = OP_FAIL;

    LWSCI_FNENTRY("");
    if ((NULL == desc) || (len < sizeof(uint64_t)) || (NULL == ipcTable)) {
        LWSCI_ERR_STR("Invalid input parameters for Importing Ipc Table. \n");
        LWSCI_ERR_ULONG("Length \n", len);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - Desc %p Length %"PRIu64"Ipc Table Ptr %p\n",
         desc, len, ipcTable);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    hdr = desc;
    if (hdr->totalSize > len) {
        LWSCI_ERR_STR("Cannot import IPC Table with descriptor of less size.");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0U == hdr->entryCount) {
        LWSCI_ERR_STR("Cannot import IPC Table with zero entries.");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (sizeof(LwSciBufIpcTopoId) != hdr->ipcEndpointSize) {
        LWSCI_ERR_STR("Cannot import Ipc Table. Ipc Endpoint Size mismatch.");
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufCreateIpcTable(hdr->entryCount, &newTable);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to create Ipc table while importing Ipc Table.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    lenAfterImports = len - ADJUSTED_SIZEOF(*hdr);

    entryHdr = hdr->entryStart;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (lwrrIdx = 0; lwrrIdx < hdr->entryCount; lwrrIdx++) {
        const uint8_t* dataStart = NULL;
        size_t totalImportedLen = 0U;

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        u64Sub(lenAfterImports, ADJUSTED_SIZEOF(*entryHdr), &lenAfterImports, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        ipcTableEntry = &(newTable->ipcTableEntryArr[lwrrIdx]);

        sciErr = LwSciBufIpcRouteImportInternal(ipcEndpoint, entryHdr->desc,
            lenAfterImports, &ipcRouteLen, false, &ipcTableEntry->ipcRoute);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to import Route while importing Ipc Table.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_table;
        }

        u64Add(totalImportedLen, ipcRouteLen, &totalImportedLen, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_table;
        }

        lwListInit(&ipcTableEntry->ipcAttrEntryHead);
        newTable->validEntryCount++;

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        dataStart = (entryHdr->desc + ipcRouteLen);
        lenAfterImports -= ipcRouteLen;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
        for (keyIdx = 0;
            keyIdx < entryHdr->keyCount;
            keyIdx++) {
            sciErr = LwSciBufIpcTableAttrDataImport(dataStart,
                          lenAfterImports, &ipcDataLen, &data);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Unable to import Data while importing IpcTable.\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_table;
            }
            lwListAppend(&data->listEntry, &ipcTableEntry->ipcAttrEntryHead);

            u64Add(totalImportedLen, ipcDataLen, &totalImportedLen, &status);
            if (OP_SUCCESS != status) {
                sciErr = LwSciError_Overflow;
                LWSCI_ERR_STR("Buffer overflow");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_table;
            }

            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            dataStart += ipcDataLen;
            lenAfterImports -= ipcDataLen;

        }

        if (0U != ipcEndpoint) {
            u64Add(totalImportedLen, sizeof(LwSciBufIpcTopoId),
                &totalImportedLen, &status);
            if (OP_SUCCESS != status) {
                sciErr = LwSciError_Overflow;
                LWSCI_ERR_STR("Buffer overflow");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                    "LwSciBuf-ADV-MISRAC2012-015")
                goto free_table;
            }
        }

        ipcTableEntry->entryExportSize = totalImportedLen;
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        entryHdr = (const LwSciBufIpcTableEntryExportHeader *)((const void*)((const uint8_t *)entryHdr+
            ADJUSTED_SIZEOF(LwSciBufIpcTableEntryExportHeader) + entryHdr->entrySize));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    }

    *ipcTable = newTable;
    LWSCI_INFO("Outputs - Imported Table %p with Entry Count %"PRIu64"\n",
                newTable, newTable->validEntryCount);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_table:
    LwSciBufFreeIpcTable(&newTable);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufIpcTableClone(
    const LwSciBufIpcTable* const * srcIpcTableAddr,
    LwSciBufIpcTable** dstIpcTableAddr)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufIpcTable* srcIpcTable = NULL;
    LwSciBufIpcTable* dstIpcTable = NULL;
    size_t lwrrIdx;
    LwSciBufIpcTableEntry* dstIpcTableEntry = NULL;
    const LwSciBufIpcTableAttrData *srcData = NULL;
    LwSciBufIpcTableAttrData *dstData = NULL;
    uint8_t status = OP_FAIL;

    LWSCI_FNENTRY("");

    if ((NULL == srcIpcTableAddr) || (NULL == dstIpcTableAddr)) {
        LWSCI_ERR_STR("Invalid parameters to clone IPC Table.\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - srcIpcTableptr %p dstIpcTableptr %p \n",
                           srcIpcTableAddr, dstIpcTableAddr);

    srcIpcTable = *srcIpcTableAddr;
    if ((NULL == srcIpcTable) || (0U == srcIpcTable->validEntryCount)) {
        LWSCI_INFO("Skipping cloning of NULL Ipc Table.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    sciErr = LwSciBufCreateIpcTable(srcIpcTable->allocEntryCount, &dstIpcTable);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to create Ipc table while cloning Ipc Table.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (lwrrIdx = 0; lwrrIdx < srcIpcTable->validEntryCount; lwrrIdx++) {
        const LwSciBufIpcTableEntry* srcIpcTableEntry =
            &srcIpcTable->ipcTableEntryArr[lwrrIdx];
        const LwSciBufIpcRoute* ipcRoute = srcIpcTableEntry->ipcRoute;

        dstIpcTableEntry = &dstIpcTable->ipcTableEntryArr[lwrrIdx];
        sciErr = LwSciBufIpcRouteClone(&ipcRoute,
                              &dstIpcTableEntry->ipcRoute);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to clone Ipc Route on dstIpcTable.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ret;
        }
        dstIpcTableEntry->entryExportSize = srcIpcTableEntry->entryExportSize;
        lwListInit(&dstIpcTableEntry->ipcAttrEntryHead);
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        lwListForEachEntry(srcData, &(srcIpcTableEntry->ipcAttrEntryHead),
                            listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 15_4))
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            dstData = LwSciCommonCalloc(1, sizeof(*dstData));
            if (NULL == dstData) {
                LWSCI_ERR_STR("Unable to allocate data while cloning IPC Table.\n");
                sciErr = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_ret;
            }
            dstData->value = LwSciCommonCalloc(1, srcData->len);
            if (NULL == dstData->value) {
                LWSCI_ERR_STR("Alloc failed for value while cloning IPC Table.\n");
                sciErr = LwSciError_InsufficientMemory;
                LwSciCommonFree(dstData);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_ret;
            }
            LwSciCommonMemcpyS(dstData->value, srcData->len, srcData->value,
                srcData->len);
            dstData->key = srcData->key;
            dstData->len = srcData->len;
            lwListAppend(&dstData->listEntry,
                         &dstIpcTableEntry->ipcAttrEntryHead);
        }
        u64Add(dstIpcTable->validEntryCount, 1, &dstIpcTable->validEntryCount, &status);
        if (OP_SUCCESS != status) {
            sciErr = LwSciError_Overflow;
            LWSCI_ERR_STR("Buffer overflow\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

output:
    *dstIpcTableAddr = dstIpcTable;
    LWSCI_INFO("Outputs - SrcIpcTable %p DstIpcTable %p\n",
                srcIpcTable, dstIpcTable);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_ret:
    LwSciBufFreeIpcTable(&dstIpcTable);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

void LwSciBufFreeIpcTable(
    LwSciBufIpcTable* const * valPtr)
{
    LwSciBufIpcTableAttrData* data = NULL;
    LwSciBufIpcTableAttrData* tmp = NULL;
    LwSciBufIpcTable* ipcTable = NULL;
    size_t lwrrIdx = 0;

    LWSCI_FNENTRY("");

    if (NULL == valPtr) {
        LWSCI_INFO("Trying to free NULL Ipc table.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    ipcTable = *valPtr;

    if (NULL == ipcTable) {
        /* This is valid since we may not have exported the Attribute List, so
         * there may not be an IPC Table. As we do not check whether an IPC
         * Table has been allocated or not, calling free on a empty IPC Table
         * should be a no-op. */
        LWSCI_INFO("Trying to free NULL Ipc table.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    for (lwrrIdx = 0; lwrrIdx < ipcTable->validEntryCount; lwrrIdx++) {
        LwSciBufIpcTableEntry* ipcTableEntry = &ipcTable->ipcTableEntryArr[lwrrIdx];

        LwSciBufFreeIpcRoute(&ipcTableEntry->ipcRoute);
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
        lwListForEachEntry_safe(data, tmp, &(ipcTableEntry->ipcAttrEntryHead), listEntry) {
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
            lwListDel(&data->listEntry);
            LwSciCommonFree(data->value);
            LwSciCommonFree(data);

            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
            lwListForEachEntryEnd_safe(data, tmp, listEntry);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
        }
    }

    LwSciCommonFree(ipcTable->ipcTableEntryArr);
    LwSciCommonFree(ipcTable);

ret:
    LWSCI_FNEXIT("");
}

void LwSciBufFreeIpcIter(
    LwSciBufIpcTableIter* ipcIter)
{
    LWSCI_FNENTRY("");
    LwSciCommonFree(ipcIter);
    LWSCI_FNEXIT("");
}

/******************************************************
 *    All IpcRoute related Public Functions           *
 ******************************************************/

LwSciError LwSciBufIpcRouteExport(
    const LwSciBufIpcRoute* ipcRoute,
    void** desc,
    size_t* len,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufIpcRoute* tmpIpcRoute;
    size_t allocSize = 0, copySize = 0;

    /* Exporting NULL IPC route is allowed.
     * Because when attrlist is created, there is no
     * known ipc route.
     */
    LwSciBufIpcRoute nullIpcRoute = { NULL, 0 };

    LWSCI_FNENTRY("");
    if ((NULL == desc) || (NULL == len)) {
        LWSCI_ERR_STR("Invalid input parameters to export IPC Route. \n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs- Desc ptr: %p, Length ptr: %p, IpcRoute: \n",
                   desc, len);

    if (NULL == ipcRoute) {
        LWSCI_INFO(" Export NULL ipcRoute.\n");
        tmpIpcRoute = &nullIpcRoute;
    } else {
        tmpIpcRoute = ipcRoute;
        LwSciBufPrintIpcRoute(tmpIpcRoute);
    }

    allocSize = LwSciBufIpcRouteExportSize(tmpIpcRoute, ipcEndpoint);
    *desc = LwSciCommonCalloc(1, allocSize);
    if (NULL == *desc) {
        LWSCI_ERR_STR("Failed to allocate IPC Route export descriptor. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufIpcRouteExportInternal(tmpIpcRoute, *desc, allocSize,
                                            &copySize, ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_desc;
    }
    *len = copySize;
    LWSCI_INFO("Outputs - Export descriptor: %p Length: %"PRIu64"\n",
               *desc, *len);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_desc:
    LwSciCommonFree(desc);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufIpcRouteImport(
    LwSciIpcEndpoint ipcEndpoint,
    const void* desc,
    size_t len,
    LwSciBufIpcRoute** ipcRoute)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == ipcRoute) || (NULL == desc) || (len < sizeof(uint64_t))) {
        LWSCI_ERR_STR("Invalid input parameters to import IPC Route. \n");
        LWSCI_ERR_ULONG("Length: \n", len);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs- Desc ptr: %p, Length : %"PRIu64", IpcRoute: %p \n",
                   desc, len, ipcRoute);

    sciErr = LwSciBufIpcRouteImportInternal(ipcEndpoint, desc, len, NULL,
                                            true, ipcRoute);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to import IPC Route. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Output: Imported IPC route: %p", *ipcRoute);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LwSciError LwSciBufIpcRouteClone(
    const LwSciBufIpcRoute* const * srcIpcRouteAddr,
    LwSciBufIpcRoute** dstIpcRouteAddr)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufIpcRoute* srcIpcRoute = NULL;
    LwSciBufIpcRoute* dstIpcRoute = NULL;
    size_t len = 0;

    LWSCI_FNENTRY("");

    if ((NULL == srcIpcRouteAddr) || (NULL == dstIpcRouteAddr)) {
        LWSCI_ERR_STR("Invalid parameters to clone IPC route.\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("Inputs - srcIpcRouteptr %p dstIpcRouteptr %p \n",
                           srcIpcRouteAddr, dstIpcRouteAddr);

    srcIpcRoute = *srcIpcRouteAddr;
    if (NULL == srcIpcRoute) {
        *dstIpcRouteAddr = NULL;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    dstIpcRoute = LwSciCommonCalloc(1, sizeof(*dstIpcRoute));
    if (NULL == dstIpcRoute) {
        LWSCI_ERR_STR("Failed to allocate memory for dstIpcRoute. \n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    dstIpcRoute->endpointCount = srcIpcRoute->endpointCount;
    if (0U == dstIpcRoute->endpointCount) {
        dstIpcRoute->ipcEndpointList = NULL;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    len = sizeof(LwSciBufIpcTopoId) * dstIpcRoute->endpointCount;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    dstIpcRoute->ipcEndpointList = LwSciCommonCalloc(1, len);
    if (NULL == dstIpcRoute->ipcEndpointList) {
        LWSCI_ERR_STR("Failed to allocate IPC Endpoint list while Cloning\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_dstIpcRoute;
    }

    LwSciCommonMemcpyS(dstIpcRoute->ipcEndpointList, len,
                      srcIpcRoute->ipcEndpointList, len);

output:
    *dstIpcRouteAddr = dstIpcRoute;

    LWSCI_INFO("Outputs: \n");
    LwSciBufPrintIpcRoute(srcIpcRoute);
    LwSciBufPrintIpcRoute(dstIpcRoute);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_dstIpcRoute:
    LwSciCommonFree(dstIpcRoute);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

void LwSciBufFreeIpcRoute(
    LwSciBufIpcRoute* const * valPtr)
{
    LwSciBufIpcRoute* ipcRoute = NULL;

    LWSCI_FNENTRY("");

    if (NULL == valPtr) {
        LWSCI_INFO("Trying to Free NULL Ipc Route. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    ipcRoute = *valPtr;

    if (NULL == ipcRoute) {
        LWSCI_INFO("Trying to Free NULL Ipc Route. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Input: ipcRoute: \n");
    LwSciBufPrintIpcRoute(ipcRoute);

    LwSciCommonFree(ipcRoute->ipcEndpointList);
    LwSciCommonFree(ipcRoute);

ret:
    LWSCI_FNEXIT("");
}

void LwSciBufIpcRouteMatchAffinity(
    const LwSciBufIpcRoute* ipcRoute,
    LwSciBufIpcRouteAffinity routeAffinity,
    LwSciIpcEndpoint ipcEndpoint,
    bool localPeer,
    bool* isMatch)
{
    LwSciError err = LwSciError_Success;
    LwSciBufIpcTopoId tmpTopoId = {};

    LWSCI_FNENTRY("");

    if ((LwSciBufIpcRoute_Max <= routeAffinity) || (NULL == isMatch) ||
        ((false == localPeer) && (0U == ipcEndpoint))) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufIpcRouteMatchAffinity()");
        LwSciCommonPanic();
    }

    *isMatch = false;

    if (false == localPeer) {
        err = getTopoId(ipcEndpoint, &tmpTopoId);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("getTopoId() failed.");
            LwSciCommonPanic();
        }
    }

    switch(routeAffinity) {
        case LwSciBufIpcRoute_AffinityNone:
        {
            /* No need to check the IPC route for affinity */
            *isMatch = true;
            break;
        }

        case LwSciBufIpcRoute_OwnerAffinity:
        {
            if (true == localPeer) {
                /* For LwSciBufIpcRoute_OwnerAffinity, the IPC route belongs to
                 * peer if the IPC route is NULL (meaning that the unreconciled
                 * list is created by the peer) OR if the first endpoint and last
                 * endpoint in the IPC route match.
                 */
                *isMatch = ((NULL == ipcRoute) ||
                    (NULL == ipcRoute->ipcEndpointList) ||
                    (0 == LwSciCommonMemcmp(&ipcRoute->ipcEndpointList[0],
                        &ipcRoute->ipcEndpointList[ipcRoute->endpointCount - 1],
                        sizeof(ipcRoute->ipcEndpointList[0]))));
            } else {
                *isMatch = ((2U <= ipcRoute->endpointCount) &&
                    ((LWSCIIPC_SELF_SOCID ==
                    ipcRoute->ipcEndpointList[1].topoId.SocId) &&
                    (LWSCIIPC_SELF_VMID ==
                    ipcRoute->ipcEndpointList[1].topoId.VmId) &&
                    (tmpTopoId.vuId == ipcRoute->ipcEndpointList[1].vuId)));
            }

            break;
        }

        case LwSciBufIpcRoute_SocAffinity:
        {
            if (true == localPeer) {
                /* For LwSciBufIpcRoute_SocAffinity, the IPC route belongs to
                 * SoC if the IPC route is NULL (meaning that the unreconciled
                 * list is created by the peer in this SoC) OR the SocId of the
                 * first endpoint is LWSCIIPC_SELF_SOCID (meaning that the IPC route
                 * originated from this SoC).
                 */
                *isMatch = ((NULL == ipcRoute) ||
                            (NULL == ipcRoute->ipcEndpointList) ||
                            (LWSCIIPC_SELF_SOCID ==
                                ipcRoute->ipcEndpointList[0].topoId.SocId));
            } else {
                *isMatch = ((2U <= ipcRoute->endpointCount) &&
                    ((LWSCIIPC_SELF_SOCID ==
                    ipcRoute->ipcEndpointList[1].topoId.SocId) &&
                    (LWSCIIPC_SELF_VMID ==
                    ipcRoute->ipcEndpointList[1].topoId.VmId) &&
                    (tmpTopoId.vuId == ipcRoute->ipcEndpointList[1].vuId) &&
                    (LWSCIIPC_SELF_SOCID !=
                    ipcRoute->ipcEndpointList[0].topoId.SocId)));
            }

            break;
        }

        case LwSciBufIpcRoute_RouteAffinity:
        {
            if (true == localPeer) {
                *isMatch = true;
            } else {
                size_t index = 0U;

                for (index = 0U; index < ipcRoute->endpointCount; index++) {
                    if ((LWSCIIPC_SELF_SOCID ==
                        ipcRoute->ipcEndpointList[index].topoId.SocId) &&
                        (LWSCIIPC_SELF_VMID ==
                        ipcRoute->ipcEndpointList[index].topoId.VmId) &&
                        (tmpTopoId.vuId ==
                        ipcRoute->ipcEndpointList[index].vuId)) {
                        *isMatch = true;
                    }
                }
            }

            break;
        }

        default:
        {
            /* Unsupported case */
            LwSciCommonPanic();
        }
    }

    LWSCI_FNEXIT("");
}

void LwSciBufIpcGetRouteFromIter(
    const LwSciBufIpcTableIter* iter,
    const LwSciBufIpcRoute** ipcRoute)
{
    if ((NULL == iter) || (NULL == ipcRoute)) {
        LwSciCommonPanic();
    }

    LWSCI_FNENTRY("");

    *ipcRoute = iter->ipcRoute;

    LWSCI_FNEXIT("");
}
