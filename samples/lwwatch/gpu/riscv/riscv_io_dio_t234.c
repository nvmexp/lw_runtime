/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "riscv_io_dio.h"
#include "riscv_prv.h"
#include "t23x/t234/dev_sec_prgnlcl.h"

/* =============================================================================
 * DIO additional macros
 * ========================================================================== */
#define DIO_MAX_WAIT 0xfffffffe

#define LW_PRGNLCL_FALCON_DOC_D0_READ      16:16
#define LW_PRGNLCL_FALCON_DOC_D0_ADDR      15:0
#define LW_PRGNLCL_FALCON_DOC_D1_WDATA     31:0
#define LW_PRGNLCL_FALCON_DIC_D0_RDATA     31:0

#define LW_PRGNLCL_FALCON_DIO_DOC_D0_WDATA 31:0
#define LW_PRGNLCL_FALCON_DIO_DOC_D1_ADDR  31:0
#define LW_PRGNLCL_FALCON_DIO_DOC_D2_READ   0:0
#define LW_PRGNLCL_FALCON_DIO_DIC_D0_RDATA 31:0

/* =============================================================================
 * DIO parsing functions given register value and DIO type
 * ========================================================================== */
LwBool _dioParseDocCtrlEmpty(DIO_PORT dioPort, LwU32 docCtrl);
LwU32 _dioParseDocCtrlFinished(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 docCtrl);
LwU32 _dioParseDocCtrlError(DIO_PORT dioPort, LwU32 docCtrl);
LwU32 _dioParseDicCtrlCount(DIO_PORT dioPort, LwU32 dicCtrl);

/* =============================================================================
 * DIO core helper functions
 * ========================================================================== */
LwU32 _dioReadDocCtrl(DIO_PORT dioPort);
LwU32 _dioReadDicCtrl(DIO_PORT dioPort);
LW_STATUS _dioReadWrite(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 addr, LwU32 *pData);
LW_STATUS _dioWaitForDocEmpty(DIO_PORT dioPort, DIO_OPERATION dioOp);
LW_STATUS _dioWaitForOperationComplete(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 *pData);
LwBool _dioIfTimeout(LwU32 *pTimerCount);
LW_STATUS _dioReset(DIO_PORT dioPort);
LW_STATUS _dioPopDataAndClear(DIO_PORT dioPort, LwU32 *pData);

/* =============================================================================
 * Other helper function implementations
 * ========================================================================== */
LwU32 _localRead(LwU32 address);
void _localWrite(LwU32 address, LwU32 data);

/* =============================================================================
 * DIO API implementations
 * ========================================================================== */
LW_STATUS riscvDioReadWrite_T234(void* pDioPort, void * pDioOp, LwU32 addr, LwU32 *pData)
{
    DIO_PORT dioPort = *((DIO_PORT *) pDioPort);
    DIO_OPERATION dioOp = *((DIO_OPERATION *) pDioOp);
    return _dioReadWrite(dioPort, dioOp, addr, pData);
}

/* =============================================================================
 * Other helper function implementations
 * ========================================================================== */
LwU32 _localRead(LwU32 address)
{
    LwU32 data;
    LW_STATUS status = riscvIcdRdm(address, &data, ICD_WIDTH_32);
    if (status == LW_OK)
    {
        printf("rd [0x%x] = 0x%x\n", address, data);
        return data;
    }
    printf("ERROR running riscvIcdRdm.\n");
    return 0;
}

void _localWrite(LwU32 address, LwU32 data)
{
    LW_STATUS status = riscvIcdWdm(address, data, ICD_WIDTH_32);
    if (status != LW_OK)
    {
        printf("ERROR running riscvIcdWdm.\n");
    }
    printf("wr [0x%x] = 0x%x\n", address, data);
}

/* =============================================================================
 * DIO core helper functions implementation
 * ========================================================================== */
LwU32 _dioReadDocCtrl(DIO_PORT dioPort)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            return _localRead(LW_PRGNLCL_FALCON_DOC_CTRL);
        case DIO_TYPE_EXTRA:
            return _localRead(LW_PRGNLCL_FALCON_DIO_DOC_CTRL(dioPort.portIdx));
        default:
            return 0;
    }
}

LwU32 _dioReadDicCtrl(DIO_PORT dioPort)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            return _localRead(LW_PRGNLCL_FALCON_DIC_CTRL);
        case DIO_TYPE_EXTRA:
            return _localRead(LW_PRGNLCL_FALCON_DIO_DIC_CTRL(dioPort.portIdx));
        default:
            return 0;
    }
}

LW_STATUS _dioReadWrite(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 addr, LwU32 *pData)
{
    LW_STATUS status = LW_OK;

    status = _dioWaitForDocEmpty(dioPort, dioOp);
    if (status != LW_OK)
    {
        return status;
    }

    // Sent request by populating DOC interface based on DIO type
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            _localWrite(LW_PRGNLCL_FALCON_DOC_D1, DRF_NUM(_PRGNLCL, _FALCON_DOC_D1, _WDATA, *pData));
            _localWrite(LW_PRGNLCL_FALCON_DOC_D0,
                DRF_NUM(_PRGNLCL, _FALCON_DOC_D0, _READ, (dioOp == DIO_OPERATION_RD)) |
                DRF_NUM(_PRGNLCL, _FALCON_DOC_D0, _ADDR, addr));
            break;
        case DIO_TYPE_EXTRA:
            _localWrite(LW_PRGNLCL_FALCON_DIO_DOC_D2(dioPort.portIdx), DRF_NUM(_PRGNLCL, _FALCON_DIO_DOC_D2, _READ, (dioOp == DIO_OPERATION_RD)));
            _localWrite(LW_PRGNLCL_FALCON_DIO_DOC_D1(dioPort.portIdx), DRF_NUM(_PRGNLCL, _FALCON_DIO_DOC_D1, _ADDR, addr));
            _localWrite(LW_PRGNLCL_FALCON_DIO_DOC_D0(dioPort.portIdx), DRF_NUM(_PRGNLCL, _FALCON_DIO_DOC_D0, _WDATA, *pData));
            break;
        default:
            status = LW_ERR_ILWALID_ARGUMENT;
    }

    if (status != LW_OK)
    {
        return status;
    }

    return _dioWaitForOperationComplete(dioPort, dioOp, pData);
}

/*!
 * @brief    Wait for free entry in DOC
 * @details  The function tries to take a free entry in DOC and exit with no DIO errors.
 *
 * @ref      https://confluence.lwpu.com/display/LW/liblwriscv+DIO+driver#liblwriscvDIOdriver-Processflowdiagram
 */
LW_STATUS _dioWaitForDocEmpty(DIO_PORT dioPort, DIO_OPERATION dioOp)
{
    LwU32 timerCount = 0;
    LwU32 docCtrl = 0;
    LwBool bDocCtrlEmpty = 0;
    do
    {
        docCtrl = _dioReadDocCtrl(dioPort);
        bDocCtrlEmpty = _dioParseDocCtrlEmpty(dioPort, docCtrl);
        if (_dioParseDocCtrlError(dioPort, docCtrl) != 0 ||
            _dioIfTimeout(&timerCount))
        {
            _dioReset(dioPort);
            return LW_ERR_GENERIC;
        }
    } while (!bDocCtrlEmpty);

    return LW_OK;
}

/*!
 * @brief    Wait for operation to complete and get response for read.
 * @details  We make sure no error is caused by the operation.
 *
 * @ref      https://confluence.lwpu.com/display/LW/liblwriscv+DIO+driver#liblwriscvDIOdriver-Processflowdiagram
 */
LW_STATUS _dioWaitForOperationComplete(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 *pData)
{
    LwU32 timerCount = 0;
    LwU32 docCtrl = 0;
    LwU32 docCtrlFinished = 0;
    do
    {
        docCtrl = _dioReadDocCtrl(dioPort);
        docCtrlFinished = _dioParseDocCtrlFinished(dioPort, dioOp, docCtrl);
        if (_dioParseDocCtrlError(dioPort, docCtrl) != 0 ||
            _dioIfTimeout(&timerCount))
        {
            _dioReset(dioPort);
            return LW_ERR_GENERIC;
        }
    } while (!docCtrlFinished);

    if (dioOp == DIO_OPERATION_RD)
    {
        LwU32 dicCtrl = 0;
        LwU32 dicCtrlCount = 0;
        do
        {
            // check dicCtrl once docCtrl indicate operation done
            dicCtrl = _dioReadDicCtrl(dioPort);
            dicCtrlCount = _dioParseDicCtrlCount(dioPort, dicCtrl);
            if (_dioIfTimeout(&timerCount))
            {
                _dioReset(dioPort);
                return LW_ERR_GENERIC;
            }

        } while (dicCtrlCount == 0);

        _dioPopDataAndClear(dioPort, pData);
    }

    return LW_OK;
}

LwBool _dioIfTimeout(LwU32 *pTimerCount)
{
    if (*pTimerCount > DIO_MAX_WAIT)
    {
        return LW_TRUE;
    }
    (*pTimerCount)++;
    return LW_FALSE;
}

LW_STATUS _dioPopDataAndClear(DIO_PORT dioPort, LwU32 *pData)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            _localWrite(LW_PRGNLCL_FALCON_DIC_CTRL, DRF_NUM(_PRGNLCL, _FALCON_DIC_CTRL, _POP, 0x1));
            *pData = _localRead(LW_PRGNLCL_FALCON_DIC_D0);
            // set valid bit to clear the data
            _localWrite(LW_PRGNLCL_FALCON_DIC_CTRL, DRF_NUM(_PRGNLCL, _FALCON_DIC_CTRL, _VALID, 0x1));
            break;
        case DIO_TYPE_EXTRA:
            _localWrite(LW_PRGNLCL_FALCON_DIO_DIC_CTRL(dioPort.portIdx), DRF_NUM(_PRGNLCL, _FALCON_DIO_DIC_CTRL, _POP, 0x1));
            *pData = _localRead(LW_PRGNLCL_FALCON_DIO_DIC_D0(dioPort.portIdx));
            // set valid bit to clear the data
            _localWrite(LW_PRGNLCL_FALCON_DIO_DIC_CTRL(dioPort.portIdx), DRF_NUM(_PRGNLCL, _FALCON_DIO_DIC_CTRL, _VALID, 0x1));
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }
    return LW_OK;
}

/*!
 * @brief  Set clear bit and wait it to be cleared by hw on finish
 */
LW_STATUS _dioReset(DIO_PORT dioPort)
{
    LW_STATUS status = LW_OK;
    LwU32 timerCount = 0;

    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            _localWrite(LW_PRGNLCL_FALCON_DOC_CTRL, DRF_NUM(_PRGNLCL, _FALCON_DOC_CTRL, _RESET, 0x1));

            while (FLD_TEST_DRF_NUM(_PRGNLCL, _FALCON_DOC_CTRL, _RESET, 0x1, _localRead(LW_PRGNLCL_FALCON_DOC_CTRL)))
            {
                if (_dioIfTimeout(&timerCount))
                {
                    return LW_ERR_GENERIC;
                }
            }
            break;
        case DIO_TYPE_EXTRA:
            _localWrite(LW_PRGNLCL_FALCON_DIO_DOC_CTRL(dioPort.portIdx),
                    DRF_NUM(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _RESET, 0x1));

            while (FLD_TEST_DRF_NUM(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _RESET, 0x1,
                                    _localRead(LW_PRGNLCL_FALCON_DIO_DOC_CTRL(dioPort.portIdx))))
            {
                if (_dioIfTimeout(&timerCount))
                {
                    return LW_ERR_GENERIC;
                }
            }
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }

    return status;
}

/* =============================================================================
 * DIO parsing functions implementation
 * ========================================================================== */
LwBool _dioParseDocCtrlEmpty(DIO_PORT dioPort, LwU32 docCtrl)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            return DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _EMPTY, docCtrl);
        case DIO_TYPE_EXTRA:
            return DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _EMPTY, docCtrl);
        default:
            return 0;
    }
}

LwU32 _dioParseDocCtrlFinished(DIO_PORT dioPort, DIO_OPERATION dioOp, LwU32 docCtrl)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            return dioOp == DIO_OPERATION_RD ?
                DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _RD_FINISHED, docCtrl) :
                DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _WR_FINISHED, docCtrl);
        case DIO_TYPE_EXTRA:
            return dioOp == DIO_OPERATION_RD ?
                DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _RD_FINISHED, docCtrl) :
                DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _WR_FINISHED, docCtrl);
        default:
            return 0;
    }
}

LwU32 _dioParseDocCtrlError(DIO_PORT dioPort, LwU32 docCtrl)
{
    LwU32 dioError;
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            if (DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _WR_ERROR, docCtrl) ||
                DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _RD_ERROR, docCtrl) ||
                DRF_VAL(_PRGNLCL, _FALCON_DOC_CTRL, _PROTOCOL_ERROR, docCtrl))
            {
                dioError = _localRead(LW_PRGNLCL_FALCON_DIO_ERR);
                return dioError;
            }
            break;
        case DIO_TYPE_EXTRA:
            if (DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _WR_ERROR, docCtrl) ||
                DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _RD_ERROR, docCtrl) ||
                DRF_VAL(_PRGNLCL, _FALCON_DIO_DOC_CTRL, _PROTOCOL_ERROR, docCtrl))
            {
                dioError = _localRead(LW_PRGNLCL_FALCON_DIO_DIO_ERR(dioPort.portIdx));
                return dioError;
            }
            break;
        default:
            break;
    }
    return 0;
}

LwU32 _dioParseDicCtrlCount(DIO_PORT dioPort, LwU32 dicCtrl)
{
    switch (dioPort.dioType)
    {
        case DIO_TYPE_SE:
            return DRF_VAL(_PRGNLCL, _FALCON_DIC_CTRL, _COUNT, dicCtrl);
        case DIO_TYPE_EXTRA:
            return DRF_VAL(_PRGNLCL, _FALCON_DIO_DIC_CTRL, _COUNT, dicCtrl);
        default:
            return 0;
    }
}
