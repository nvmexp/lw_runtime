 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2017-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: error.cpp                                                         *|
|*                                                                            *|
 \****************************************************************************/
#include "precomp.h"

//******************************************************************************

CString
statusString
(
    ULONG               ulStatus
)
{
    CString             sStatus(MAX_COMMAND_STRING);

    // Build the status error string
    sStatus.sprintf("0x%08x", ulStatus);
    sStatus = exec(sStatus, buildModCommand("error", "ext", sStatus));

    return sStatus;

} // statusString

//******************************************************************************

CString
rmString
(
    ULONG               rmStatus
)
{
    CString             sStatus;

    // Switch on the RM status value
    switch(rmStatus)
    {
        case LW_OK:                                 sStatus = "Success";                                                    break;
        case LW_ERR_GENERIC:                        sStatus = "Error: Generic Error";                                       break;

        case LW_ERR_BROKEN_FB:                      sStatus = "Error: Frame-Buffer broken";                                 break;
        case LW_ERR_BUFFER_TOO_SMALL:               sStatus = "Error: Buffer passed in is too small";                       break;
        case LW_ERR_BUSY_RETRY:                     sStatus = "Error: System is busy, retry later";                         break;
        case LW_ERR_CALLBACK_NOT_SCHEDULED:         sStatus = "Error: The requested callback API not scheduled";            break;
        case LW_ERR_CARD_NOT_PRESENT:               sStatus = "Error: Card not detected";                                   break;
        case LW_ERR_CYCLE_DETECTED:                 sStatus = "Error: Call cycle detected";                                 break;
        case LW_ERR_DMA_IN_USE:                     sStatus = "Error: Requested DMA is in use";                             break;
        case LW_ERR_DMA_MEM_NOT_LOCKED:             sStatus = "Error: Requested DMA memory is not locked";                  break;
        case LW_ERR_DMA_MEM_NOT_UNLOCKED:           sStatus = "Error: Requested DMA memory is not unlocked";                break;
        case LW_ERR_DUAL_LINK_INUSE:                sStatus = "Error: Dual-Link is in use";                                 break;
        case LW_ERR_ECC_ERROR:                      sStatus = "Error: ECC error";                                           break;
        case LW_ERR_FIFO_BAD_ACCESS:                sStatus = "Error: FIFO: Invalid access";                                break;
        case LW_ERR_FREQ_NOT_SUPPORTED:             sStatus = "Error: Requested frequency is not supported";                break;
        case LW_ERR_GPU_DMA_NOT_INITIALIZED:        sStatus = "Error: Requested DMA not initialized";                       break;
        case LW_ERR_GPU_IS_LOST:                    sStatus = "Error: GPU lost from the bus";                               break;
        case LW_ERR_GPU_IN_FULLCHIP_RESET:          sStatus = "Error: GPU lwrrently in full-chip reset";                    break;
        case LW_ERR_GPU_NOT_FULL_POWER:             sStatus = "Error: GPU not in full power";                               break;
        case LW_ERR_GPU_UUID_NOT_FOUND:             sStatus = "Error: GPU UUID not found";                                  break;
        case LW_ERR_HOT_SWITCH:                     sStatus = "Error: System in hot switch";                                break;
        case LW_ERR_I2C_ERROR:                      sStatus = "Error: I2C Error";                                           break;
        case LW_ERR_I2C_SPEED_TOO_HIGH:             sStatus = "Error: I2C speed too high";                                  break;
        case LW_ERR_ILLEGAL_ACTION:                 sStatus = "Error: Current action is not allowed";                       break;
        case LW_ERR_IN_USE:                         sStatus = "Error: In use error";                                        break;
        case LW_ERR_INFLATE_COMPRESSED_DATA_FAILED: sStatus = "Error: Failed to inflate compressed data";                   break;
        case LW_ERR_INSERT_DUPLICATE_NAME:          sStatus = "Error: Found a duplicate entry in the requested btree";      break;
        case LW_ERR_INSUFFICIENT_RESOURCES:         sStatus = "Error: Ran out of a critical resource, other than memory";   break;
        case LW_ERR_INSUFFICIENT_PERMISSIONS:       sStatus = "Error: The requester does not have sufficient permissions";  break;
        case LW_ERR_INSUFFICIENT_POWER:             sStatus = "Error: Insufficient power";                                  break;
        case LW_ERR_ILWALID_ACCESS_TYPE:            sStatus = "Error: This type of access is not allowed";                  break;
        case LW_ERR_ILWALID_ADDRESS:                sStatus = "Error: Address not valid";                                   break;
        case LW_ERR_ILWALID_ARGUMENT:               sStatus = "Error: Invalid argument to call";                            break;
        case LW_ERR_ILWALID_BASE:                   sStatus = "Error: Invalid base";                                        break;
        case LW_ERR_ILWALID_CHANNEL:                sStatus = "Error: Given channel-id not valid";                          break;
        case LW_ERR_ILWALID_CLASS:                  sStatus = "Error: Given class-id not valid";                            break;
        case LW_ERR_ILWALID_CLIENT:                 sStatus = "Error: Given client not valid";                              break;
        case LW_ERR_ILWALID_COMMAND:                sStatus = "Error: Command passed is not valid";                         break;
        case LW_ERR_ILWALID_DATA:                   sStatus = "Error: Invalid data passed";                                 break;
        case LW_ERR_ILWALID_DEVICE:                 sStatus = "Error: Current device is not valid";                         break;
        case LW_ERR_ILWALID_DMA_SPECIFIER:          sStatus = "Error: The requested DMA specifier is not valid";            break;
        case LW_ERR_ILWALID_EVENT:                  sStatus = "Error: Invalid event oclwred";                               break;
        case LW_ERR_ILWALID_FLAGS:                  sStatus = "Error: Invalid flags passed";                                break;
        case LW_ERR_ILWALID_FUNCTION:               sStatus = "Error: Called function is not valid";                        break;
        case LW_ERR_ILWALID_HEAP:                   sStatus = "Error: Heap corrupted";                                      break;
        case LW_ERR_ILWALID_INDEX:                  sStatus = "Error: Index invalid";                                       break;
        case LW_ERR_ILWALID_IRQ_LEVEL:              sStatus = "Error: Requested IRQ level is not valid";                    break;
        case LW_ERR_ILWALID_LIMIT:                  sStatus = "Error: Invalid limit";                                       break;
        case LW_ERR_ILWALID_LOCK_STATE:             sStatus = "Error: Requested lock state not valid";                      break;
        case LW_ERR_ILWALID_METHOD:                 sStatus = "Error: Requested method not valid";                          break;
        case LW_ERR_ILWALID_OBJECT:                 sStatus = "Error: Object not valid";                                    break;
        case LW_ERR_ILWALID_OBJECT_BUFFER:          sStatus = "Error: Object buffer passed is not valid";                   break;
        case LW_ERR_ILWALID_OBJECT_HANDLE:          sStatus = "Error: Object handle is not valid";                          break;
        case LW_ERR_ILWALID_OBJECT_NEW:             sStatus = "Error: New object is not valid";                             break;
        case LW_ERR_ILWALID_OBJECT_OLD:             sStatus = "Error: Old object is not valid";                             break;
        case LW_ERR_ILWALID_OBJECT_PARENT:          sStatus = "Error: Object parent is not valid";                          break;
        case LW_ERR_ILWALID_OFFSET:                 sStatus = "Error: The offset passed is not valid";                      break;
        case LW_ERR_ILWALID_OPERATION:              sStatus = "Error: Requested operation is not valid";                    break;
        case LW_ERR_ILWALID_OWNER:                  sStatus = "Error: Owner not valid";                                     break;
        case LW_ERR_ILWALID_PARAM_STRUCT:           sStatus = "Error: Invalid structure parameter";                         break;
        case LW_ERR_ILWALID_PARAMETER:              sStatus = "Error: At least one of the parameters is invalid";           break;
        case LW_ERR_ILWALID_PATH:                   sStatus = "Error: The requested path is not valid";                     break;
        case LW_ERR_ILWALID_POINTER:                sStatus = "Error: Pointer not valid";                                   break;
        case LW_ERR_ILWALID_REGISTRY_KEY:           sStatus = "Error: Found an invalid registry key";                       break;
        case LW_ERR_ILWALID_REQUEST:                sStatus = "Error: Invalid request";                                     break;
        case LW_ERR_ILWALID_STATE:                  sStatus = "Error: Invalid state";                                       break;
        case LW_ERR_ILWALID_STRING_LENGTH:          sStatus = "Error: The string length is not valid";                      break;
        case LW_ERR_ILWALID_READ:                   sStatus = "Error: The requested read operation is not valid";           break;
        case LW_ERR_ILWALID_WRITE:                  sStatus = "Error: The requested write operation is not valid";          break;
        case LW_ERR_ILWALID_XLATE:                  sStatus = "Error: The requested translate operation is not valid";      break;
        case LW_ERR_IRQ_NOT_FIRING:                 sStatus = "Error: Requested IRQ is not firing";                         break;
        case LW_ERR_IRQ_EDGE_TRIGGERED:             sStatus = "Error: IRQ is edge triggered";                               break;
        case LW_ERR_MEMORY_TRAINING_FAILED:         sStatus = "Error: Failed memory training sequence";                     break;
        case LW_ERR_MISMATCHED_SLAVE:               sStatus = "Error: Slave mismatch";                                      break;
        case LW_ERR_MISMATCHED_TARGET:              sStatus = "Error: Target mismatch";                                     break;
        case LW_ERR_MISSING_TABLE_ENTRY:            sStatus = "Error: Requested entry missing not found in the table";      break;
        case LW_ERR_MODULE_LOAD_FAILED:             sStatus = "Error: Failed to load the requested module";                 break;
        case LW_ERR_MORE_DATA_AVAILABLE:            sStatus = "Error: There is more data available";                        break;
        case LW_ERR_MORE_PROCESSING_REQUIRED:       sStatus = "Error: More processing required for the given call";         break;
        case LW_ERR_MULTIPLE_MEMORY_TYPES:          sStatus = "Error: Multiple memory types found";                         break;
        case LW_ERR_NO_FREE_FIFOS:                  sStatus = "Error: No more free FIFOs found";                            break;
        case LW_ERR_NO_INTR_PENDING:                sStatus = "Error: No interrupt pending";                                break;
        case LW_ERR_NO_MEMORY:                      sStatus = "Error: Out of memory";                                       break;
        case LW_ERR_NO_SUCH_DOMAIN:                 sStatus = "Error: Requested domain does not exist";                     break;
        case LW_ERR_NO_VALID_PATH:                  sStatus = "Error: Caller did not speficy a valid path";                 break;
        case LW_ERR_NOT_COMPATIBLE:                 sStatus = "Error: Incompatible types";                                  break;
        case LW_ERR_NOT_READY:                      sStatus = "Error: Not ready";                                           break;
        case LW_ERR_NOT_SUPPORTED:                  sStatus = "Error: Call not supported";                                  break;
        case LW_ERR_OBJECT_NOT_FOUND:               sStatus = "Error: Requested object not found";                          break;
        case LW_ERR_OBJECT_TYPE_MISMATCH:           sStatus = "Error: Specified objects do not match";                      break;
        case LW_ERR_OPERATING_SYSTEM:               sStatus = "Error: Generic operating system error";                      break;
        case LW_ERR_OTHER_DEVICE_FOUND:             sStatus = "Error: Found other device instead of requested one";         break;
        case LW_ERR_OUT_OF_RANGE:                   sStatus = "Error: The specified value is out of bounds";                break;
        case LW_ERR_OVERLAPPING_UVM_COMMIT:         sStatus = "Error: Overlapping unified virtual memory commit";           break;
        case LW_ERR_PAGE_TABLE_NOT_AVAIL:           sStatus = "Error: Requested page table not available";                  break;
        case LW_ERR_PID_NOT_FOUND:                  sStatus = "Error: Process-Id not found";                                break;
        case LW_ERR_PROTECTION_FAULT:               sStatus = "Error: Protection fault";                                    break;
        case LW_ERR_RC_ERROR:                       sStatus = "Error: RC error";                                            break;
        case LW_ERR_REJECTED_VBIOS:                 sStatus = "Error: Given Video BIOS rejected/invalid";                   break;
        case LW_ERR_RESET_REQUIRED:                 sStatus = "Error: Reset required";                                      break;
        case LW_ERR_STATE_IN_USE:                   sStatus = "Error: State in use";                                        break;
        case LW_ERR_SIGNAL_PENDING:                 sStatus = "Error: Signal pending";                                      break;
        case LW_ERR_TIMEOUT:                        sStatus = "Error: Call timed out";                                      break;
        case LW_ERR_TIMEOUT_RETRY:                  sStatus = "Error: Call timed out, please retry later";                  break;
        case LW_ERR_TOO_MANY_PRIMARIES:             sStatus = "Error: Too many primaries";                                  break;
        case LW_ERR_UVM_ADDRESS_IN_USE:             sStatus = "Error: UVM requested address already in use";                break;
        case LW_ERR_MAX_SESSION_LIMIT_REACHED:      sStatus = "Error: Maximum number of sessions reached";                  break;
        case LW_ERR_LIB_RM_VERSION_MISMATCH:        sStatus = "Error: Library version doesn't match driver version";        break;

        case LW_WARN_HOT_SWITCH:                    sStatus = "Warning: Hot switch";                                        break;
        case LW_WARN_INCORRECT_PERFMON_DATA:        sStatus = "Warning: Incorrect performance monitor data";                break;
        case LW_WARN_MISMATCHED_SLAVE:              sStatus = "Warning: Slave mismatch";                                    break;
        case LW_WARN_MISMATCHED_TARGET:             sStatus = "Warning: Target mismatch";                                   break;
        case LW_WARN_MORE_PROCESSING_REQUIRED:      sStatus = "Warning: More processing required for the call";             break;
        case LW_WARN_NOTHING_TO_DO:                 sStatus = "Warning: Nothing to do";                                     break;
        case LW_WARN_NULL_OBJECT:                   sStatus = "Warning: NULL object found";                                 break;
        case LW_WARN_OUT_OF_RANGE:                  sStatus = "Warning: Value out of range";                                break;
        default:                                    sStatus = "Unknown RM status";                                          break;
    }
    return sStatus;

} // rmString

//******************************************************************************

CString
regString
(
    int                 reResult,
    regex_t            *pRegEx,
    const char         *pRegExpr
)
{
    CString             sError(MAX_DBGPRINTF_STRING);
    CString             sRegEx(MAX_DBGPRINTF_STRING);

    // Try to get the regular expression error
    if (regerror(reResult, pRegEx, sError.data(), sError.capacity()))
    {
        // Build the regular expression error
        sRegEx.sprintf("Error compiling regular expression '%s': %s (%d)!", pRegExpr, DML(sError), reResult);
    }
    else    // Unable to get regular expression error
    {
        // Build the regular expression error
        sRegEx.sprintf("Error compiling regular expression '%s' (%d)!", pRegExpr, reResult);
    }
    return sRegEx;

} // regString

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
