/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
 #pragma once

#include <string>

#include "lwlink_errors.h"

/*****************************************************************************/
/*  Fabric Manager LWLink error types                                        */
/*****************************************************************************/

/* All these LWL_xxx errors are returned by lwlink driver and is defined in
 * drivers\lwlink\interface\lwlink_errors.h. Right now, these errors are propagated
 * to GFM as it is without any translation. But the getLinkErrorCode() is used
 * in the code to provide an opportunity for mangling this error code in the future 
 * if needed. In that case, this enum values will change instead of this direct 
 * LWL_xxx assignment.
 *
 * In addition to driver returned errors, LFM or GFM introduced errors are 
 * defined here.
 */
typedef enum {

    // existing LWLink IOCTL specific errors
    FM_LWLINK_ST_SUCCESS                                = LWL_SUCCESS,
    FM_LWLINK_ST_BAD_ARGS                               = LWL_BAD_ARGS,
    FM_LWLINK_ST_LWL_NO_MEM                             = LWL_NO_MEM,
    FM_LWLINK_ST_LWL_NOT_FOUND                          = LWL_NOT_FOUND,
    FM_LWLINK_ST_LWL_INITIALIZATION_PARTIAL_FAILURE     = LWL_INITIALIZATION_PARTIAL_FAILURE,
    FM_LWLINK_ST_LWL_INITIALIZATION_TOTAL_FAILURE       = LWL_INITIALIZATION_TOTAL_FAILURE,
    FM_LWLINK_ST_LWL_PCI_ERROR                          = LWL_PCI_ERROR,
    FM_LWLINK_ST_LWL_ERR_GENERIC                        = LWL_ERR_GENERIC,
    FM_LWLINK_ST_LWL_ERR_ILWALID_STATE                  = LWL_ERR_ILWALID_STATE,
    FM_LWLINK_ST_LWL_UNBOUND_DEVICE                     = LWL_UNBOUND_DEVICE,
    FM_LWLINK_ST_LWL_MORE_PROCESSING_REQUIRED           = LWL_MORE_PROCESSING_REQUIRED,
    FM_LWLINK_ST_LWL_IO_ERROR                           = LWL_IO_ERROR,

    // GFM/LFM transaction related errors
    FM_LWLINK_ST_TIMEOUT,
    FM_LWLINK_ST_SLAVE_FM_SOCKET_ERR,
    FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
} LWLinkErrorCodes;

/*
 * This class will translate the error code to human readable format.
 * All the members are defined as static as there is no specific member 
 * state to hold and no other class has to create an instance for this class.
 */
class FMLWLinkError
{
public:
    static int getLinkErrorCode (LwlStatus status);

    static const char* getLinkErrorString(LWLinkErrorCodes errorCode);
};
