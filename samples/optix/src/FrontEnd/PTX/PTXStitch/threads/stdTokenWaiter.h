/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2016, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdTokenWaiter.h
 *
 *  Description              :
 *     
 *         This module provides a data type that can be used for
 *         waiting on a specified number of abstract 'tokens'.
 *         Tokens can be 'provided' by other threads.
 */

#ifndef stdTokenWaiter_INCLUDED
#define stdTokenWaiter_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdThreads.h"

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------- Types  ----------------------------------*/

typedef struct stdTokenWaiterRec {
    uInt         tokens;  
    stdSem_t     wait;  
} *stdTokenWaiter_t;

/*---------------------------- External Functions ----------------------------*/

/*
 * Function        : Create a tokens waiter
 * Parameters      : tokens         (I) The amount of tokens to wait for
 * Function Result : Requested new token waiter
 */
static inline
stdTokenWaiter_t tkwCreate( uInt tokens )
{
    stdTokenWaiter_t result;

    stdNEW(result);

    result->tokens = tokens;
    result->wait   = stdSemCreate(0);
        
    return result;
}

/*
 * Function        : Provide one token to token waiter
 * Parameters      : tkw  (I) Token waiter to update
 * Function Result : True iff. this was the final token
 *                   for this object, and hence the waiter 
 *                   is released
 */
static inline
Bool tkwProvideToken( stdTokenWaiter_t tkw )
{
    if (stdAtomicFetchAndAdd(&tkw->tokens,-1) == 1) {
        stdSemV(tkw->wait);
        return True;
    } else {
        return False;
    }
}

/*
 * Function        : Wait until requested number of tokens have
 *                   been provided, and then delete the specified
 *                   token waiter object.
 * Parameters      : tkw  (I) Token Waiter to wait on
 */
static inline
void tkwWaitTokens( stdTokenWaiter_t tkw )
{
    stdSemP(tkw->wait);
    
    stdSemDelete(tkw->wait);
    stdFREE(tkw);
}




#ifdef __cplusplus
}
#endif

#endif


