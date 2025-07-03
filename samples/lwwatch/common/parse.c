/*
 * Copyright 2003-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "lwwatch.h"

#include <stdio.h>
#include <string.h>

#include "os.h"
#include "hal.h"

#if defined(USERMODE) && defined(LW_WINDOWS) && !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#include <usermode.h>
#include <wdbgexts.h>
#include <dbgeng.h>
#endif

#ifndef CLIENT_SIDE_RESMAN
#if defined(USERMODE)

#elif defined(LW_WINDOWS)
#include "lwwatch.h"

#elif LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
#include "lwstring.h"
#endif
#endif


//-----------------------------------------------------
// getToken
//-----------------------------------------------------
char *getToken(char *ptr, char *token, char **tokenPtr)
{
    // Make sure we have valid arguments
    if ((ptr == NULL) || (token == NULL))
        return NULL;
    if (*ptr == '\0')
        return NULL;

    // Initialize token string and skip all leading whitespace
    *token = '\0';
    while (*ptr == ' ' || *ptr == '\t')
        ptr++;

    // Fail if nothing but whitespace was remaining
    if (*ptr == '\0')
        return NULL;

    // Save pointer to token location if given argument
    if (tokenPtr)
        *tokenPtr = ptr;

    // Loop gathering the next token until whitespace or string end
    while (*ptr != ' ' && *ptr != '\t' && *ptr != '\0')
    {
        *token = *ptr;
        ptr++, token++;
    }
    // Terminate the token string
    *token = '\0';

    // Skip any remaining whitespace after the token
    while (*ptr == ' ' || *ptr == '\t')
        ptr++;

    // Return location to start the next token scan
    return ptr;
}

//-----------------------------------------------------
// parseCmd
//-----------------------------------------------------
int parseCmd(char *args, char *op, int numParam, char **param)
{
    //
    // ARG as Input - Assume numParam = 2
    // "xxxx -k   aaaa bbbb cccc"
    //       |    |         |
    //       ptr1 ptr2      ptr
    // ARG as Output
    // "xxxx cccc""aaaa bbbb"
    //             |
    //             *param
    //
    char token[128], *ptr, *ptr1, *ptr2 = NULL, *ptr3;
    int count;

    // Make sure the arguments are valid
    if ((args == NULL) || (op == NULL) || ((numParam > 0) && (param == NULL)))
        return 0;
    if (*args == '\0')
        return 0;

    // Looping parsing the tokens in the argument list
    for (ptr = getToken(args, token, &ptr1); ptr != NULL; ptr = getToken(ptr, token, &ptr1))
    {
        // Check next token to see if it is the requested option
        if (token[0] == '-' && !strcmp(token + 1, op))
        {
            // Save pointer to start of parameters and stop token parsing
            ptr2 = ptr;
            break;
        }
    }
    // Fail if no matching option was located
    if (ptr2 == NULL)
        return 0;

    // Loop parsing off the requested number of parameters
    for (count = 0; count < numParam; count++)
    {
        // Get the next option parameter
        ptr = getToken(ptr, token, &ptr3);

        // Exit parameter loop if no more tokens or start of new option
        if ((ptr == NULL) || (token[0] == '-'))
            break;
    }
    // Fail if the requested number of parameters is not found
    if (count < numParam)
        return 0;

    // Terminate the parameters found if more tokens remain 
    if (*ptr != '\0')
        *(ptr - 1) = '\0';

    // Copy the parameters found to the token buffer
    strcpy(token, ptr2);

    // Remove the option and parameters from the argument string
    memmove(ptr1, ptr, (strlen(ptr) + 1));

    // Check for parameters found (Need to remove actual option from argument string)
    if (numParam > 0)
    {
        // Skip over the remaining argument list and copy parameters from token buffer as separate string
        // (Parameter string follows remaining argument string)
        ptr1 += strlen(ptr1);
        strcpy(++ptr1, token);
        *param = ptr1;
    }
    // No parameters, clean parameter pointer if given
    else if (param != NULL)
        *param = NULL;

#ifdef USERMODE
    argstaken = numParam + 1 + 1;   // next argv[] entry for GetSafeExpression()
#endif
    // Return parsing success
    return 1;
}

