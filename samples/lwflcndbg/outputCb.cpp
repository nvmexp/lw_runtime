/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004
// OutputCb.cpp - Output Callback class methods definitions
// 
//*****************************************************

#include <stdio.h>
#include "outputCb.h"
#include "regMods.h"

//
// globals
//

//
// lwwatchFp - used to store debug messages
//
#ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
FILE *lwwatchFp = NULL;
#endif


//
// This flag determines if we should log the communication because the extension
// and the engine in LWWATCH_OC_DEBUG_FILE
// 
extern "C" extern int debugOCMFlag;

//
// Comments about LWWATCH_DEBUG_OUTPUT_CALLBACK are here since I could not find
// a better place to put them: This is enabled in sources file. When enabled, it
// will log the communication between the extension and the engine in
// LWWATCH_OC_DEBUG_FILE. You can turn it off as mentioned in sources file.
// 

// 
// LWWATCH_OC_DEBUG_FILE records messages recorded during output callback.
// When LWWATCH_DEBUG_OUTPUT_CALLBACK is enabled in sources file.  When enabled,
// it will log the communication between the extension and the engine in
// LWWATCH_OC_DEBUG_FILE. You can turn it off as mentioned in sources file
//
const char * const LWWATCH_OC_DEBUG_FILE="C:\\lwwatch.txt";



OutputCb::OutputCb()
{
    //
    // O references when created
    // 
    ref = 0;
    
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        lwwatchFp = fopen(LWWATCH_OC_DEBUG_FILE,"wc");
    #endif

}            

OutputCb::~OutputCb()
{
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp) 
        {
            fclose(lwwatchFp);
            lwwatchFp = NULL;
        }
    #endif   
}            

//-----------------------------------------------------
// OutputCb::Output
// + Our output callback function
// + CAUTION: Do not use dprintf in the output
// callback. That will trigger another call to itself
// and go into an infinite loop.
//
//-----------------------------------------------------
STDMETHODIMP 
OutputCb::Output
(
    THIS_
    IN ULONG Mask,
    IN PCSTR Text
) 
{
    char *Text2;
    
    if (NULL == Text){
        return S_OK;
    }
    
    //
    // Print for DEBUG mode
    //
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag>=1) 
        {   
            fprintf(lwwatchFp,"\n----IN OUTPUT CALLBACK----\n");
            fprintf(lwwatchFp,"Read command: ");
            switch(readStruct.readSizeCmd)
            {
                case READ00:
                    fprintf(lwwatchFp,"<nil>\n");
                    break;
                case READ08:
                    fprintf(lwwatchFp,"READ08\n");
                    break;
                case READ16:
                    fprintf(lwwatchFp,"READ16\n");
                    break;
                case READ32:
                    fprintf(lwwatchFp,"READ32\n");
                    break;
            }
            fprintf(lwwatchFp,"Text: %s\n", Text);
            fflush(lwwatchFp);
        }
    #endif

    
    //
    // Check if we are called for FORCE_REG_OP_STRING2.
    //
    if (strncmp(Text,FORCE_REG_OP_STRING2, strlen(FORCE_REG_OP_STRING2))==0)
    {
        //
        // Last time this func was called, it recorded the register read
        // value and set readIsValid.
        // We don't wanna go to errorOut from heresince it will nullify 
        // readIsValid
        //
        return S_OK;
    }
    
    //
    //Don't read anything from Text if readSizeCmd says so.
    //
    if (READ00 == readStruct.readSizeCmd) 
    {
        return S_OK;
    }

    Text2 = (char *)malloc(sizeof(char)*(strlen(Text)+1));

    if (NULL == Text2)
    {
        #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
            if (lwwatchFp && debugOCMFlag) 
            {   fprintf(lwwatchFp,"Unable to allocate memory in " __FUNCTION__ ".\n");
                fflush(lwwatchFp);
            }
        #endif
        goto errorOut;
    }
    
    //
    // strtok modifies the string it operates on. We don't want the input
    // string Text to change. So, make a copy.
    //
    strcpy(Text2,Text);

    //
    // We want tokens from Text
    //
    char *token = strtok(Text2," \n");

    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && token && debugOCMFlag>=2) 
        {   
            fprintf(lwwatchFp,"Token: %s\n",token);
            fflush(lwwatchFp);
        }
    #endif
    
    //
    // Using a state machine to find our register value
    // Debugger engine spits out string like ".call returns unsigned int 0x45ff"
    // Find first oclwrence of int/char/short and grab the number right after it
    //
    UINT32 regVal = ILWALID_REG32;
    int successFlag = FALSE;
    int state = 1;
    int error = 0;
    while (token != NULL)
    {        
        switch(state)
        {
            case 1:
                if (strcmp(token,"int")==0   ||
                    strcmp(token,"short")==0 ||
                    strcmp(token,"char")==0   )  
                {
                    state = 2;
                }
                break;
            case 2: // success state
                error = sscanf(token,"%10x",&regVal);
                if (error <=0) 
                {
                    state = 1;
                }
                else 
                {  
                    successFlag = TRUE;
                }
                break;
            default:
                break;
        }
        token = strtok(NULL," \n");
        #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
            if (lwwatchFp && token && debugOCMFlag>=2) 
            {
                fprintf(lwwatchFp,"Token: %s.\n",token);
                fflush(lwwatchFp);
            }
        #endif
        if (TRUE == successFlag) break;
    }

    if (FALSE==successFlag) goto errorOut;
    
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag>=1) 
        {
            fprintf(lwwatchFp,"Register value read: %08lx\n", regVal);
            fflush(lwwatchFp);
        }
    #endif

    switch(readStruct.readSizeCmd)
    {
        case READ32:
            readStruct.readU032 = regVal;
            break;
        case READ16:
            readStruct.readU016 = (UINT16)regVal;
            break;
        case READ08:
            readStruct.readU008 = (UINT8)regVal;
            break;
        default:
            #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
                if (lwwatchFp && debugOCMFlag) 
                {
                    fprintf(lwwatchFp,"Output: Bug detected.\n");
                    fflush(lwwatchFp);
                }
            #endif
            goto errorOut;
    }
            
    //
    // signal to RegRd that read was successfull
    //
    readStruct.readIsValid = VALID;
    if (Text2) free(Text2);
    return S_OK;
    
//
//set globals to error vals and return
//
errorOut:
    //
    // signal to RegRd that read was unsuccessfull
    //
    readStruct.readIsValid = INVALID;
    if (Text2) free(Text2);

    //
    // Returning OK because the Output callback successfully completed
    //
    return S_OK;

    UNREFERENCED_PARAMETER(Mask);
}


STDMETHODIMP 
OutputCb::QueryInterface
(
    THIS_
    __in REFIID InterfaceId,
    __out PVOID* Interface
) 
{

    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag) 
        {
            fprintf(lwwatchFp,"QueryInterface called.\n");
            fflush(lwwatchFp);
        }
    #endif

    #if _MSC_VER >= 1100
        if (IsEqualIID(InterfaceId, __uuidof(IUnknown)) ||
            IsEqualIID(InterfaceId, __uuidof(IDebugOutputCallbacks)))
    #else
        if (IsEqualIID(InterfaceId, IID_IUnknown) ||
            IsEqualIID(InterfaceId, IDebugOutputCallbacks))
    #endif
        {
            *Interface = this;
            AddRef();
            return S_OK;
        }
        else
        {
            *Interface = NULL;
            return E_NOINTERFACE;
        }
}
 
STDMETHODIMP_(ULONG) 
OutputCb::AddRef
(
    THIS
)    
{
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag >=2)
        {   
            fprintf(lwwatchFp,"AddRef called.\n");
            fflush(lwwatchFp);
        }
    #endif
   return ++ref;;
}

STDMETHODIMP_(ULONG) 
OutputCb::Release
(
    THIS
)
{
    #ifdef LWWATCH_DEBUG_OUTPUT_CALLBACK
        if (lwwatchFp && debugOCMFlag>=2) 
        {
            fprintf(lwwatchFp,"Release called.\n");
            fflush(lwwatchFp);
        }
    #endif
    
    if (0L != --ref)
    {
        return ref;
    }

    delete this;
    return 0; 
}
