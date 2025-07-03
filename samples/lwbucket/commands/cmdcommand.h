 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
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
|*  Module: cmdcommand.h                                                      *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _CMDCOMMAND_H
#define _CMDCOMMAND_H

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define NO_GLOBALS      false                       // Do not initialize the globals
#define INIT_GLOBALS    true                        // Initialize the globals

//******************************************************************************
//
//  cmd namespace
//
//******************************************************************************
namespace cmd
{

//******************************************************************************
//
//  Forwards
//
//******************************************************************************
class CCommand;

//******************************************************************************
//
// class CCommand
//
// Class for dealing with debugger commands
//
//******************************************************************************
class CCommand
{
private:
        PDEBUG_CLIENT   m_pDbgClient;               // Local debug client for this command
        PCSTR           m_args;                     // Pointer to command argument string
        int*            m_argc;                     // Pointer to argument count variable
        char**          m_argv;                     // Pointer to argument values
        char*           m_pCommand;                 // Pointer to command name
        CString         m_sArgs;                    // Argument string
        CDmlState       m_DmlState;                 // DML state
        CProgressState  m_ProgressState;            // Progress state
        CEffectiveProcessor m_EffectiveProcessor;   // Effective processor type

public:
                        CCommand(PDEBUG_CLIENT pDbgClient, PCSTR args, int* argc, char** argv, char* pCommand, bool bInitGlobals = INIT_GLOBALS, ULONG ulEffectiveProcessor = IMAGE_FILE_MACHINE_UNKNOWN);
                       ~CCommand();

        void            updateDmlState();
        void            restoreDmlState();

        void            updateProgressState();
        void            restoreProgressState();

}; // class CCommand

//******************************************************************************
//
//  Functions
//
//******************************************************************************





} // cmd namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _CMDCOMMAND_H
