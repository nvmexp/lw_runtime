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
|*  Module: cmdcommand.cpp                                                    *|
|*                                                                            *|
 \****************************************************************************/
#include "cmdprecomp.h"
#include "cmdcommand.h"

//******************************************************************************
//
//  cmd namespace
//
//******************************************************************************
namespace cmd
{

//******************************************************************************

CCommand::CCommand
(
    PDEBUG_CLIENT   pDbgClient,                 // Local debug client for this command
    PCSTR           args,                       // Pointer to command argument string
    int            *argc,                       // Pointer to argument count variable
    char          **argv,                       // Pointer to argument values
    char           *pCommand,                   // Pointer to command name
    bool            bInitGlobals,               // Flag to indicate global initialization
    ULONG           ulEffectiveProcessor        // Effective processor type
)
:   m_pDbgClient(pDbgClient),
    m_args(args),
    m_argc(argc),
    m_argv(argv),
    m_pCommand(pCommand),
    m_sArgs(args),
    m_DmlState(),
    m_ProgressState(),
    m_EffectiveProcessor(ulEffectiveProcessor)
{
    UNREFERENCED_PARAMETER(pDbgClient);

    assert(pDbgClient != NULL);
    assert(args != NULL);
    assert(argc != NULL);
    assert(argv != NULL);
    assert(pCommand != NULL);

    // Initialize the arguments for this command
    initializeArguments(m_sArgs.data(), pCommand, argc, argv);

    // Initialize the globals (if requested)
    if (bInitGlobals)
    {
        // Initialize globals
        initializeGlobals();
    }

} // CCommand

//******************************************************************************

CCommand::~CCommand()
{

} // ~CCommand

//******************************************************************************

void
CCommand::updateDmlState()
{
    // Update the DML state
    m_DmlState.update();

} // updateDmlState

//******************************************************************************

void
CCommand::restoreDmlState()
{
    // Restore the DML state
    m_DmlState.restore();

} // restoreDmlState

//******************************************************************************

void
CCommand::updateProgressState()
{
    // Update the progress state
    m_ProgressState.update();

} // updateProgressState

//******************************************************************************

void
CCommand::restoreProgressState()
{
    // Restore the progress state
    m_ProgressState.restore();

} // restoreProgressState

} // cmd namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
