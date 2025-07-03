/*
 *  Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 * 
 *  NOTICE TO USER: The source code, and related code and software
 *  ("Code"), is copyrighted under U.S. and international laws.  
 * 
 *  LWPU Corporation owns the copyright and any patents issued or 
 *  pending for the Code.  
 * 
 *  LWPU CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY 
 *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
 *  OR IMPLIED WARRANTY OF ANY KIND.  LWPU CORPORATION DISCLAIMS ALL
 *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND 
 *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE.  IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
 *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
 *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE 
 *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO LWPU
 *  CORPORATION.
 * 
 *  Module name              : stdCommandParse.inc
 *
 *  Description              :
 *         This file contains some C functions that form
 *         standard idiom for all tools that use this command parsing
 *         library: it defines a number of standard options, and passes
 *         the tool name to the message generation library.
 *
 *         It should be included in the main tool module instead
 *         of used as a library, because it includes the tool specific
 *         command option definition file (options.h).
 */

/*-------------------------------- Includes ----------------------------------*/

#include "stdLocal.h"
#include "stdCmdOpt.h"
#include "stdMessages.h"
#include "stdFileNames.h"


#ifndef TOOL_IDENT
# define TOOL_IDENT     ""
#endif

#ifndef USAGE
# define USAGE          "%s"
#endif

#ifndef PARSEFLAGS
# define PARSEFLAGS     0
#endif

/*-------------------------------- Functions ---------------------------------*/


static void __version(void)
{
    printf("%s: LWPU (R) %s\n", msgToolName, TOOL_IDENT );
    printf("Copyright (c) 2017, LWPU CORPORATION. All rights reserved.\n");
    printf("Built on "stdSTRING(STD_DATE)"\n");
  #ifdef TOOLS_VERSION
    printf(TOOLS_VERSION"\n");
  #endif
}


static void __usage( cmdoptDescr_t options, Bool hidden )
{
    printf("\n");
    printf("Usage  : ");
    printf(USAGE, msgToolName);
    
    cmdoptPrint(options,hidden,False);
    printf("\n");
    
    stdEXIT(0); 
}


static cmdoptDescr_t __parseOptions( Int argc, String argv[] )
{
    cmdoptDescr_t options= cmdoptCreate();
    String        toolName;
    
 #include "options.h"
    
    cmdoptAddOption( 
          options,
          "tool-name", "tool-name", cmdoptString, cmdoptSingleValue, cmdoptHidden, Nil, Nil,
          Nil,
          Nil,
          "<tool name>",
         "Change tool name to specified string"
    );

    cmdoptAddOption( 
          options,
          "help", "h", cmdoptBool, cmdoptNoValue, 0, Nil, Nil,
          Nil,
          Nil,
          Nil,
         "Print this help information on this tool"
    );

    cmdoptAddOption( 
          options,
          "version", "V", cmdoptBool, cmdoptNoValue, 0, Nil, Nil,
          Nil,
          Nil,
          Nil,
          "Print version information on this tool"
    );

    cmdoptAddOption( 
          options,
          "options-file", "optf", cmdoptOptionsFile, cmdoptListValue, 0, Nil, Nil,
          Nil,
          Nil,
          "<file>",
          "Include command line options from specified file"
    );

   cmdoptAddOption(options, "trap-into-debugger", "_trap_",  cmdoptBool, cmdoptNoValue, 0
#ifdef RELEASE
    | cmdoptDisabled
#endif
   ,Nil, Nil, Nil, Nil, Nil,
    "Trap into debugger upon assertion failures and application crashes" );

    fnamDecomposePath( argv[0], Nil, &msgToolName, Nil );
    cmdoptParse(options, argc, argv, PARSEFLAGS);
    
    cmdoptGetOptiolwalue( options, "tool-name", toolName );
    if (toolName) { fnamDecomposePath( toolName, Nil, &msgToolName, Nil ); }
    
    if (cmdoptOnCommandLine( options, "trap-into-debugger" )) { procSetTrapOnError();   }
    if (cmdoptOnCommandLine( options, "help"               )) { __usage(options,False); }
    if (cmdoptOnCommandLine( options, "version"            )) { __version();stdEXIT(0);    }
   
    return options;
}
    

