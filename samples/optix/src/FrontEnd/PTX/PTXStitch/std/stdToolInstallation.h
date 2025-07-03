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
 *  Module name              : stdToolInstallation.h
 *
 *  Last update              :
 *
 *  Description              :
 *     
 *        This module infers the location of the tool installation from the
 *        file name of one of its tools, plus the contents of the current
 *        exelwtable search path. It is an alternative for stdToolPatch, when
 *        the exelwtable resides at its 'proper' place in the tool installation.
 */

/*------------------------------- Includes -----------------------------------*/

#include <stdTypes.h>

#ifndef stdToolInstallation_INCLUDED
#define stdToolInstallation_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Infer installation path from specified exelwtable file name
 *                   plus the contents of the current exelwtable search path.          
 * Parameters      : fileName   (I) File name to infer from. 
 *                                  SDK tools typically pass argv[0] for 
 *                                  finding out where their installation resides.
 * Function Result : Tool installation path, when recognized, or NULL
 */
String tlpInstallPath( String fileName );


#ifdef __cplusplus
}
#endif

#endif
