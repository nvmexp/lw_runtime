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
 *  Module name              : stdSharedLibraries.c
 *
 *  Last update              :
 *
 *  Description              :
 */

/*--------------------------------- Includes ---------------------------------*/

#include <stdSharedLibraries.h>
#include <stdLocal.h>

#ifdef STD_OS_win32
#else
#include <dlfcn.h>
#endif

/*-------------------- Search Path Manipulation Functions --------------------*/

/*
 * Function        : Open shared library.
 * Parameters      : libName (I) Name of library to load
 * Function Result : Handle to opened library, or NULL when open failed.
 */
slibSharedLibrary_t  slibOpenLibrary( String libName )
{
#ifdef STD_OS_win32
    return (slibSharedLibrary_t) LoadLibrary(libName);
#else
    return (slibSharedLibrary_t) dlopen     (libName,RTLD_NOW);
#endif
}


/*
 * Function        : Close previously opened shared library.
 * Parameters      : handle (I) Handle of library to close
 * Function Result : 
 */
void  slibCloseLibrary( slibSharedLibrary_t handle )
{
#ifdef STD_OS_win32
    FreeLibrary((HANDLE)handle);
#else
    dlclose((Pointer)handle);
#endif
}


/*
 * Function        : Obtain address of symbol in shared library.
 * Parameters      : handle (I) Handle of library to search
 *                   symbol (I) Name of symbol to locate
 * Function Result : Address of specified symbol in library,
 *                   or NULL if not found
 */
Pointer  slibGetSymbolAddress( slibSharedLibrary_t handle, String symbol )
{
#ifdef STD_OS_win32
    return GetProcAddress((HANDLE)handle,symbol);
#else
    return dlsym((Pointer)handle,symbol);
#endif
}

