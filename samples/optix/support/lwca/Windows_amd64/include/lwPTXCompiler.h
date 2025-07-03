/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef lwPTXCompiler_INCLUDED
#define lwPTXCompiler_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/* --- Dependency --- */
#include <stddef.h>   /* For size_t */

/*************************************************************************//**
 *
 * \defgroup handle PTX-Compiler Handle
 *
 ****************************************************************************/


/**
 * \ingroup handle
 * \brief   lwPTXCompilerHandle represents a handle to the PTX Compiler.
 *
 * To compile a PTX program string, an instance of lwPTXCompiler
 * must be created and the handle to it must be obtained using the
 * API lwPTXCompilerCreate(). Then the compilation can be done
 * using the API lwPTXCompilerCompile().
 *
 */
typedef struct lwPTXCompiler* lwPTXCompilerHandle;

/**
 *
 * \defgroup error Error codes
 *
 */

/** \ingroup error
 *
 * \brief     The lwPTXCompiler APIs return the lwPTXCompileResult codes to indicate the call result
 */

typedef enum {

    /* Indicates the API completed successfully */
    LWPTXCOMPILE_SUCCESS = 0,

    /* Indicates an invalid lwPTXCompilerHandle was passed to the API */
    LWPTXCOMPILE_ERROR_ILWALID_COMPILER_HANDLE = 1,

    /* Indicates invalid inputs were given to the API  */
    LWPTXCOMPILE_ERROR_ILWALID_INPUT = 2,

    /* Indicates that the compilation of the PTX program failed */
    LWPTXCOMPILE_ERROR_COMPILATION_FAILURE = 3,

    /* Indicates that something went wrong internally */
    LWPTXCOMPILE_ERROR_INTERNAL = 4,

    /* Indicates that the API was unable to allocate memory */
    LWPTXCOMPILE_ERROR_OUT_OF_MEMORY = 5,

    /* Indicates that the handle was passed to an API which expected */
    /* the lwPTXCompilerCompile() to have been called previously */
    LWPTXCOMPILE_ERROR_COMPILER_ILWOCATION_INCOMPLETE = 6,

    /* Indicates that the PTX version encountered in the PTX is not */
    /* supported by the current compiler */
    LWPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION = 7,
} lwPTXCompileResult;

/* ----------------------------- PTX Compiler APIs ---------------------------- */

/**
 *
 * \defgroup versioning API Versioning
 *
 * The PTX compiler APIs are versioned so that any new features or API
 * changes can be done by bumping up the API version.
 */

/** \ingroup versioning
 *
 * \brief            Queries the current \p major and \p minor version of
 *                   PTX Compiler APIs being used
 *
 * \param            [out] major   Major version of the PTX Compiler APIs
 * \param            [out] minor   Minor version of the PTX Compiler APIs
 * \note                           The version of PTX Compiler APIs follows the LWCA Toolkit versioning.
 *                                 The PTX ISA version supported by a PTX Compiler API version is listed
 *                                 <a href="https://docs.lwpu.com/lwca/parallel-thread-exelwtion/#release-notes">here</a>.
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 */
lwPTXCompileResult lwPTXCompilerGetVersion (unsigned int* major, unsigned int* minor);

/**
 *
 * \defgroup compilation Compilation APIs
 *
 */

/** \ingroup compilation
 *
 * \brief            Obtains the handle to an instance of the PTX compiler
 *                   initialized with the given PTX program \p ptxCode
 *
 * \param            [out] compiler  Returns a handle to PTX compiler initialized
 *                                   with the PTX program \p ptxCode
 * \param            [in] ptxCodeLen Size of the PTX program \p ptxCode passed as string
 * \param            [in] ptxCode    The PTX program which is to be compiled passed as string.
 *
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 */
lwPTXCompileResult lwPTXCompilerCreate (lwPTXCompilerHandle *compiler, size_t ptxCodeLen, const char* ptxCode);

/** \ingroup compilation
 *
 * \brief            Destroys and cleans the already created PTX compiler
 *
 * \param            [in] compiler  A handle to the PTX compiler which is to be destroyed
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerDestroy (lwPTXCompilerHandle *compiler);

/** \ingroup compilation
 *
 * \brief          Compile a PTX program with the given compiler options
 *
 * \param            [in,out] compiler      A handle to PTX compiler initialized with the
 *                                          PTX program which is to be compiled.
 *                                          The compiled program can be accessed using the handle
 * \param            [in] numCompileOptions Length of the array \p compileOptions
 * \param            [in] compileOptions   Compiler options with which compilation should be done.
 *                                         The compiler options string is a null terminated character array.
 *                                         A valid list of compiler options is at
 *                                         <a href="http://docs.lwpu.com/lwca/ptx-compiler-api/index.html#compile-options">link</a>.
 * \note                                   --gpu-name (-arch) is a mandatory option.
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_COMPILATION_FAILURE  \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION  \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerCompile (lwPTXCompilerHandle compiler, int numCompileOptions, const char* const * compileOptions);

/** \ingroup compilation
 *
 * \brief            Obtains the size of the image of the compiled program
 *
 * \param            [in] compiler          A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] binaryImageSize  The size of the image of the compiled program
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_COMPILER_ILWOCATION_INCOMPLETE \endlink
 *
 * \note             lwPTXCompilerCompile() API should be ilwoked for the handle before calling this API.
 *                   Otherwise, LWPTXCOMPILE_ERROR_COMPILER_ILWOCATION_INCOMPLETE is returned.
 */
lwPTXCompileResult lwPTXCompilerGetCompiledProgramSize (lwPTXCompilerHandle compiler, size_t* binaryImageSize);

/** \ingroup compilation
 *
 * \brief            Obtains the image of the compiled program
 *
 * \param            [in] compiler          A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] binaryImage      The image of the compiled program.
 *                                         Client should allocate memory for \p binaryImage
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_COMPILER_ILWOCATION_INCOMPLETE \endlink
 *
 * \note             lwPTXCompilerCompile() API should be ilwoked for the handle before calling this API.
 *                   Otherwise, LWPTXCOMPILE_ERROR_COMPILER_ILWOCATION_INCOMPLETE is returned.
 *
 */

lwPTXCompileResult lwPTXCompilerGetCompiledProgram (lwPTXCompilerHandle compiler, void*   binaryImage);

/** \ingroup compilation
 *
 * \brief            Query the size of the error message that was seen previously for the handle
 *
 * \param            [in] compiler          A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] errorLogSize     The size of the error log in bytes which was produced
 *                                          in previous call to lwPTXCompilerCompiler().
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerGetErrorLogSize (lwPTXCompilerHandle compiler, size_t* errorLogSize);

/** \ingroup compilation
 *
 * \brief            Query the error message that was seen previously for the handle
 *
 * \param            [in] compiler         A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] errorLog        The error log which was produced in previous call to lwPTXCompilerCompiler().
 *                                         Clients should allocate memory for \p errorLog
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerGetErrorLog (lwPTXCompilerHandle compiler, char*   errorLog);

/** \ingroup compilation
 *
 * \brief            Query the size of the information message that was seen previously for the handle
 *
 * \param            [in] compiler        A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] infoLogSize    The size of the information log in bytes which was produced
 *                                         in previous call to lwPTXCompilerCompiler().
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerGetInfoLogSize (lwPTXCompilerHandle compiler, size_t* infoLogSize);

/** \ingroup compilation
 *
 * \brief           Query the information message that was seen previously for the handle
 *
 * \param            [in] compiler        A handle to PTX compiler on which lwPTXCompilerCompile() has been performed.
 * \param            [out] infoLog        The information log which was produced in previous call to lwPTXCompilerCompiler().
 *                                        Clients should allocate memory for \p infoLog
 *
 * \return
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_SUCCESS \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_INTERNAL \endlink
 *   - \link #lwPTXCompileResult LWPTXCOMPILE_ERROR_ILWALID_PROGRAM_HANDLE \endlink
 *
 */
lwPTXCompileResult lwPTXCompilerGetInfoLog (lwPTXCompilerHandle compiler, char*   infoLog);

#ifdef __cplusplus
}
#endif

#endif // lwPTXCompiler_INCLUDED
