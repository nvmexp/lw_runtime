//
// LWIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2014-2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
// LWIDIA_COPYRIGHT_END
//

#ifndef LWVM_H
#define LWVM_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*****************************//**
 *
 * \defgroup error Error Handling
 *
 ********************************/


/**
 * \ingroup error
 * \brief   LWVM API call result code.
 */
typedef enum {
  LWVM_SUCCESS = 0,
  LWVM_ERROR_OUT_OF_MEMORY = 1,
  LWVM_ERROR_PROGRAM_CREATION_FAILURE = 2,
  LWVM_ERROR_IR_VERSION_MISMATCH = 3,
  LWVM_ERROR_ILWALID_INPUT = 4,
  LWVM_ERROR_ILWALID_PROGRAM = 5,
  LWVM_ERROR_ILWALID_IR = 6,
  LWVM_ERROR_ILWALID_OPTION = 7,
  LWVM_ERROR_NO_MODULE_IN_PROGRAM = 8,
  LWVM_ERROR_COMPILATION = 9
} lwvmResult;


/**
 * \ingroup error
 * \brief   Get the message string for the given #lwvmResult code.
 *
 * \param   [in] result LWVM API result code.
 * \return  Message string for the given #lwvmResult code.
 */
const char *lwvmGetErrorString(lwvmResult result);


/****************************************//**
 *
 * \defgroup query General Information Query
 *
 *******************************************/


/**
 * \ingroup query
 * \brief   Get the LWVM version.
 *
 * \param   [out] major LWVM major version number.
 * \param   [out] minor LWVM minor version number.
 * \return 
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *
 */
lwvmResult lwvmVersion(int *major, int *minor);


/**
 * \ingroup query
 * \brief   Get the LWVM IR version.
 *
 * \param   [out] majorIR  LWVM IR major version number.
 * \param   [out] minorIR  LWVM IR minor version number.
 * \param   [out] majorDbg LWVM IR debug metadata major version number.
 * \param   [out] minorDbg LWVM IR debug metadata minor version number.
 * \return 
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *
 */
lwvmResult lwvmIRVersion(int *majorIR, int *minorIR, int *majorDbg, int *minorDbg);


/********************************//**
 *
 * \defgroup compilation Compilation
 *
 ***********************************/

/**
 * \ingroup compilation
 * \brief   LWVM Program 
 *
 * An opaque handle for a program 
 */
typedef struct _lwvmProgram *lwvmProgram;

/**
 * \ingroup compilation
 * \brief   Create a program, and set the value of its handle to *prog.
 *
 * \param   [in] prog LWVM program. 
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     lwvmDestroyProgram()
 */
lwvmResult lwvmCreateProgram(lwvmProgram *prog);


/**
 * \ingroup compilation
 * \brief   Destroy a program.
 *
 * \param    [in] prog LWVM program. 
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     lwvmCreateProgram()
 */
lwvmResult lwvmDestroyProgram(lwvmProgram *prog);


/**
 * \ingroup compilation
 * \brief   Add a module level LWVM IR to a program. 
 *
 * The buffer should contain an LWVM IR module.
 * The module should have LWVM IR version 1.6 either in the LLVM 7.0.1 bitcode
 * representation or in the LLVM 7.0.1 text representation. Support for reading
 * the text representation of LWVM IR is deprecated and may be removed in a
 * later version.
 *
 * \param   [in] prog   LWVM program.
 * \param   [in] buffer LWVM IR module in the bitcode or text
 *                      representation.
 * \param   [in] size   Size of the LWVM IR module.
 * \param   [in] name   Name of the LWVM IR module.
 *                      If NULL, "<unnamed>" is used as the name.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_INPUT \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmAddModuleToProgram(lwvmProgram prog, const char *buffer, size_t size, const char *name);

/**
 * \ingroup compilation
 * \brief   Add a module level LWVM IR to a program. 
 *
 * The buffer should contain an LWVM IR module. The module should have LWVM IR
 * version 1.6 in LLVM 7.0.1 bitcode representation.
 *
 * A module added using this API is lazily loaded - the only symbols loaded
 * are those that are required by module(s) loaded using
 * lwvmAddModuleToProgram. It is an error for a program to have
 * all modules loaded using this API. Compiler may also optimize entities
 * in this module by making them internal to the linked LWVM IR module,
 * making them eligible for other optimizations. Due to these
 * optimizations, this API to load a module is more efficient and should
 * be used where possible.
 * 
 * \param   [in] prog   LWVM program.
 * \param   [in] buffer LWVM IR module in the bitcode representation.
 * \param   [in] size   Size of the LWVM IR module.
 * \param   [in] name   Name of the LWVM IR module.
 *                      If NULL, "<unnamed>" is used as the name.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_INPUT \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmLazyAddModuleToProgram(lwvmProgram prog, const char *buffer, size_t size, const char *name);

/**
 * \ingroup compilation
 * \brief   Compile the LWVM program.
 *
 * The LWVM IR modules in the program will be linked at the IR level.
 * The linked IR program is compiled to PTX.
 *
 * The target datalayout in the linked IR program is used to
 * determine the address size (32bit vs 64bit).
 *
 * The valid compiler options are:
 *
 *   - -g (enable generation of debugging information, valid only with -opt=0)
 *   - -generate-line-info (generate line number information)
 *   - -opt=
 *     - 0 (disable optimizations)
 *     - 3 (default, enable optimizations)
 *   - -arch=
 *     - compute_35
 *     - compute_37
 *     - compute_50
 *     - compute_52 (default)
 *     - compute_53
 *     - compute_60
 *     - compute_61
 *     - compute_62
 *     - compute_70
 *     - compute_72
 *     - compute_75
 *     - compute_80
 *   - -ftz=
 *     - 0 (default, preserve denormal values, when performing
 *          single-precision floating-point operations)
 *     - 1 (flush denormal values to zero, when performing
 *          single-precision floating-point operations)
 *   - -prec-sqrt=
 *     - 0 (use a faster approximation for single-precision
 *          floating-point square root)
 *     - 1 (default, use IEEE round-to-nearest mode for
 *          single-precision floating-point square root)
 *   - -prec-div=
 *     - 0 (use a faster approximation for single-precision
 *          floating-point division and reciprocals)
 *     - 1 (default, use IEEE round-to-nearest mode for
 *          single-precision floating-point division and reciprocals)
 *   - -fma=
 *     - 0 (disable FMA contraction)
 *     - 1 (default, enable FMA contraction)
 *
 * \param   [in] prog       LWVM program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::lwvmResult LWVM_ERROR_IR_VERSION_MISMATCH \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_OPTION \endlink
 *   - \link ::lwvmResult LWVM_ERROR_NO_MODULE_IN_PROGRAM \endlink
 *   - \link ::lwvmResult LWVM_ERROR_COMPILATION \endlink
 */
lwvmResult lwvmCompileProgram(lwvmProgram prog, int numOptions, const char **options);   

/**
 * \ingroup compilation
 * \brief   Verify the LWVM program.
 *
 * The valid compiler options are:
 *
 * Same as for lwvmCompileProgram().
 *
 * \param   [in] prog       LWVM program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_OUT_OF_MEMORY \endlink
 *   - \link ::lwvmResult LWVM_ERROR_IR_VERSION_MISMATCH \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_IR \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_OPTION \endlink
 *   - \link ::lwvmResult LWVM_ERROR_NO_MODULE_IN_PROGRAM \endlink
 *
 * \see     lwvmCompileProgram()
 */
lwvmResult lwvmVerifyProgram(lwvmProgram prog, int numOptions, const char **options);

/**
 * \ingroup compilation
 * \brief   Get the size of the compiled result.
 *
 * \param   [in]  prog          LWVM program.
 * \param   [out] bufferSizeRet Size of the compiled result (including the
 *                              trailing NULL).
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmGetCompiledResultSize(lwvmProgram prog, size_t *bufferSizeRet);


/**
 * \ingroup compilation
 * \brief   Get the compiled result.
 *
 * The result is stored in the memory pointed by 'buffer'.
 *
 * \param   [in]  prog   LWVM program.
 * \param   [out] buffer Compiled result.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmGetCompiledResult(lwvmProgram prog, char *buffer);


/**
 * \ingroup compilation
 * \brief   Get the Size of Compiler/Verifier Message.
 *
 * The size of the message string (including the trailing NULL) is stored into
 * 'buffer_size_ret' when the return value is LWVM_SUCCESS.
 *   
 * \param   [in]  prog          LWVM program.
 * \param   [out] bufferSizeRet Size of the compilation/verification log
                                (including the trailing NULL).
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmGetProgramLogSize(lwvmProgram prog, size_t *bufferSizeRet);


/**
 * \ingroup compilation
 * \brief   Get the Compiler/Verifier Message
 *
 * The NULL terminated message string is stored in the memory pointed by
 * 'buffer' when the return value is LWVM_SUCCESS.
 *   
 * \param   [in]  prog   LWVM program program.
 * \param   [out] buffer Compilation/Verification log.
 * \return
 *   - \link ::lwvmResult LWVM_SUCCESS \endlink
 *   - \link ::lwvmResult LWVM_ERROR_ILWALID_PROGRAM \endlink
 */
lwvmResult lwvmGetProgramLog(lwvmProgram prog, char *buffer);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LWVM_H */
