//
// LWIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2014-2021, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
// LWIDIA_COPYRIGHT_END
//

#ifndef __LWRTC_H__
#define __LWRTC_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*************************************************************************//**
 *
 * \defgroup error Error Handling
 *
 * LWRTC defines the following enumeration type and function for API call
 * error handling.
 *
 ****************************************************************************/


/**
 * \ingroup error
 * \brief   The enumerated type lwrtcResult defines API call result codes.
 *          LWRTC API functions return lwrtcResult to indicate the call
 *          result.
 */
typedef enum {
  LWRTC_SUCCESS = 0,
  LWRTC_ERROR_OUT_OF_MEMORY = 1,
  LWRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  LWRTC_ERROR_ILWALID_INPUT = 3,
  LWRTC_ERROR_ILWALID_PROGRAM = 4,
  LWRTC_ERROR_ILWALID_OPTION = 5,
  LWRTC_ERROR_COMPILATION = 6,
  LWRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  LWRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  LWRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  LWRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  LWRTC_ERROR_INTERNAL_ERROR = 11
} lwrtcResult;


/**
 * \ingroup error
 * \brief   lwrtcGetErrorString is a helper function that returns a string
 *          describing the given lwrtcResult code, e.g., LWRTC_SUCCESS to
 *          \c "LWRTC_SUCCESS".
 *          For unrecognized enumeration values, it returns
 *          \c "LWRTC_ERROR unknown".
 *
 * \param   [in] result LWCA Runtime Compilation API result code.
 * \return  Message string for the given #lwrtcResult code.
 */
const char *lwrtcGetErrorString(lwrtcResult result);


/*************************************************************************//**
 *
 * \defgroup query General Information Query
 *
 * LWRTC defines the following function for general information query.
 *
 ****************************************************************************/


/**
 * \ingroup query
 * \brief   lwrtcVersion sets the output parameters \p major and \p minor
 *          with the LWCA Runtime Compilation version number.
 *
 * \param   [out] major LWCA Runtime Compilation major version number.
 * \param   [out] minor LWCA Runtime Compilation minor version number.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *
 */
lwrtcResult lwrtcVersion(int *major, int *minor);


/**
 * \ingroup query
 * \brief   lwrtcGetNumSupportedArchs sets the output parameter \p numArchs 
 *          with the number of architectures supported by LWRTC. This can 
 *          then be used to pass an array to ::lwrtcGetSupportedArchs to
 *          get the supported architectures.
 *
 * \param   [out] numArchs number of supported architectures.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *
 * see    ::lwrtcGetSupportedArchs
 */
lwrtcResult lwrtcGetNumSupportedArchs(int* numArchs);


/**
 * \ingroup query
 * \brief   lwrtcGetSupportedArchs populates the array passed via the output parameter 
 *          \p supportedArchs with the architectures supported by LWRTC. The array is
 *          sorted in the ascending order. The size of the array to be passed can be
 *          determined using ::lwrtcGetNumSupportedArchs.
 *
 * \param   [out] supportedArchs sorted array of supported architectures.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *
 * see    ::lwrtcGetNumSupportedArchs
 */
lwrtcResult lwrtcGetSupportedArchs(int* supportedArchs);


/*************************************************************************//**
 *
 * \defgroup compilation Compilation
 *
 * LWRTC defines the following type and functions for actual compilation.
 *
 ****************************************************************************/


/**
 * \ingroup compilation
 * \brief   lwrtcProgram is the unit of compilation, and an opaque handle for
 *          a program.
 *
 * To compile a LWCA program string, an instance of lwrtcProgram must be
 * created first with ::lwrtcCreateProgram, then compiled with
 * ::lwrtcCompileProgram.
 */
typedef struct _lwrtcProgram *lwrtcProgram;


/**
 * \ingroup compilation
 * \brief   lwrtcCreateProgram creates an instance of lwrtcProgram with the
 *          given input parameters, and sets the output parameter \p prog with
 *          it.
 *
 * \param   [out] prog         LWCA Runtime Compilation program.
 * \param   [in]  src          LWCA program source.
 * \param   [in]  name         LWCA program name.\n
 *                             \p name can be \c NULL; \c "default_program" is
 *                             used when \p name is \c NULL or "".
 * \param   [in]  numHeaders   Number of headers used.\n
 *                             \p numHeaders must be greater than or equal to 0.
 * \param   [in]  headers      Sources of the headers.\n
 *                             \p headers can be \c NULL when \p numHeaders is
 *                             0.
 * \param   [in]  includeNames Name of each header by which they can be
 *                             included in the LWCA program source.\n
 *                             \p includeNames can be \c NULL when \p numHeaders
 *                             is 0.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcDestroyProgram
 */
lwrtcResult lwrtcCreateProgram(lwrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames);


/**
 * \ingroup compilation
 * \brief   lwrtcDestroyProgram destroys the given program.
 *
 * \param    [in] prog LWCA Runtime Compilation program.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcCreateProgram
 */
lwrtcResult lwrtcDestroyProgram(lwrtcProgram *prog);


/**
 * \ingroup compilation
 * \brief   lwrtcCompileProgram compiles the given program.
 *
 * \param   [in] prog       LWCA Runtime Compilation program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.\n
 *                          \p options can be \c NULL when \p numOptions is 0.
 *
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_OPTION \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_COMPILATION \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
 *
 * It supports compile options listed in \ref options.
 */
lwrtcResult lwrtcCompileProgram(lwrtcProgram prog,
                                int numOptions, const char * const *options);


/**
 * \ingroup compilation
 * \brief   lwrtcGetPTXSize sets \p ptxSizeRet with the size of the PTX
 *          generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * \param   [in]  prog       LWCA Runtime Compilation program.
 * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
 *                           \c NULL).
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetPTX
 */
lwrtcResult lwrtcGetPTXSize(lwrtcProgram prog, size_t *ptxSizeRet);


/**
 * \ingroup compilation
 * \brief   lwrtcGetPTX stores the PTX generated by the previous compilation
 *          of \p prog in the memory pointed by \p ptx.
 *
 * \param   [in]  prog LWCA Runtime Compilation program.
 * \param   [out] ptx  Compiled result.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetPTXSize
 */
lwrtcResult lwrtcGetPTX(lwrtcProgram prog, char *ptx);


/**
 * \ingroup compilation
 * \brief   lwrtcGetLWBINSize sets \p lwbinSizeRet with the size of the lwbin
 *          generated by the previous compilation of \p prog. The value of
 *          lwbinSizeRet is set to 0 if the value specified to \c -arch is a
 *          virtual architecture instead of an actual architecture.
 *
 * \param   [in]  prog       LWCA Runtime Compilation program.
 * \param   [out] lwbinSizeRet Size of the generated lwbin.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetLWBIN
 */
lwrtcResult lwrtcGetLWBINSize(lwrtcProgram prog, size_t *lwbinSizeRet);


/**
 * \ingroup compilation
 * \brief   lwrtcGetLWBIN stores the lwbin generated by the previous compilation
 *          of \p prog in the memory pointed by \p lwbin. No lwbin is available
 *          if the value specified to \c -arch is a virtual architecture instead
 *          of an actual architecture.
 *
 * \param   [in]  prog LWCA Runtime Compilation program.
 * \param   [out] lwbin  Compiled and assembled result.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetLWBINSize
 */
lwrtcResult lwrtcGetLWBIN(lwrtcProgram prog, char *lwbin);


/**
 * \ingroup compilation
 * \brief   lwrtcGetProgramLogSize sets \p logSizeRet with the size of the
 *          log generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * Note that compilation log may be generated with warnings and informative
 * messages, even when the compilation of \p prog succeeds.
 *
 * \param   [in]  prog       LWCA Runtime Compilation program.
 * \param   [out] logSizeRet Size of the compilation log
 *                           (including the trailing \c NULL).
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetProgramLog
 */
lwrtcResult lwrtcGetProgramLogSize(lwrtcProgram prog, size_t *logSizeRet);


/**
 * \ingroup compilation
 * \brief   lwrtcGetProgramLog stores the log generated by the previous
 *          compilation of \p prog in the memory pointed by \p log.
 *
 * \param   [in]  prog LWCA Runtime Compilation program.
 * \param   [out] log  Compilation log.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_INPUT \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_ILWALID_PROGRAM \endlink
 *
 * \see     ::lwrtcGetProgramLogSize
 */
lwrtcResult lwrtcGetProgramLog(lwrtcProgram prog, char *log);


/**
 * \ingroup compilation
 * \brief   lwrtcAddNameExpression notes the given name expression
 *          denoting the address of a __global__ function 
 *          or __device__/__constant__ variable.
 *
 * The identical name expression string must be provided on a subsequent
 * call to lwrtcGetLoweredName to extract the lowered name.
 * \param   [in]  prog LWCA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of
 *               a __global__ function or __device__/__constant__ variable.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION \endlink
 *
 * \see     ::lwrtcGetLoweredName
 */
lwrtcResult lwrtcAddNameExpression(lwrtcProgram prog,
                                   const char * const name_expression);

/**
 * \ingroup compilation
 * \brief   lwrtcGetLoweredName extracts the lowered (mangled) name
 *          for a __global__ function or __device__/__constant__ variable,
 *          and updates *lowered_name to point to it. The memory containing
 *          the name is released when the LWRTC program is destroyed by 
 *          lwrtcDestroyProgram.
 *          The identical name expression must have been previously
 *          provided to lwrtcAddNameExpression.
 *
 * \param   [in]  prog LWCA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of 
 *               a __global__ function or __device__/__constant__ variable.
 * \param   [out] lowered_name initialized by the function to point to a
 *               C string containing the lowered (mangled)
 *               name corresponding to the provided name expression.
 * \return
 *   - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION \endlink
 *   - \link #lwrtcResult LWRTC_ERROR_NAME_EXPRESSION_NOT_VALID \endlink
 *
 * \see     ::lwrtcAddNameExpression
 */
lwrtcResult lwrtcGetLoweredName(lwrtcProgram prog,
                                const char *const name_expression,
                                const char** lowered_name);


/**
 * \defgroup options Supported Compile Options
 *
 * LWRTC supports the compile options below.
 * Option names with two preceding dashs (\c --) are long option names and
 * option names with one preceding dash (\c -) are short option names.
 * Short option names can be used instead of long option names.
 * When a compile option takes an argument, an assignment operator (\c =)
 * is used to separate the compile option argument from the compile option
 * name, e.g., \c "--gpu-architecture=compute_60".
 * Alternatively, the compile option name and the argument can be specified in
 * separate strings without an assignment operator, .e.g,
 * \c "--gpu-architecture" \c "compute_60".
 * Single-character short option names, such as \c -D, \c -U, and \c -I, do
 * not require an assignment operator, and the compile option name and the
 * argument can be present in the same string with or without spaces between
 * them.
 * For instance, \c "-D=<def>", \c "-D<def>", and \c "-D <def>" are all
 * supported.
 *
 * The valid compiler options are:
 *
 *   - Compilation targets
 *     - \c --gpu-architecture=\<arch\> (\c -arch)\n
 *       Specify the name of the class of GPU architectures for which the
 *       input must be compiled.\n
 *       - Valid <c>\<arch\></c>s:
 *         - \c compute_35
 *         - \c compute_37
 *         - \c compute_50
 *         - \c compute_52
 *         - \c compute_53
 *         - \c compute_60
 *         - \c compute_61
 *         - \c compute_62
 *         - \c compute_70
 *         - \c compute_72
 *         - \c compute_75
 *         - \c compute_80
 *         - \c sm_35
 *         - \c sm_37
 *         - \c sm_50
 *         - \c sm_52
 *         - \c sm_53
 *         - \c sm_60
 *         - \c sm_61
 *         - \c sm_62
 *         - \c sm_70
 *         - \c sm_72
 *         - \c sm_75
 *         - \c sm_80
 *       - Default: \c compute_52
 *   - Separate compilation / whole-program compilation
 *     - \c --device-c (\c -dc)\n
 *       Generate relocatable code that can be linked with other relocatable
 *       device code.  It is equivalent to --relocatable-device-code=true.
 *     - \c --device-w (\c -dw)\n
 *       Generate non-relocatable code.  It is equivalent to
 *       \c --relocatable-device-code=false.
 *     - \c --relocatable-device-code={true|false} (\c -rdc)\n
 *       Enable (disable) the generation of relocatable device code.
 *       - Default: \c false
 *     - \c --extensible-whole-program (\c -ewp)\n
 *       Do extensible whole program compilation of device code.
 *       - Default: \c false
 *   - Debugging support
 *     - \c --device-debug (\c -G)\n
 *       Generate debug information.
 *     - \c --generate-line-info (\c -lineinfo)\n
 *       Generate line-number information.
 *   - Code generation
 *     - \c --ptxas-options \<options\> (\c -Xptxas)\n
 *       Specify options directly to ptxas, the PTX optimizing assembler.
 *     - \c --maxrregcount=\<N\> (\c -maxrregcount)\n
 *       Specify the maximum amount of registers that GPU functions can use.
 *       Until a function-specific limit, a higher value will generally
 *       increase the performance of individual GPU threads that execute this
 *       function.  However, because thread registers are allocated from a
 *       global register pool on each GPU, a higher value of this option will
 *       also reduce the maximum thread block size, thereby reducing the amount
 *       of thread parallelism.  Hence, a good maxrregcount value is the result
 *       of a trade-off.  If this option is not specified, then no maximum is
 *       assumed.  Value less than the minimum registers required by ABI will
 *       be bumped up by the compiler to ABI minimum limit.
 *     - \c --ftz={true|false} (\c -ftz)\n
 *       When performing single-precision floating-point operations, flush
 *       denormal values to zero or preserve denormal values.
 *       \c --use_fast_math implies \c --ftz=true.
 *       - Default: \c false
 *     - \c --prec-sqrt={true|false} (\c -prec-sqrt)\n
 *       For single-precision floating-point square root, use IEEE
 *       round-to-nearest mode or use a faster approximation.
 *       \c --use_fast_math implies \c --prec-sqrt=false.
 *       - Default: \c true
 *     - \c --prec-div={true|false} (\c -prec-div)\n
 *       For single-precision floating-point division and reciprocals, use IEEE
 *       round-to-nearest mode or use a faster approximation.
 *       \c --use_fast_math implies \c --prec-div=false.
 *       - Default: \c true
 *     - \c --fmad={true|false} (\c -fmad)\n
 *       Enables (disables) the contraction of floating-point multiplies and
 *       adds/subtracts into floating-point multiply-add operations (FMAD,
 *       FFMA, or DFMA).  \c --use_fast_math implies \c --fmad=true.
 *       - Default: \c true
 *     - \c --use_fast_math (\c -use_fast_math)\n
 *       Make use of fast math operations.
 *       \c --use_fast_math implies \c --ftz=true \c --prec-div=false
 *       \c --prec-sqrt=false \c --fmad=true.
 *     - \c --extra-device-vectorization (\c -extra-device-vectorization)\n
 *       Enables more aggressive device code vectorization in the LWVM optimizer.
 *     - \c --modify-stack-limit={true|false} (\c -modify-stack-limit)\n
 *       On Linux, during compilation, use \c setrlimit() to increase stack size 
 *       to maximum allowed. The limit is reset to the previous value at the
 *       end of compilation.
 *       Note: \c setrlimit() changes the value for the entire process.
 *       - Default: \c true
 *   - Preprocessing
 *     - \c --define-macro=\<def\> (\c -D)\n
 *       \c \<def\> can be either \c \<name\> or \c \<name=definitions\>.
 *       - \c \<name\> \n
 *         Predefine \c \<name\> as a macro with definition \c 1.
 *       - \c \<name\>=\<definition\> \n
 *         The contents of \c \<definition\> are tokenized and preprocessed
 *         as if they appeared during translation phase three in a \c \#define
 *         directive.  In particular, the definition will be truncated by
 *         embedded new line characters.
 *     - \c --undefine-macro=\<def\> (\c -U)\n
 *       Cancel any previous definition of \c \<def\>.
 *     - \c --include-path=\<dir\> (\c -I)\n
 *       Add the directory \c \<dir\> to the list of directories to be
 *       searched for headers.  These paths are searched after the list of
 *       headers given to ::lwrtcCreateProgram.
 *     - \c --pre-include=\<header\> (\c -include)\n
 *       Preinclude \c \<header\> during preprocessing.
 *   - Language Dialect
 *     - \c --std={c++03|c++11|c++14|c++17} (\c -std={c++11|c++14|c++17})\n
 *       Set language dialect to C++03, C++11, C++14 or C++17
 *     - \c --builtin-move-forward={true|false} (\c -builtin-move-forward)\n
 *       Provide builtin definitions of \c std::move and \c std::forward,
 *       when C++11 language dialect is selected.
 *       - Default: \c true
 *     - \c --builtin-initializer-list={true|false}
 *       (\c -builtin-initializer-list)\n
 *       Provide builtin definitions of \c std::initializer_list class and
 *       member functions when C++11 language dialect is selected.
 *       - Default: \c true
 *   - Misc.
 *     - \c --disable-warnings (\c -w)\n
 *       Inhibit all warning messages.
 *     - \c --restrict (\c -restrict)\n
 *       Programmer assertion that all kernel pointer parameters are restrict
 *       pointers.
 *     - \c --device-as-default-exelwtion-space
 *       (\c -default-device)\n
 *       Treat entities with no exelwtion space annotation as \c __device__
 *       entities.
 *     - \c --optimization-info=\<kind\> (\c -opt-info)\n
 *       Provide optimization reports for the specified kind of optimization.
 *       The following kind tags are supported:
 *         - \c inline : emit a remark when a function is inlined.
 *     - \c --version-ident={true|false} (\c -dQ)\n
 *       Embed used compiler's version info into generated PTX/LWBIN 
 *       - Default: \c false
 *     - \c --display-error-number (\c -err-no)\n
 *       Display diagnostic number for warning messages.
 *     - \c --diag-error=<error-number>,... (\c -diag-error)\n
 *       Emit error for specified diagnostic message number(s). Message numbers can be separated by comma.
 *     - \c --diag-suppress=<error-number>,... (\c -diag-suppress)\n
 *       Suppress specified diagnostic message number(s). Message numbers can be separated by comma.
 *     - \c --diag-warn=<error-number>,... (\c -diag-warn)\n
 *       Emit warning for specified diagnostic message number(s). Message numbers can be separated by comma.
 *
 */


#ifdef __cplusplus
}
#endif /* __cplusplus */


/* The utility function 'lwrtcGetTypeName' is not available by default. Define
   the macro 'LWRTC_GET_TYPE_NAME' to a non-zero value to make it available.
*/
   
#if LWRTC_GET_TYPE_NAME || __DOXYGEN_ONLY__

#if LWRTC_USE_CXXABI || __clang__ || __GNUC__ || __DOXYGEN_ONLY__
#include <cxxabi.h>
#include <cstdlib>

#elif defined(_WIN32)
#include <Windows.h>
#include <DbgHelp.h>
#endif /* LWRTC_USE_CXXABI || __clang__ || __GNUC__ */


#include <string>
#include <typeinfo>

template <typename T> struct __lwrtcGetTypeName_helper_t { };

/*************************************************************************//**
 *
 * \defgroup hosthelper Host Helper
 *
 * LWRTC defines the following functions for easier interaction with host code.
 *
 ****************************************************************************/

/**
 * \ingroup hosthelper
 * \brief   lwrtcGetTypeName stores the source level name of a type in the given 
 *          std::string location. 
 *
 * This function is only provided when the macro LWRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return LWRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - lwrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] tinfo: reference to object of type std::type_info for a given type.
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *  - \link #lwrtcResult LWRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
inline lwrtcResult lwrtcGetTypeName(const std::type_info &tinfo, std::string *result)
{
#if USE_CXXABI || __clang__ || __GNUC__
  const char *name = tinfo.name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  if (status == 0) {
    *result = undecorated_name;
    free(undecorated_name);
    return LWRTC_SUCCESS;
  }
#elif defined(_WIN32)
  const char *name = tinfo.raw_name();
  if (!name || *name != '.') {
    return LWRTC_ERROR_INTERNAL_ERROR;
  }
  char undecorated_name[4096];
  //name+1 skips over the '.' prefix
  if(UnDecorateSymbolName(name+1, undecorated_name,
                          sizeof(undecorated_name) / sizeof(*undecorated_name),
                           //note: doesn't seem to work correctly without UNDNAME_NO_ARGUMENTS.
                           UNDNAME_NO_ARGUMENTS | UNDNAME_NAME_ONLY ) ) {
    *result = undecorated_name;
    return LWRTC_SUCCESS;
  }
#endif  /* USE_CXXABI || __clang__ || __GNUC__ */

  return LWRTC_ERROR_INTERNAL_ERROR;
}

/**
 * \ingroup hosthelper
 * \brief   lwrtcGetTypeName stores the source level name of the template type argument
 *          T in the given std::string location.
 *
 * This function is only provided when the macro LWRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return LWRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - lwrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #lwrtcResult LWRTC_SUCCESS \endlink
 *  - \link #lwrtcResult LWRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
 
template <typename T>
lwrtcResult lwrtcGetTypeName(std::string *result)
{
  lwrtcResult res = lwrtcGetTypeName(typeid(__lwrtcGetTypeName_helper_t<T>), 
                                     result);
  if (res != LWRTC_SUCCESS) 
    return res;

  std::string repr = *result;
  std::size_t idx = repr.find("__lwrtcGetTypeName_helper_t");
  idx = (idx != std::string::npos) ? repr.find("<", idx) : idx;
  std::size_t last_idx = repr.find_last_of('>');
  if (idx == std::string::npos || last_idx == std::string::npos) {
    return LWRTC_ERROR_INTERNAL_ERROR;
  }
  ++idx;
  *result = repr.substr(idx, last_idx - idx);
  return LWRTC_SUCCESS;
}

#endif  /* LWRTC_GET_TYPE_NAME */

#endif /* __LWRTC_H__ */
