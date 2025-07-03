/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/** \file lwphy.h
 *  \brief PHY Layer library header file
 *
 *  Header file for the lwPHY API
 */

#if !defined(LWPHY_H_INCLUDED_)
#define LWPHY_H_INCLUDED_

#include <lwda_runtime.h>
#include <stdint.h>
#include "lwComplex.h"
#include "lwda_fp16.h"

#ifndef LWPHYWINAPI
#ifdef _WIN32
#define LWPHYWINAPI __stdcall
#else
#define LWPHYWINAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */
#define MAX_N_DMRSSYMS_SUPPORTED 4
#define MAX_ENCODED_CODE_BLOCK_BIT_SIZE 25344
#define MAX_DECODED_CODE_BLOCK_BIT_SIZE 8448
#define MAX_N_BBU_LAYERS_SUPPORTED 16
#define MAX_N_ANTENNAS_SUPPORTED 16
#define MAX_N_CARRIERS_SUPPORTED 1
#define MAX_NF_SUPPORTED (3276 * MAX_N_CARRIERS_SUPPORTED)
#define MAX_NH_SUPPORTED 1
#define MAX_ND_SUPPORTED 14
#define MAX_N_TBS_SUPPORTED 10         // maximum number of transport blocks supported
#define MAX_N_CBS_PER_TB_SUPPORTED 500 // maximum number of code blocks per transport block supported
#define MAX_TOTAL_N_CBS_SUPPORTED ((MAX_N_TBS_SUPPORTED) * (MAX_N_CBS_PER_TB_SUPPORTED))
#define MAX_BYTES_PER_TRANSPORT_BLOCK 144384
#define MAX_WORDS_PER_TRANSPORT_BLOCK (MAX_BYTES_PER_TRANSPORT_BLOCK / sizeof(uint32_t))

#define LWPHY_LDPC_MAX_LIFTING_SIZE (384)
#define LWPHY_LDPC_MAX_BG1_PARITY_NODES (46)
#define LWPHY_LDPC_MAX_BG2_PARITY_NODES (42)
#define LWPHY_LDPC_MAX_BG1_VAR_NODES (68)
#define LWPHY_LDPC_MAX_BG2_VAR_NODES (52)
#define LWPHY_LDPC_BG1_INFO_NODES (22)
#define LWPHY_LDPC_MAX_BG2_INFO_NODES (10)
#define LWPHY_LDPC_NUM_PUNCTURED_NODES (2)
#define LWPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES (LWPHY_LDPC_MAX_BG1_VAR_NODES - LWPHY_LDPC_NUM_PUNCTURED_NODES)
#define LWPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES (LWPHY_LDPC_MAX_BG2_VAR_NODES - LWPHY_LDPC_NUM_PUNCTURED_NODES)

#define QAM_STRIDE 8 // Stride for UE QAM symbols in the equalizer output

// Transport block parameters
typedef struct tb_pars
{
    // MIMO
    uint32_t numLayers;
    uint64_t layerMap; //[MAXLAYERS];    This field is a bit map for now

    // Resource allocation
    uint32_t startPrb;
    uint32_t numPrb;
    uint32_t startSym;
    uint32_t numSym;
    uint32_t dataScramId;
    // Back-end parameters
    uint32_t mcsTableIndex;
    uint32_t mcsIndex;
    uint32_t rv;

    // DMRS parameters
    uint32_t dmrsType;
    uint32_t dmrsAddlPosition;
    uint32_t dmrsMaxLength;
    uint32_t dmrsScramId;
    uint32_t dmrsEnergy;

    uint32_t dmrsCfg;
    uint32_t nRnti;

    uint32_t nPortIndex; //up to 8 layers encoded in an uint32_t, in groups of 4 bits
    uint32_t nSCID;
} tb_pars;

typedef struct gnb_pars
{
    uint32_t fc;
    uint32_t mu;
    uint32_t nRx;
    uint32_t nPrb;
    uint32_t cellId;
    uint32_t slotNumber; // @todo: slotNumber to be removed since its a slot level parameter
    uint32_t Nf;
    uint32_t Nt;
    uint32_t df;
    uint32_t dt;
    uint32_t numBsAnt;
    uint32_t numBbuLayers;
    uint32_t numTb;
    uint32_t ldpcnIterations;
    uint32_t ldpcEarlyTermination;
    uint32_t ldpcAlgoIndex;
    uint32_t ldpcFlags;
    uint32_t ldplwseHalf;
    uint32_t slotType; // 0 UL, 1 DL
} gnb_pars;

/**
 * lwPHY error codes
 */
typedef enum
{
    LWPHY_STATUS_SUCCESS            = 0,  /*!< The API call returned with no errors.                                */
    LWPHY_STATUS_INTERNAL_ERROR     = 1,  /*!< An unexpected, internal error oclwrred.                              */
    LWPHY_STATUS_NOT_SUPPORTED      = 2,  /*!< The requested function is not lwrrently supported.                   */
    LWPHY_STATUS_ILWALID_ARGUMENT   = 3,  /*!< One or more of the arguments provided to the function was invalid.   */
    LWPHY_STATUS_ARCH_MISMATCH      = 4,  /*!< The requested operation is not supported on the current architecture.*/
    LWPHY_STATUS_ALLOC_FAILED       = 5,  /*!< A memory allocation failed.                                          */
    LWPHY_STATUS_SIZE_MISMATCH      = 6,  /*!< The size of the operands provided to the function do not match.      */
    LWPHY_STATUS_MEMCPY_ERROR       = 7,  /*!< An error oclwrred during a memcpy operation.                         */
    LWPHY_STATUS_ILWALID_COLWERSION = 8,  /*!< An invalid colwersion operation was requested.                       */
    LWPHY_STATUS_UNSUPPORTED_TYPE   = 9,  /*!< An operation was requested on an unsupported type.                   */
    LWPHY_STATUS_UNSUPPORTED_LAYOUT = 10, /*!< An operation was requested on an unsupported layout.                 */
    LWPHY_STATUS_UNSUPPORTED_RANK   = 11, /*!< An operation was requested on an unsupported rank.                   */
    LWPHY_STATUS_UNSUPPORTED_CONFIG = 12  /*!< An operation was requested on an unsupported configuration.          */
} lwphyStatus_t;

// Note: LWDA_R/C values are defined in LWCA library_types.h header

/**
 * lwPHY data types
 */
typedef enum
{
    LWPHY_VOID  = -1,         /*!< uninitialized type                       */
    LWPHY_BIT   = 20,         /*!< 1-bit value                              */
    LWPHY_R_8I  = LWDA_R_8I,  /*!< 8-bit signed integer real values         */
    LWPHY_C_8I  = LWDA_C_8I,  /*!< 8-bit signed integer complex values      */
    LWPHY_R_8U  = LWDA_R_8U,  /*!< 8-bit unsigned integer real values       */
    LWPHY_C_8U  = LWDA_C_8U,  /*!< 8-bit unsigned integer complex values    */
    LWPHY_R_16I = 21,         /*!< 16-bit signed integer real values        */
    LWPHY_C_16I = 22,         /*!< 16-bit signed integer complex values     */
    LWPHY_R_16U = 23,         /*!< 16-bit unsigned integer real values      */
    LWPHY_C_16U = 24,         /*!< 16-bit unsigned integer complex values   */
    LWPHY_R_32I = LWDA_R_32I, /*!< 32-bit signed integer real values        */
    LWPHY_C_32I = LWDA_C_32I, /*!< 32-bit signed integer complex values     */
    LWPHY_R_32U = LWDA_R_32U, /*!< 32-bit unsigned integer real values      */
    LWPHY_C_32U = LWDA_C_32U, /*!< 32-bit unsigned integer complex values   */
    LWPHY_R_16F = LWDA_R_16F, /*!< half precision (16-bit) real values      */
    LWPHY_C_16F = LWDA_C_16F, /*!< half precision (16-bit) complex values   */
    LWPHY_R_32F = LWDA_R_32F, /*!< single precision (32-bit) real values    */
    LWPHY_C_32F = LWDA_C_32F, /*!< single precision (32-bit) complex values */
    LWPHY_R_64F = LWDA_R_64F, /*!< single precision (64-bit) real values    */
    LWPHY_C_64F = LWDA_C_64F  /*!< double precision (64-bit) complex values */
} lwphyDataType_t;

typedef struct
{
    int type;
    union
    {
        unsigned int    b1;   /*!< LWPHY_BIT   (1-bit value)                              */
        signed char     r8i;  /*!< LWPHY_R_8I  (8-bit signed integer real values)         */
        char2           c8i;  /*!< LWPHY_C_8I  (8-bit signed integer complex values)      */
        unsigned char   r8u;  /*!< LWPHY_R_8U  (8-bit unsigned integer real values)       */
        uchar2          c8u;  /*!< LWPHY_C_8U  (8-bit unsigned integer complex values)    */
        short           r16i; /*!< LWPHY_R_16I (16-bit signed integer real values)        */
        short2          c16i; /*!< LWPHY_C_16I (16-bit signed integer complex values)     */
        unsigned short  r16u; /*!< LWPHY_R_16U (16-bit unsigned integer real values)      */
        ushort2         c16u; /*!< LWPHY_C_16U (16-bit unsigned integer complex values)   */
        int             r32i; /*!< LWPHY_R_32I (32-bit signed integer real values)        */
        int2            c32i; /*!< LWPHY_C_32I (32-bit signed integer complex values)     */
        unsigned int    r32u; /*!< LWPHY_R_32U (32-bit unsigned integer real values)      */
        uint2           c32u; /*!< LWPHY_C_32U (32-bit unsigned integer complex values)   */
        unsigned short  r16f; /*!< LWPHY_R_16F (half precision (16-bit) real values)      */
        ushort2         c16f; /*!< LWPHY_C_16F (half precision (16-bit) complex values)   */
        float           r32f; /*!< LWPHY_R_32F (single precision (32-bit) real values)    */
        lwComplex       c32f; /*!< LWPHY_C_32F (single precision (32-bit) complex values) */
        double          r64f; /*!< LWPHY_R_64F (single precision (64-bit) real values)    */
        lwDoubleComplex c64f; /*!< LWPHY_C_64F (double precision (64-bit) complex values) */
    } value;
} lwphyVariant_t;

/**
 * \defgroup LWPHY_ERROR Error Handling
 *
 * This section describes the error handling functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code.  If the error
 * code is not recognized, "Unknown status code" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwphyGetErrorName, ::lwphyStatus_t
 */
const char* LWPHYWINAPI lwphyGetErrorString(lwphyStatus_t status);

/******************************************************************/ /**
 * \brief Returns a string version of an error code enumeration value
 *
 * Returns a string version of an error code.  If the error
 * code is not recognized, "LWPHY_UNKNOWN_STATUS" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwphyGetErrorString, ::lwphyStatus_t
 */
const char* LWPHYWINAPI lwphyGetErrorName(lwphyStatus_t status);

/** @} */ /* END LWPHY_ERROR */

/**
 * Maximum supported number of tensor dimensions 
 */
#define LWPHY_DIM_MAX 4

/* lwphySetTensorDescriptor() flags */
/* Use strides if provided, otherwise TIGHT                */
#define LWPHY_TENSOR_ALIGN_DEFAULT 0x00
/* Pack tightly, regardless of stride values               */
#define LWPHY_TENSOR_ALIGN_TIGHT 0x01
/* Align 2nd dimension for coalesced I/O (ignore strides)  */
#define LWPHY_TENSOR_ALIGN_COALESCE 0x02
/* Interpret strides as dimension orderings                */
#define LWPHY_TENSOR_STRIDES_AS_ORDER 0x04

/* QAM levels - set value set to log2(QAM) value */
#define LWPHY_QAM_4 (2)
#define LWPHY_QAM_16 (4)
#define LWPHY_QAM_64 (6)
#define LWPHY_QAM_256 (8)

/* # of subcarriers/tones per PRB */
#define LWPHY_N_TONES_PER_PRB (12)

/* DMRS configurations supported */
#define LWPHY_DMRS_CFG0 (0) // 1 layer : DMRS grid 0   ; fOCC = [+1, +1]          ; 1 DMRS symbol
#define LWPHY_DMRS_CFG1 (1) // 2 layers: DMRS grids 0,1; fOCC = [+1, +1]          ; 1 DMRS symbol
#define LWPHY_DMRS_CFG2 (2) // 4 layers: DMRS grids 0,1; fOCC = [+1, +1], [+1, -1]; 1 DMRS symbol
#define LWPHY_DMRS_CFG3 (3) // 8 layers: DMRS grids 0,1; fOCC/tOCC = [+1, +1], [+1, -1]; 4 DMRS symbols

/* Maximum number of downlink layers per transport block (TB) */
#define MAX_DL_LAYERS_PER_TB 16 //same as MAX_N_BBU_LAYERS_SUPPORTED 16

/**
 * \defgroup LWPHY_CONTEXT Library context
 *
 * This section describes the context functions of the lwPHY application
 * programming interface.
 *
 * @{
 */

struct lwphyContext;
/**
 * lwPHY context
 */
typedef struct lwphyContext* lwphyContext_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a lwPHY context
 *
 * Allocates a lwPHY library context and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p pcontext is NULL.
 * 
 * Returns ::LWPHY_STATUS_ALLOC_FAILED if a context cannot be allocated
 * on the host.
 *
 * Returns ::LWPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pcontext - Address to return the new ::lwphyContext_t instance
 * \param flags - Creation flags (lwrrently unused)
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ALLOC_FAILED,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyGetErrorName,::lwphyGetErrorString,::lwphyDestroyContext
 */
lwphyStatus_t LWPHYWINAPI lwphyCreateContext(lwphyContext_t* pcontext,
                                             unsigned int    flags);

/******************************************************************/ /**
 * \brief Destroys a lwPHY context
 *
 * Destroys a lwPHY context object that was previously created by a call
 * to ::lwphyCreateContext. The handle provided to this function should
 * not be used for any operations after this function returns.
 * 
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p decoder is NULL.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param ctx - previously allocated ::lwphyContext_t instance
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateContext
 */
lwphyStatus_t LWPHYWINAPI lwphyDestroyContext(lwphyContext_t ctx);

/** @} */ /* END LWPHY_CONTEXT */

/**
 * \defgroup LWPHY_TENSOR_DESC Tensor Descriptors
 *
 * This section describes the tensor descriptor functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

struct lwphyTensorDescriptor;
/**
 * lwPHY Tensor Descriptor handle
 */
typedef struct lwphyTensorDescriptor* lwphyTensorDescriptor_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a lwPHY tensor descriptor
 *
 * Allocates a lwPHY tensor descriptor and returns a handle in the address
 * provided by the caller.
 *
 * The allocated descriptor will have type ::LWPHY_VOID, and (in most
 * cases) cannot be used for operations until the tensor state has been
 * initialized by calling ::lwphySetTensorDescriptor.
 *
 * Upon successful return the tensor descriptor will have a rank of 0.
 * 
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p ptensorDesc is NULL.
 * 
 * Returns ::LWPHY_STATUS_ALLOC_FAILED if a tensor descriptor cannot be
 * allocated on the host.
 *
 * Returns ::LWPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param ptensorDesc - Address for the new ::lwphyTensorDescriptor_t
 * instance
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ALLOC_FAILED,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyGetErrorName,::lwphyGetErrorString
 */
lwphyStatus_t LWPHYWINAPI lwphyCreateTensorDescriptor(lwphyTensorDescriptor_t* ptensorDesc);

/******************************************************************/ /**
 * \brief Destroys a lwPHY tensor descriptor
 *
 * Destroys a lwPHY tensor descriptor that was previously allocated by
 * a call to ::lwphyCreateTensorDescriptor. The handle provided to this
 * function should not be used for any operations after this function
 * returns.
 * 
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p tensorDesc is NULL.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param tensorDesc - previously allocated ::lwphyTensorDescriptor_t
 * instance
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateTensorDescriptor
 */
lwphyStatus_t LWPHYWINAPI lwphyDestroyTensorDescriptor(lwphyTensorDescriptor_t tensorDesc);

/******************************************************************/ /**
 * \brief Provide values for the internal state of a lwPHY tensor descriptor
 *
 * Sets the internal state of a tensor descriptor that was created via
 * the ::lwphyCreateTensorDescriptor function.
 *
 * Note that a tensor descriptor is not associated with a specific memory
 * allocation or address. A tensor descriptor provides the lwPHY library with
 * values that can be used "interpret" a range of memory as a tensor with
 * the specified properties. A tensor descriptor can be used with multiple
 * different addresses, and an address can be accessed with multiple different
 * tensor descriptors.
 *
 * \param tensorDesc - previously allocated ::lwphyTensorDescriptor_t
 * instance
 * \param type - ::lwphyDataType_t enumeration with the desired tensor
 * element type
 * \param numDimensions - the desired tensor rank
 * \param dimensions - an array of dimensions for the tensor descriptor
 * \param strides - an array of strides (may be NULL)
 * \param flags - tensor descriptor flags
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if:
 * <ul>
 *   <li>\p tensorDesc is NULL.</li>
 *   <li>\p dimensions is NULL.</li>
 *   <li>\p numDimensions <= 0.</li>
 *   <li>\p numDimensions > ::LWPHY_DIM_MAX.</li>
 *   <li>\p flags has the LWPHY_TENSOR_STRIDES_AS_ORDER bit set and \p str
 *          is NULL.</li>
 *   <li>\p type is ::LWPHY_VOID.</li>
 *   <li>Any element of the dimensions array is less than equal to 0.</li>
 * </ul>
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the state update was successful.
 * 
 * The stride of a given dimension describes the distance between two
 * elements that differ by 1 in that dimension. For example, a 2-dimensional,
 * (10 x 8) matrix with no padding would have a stride[0] = 1 and stride[1] =
 * 10.
 *
 * There is no requirement that strides be in ascending order.
 * 
 * The \p flags argument can be used to request that the lwPHY library
 * automatically callwlate values for the tensor strides, as a
 * colwenience. The values allowed for \p flags are:
 * <ul>
 *   <li>LWPHY_TENSOR_ALIGN_DEFAULT: If strides are provided, they will
 *       be used. Otherwise, set the strides for tight packing.</li>
 *   <li>LWPHY_TENSOR_ALIGN_TIGHT: Set the strides so that no padding is
 *       present. stride[0] = 1, and stride[i] = dimensions[i - 1] * strides[i - 1]</li>
 *   <li>LWPHY_TENSOR_ALIGN_COALESCE: Set the strides for the first dimension
 *       based on the element type, so that the stride (in bytes) will be a
 *       multiple of 128.</li>
 *   <li>LWPHY_TENSOR_STRIDES_AS_ORDER: Interpret the values provided for the
 *       \p strides parameter as indices into the \p dimensions parameter,
 *       and set the strides to reflect that order of dimensions.</li>
 * </ul>
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateTensorDescriptor
 */
lwphyStatus_t LWPHYWINAPI lwphySetTensorDescriptor(lwphyTensorDescriptor_t tensorDesc,
                                                   lwphyDataType_t         type,
                                                   int                     numDimensions,
                                                   const int               dimensions[],
                                                   const int               strides[],
                                                   unsigned int            flags);

/******************************************************************/ /**
 * \brief Query values for the internal state of a lwPHY tensor descriptor
 *
 * Retrieves the internal state of a tensor descriptor that was created via
 * the ::lwphyCreateTensorDescriptor function and initialized with the
 * ::lwphySetTensorDescriptor function
 *
 * \param tensorDesc - previously allocated ::lwphyTensorDescriptor_t
 * instance
 * \param numDimsRequested - the size of the array provided by the \p dimensions
 *        parameter, and the \p strides parameter (if non-NULL)
 * \param dataType - address for the returned ::lwphyDataType_t (may be NULL)
 * \param numDims - output address for the rank of the tensor descriptor (may be NULL)
 * \param dimensions - output location for dimensions for the tensor descriptor
 * \param strides - output location for tensor strides (may be NULL)
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p tensorDesc is NULL, or if
 * \p numDimsRequested > 0 and dimensions is NULL.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the state query was successful.
 * 
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateTensorDescriptor,::lwphySetTensorDescriptor
 */
lwphyStatus_t LWPHYWINAPI lwphyGetTensorDescriptor(const lwphyTensorDescriptor_t tensorDesc,
                                                   int                           numDimsRequested,
                                                   lwphyDataType_t*              dataType,
                                                   int*                          numDims,
                                                   int                           dimensions[],
                                                   int                           strides[]);

/******************************************************************/ /**
 * \brief Returns the size of an allocation for a tensor descriptor
 *
 * Callwlates the size (in bytes) of an allocation that would be required
 * to represent a tensor described by the given descriptor.
 *
 * \param tensorDesc - previously allocated ::lwphyTensorDescriptor_t
 * instance
 * \param psz - address to hold the callwlated size output
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p tensorDesc is NULL, or if
 * \p psz is NULL.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the size callwlation was successful.
 * 
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateTensorDescriptor,::lwphySetTensorDescriptor
 */
lwphyStatus_t LWPHYWINAPI lwphyGetTensorSizeInBytes(const lwphyTensorDescriptor_t tensorDesc,
                                                    size_t*                       psz);

/******************************************************************/ /**
 * \brief Returns a string value for a given data type
 *
 * Returns a string for the given ::lwphyDataType_t, or "UNKNOWN_TYPE"
 * if the type is unknown.
 *
 * \param type - data type (::lwphyDataType_t)
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwphyDataType_t
 */
const char* LWPHYWINAPI lwphyGetDataTypeString(lwphyDataType_t type);

/** @} */ /* END LWPHY_TENSOR_DESC */

/**
 * \defgroup LWPHY_TENSOR_OPS Tensor Operations
 *
 * This section describes the tensor operation functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Colwerts a source tensor to a different type or layout
 *
 * Colwerts an input tensor (described by an address and a tensor
 * descriptor) to an output tensor, possibly changing layout and/or
 * data type in the process.
 * The input and output tensors must have the same dimensions.
 * 
 * Tensors with identical data types, dimensions, and strides may be
 * colwerted internally using a memory copy operation.
 *
 * The following colwersions are lwrrently supported:
 * <ul>
 *   <li>Colwersion of all types to tensors with the same dimensions but different
 *   strides</li>
 *   <li>Widening colwersions (e.g. colwersion of a signed, unsigned, or
 *   floating point fundamental type to the same fundamental type with a
 *   larger range (e.g. LWPHY_R_8I to LWPHY_R_32I)</li>
 * </ul>
 *
 * Other colwersions are possible and may be added in the future.
 *
 * \param tensorDescDst - previously allocated ::lwphyTensorDescriptor_t for
 * the destination (output)
 * \param dstAddr - tensor address for output data
 * \param tensorDescSrc - previously allocated ::lwphyTensorDescriptor_t for
 * source data
 * \param srcAddr - tensor address for input data
 * \param strm - LWCA stream for memory copy
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if any of \p tensorDescDst, \p dstAddr,
 * \p tensorDescSrc, or \p srcAddr is NULL, or if the data type of either
 * \p tensorDescDst or \p tensorDescSrc is LWPHY_VOID.
 *
 * Returns ::LWPHY_STATUS_SIZE_MISMATCH if all dimensions of tensor descriptors
 * \p tensorDescDst and \p tensorDescSrc do not match.
 *
 * Returns ::LWPHY_STATUS_MEMCPY_ERROR if an error oclwrred performing a memory
 * copy from the source to the destination.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the colwersion operation was submitted to
 * the given stream successfully.
 * 
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_SIZE_MISMATCH
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateTensorDescriptor,::lwphySetTensorDescriptor
 */
lwphyStatus_t LWPHYWINAPI lwphyColwertTensor(const lwphyTensorDescriptor_t tensorDescDst,
                                             void*                         dstAddr,
                                             lwphyTensorDescriptor_t       tensorDescSrc,
                                             const void*                   srcAddr,
                                             lwdaStream_t                  strm);
/** @} */ /* END LWPHY_TENSOR_OPS */

/**
 * \defgroup LWPHY_CHANNEL_ESTIMATION Channel Estimation
 *
 * This section describes the channel estimation functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Performs 1-D time/frequency channel estimation
 *
 * Performs MMSE channel estimation using 1-D interpolation in the time
 * and frequency dimensions
 *
 * \param tensorDescDst - tensor descriptor for output
 * \param dstAddr - address for tensor output
 * \param tensorDescSymbols - tensor descriptor for input symbol data
 * \param symbolsAddr - address for input symbol data
 * \param tensorDescFreqFilters - tensor descriptor for input frequency filters
 * \param freqFiltersAddr - address for input frequency filters
 * \param tensorDescTimeFilters - tensor descriptor for input time filters
 * \param timeFiltersAddr - address for input time filters
 * \param tensorDescFreqIndices - tensor descriptor for pilot symbol frequency indices 
 * \param freqIndicesAddr - address for pilot symbol frequency indices
 * \param tensorDescTimeIndices - tensor descriptor for pilot symbol time indices
 * \param timeIndicesAddr - address for pilot symbol time indices
 * \param strm - LWCA stream for kernel launch
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if any of the tensor descriptors or
 * address values are NULL.
 *
 * Returns ::LWPHY_STATUS_SUCCESS if submission of the kernel was successful
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t
 */
lwphyStatus_t LWPHYWINAPI lwphyChannelEst1DTimeFrequency(const lwphyTensorDescriptor_t tensorDescDst,
                                                         void*                         dstAddr,
                                                         const lwphyTensorDescriptor_t tensorDescSymbols,
                                                         const void*                   symbolsAddr,
                                                         const lwphyTensorDescriptor_t tensorDescFreqFilters,
                                                         const void*                   freqFiltersAddr,
                                                         const lwphyTensorDescriptor_t tensorDescTimeFilters,
                                                         const void*                   timeFiltersAddr,
                                                         const lwphyTensorDescriptor_t tensorDescFreqIndices,
                                                         const void*                   freqIndicesAddr,
                                                         const lwphyTensorDescriptor_t tensorDescTimeIndices,
                                                         const void*                   timeIndicesAddr,
                                                         lwdaStream_t                  strm);

/******************************************************************/ /**
 * \brief Performs channel estimation
 *
 * Performs MMSE channel estimation
 *
 * \param cellId - cell identifier (for callwlation of initial seed in descrambler sequence generation)
 * \param slotNum - current slot number (for callwlation of initial seed in descrambler sequence generation)
 * \param nBSAnts - number of base station antennas
 * \param nLayers - number of layers
 * \param nDMRSSyms - number of DMRS symbols
 * \param nDMRSGridsPerPRB - number of DMRS grids per PRB
 * \param nTotalDMRSPRB - total number of DMRS PRBs
 * \param nTotalDataPRB - total number of data PRBs
 * \param Nh - number of time domain channel estimates (lwrrently limited to Nh = 1)
 * \param activeDMRSGridBmsk - active DMRS grid bitmask
 * \param tDescDataRx - tensor descriptor for received data
 * \param dataRxAddr - address of received data
 * \param tDescWFreq - tensor descriptor for frequency domain filter
 * \param WFreqAddr - address of frequency domain filter
 * \param tDescDescrShiftSeq - tensor descriptor for descrambling shift sequence
 * \param descrShiftSeqAddr - address for descrambling shift sequence data
 * \param tDeslwnShiftSeq - tensor descriptor for unshift sequence
 * \param unShiftSeqAddr - address for unshift sequence
 * \param tDescH - descriptor for output tensor
 * \param HAddr - address for output tensor
 * \param tDescDbg - tensor descriptor for debug buffer
 * \param dbgAddr - address of debug buffer
 * \param strm - LWCA stream for kernel launch
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if any of the tensor descriptors or
 * address values are NULL.
 *
 * Returns ::LWPHY_STATUS_SUCCESS if submission of the kernel was successful
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t
 */
lwphyStatus_t LWPHYWINAPI lwphyChannelEst(unsigned int                  cellId,
                                          unsigned int                  slotNum,
                                          unsigned int                  nBSAnts,
                                          unsigned int                  nLayers,
                                          unsigned int                  nDMRSSyms,
                                          unsigned int                  nDMRSGridsPerPRB,
                                          unsigned int                  nTotalDMRSPRB,
                                          unsigned int                  nTotalDataPRB,
                                          unsigned int                  Nh,
                                          unsigned int                  activeDMRSGridBmsk,
                                          const lwphyTensorDescriptor_t tDescDataRx,
                                          const void*                   dataRxAddr,
                                          const lwphyTensorDescriptor_t tDescWFreq,
                                          const void*                   WFreqAddr,
                                          const lwphyTensorDescriptor_t tDescDescrShiftSeq,
                                          const void*                   descrShiftSeqAddr,
                                          const lwphyTensorDescriptor_t tDeslwnShiftSeq,
                                          const void*                   unShiftSeqAddr,
                                          const lwphyTensorDescriptor_t tDescH,
                                          void*                         HAddr,
                                          const lwphyTensorDescriptor_t tDescDbg,
                                          void*                         dbgAddr,
                                          lwdaStream_t                  strm);

/** @} */ /* END LWPHY_CHANNEL_ESTIMATION */

/**
 * \defgroup LWPHY_CHANNEL_EQUALIZATION Channel Equalization
 *
 * This section describes the channel equalization functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */
lwphyStatus_t LWPHYWINAPI lwphyChannelEqCoefCompute(unsigned int                  nBSAnts,
                                                    unsigned int                  nLayers,
                                                    unsigned int                  Nh,
                                                    unsigned int                  Nprb,
                                                    const lwphyTensorDescriptor_t tDescH,
                                                    const void*                   HAddr,
                                                    const lwphyTensorDescriptor_t tDescNoisePwr,
                                                    const void*                   noisePwrAddr,
                                                    lwphyTensorDescriptor_t       tDescCoef,
                                                    void*                         coefAddr,
                                                    lwphyTensorDescriptor_t       tDescReeDiag,
                                                    void*                         reeDiagAddr,
                                                    lwphyTensorDescriptor_t       tDescDbg,
                                                    void*                         dbgAddr,
                                                    lwdaStream_t                  strm);

lwphyStatus_t LWPHYWINAPI lwphyChannelEqSoftDemap(unsigned int                  nBSAnts,
                                                  unsigned int                  nLayers,
                                                  unsigned int                  Nh,
                                                  unsigned int                  Nd,
                                                  unsigned int                  Nprb,
                                                  const lwphyTensorDescriptor_t tDescDataSymbLoc,
                                                  const void*                   dataSymbLocAddr,
                                                  const lwphyTensorDescriptor_t tDescQamInfo,
                                                  const void*                   qamAddrInfo,
                                                  const lwphyTensorDescriptor_t tDescCoef,
                                                  const void*                   coefAddr,
                                                  const lwphyTensorDescriptor_t tDescReeDiagIlw,
                                                  const void*                   reeDiagIlwAddr,
                                                  const lwphyTensorDescriptor_t tDescDataRx,
                                                  const void*                   dataRxAddr,
                                                  lwphyTensorDescriptor_t       tDescDataEq,
                                                  void*                         dataEqAddr,
                                                  lwphyTensorDescriptor_t       tDescLlr,
                                                  void*                         llrAddr,
                                                  lwphyTensorDescriptor_t       tDescDbg,
                                                  void*                         dbgAddr,
                                                  lwdaStream_t                  strm);

lwphyStatus_t LWPHYWINAPI lwphyChannelEq(unsigned int                  nBSAnts,
                                         unsigned int                  nLayers,
                                         unsigned int                  Nh,
                                         unsigned int                  Nf,
                                         unsigned int                  Nd,
                                         unsigned int                  qam,
                                         const lwphyTensorDescriptor_t tDescDataSymLoc,
                                         const void*                   dataSymLocAddr,
                                         const lwphyTensorDescriptor_t tDescDataRx,
                                         const void*                   dataRxAddr,
                                         const lwphyTensorDescriptor_t tDescH,
                                         const void*                   HAddr,
                                         const lwphyTensorDescriptor_t tDescNoisePwr,
                                         const void*                   noisePwrAddr,
                                         lwphyTensorDescriptor_t       tDescDataEq,
                                         void*                         dataEqAddr,
                                         lwphyTensorDescriptor_t       tDescReeDiag,
                                         void*                         reeDiagAddr,
                                         lwphyTensorDescriptor_t       tDescLLR,
                                         void*                         LLRAddr,
                                         lwdaStream_t                  strm);

/** @} */ /* END LWPHY_CHANNEL_EQUALIZATION */

/**
 * \defgroup LWPHY_SS Generate Synchronization Signals
 *
 * This section describes the synchronization singals generation 
 * of the lwPHY application programming interface.
 *
 * @{
 */

struct SSTxParams
{
    uint16_t NID;        // Physical cell id
    uint16_t nHF;        // Half frame index (0 or 1)
    uint16_t Lmax;       // Max number of ss blocks in pbch period (4,8,or 64)
    uint16_t blockIndex; // SS block index (0 - L_max)
    uint16_t f0;         // Index of initial ss subcarrier
    uint16_t t0;         // Index of initial ss ofdm symbol
    uint16_t nF;
    uint16_t nT;
    uint16_t slotIdx;
};

lwphyStatus_t lwphySSTxPipelinePrepare(void** workspace);
lwphyStatus_t lwphySSTxPipelineFinalize(void** workspace);

lwphyStatus_t lwphySSTxPipeline(__half2*          d_xQam,
                                int16_t*          d_PSS,
                                int16_t*          d_SSS,
                                __half2*          d_dmrs,
                                uint32_t*         d_c,
                                uint32_t*         d_dmrsIdx,
                                uint32_t*         d_qamIdx,
                                uint32_t*         d_pssIdx,
                                uint32_t*         d_sssIdx,
                                __half2*          d_tfSignalSS,
                                const uint32_t*   d_x_scram,
                                const SSTxParams* param,
                                __half2*          d_tfSignal,
                                void*             workspace,
                                lwdaStream_t      stream = 0);

lwphyStatus_t lwphyGenerateSyncSignal(const uint32_t NID,
                                      int16_t*       outputPSS,
                                      int16_t*       outputSSS);

/** @} */ /* END LWPHY_SS */

/**
 * \defgroup LWPHY_CRC CRC Computation
 *
 * This section describes the CRC computation functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */
struct PerTbParams;

lwphyStatus_t lwphyCRCDecode(
    /* DEVICE MEMORY*/
    uint32_t* d_outputCBCRCs, // output buffer containing result of CRC check for each input code block (one uint32_t value per code block): 0 if the CRC check passed, a value different than zero otherwise
    uint32_t* d_outputTBCRCs, // output buffer containing result of CRC check for each input transport block (one uint32_t value per transport block): 0 if the CRC check passed, a value different than zero otherwise

    uint8_t*           d_outputTransportBlocks, // output buffer containing the information bytes of each input transport block
    const uint32_t*    d_inputCodeBlocks,       // input buffer containing the input code blocks
    const PerTbParams* d_tbPrmsArray,           // array of PerTbParams structs describing each input transport block
    /* END DEVICE MEMORY*/
    uint32_t     nTBs,           // total number of input transport blocks
    uint32_t     maxNCBsPerTB,   // Maximum number of code blocks per transport block for current launch
    uint32_t     maxTBByteSize,  // Maximum size in bytes of transport block for current launch
    int          reverseBytes,   // reverse order of bytes in each word before computing the CRC
    int          timeIt,         // run NRUNS times and report average running time
    uint32_t     NRUNS,          // number of iterations used to compute average running time
    uint32_t     codeBlocksOnly, // Only compute CRC of code blocks. Skip transport block CRC computation
    lwdaStream_t strm);

lwphyStatus_t lwphyCRCEncode(
    /* DEVICE MEMORY*/
    uint32_t*          d_outputCBCRCs,        // output buffer containing result of CRC check for each input code block (one uint32_t value per code block): 0 if the CRC check passed, a value different than zero otherwise
    uint32_t*          d_outputTBCRCs,        // output buffer containing result of CRC check for each input transport block (one uint32_t value per transport block): 0 if the CRC check passed, a value different than zero otherwise
    uint8_t*           d_outputCodeBlocks,    // output buffer containing assembled transport blocks
    const uint32_t*    d_inputTransferBlocks, // Aarray containing list of input code blocks
    const PerTbParams* d_tbPrmsArray,         // array of PerTbParams structs describing each input transport block
    /* END DEVICE MEMORY*/
    uint32_t     nTBs,           // total number of input transport blocks
    uint32_t     maxNCBsPerTB,   // Maximum number of code blocks per transport block for current launch
    uint32_t     maxTBByteSize,  // Maximum size in bytes of transport block for current launch
    int          reverseBytes,   // reverse order of bytes in each word before computing the CRC
    int          timeIt,         // run NRUNS times and report average running time
    uint32_t     NRUNS,          // number of iterations used to compute average running time
    uint32_t     codeBlocksOnly, // Only compute CRC of code blocks. Skip transport block CRC computation
    lwdaStream_t strm);

/** @} */ /* END LWPHY_CRC */

/**
 * \defgroup LWPHY_SCRAMBLE Scrambling/Descrambling
 *
 * This section describes the scrambling/descrambling functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

// FIXME: this is a stand-alone implementation of descrambling used as initial reference.
// It will be removed eventually
void lwphyDescrambleInit(void** descrambleElw);

void lwphyDescrambleCleanUp(void** descrambleElw);

lwphyStatus_t lwphyDescrambleLoadParams(void**          descrambleElw,
                                        uint32_t        nTBs,
                                        uint32_t        maxNCodeBlocks,
                                        const uint32_t* tbBoundaryArray,
                                        const uint32_t* cinitArray);

lwphyStatus_t lwphyDescrambleLoadInput(void** descrambleElw,
                                       float* llrs);

lwphyStatus_t lwphyDescramble(void**       descrambleElw,
                              float*       d_llrs,
                              bool         timeIt,
                              uint32_t     NRUNS,
                              lwdaStream_t strm);

lwphyStatus_t lwphyDescrambleStoreOutput(void** descrambleElw,
                                         float* llrs);

lwphyStatus_t lwphyDescrambleAllParams(float*          llrs,
                                       const uint32_t* tbBoundaryArray,
                                       const uint32_t* cinitArray,
                                       uint32_t        nTBs,
                                       uint32_t        maxNCodeBlocks,
                                       int             timeIt,
                                       uint32_t        NRUNS,
                                       lwdaStream_t    stream);
/** @} */ /* END LWPHY_SCRAMBLE */

/**
 * \defgroup LWPHY_RATE_MATCHING Rate Matching
 *
 * This section describes the rate matching functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

void rate_matchingFP16(
    uint32_t           CMax,           // maximum number of code blocks per transport blocks
    uint32_t           EMax,           // maximum input code block size in "soft-bits"
    uint32_t           nTb,            // number of input transport blocks
    uint32_t           nBBULayers,     // number of BBU layers
    const PerTbParams* d_tbPrmsArray,  // array of PerTbParams structs describing each input transport block
    float*             in,             // array of input LLRS
    __half*            out,            // rate-dematched, descrambled and layer de-mapped output LLRs
    int                descramblingOn, // enable/disable descrambling
    lwdaStream_t       strm);

void rate_matchingFP32(
    uint32_t           CMax,           // maximum number of code blocks per transport blocks
    uint32_t           EMax,           // maximum input code block size in "soft-bits"
    uint32_t           nTb,            // number of input transport blocks
    uint32_t           nBBULayers,     // number of BBU layers
    const PerTbParams* d_tbPrmsArray,  // array of PerTbParams structs describing each input transport block
    float*             in,             // array of input LLRS
    float*             out,            // rate-dematched, descrambled and layer de-mapped output LLRs
    int                descramblingOn, // enable/disable descrambling
    lwdaStream_t       strm);

/** @} */ /* END LWPHY_RATE_MATCHING */

/**
 * \defgroup DL_LWPHY_RATE_MATCHING DL Rate Matching
 *
 * This section describes the downlink rate matching functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */
struct dl_rateMatchingElw;

/**
 * @brief Struct that tracks configuration information at a per TB (Transport Block) granularity
 *        for the downlink rate matching component.
 */
struct PerTbParams
{
    uint32_t rv;      /*!< redundancy version per TB; [0, 3] */
    uint32_t Qm;      /*!< modulation order per TB: [2, 4, 6, 8] */
    uint32_t bg;      /*!< base graph per TB; options are 1 or 2 */
    uint32_t Nl;      /*!< number of transmission layers per TB; [1, MAX_DL_LAYERS_PER_TB] in DL */
    uint32_t num_CBs; /*!< number of code blocks (CBs) per TB */
    uint32_t Zc;      /*!< lifting factor per TB */

    uint32_t N;     /*!< # bits in code block per TB */
    uint32_t Ncb;   /*!< same as N for now */
    uint32_t G;     /*!< number of rate-matched bits available for TB transmission */
    uint32_t K;     /*!< non punctured systematic bits */
    uint32_t F;     /*!< filler bits */
    uint32_t cinit; /*!< used to generate scrambling sequence; seed2 arg. of gold32 */

    uint32_t firstCodeBlockIndex;                   // for symbol-by-symbol processing
    uint32_t encodedSize;                           // Size in bytes of encoded Tb
    uint32_t layer_map_array[MAX_DL_LAYERS_PER_TB]; /*!< first Nl elements of array specify the
    layer(s) this TB maps to. TODO potentially colwert to bitmap. */
};

/**
 * @brief Update PerTbParams struct that tracks configuration information at per TB
 *        granularity from tb_pars array and gnb_params struct.
 * @param[in,out] tb_params_struct: pointer to a PerTbParams configuration struct
 * @param[in] tb_params: array of tb_pars structs
 * @param[in] gnb_params: pointer to gnb_pars struct
 * @return LWPHY_STATUS_SUCCESS or LWPHY_STATUS_ILWALID_ARGUMENT.
 */
lwphyStatus_t lwphySetTBParamsFromStructs(PerTbParams* tb_params_struct,
                                          tb_pars*     tb_params,
                                          gnb_pars*    gnb_params);

/**
 * @brief Update PerTbParams struct that tracks configuration information at per TB
 *        granularity. Check that configuration values are valid. layer_map_array contents
 *        should be set separately.
 * @param[in,out] tb_params_struct: pointer to a PerTbParams configuration struct
 * @param[in] cfg_rv: redundancy version
 * @param[in] cfg_Qm: modulation order
 * @param[in] cfg_bg: base graph
 * @param[in] cfg_Nl: number of layers per Tb (at most MAX_DL_LAYERS_PER_TB for downlink)
 * @param[in] cfg_num_CBs: number of code blocks
 * @param[in] cfg_Zc: lifting factor
 * @param[in] cfg_G: number of rated matched bits available for TB transmission
 * @param[in] cfg_F: number of filler bits
 * @param[in] cfg_cinit: seed used for scrambling sequence
 * @return LWPHY_STATUS_SUCCESS or LWPHY_STATUS_ILWALID_ARGUMENT.
 */
lwphyStatus_t lwphySetTBParams(PerTbParams* tb_params_struct,
                               uint32_t     cfg_rv,
                               uint32_t     cfg_Qm,
                               uint32_t     cfg_bg,
                               uint32_t     cfg_Nl,
                               uint32_t     cfg_num_CBs,
                               uint32_t     cfg_Zc,
                               uint32_t     cfg_G,
                               uint32_t     cfg_F,
                               uint32_t     cfg_cinit);

/** @brief: Return workspace size, in bytes, needed for all configuration parameters
 *          of the rate matching component. Does not allocate any space.
 *  @param[in] num_TBs: number of Transport blocks (TBs) to be processed within a kernel launch
 *  @param[in] max_codeblocks_per_TB: maximum number of codeblocks, per TB, across all num_TBs.
 *  @return workspace size in bytes
 */
size_t lwphyDlRateMatchingWorkspaceSize(int num_TBs,
                                        int max_codeblocks_per_TB);

/** @brief: Populate environment for rate matching, e.g., load array of PerTbStruct etc. into GPU config_workspace.
 *  @param[in, out] elwHandle: handle to rate matching environment
 *  @param[in] num_TBs: number of TBs handled in a kernel launch
 *  @param[in] kernel_params: an array of previously populated PerTbParams structs
 *  @param[in, out] max_Er: maximum number of rate-matched length bits for all CBs (Code Blocks) across all TBs.
 *  @param[in, out] max_CBs: maximum number of CBs across all num_TBs TBs
 *  @param[in] num_layers: number of layers
 *  @param[in] enable_scrambling: enable scrambling
 *  @param[in] enable_layer_mapping: enable layer mapping
 *  @param[in] config_workspace: pre-allocated device buffer to hold the various config. parameters.
 *  @param[in] allocated_workspace_size: size, in bytes, of config. workspace.
 *  @param[in] h_workspace: pinned host memory for some temporary buffers
 *  @param[in] strm: LWCA stream for memory copies
 *  @return LWPHY_STATUS_SUCCESS or LWPHY_STATUS_ILWALID_ARGUMENT.
 */
lwphyStatus_t lwphyDlRateMatchingLoadParams(dl_rateMatchingElw** elwHandle,
                                            uint32_t             num_TBs,
                                            PerTbParams          kernel_params[],
                                            uint32_t*            max_Er,
                                            uint32_t*            max_CBs,
                                            uint32_t             num_layers,
                                            uint32_t             enable_scrambling,
                                            uint32_t             enable_layer_mapping,
                                            uint32_t*            config_workspace,
                                            size_t               allocated_workspace_size,
                                            uint32_t*            h_workspace,
                                            lwdaStream_t         strm);

/** @brief: Launch rate matching + scrambling + layer mapping kernel.
 *          Assumes dl_rate_matchingLoadParams has been called beforehand, so elw is properly configured.
 *  @param[in] elw: config environment.
 *  @param[in] d_rate_matching_input: LDPC encoder's output; device buffer, previously allocated.
 *  @param[in, out] d_rate_matching_output: Kernel's generated output; device pointer, preallocated.
 *                                          A call to lwphyRestructureRmOutput is needed for modulation.
 *  @param[in] strm: LWCA stream for kernel launch
 */

void lwphyDlRateMatching(dl_rateMatchingElw* elw,
                         const uint32_t*     d_rate_matching_input,
                         uint32_t*           d_rate_matching_output,
                         lwdaStream_t        strm);

void lwphyDlRateMatchingCleanUp(dl_rateMatchingElw** elwHandle);

/** @brief: Copy Er values computed as part of dl_rate_matchingLoadParams to the host.
 *  @param[in] elwHandle: handle to rate matching environment
 *  @param[in, out] Er: preallocated CPU array CMax * num_TBs elements wide.
 *  @param[in] Cmax: maximum number of code blocks (CBs) across all TBs.
 *  @param[in] num_TBs: number of transport blocks
 *  @param[in] strm: LWCA stream for memory copy.
 *  @return LWPHY_STATUS_SUCCESS or LWPHY_STATUS_ILWALID_ARGUMENT.
 */
lwphyStatus_t lwphyCopyErValuesToHost(dl_rateMatchingElw** elwHandle,
                                      uint32_t*            Er,
                                      int                  Cmax,
                                      int                  num_TBs,
                                      lwdaStream_t         strm);

/** @brief: Prepare rate matching output for modulation. Remove gaps.
 *  @param[in] elw: handle to rate matching environment
 *  @param[in] orig_d_rate_matching_output: rate matching output. There are emax bits (overprovisioned) per code block.
 *  @param[in,out] new_d_rate_matching_output: output restructured for modulation. There are Er bits per code block.
 *                                             Each layer starts at an uint32_t aligned boundary.
 *  @param[in] cmax: maximum number of code blocks (CBs) across all TBs.
 *  @param[in] emax: maximum number of rate matched bits across all CBs across all TBs.
 *  @param[in] strm: LWCA stream for kernel launch
 */
void lwphyRestructureRmOutput(dl_rateMatchingElw* elw,
                              const uint32_t*     orig_d_rate_matching_output,
                              uint32_t*           new_d_rate_matching_output,
                              uint32_t            cmax,
                              uint32_t            emax,
                              lwdaStream_t        strm);

/** @} */ /* END DL_LWPHY_RATE_MATCHING */

/**
 * LDPC Codeword Results 
 */
typedef struct
{
    unsigned char numIterations;
    unsigned char checkErrorCount;
} lwphyLDPCResults_t;

typedef struct
{
    const lwphyTensorDescriptor_t desc;
    void*                         addr;
} lwphyLDPCDiagnostic_t;

lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCEncode(lwphyTensorDescriptor_t inDesc,
                                                         void*                   inAddr,
                                                         lwphyTensorDescriptor_t outDesc,
                                                         void*                   outAddr,
                                                         int                     BG,
                                                         int                     Kb,
                                                         int                     Z,
                                                         bool                    puncture,
                                                         int                     maxParityNodes,
                                                         int                     rv,
                                                         lwdaStream_t            strm = 0);
/**
 * \defgroup LWPHY_ERROR_CORRECTION Error Correction
 *
 * This section describes the error correction functions of the lwPHY 
 * application programming interface.
 *
 * @{
 */

struct lwphyLDPCDecoder;
/**
 * lwPHY LDPC decoder handle
 */
typedef struct lwphyLDPCDecoder* lwphyLDPCDecoder_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a lwPHY LDPC decoder instance
 *
 * Allocates a lwPHY decoder instance and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p pdecoder is NULL.
 * 
 * Returns ::LWPHY_STATUS_ALLOC_FAILED if an LDPC decoder cannot be
 * allocated on the host.
 *
 * Returns ::LWPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param context - lwPHY context
 * \param pdecoder - Address for the new ::lwphyLDPCDecoder_t instance
 * \param flags - Creation flags (lwrrently unused)
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ALLOC_FAILED,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyGetErrorName,::lwphyGetErrorString,::lwphyCreateContext,::lwphyDestroyLDPCDecoder
 */
lwphyStatus_t LWPHYWINAPI lwphyCreateLDPCDecoder(lwphyContext_t      context,
                                                 lwphyLDPCDecoder_t* pdecoder,
                                                 unsigned int        flags);

/******************************************************************/ /**
 * \brief Destroys a lwPHY LDPC decoder object
 *
 * Destroys a lwPHY LDPC decoder object that was previously created by
 * a call to ::lwphyCreateLDPCDecoder. The handle provided to this
 * function should not be used for any operations after this function
 * returns.
 * 
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if \p decoder is NULL.
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param decoder - previously allocated ::lwphyLDPCDecoder_t instance
 *
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 *
 * \sa ::lwphyStatus_t,::lwphyCreateLDPCDecoder
 */
lwphyStatus_t LWPHYWINAPI lwphyDestroyLDPCDecoder(lwphyLDPCDecoder_t decoder);

/******************************************************************/ /**
 * \brief Perfom a bulk LDPC decode operation on a tensor of soft input values
 *
 * Performs a bulk LDPC decode operation on an input tensor of "soft"
 * log likelihood ration (LLR) values.
 *
 * \param decoder - lwPHY LDPC decoder instance
 * \param tensorDescDst - tensor descriptor for LDPC output
 * \param dstAddr - address for LDPC output
 * \param tensorDescLLR - tensor descriptor for soft input LLR values
 * \param LLRAddr - address for soft input LLR values
 * \param BG - base graph for LDPC decode operation (1 or 2)
 * \param Kb - number of information nodes (must be 6, 8, 9, 10 (BG2) or 22 (BG1))
 * \param mb - number of parity nodes (must be between 4 and 46 inclusive)
 * \param Z - lifting size
 * \param algoIndex - LDPC decode algorithm to use
 * \param maxNumIterations - maximum number of iterations
 * \param normalization - min-sum normalization value
 * \param earlyTermination - (lwrrently unused)
 * \param results - pointer to array of ::lwphyLDPCResults_t values (may be NULL)
 * \param workspace - address of caller-allocated workspace
 * \param flags - (lwrrently unused)
 * \param strm - LWCA stream for LDPC exelwtion
 * \param reserved - (lwrrently unused)
 *
 * If the value of \p algoIndex is zero, the library will choose the "best"
 * algorithm for the given LDPC configuration.
 *
 * The type of input tensor descriptor \p tensorDescLLR must be either ::LWPHY_R_32F or
 * ::LWPHY_R_16F, and the rank must be 2.
 *
 * The type of output tensor descriptor \p tensorDescDst must be ::LWPHY_BIT, and the
 * rank must be 2.
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if:
 * <ul>
 *   <li>\p decoder is NULL</li>
 *   <li>\p BG, \p Kb, \p mb, and \p Z do not represent a valid LDPC configuration</li>
 *   <li>\p maxNumIterations <= 0</li>
 *   <li>\p tensorDescDst is NULL</li>
 *   <li>\p tensorDescLLR is NULL</li>
 *   <li>\p dstAddr NULL</li>
 *   <li>\p LLRAddr is NULL</li>
 * </ul>
 *
 * Returns ::LWPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given LLR tensor and/or
 * algorithm index (\p algoIndex).
 *
 * Returns ::LWPHY_STATUS_UNSUPPORTED_RANK if either the input tensor descriptor (\p tensorDescLLR)
 * or output tensor descriptor (\p tensorDescDst) do not have a rank of 2.
 *
 * Returns ::LWPHY_STATUS_UNSUPPORTED_TYPE if the output tensor descriptor (\p tensorDescLLR)
 * is not of type ::LWPHY_BIT, or if the input tensor descriptor is not one of (::LWPHY_R_32F or ::LWPHY_R_16F)
 *
 * Returns ::LWPHY_STATUS_INTERNAL_ERROR if the base graph is not equal to 1. (This restriction
 * will be lifted in a future release.)
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the decode operation was submitted to the stream
 * successfully.
 * 
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 * ::LWPHY_STATUS_UNSUPPORTED_RANK
 * ::LWPHY_STATUS_UNSUPPORTED_TYPE
 * ::LWPHY_STATUS_UNSUPPORTED_CONFIG
 *
 * \sa ::lwphyStatus_t,::lwphyCreateLDPCDecoder,::lwphyDestroyLDPCDecoder,::lwphyErrorCorrectionLDPCDecodeGetWorkspaceSize
 */
lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCDecode(lwphyLDPCDecoder_t            decoder,
                                                         lwphyTensorDescriptor_t       tensorDescDst,
                                                         void*                         dstAddr,
                                                         const lwphyTensorDescriptor_t tensorDescLLR,
                                                         const void*                   LLRAddr,
                                                         int                           BG,
                                                         int                           Kb,
                                                         int                           Z,
                                                         int                           mb,
                                                         int                           maxNumIterations,
                                                         float                         normalization,
                                                         int                           earlyTermination,
                                                         lwphyLDPCResults_t*           results,
                                                         int                           algoIndex,
                                                         void*                         workspace,
                                                         int                           flags,
                                                         lwdaStream_t                  strm,
                                                         void*                         reserved);

/******************************************************************/ /**
 * \brief Returns the workspace size for and LDPC decode operation
 *
 * Callwlates the workspace size (in bytes) required to perform an LDPC
 * decode operation for the given LDPC configuration.
 *
 * If the \p algorIndex parameter is -1, the function will return the
 * maximum workspace size for all numbers of parity nodes less than or
 * equal to the value of the \p mb parameter (for the given lifting
 * size \p Z). This is useful for determining the maximum workspace
 * size across different code rates.
 *
 * \param decoder - decoder object created by ::lwphyCreateLDPCDecoder
 * \param BG - base graph for LDPC decode operation (1 or 2)
 * \param Kb - number of information nodes (must be 6, 8, 9, 10 (BG2) or 22 (BG1))
 * \param mb - number of parity nodes (must be between 4 and 46 inclusive)
 * \param Z - lifting size
 * \param numCodeWords - number of codewords to decode simultaneously
 * \param LLRtype - data type for input LLR values
 * \param algoIndex - LDPC decode algorithm to use
 * \param sizeInBytes - output address for callwlated workspace size
 *
 * Different LDPC decoding algorithms may have different workspace
 * requirements. If the value of \p algoIndex is zero, the library will
 * choose the "best" algorithm for the given LDPC configuration.
 *
 * Returns ::LWPHY_STATUS_ILWALID_ARGUMENT if:
 * <ul>
 *   <li>\p BG, \p Kb, \p mb, and \p Z do not represent a valid LDPC configuration</li>
 *   <li>\p numCodeWords <= 0</li>
 *   <li>\p sizeInBytes is NULL</li>
 * </ul>
 *
 * Returns ::LWPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given \p LLRtype and/or
 * algorithm index (\p algoIndex).
 * 
 * 
 * Returns ::LWPHY_STATUS_SUCCESS if the size callwlation was successful.
 * 
 * \return
 * ::LWPHY_STATUS_SUCCESS,
 * ::LWPHY_STATUS_ILWALID_ARGUMENT
 * ::LWPHY_STATUS_UNSUPPORTED_CONFIG
 *
 * \sa ::lwphyStatus_t,::lwphyCreateLDPCDecoder,::lwphyErrorCorrectionLDPCDecode,::lwphyDestroyLDPCDecoder
 */
lwphyStatus_t LWPHYWINAPI lwphyErrorCorrectionLDPCDecodeGetWorkspaceSize(lwphyLDPCDecoder_t decoder,
                                                                         int                BG,
                                                                         int                Kb,
                                                                         int                mb,
                                                                         int                Z,
                                                                         int                numCodeWords,
                                                                         lwphyDataType_t    LLRtype,
                                                                         int                algoIndex,
                                                                         size_t*            sizeInBytes);
/** @} */ /* END LWPHY_ERROR_CORRECTION */

/**
 * \defgroup LWPHY_MODULATION_MAPPER  Modulation Mapper
 *
 * This section describes the modulation function(s) of the lwPHY
 * application programming interface.
 *
 * @{
 */

struct PdschDmrsParams;

/** @brief: Launch modulation kernel that maps the modulation input bits
 *          to modulation symbols.
 *  @param[in] d_params: Pointer to PdschDmrsParams on the device.
 *                       If nullptr, then symbols are allocated contiguously, starting from
 *                       zero in modulation_output. If not, symbols are allocated
 *                       in the appropriate Rbs, start position, in the {273*12, 14, 16}
 *                       modulation_output tensor.
 *  @param[in] input_desc: input tensor descriptor; dimension ceil(num_bits/32.0). Not used.
 *  @param[in] modulation_input: pointer to input tensor data
 *                               Data is expected to be contiguously allocated for every layer without
 *                               any gaps. Each layer should start at a uint32_t aligned boundary.
 *  @param[in] max_num_symbols: maximum number of symbols across all TBs in modulation_input.
 *  @param[in] num_TBs:  number of Transport Blocks contained in modulation_input
 *  @param[in] workspace: pointer to # TBs PerTBParams struct on the device. Only fields G and Qm are used.
 *  @param[in] output_desc: output tensor descriptor; dimension (num_bits / modulation_order)
 *                          if d_params=nullptr or {273*12, 14, 16} otherwise. Not used.
 *  @param[in,out] modulation_output: pointer to output tensor (preallocated)
 *                                    Each symbol is a complex number using half-precision for
 *                                    the real and imaginary parts.
 *  @param[in] strm: LWCA stream for kernel launch
 *  @return LWPHY_STATUS_SUCCESS or LWPHY_STATUS_ILWALID_ARGUMENT
 */
lwphyStatus_t lwphyModulation(PdschDmrsParams*              d_params,
                              const lwphyTensorDescriptor_t input_desc,
                              const void*                   modulation_input,
                              int                           max_num_symbols,
                              int                           num_TBs,
                              PerTbParams*                  workspace,
                              lwphyTensorDescriptor_t       output_desc,
                              void*                         modulation_output,
                              lwdaStream_t                  strm);

/** @} */ /* END LWPHY_MODULATION_MAPPER */

/**
 * \defgroup UL_LWPHY_PUCCH_RECEIVER PUCCH Receiver
 *
 * This section describes the structs and functions of the uplink
 * lwPHY control channel receiver. Lwrrently only PUCCH Format 1 is supported.
 *
 * @{
 */

#define LWPHY_PUCCH_FORMAT1 1
#define NUM_PAPR_SEQUENCES 30
#define OFDM_SYMBOLS_PER_SLOT 14
#define MAX_UE_CNT 42 // 6 * OFDM_SYMBOLS_PER_SLOT / 2

/**
 * @brief Struct that tracks user equipment (UE) specific PUCCH parameters.
 */
struct PucchUeCellParams
{
    uint32_t time_cover_code_index;   /*!< time cover code index; used to remove user's code */
    uint32_t num_bits;                /*!< number of transmitted bits: 1 or 2 */
    uint32_t init_cyclic_shift_index; /*!< initial cyclic shift; used in cyclic shift index computation*/
};

/**
 * @brief Struct that tracks all necessary parameters for PUCCH receiver processing.
 *        It also includes a PucchUeCellParams struct per UE.
 */
struct PucchParams
{
    uint32_t format;       /*!< PUCCH format. Should be LWPHY_PUCCH_FORMAT1 for now. */
    uint32_t num_pucch_ue; /*!< number of user equipment (UEs) in PUCCH */

    float Wf[LWPHY_N_TONES_PER_PRB * LWPHY_N_TONES_PER_PRB];          /*!< frequency channel estimation filter */
    float Wt_cell[OFDM_SYMBOLS_PER_SLOT * OFDM_SYMBOLS_PER_SLOT / 4]; /*!< time channel estimation filter; overprovisioned */

    uint32_t start_symbol;       /*!< start symbol (in time dimension of input signal) */
    uint32_t num_symbols;        /*!< number of symbols [4, 14] */
    uint32_t PRB_index;          /*!< index of physical resource allocation */
    uint32_t low_PAPR_seq_index; /*!< sequence of low-PAPR (Peak-to-Average Power ratio) */
    uint32_t num_dmrs_symbols;   /*!< number of DMRS symbols (derived parameter); ceil(num_symbols*1.0/2) in PUCCH Format 1 */
    uint32_t num_data_symbols;   /*!< number of data symbols (derived parameters); num_symbols - num_dmrs_symbols */
    uint32_t num_bs_antennas;    /*!< number of base station antennas */
    uint32_t mu;                 /*!< numerology */
    uint32_t slot_number;        /*!< slot number */
    uint32_t hopping_id;         /*!< hopping Id */

    PucchUeCellParams cell_params[MAX_UE_CNT]; /*!< PucchUeCellParams structs; overprovisioned (first num_pucch_ue elements valid) */
};

/** @brief: Partially update PucchParams struct for Format 1 based on tb_pars and gnb_pars.
 *          NB: the following PucchParams fields are NOT updated in this function:
 *          (1) num_pucch_ue, (2) Wf, (3) Wt_cell, (4) low_PAPR_seq_index, (5) hopping_id,
 *          and (6) the cell_params array.
 *  @param[in,out] pucch_params: pointer to PUCCH configuration parameters on the host.
 *  @param[in] gnb_params: pointer to gnb_pars struct on the host.
 *  @param[in] tb_params: pointer to tb_pars struct on the host.
 */
void lwphyUpdatePucchParamsFormat1(PucchParams*    pucch_params,
                                   const gnb_pars* gnb_params,
                                   const tb_pars*  tb_params);

/** @brief: Return workspace size, in bytes, needed for all configuration parameters
 *          and intermediate computations of the PUCCH receiver. Does not allocate any space.
 *  @param[in] num_ues: number of User Equipement (UEs)
 *  @param[in] num_bs_antennas: number of Base Station (BS) antennas
 *  @param[in] num_symbols: number of symbols; sum of DMRS and data symbols.
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: LWPHY_C_32F or LWPHY_C_16F
 *  @return workspace size in bytes
 */
size_t lwphyPucchReceiverWorkspaceSize(int             num_ues,
                                       int             num_bs_antennas,
                                       int             num_symbols,
                                       lwphyDataType_t pucch_complex_data_type);

/** @brief: Copy PUCCH params from the CPU to the allocated PUCCH receiver workspace.
 *          The location of the struct in the workspace is implementation dependent.
 *  @param[in] h_pucch_params: pointer to PUCCH configuration parameters on the host.
 *  @param[in] pucch_workspace: pointer to the pre-allocated pucch receiver's workspace on the device.
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: LWPHY_C_32F or LWPHY_C_16F
 */
void lwphyCopyPucchParamsToWorkspace(const PucchParams* h_pucch_params,
                                     void*              pucch_workspace,
                                     lwphyDataType_t    pucch_complex_data_type);

/** @brief: Launch PUCCH receiver kernels that do processing at receive end of PUCCH
 *          (Physical Uplink Control Channel).
 *  @param[in] data_rx_desc: input tensor descriptor; dimensions: Nf x Nt x L_BS
 *  @param[in] data_rx_addr: pointer to input tensor data (i.e., base station received signal); each
 *                           tensor element is a complex number
 *  @param[in] bit_estimates_desc: output tensor descriptor; dimensions nUe_pucch x 2
 *  @param[in, out] bit_estimates_addr: pre-allocated device buffer with bit estimates
 *  @param[in] pucch_format: PUCCH format; lwrrently only format 1 is supported.
 *  @param[in, out] pucch_params: pointer to PUCCH config params.
 *  @param[in] strm: LWCA stream for kernel launch.
 *  @param[in, out] pucch_workspace: address of user allocated workspace
 *                  pucch params should have been already copied there via a
 *                  lwphyCopyPucchParamsToWorkspace() call.
 *  @param[in] allocated_workspace_size: size of pucch_workspace
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: LWPHY_C_32F or LWPHY_C_16F
 */
void lwphyPucchReceiver(const lwphyTensorDescriptor_t data_rx_desc,
                        const void*                   data_rx_addr,
                        lwphyTensorDescriptor_t       bit_estimates_desc,
                        void*                         bit_estimates_addr,
                        const uint32_t                pucch_format,
                        const PucchParams*            pucch_params,
                        lwdaStream_t                  strm,
                        void*                         pucch_workspace,
                        size_t                        allocated_workspace_size,
                        lwphyDataType_t               pucch_complex_data_type);

/** @} */ /* END UL_LWPHY_PUCCH_RECEIVER */

struct PdcchParams
{
    int      n_f;
    int      n_t;
    __half2* qam_payload;
    uint32_t slot_number;
    uint32_t start_rb;  // starting RB
    uint32_t n_rb;      // # of pdcch RBs
    uint32_t start_sym; // starting OFDM symbol number
    uint32_t n_sym;     // # of pdcch OFDM symbols (1-3)
    uint32_t dmrs_id;   // dmrs scrambling id
    float    beta_qam;  // power scaling of qam signal
    float    beta_dmrs; // power scaling of dmrs signal
};

void lwphyPdcchTfSignal(lwphyTensorDescriptor_t tf_signal_desc,
                        void*                   tf_signal_addr,
                        PdcchParams&            params,
                        lwdaStream_t            stream);

/**
 * \defgroup DL_LWPHY_PDSCH_DMRS  PDSCH DMRS
 *
 * This section describes the PDSCH (Physical DOwnlink Shared Channel) DMRS functions of the lwPHY
 * application programming interface.
 *
 * @{
 */

/**
 * @brief Struct that tracks all necessary parameters for PDSCH DRMS computation.
 *        This struct is also used in PDSCH modulation.
 *        Thre is one PdschDmrsParams struct per TB.
 */
struct PdschDmrsParams
{
    uint32_t Nf;               /*!< from gnb_pars.Nf */
    uint32_t Nt;               /*!< from gnb_pars.Nt */
    uint32_t slot_number;      /*!< from gnb_pars.slotNumber */
    uint32_t num_dmrs_symbols; /*!< number of DMRS symbols; lwrrently only one DMRS symbol is supported */
    uint32_t num_data_symbols; /*!< number of data symbols */
    uint32_t cell_id;          /*!< gnb_pars.cellId */
    uint32_t symbol_number;    /*!< index of DMRS symbol (0-based). Single DMRS symbol lwrrently supported. */

    uint32_t num_Rbs;    /*!< number of allocated RBs (Resource Blocks), at most 273 */
    uint32_t start_Rb;   /*!< initial RB (0 indexing) */
    float    beta_dmrs;  /*!< DMRS power scaling */
    uint32_t num_layers; /*!< number of layers */

    uint32_t port_ids[8]; /*!< at most 8 ports supported for DMRS configuration type 1; only first num_layer values are valid; should add 1000 to get port */
    uint32_t n_scid;      /*!< scrambling Id used  */
    uint32_t dmrs_scid;   /*!< DMRS scrambling Id */
};

/**
 * @brief: Populate PdschDmrsParams struct on the host.
 * @param[in, out] pdsch_dmrs_params: pointer to DMRS config params struct on the host. This struct is also used in modulation.
 * @param[in] tb_params: array of tb_pars structs on the host.
 * @param[in] gnb_params: pointer to gnb_pars struct on the host.
 */
void lwphyUpdatePdschDmrsParams(PdschDmrsParams* pdsch_dmrs_params,
                                const tb_pars*   tb_params,
                                const gnb_pars*  gnb_params);

/**
 * @brief: Compute Pdsch DMRS symbols and place then in  {273*12, 14, 16} tensor.
 * @param[in] dmrs_params: DMRS config. parameters struct array on the device, with # TBs entries.
 * @param[in] dmrs_output_desc: tensor descriptor for intermediate output; dimensions {Nf, num_TBs}.
 * @param[in, out] dmrs_output: pointer to intermediate tensor date; each tensor element is a complex number (half-precision).
 * @param[in] re_mapped_dmrs_output_desc: output tensor descriptor; dimensions {273*12, 14, 16} tensor.
 * @param[in] re_mapped_dmrs_output: pointer to output tensor data; each element is a complex number (half-precision).
 * @param[in] strm: LWCA stream for kernel launch.
 */
void lwphyPdschDmrs(PdschDmrsParams*        dmrs_params,
                    lwphyTensorDescriptor_t dmrs_output_desc,
                    void*                   dmrs_output,
                    lwphyTensorDescriptor_t re_mapped_dmrs_output_desc,
                    void*                   re_mapped_dmrs_output,
                    lwdaStream_t            strm);

/** @} */ /* END DL_LWPHY_PDSCH_DMRS */

#define LWPHY_POLAR_ENC_MAX_INFO_BITS (164)
#define LWPHY_POLAR_ENC_MAX_CODED_BITS (512)
#define LWPHY_POLAR_ENC_MAX_TX_BITS (8192)

/** @brief: Polar encoding and rate matching for control channel processing
 *  @param[in]  nInfoBits  : Number of information bits, range [1,164] 
 *  @param[in]  nTxBits    : Number of rate-matched transmit bits, range [1, 8192] 
 *  @param[in]  pInfoBits  : Pointer to GPU memory contaiing information bit stream packed in
 *                           a uint8_t array (with atleast 32b alignment), size ceiling(nInfoBits/8), upto 21 bytes (164 bits) 
 *  @param[in]  pNCodedBits: Pointer to CPU memory to store store the encoded bit length (valid values: 32,64,128,256,512)
 *  @param[out] pCodedBits : Pointer to GPU memory to store polar encoded bit stream packed in 
 *                           a uint8_t array (with atleast 32b alignment), size ceiling(nMaxCodedBits/8) = 64 bytes 
 *  @param[out] pTxBits    : Pointer to device memory for storing polar rate-matched transmit bit stream
 *                           packed in a uint8_t array (with atleast 32b alignment), size must be a multiple
 *                           of 4 bytes (padded to nearest 32b boundary) with max size being ceiling(nTxBits/8), upto 1024 bytes
 * @param[in]   strm       : LWCA stream for kernel launch.
 *
 */
lwphyStatus_t LWPHYWINAPI lwphyPolarEncRateMatch(uint32_t       nInfoBits,
                                                 uint32_t       nTxBits,
                                                 uint8_t const* pInfoBits,
                                                 uint32_t*      pNCodedBits,
                                                 uint8_t*       pCodedBits,
                                                 uint8_t*       pTxBits,
                                                 lwdaStream_t   strm);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif /* !defined(LWPHY_H_INCLUDED_) */
