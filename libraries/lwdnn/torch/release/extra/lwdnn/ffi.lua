require 'lwtorch'
local ffi = require 'ffi'

ffi.cdef[[


typedef enum {
        LWDNN_MAJOR  =    5,
        LWDNN_MINOR  =    0,
        LWDNN_PATCHLEVEL  = 4,
        LWDNN_VERSION  =  (LWDNN_MAJOR * 1000 + LWDNN_MINOR * 100 + LWDNN_PATCHLEVEL)
} lwdnlwerFakeEnum;

struct lwdnnContext;
typedef struct lwdnnContext *lwdnnHandle_t;

size_t             lwdnnGetVersion(void);

/*
 * LWDNN return codes
 */
typedef enum
{
    LWDNN_STATUS_SUCCESS          = 0,
    LWDNN_STATUS_NOT_INITIALIZED  = 1,
    LWDNN_STATUS_ALLOC_FAILED     = 2,
    LWDNN_STATUS_BAD_PARAM        = 3,
    LWDNN_STATUS_INTERNAL_ERROR   = 4,
    LWDNN_STATUS_ILWALID_VALUE    = 5,
    LWDNN_STATUS_ARCH_MISMATCH    = 6,
    LWDNN_STATUS_MAPPING_ERROR    = 7,
    LWDNN_STATUS_EXELWTION_FAILED = 8,
    LWDNN_STATUS_NOT_SUPPORTED    = 9,
    LWDNN_STATUS_LICENSE_ERROR    = 10
} lwdnnStatus_t;

/* human-readable error messages*/
const char *              lwdnnGetErrorString(lwdnnStatus_t status);

lwdnnStatus_t             lwdnnCreate        (lwdnnHandle_t *handle);
lwdnnStatus_t             lwdnnDestroy       (lwdnnHandle_t handle);
lwdnnStatus_t             lwdnnSetStream     (lwdnnHandle_t handle, lwdaStream_t streamId);
lwdnnStatus_t             lwdnnGetStream     (lwdnnHandle_t handle, lwdaStream_t *streamId);


/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct lwdnnTensorStruct*          lwdnnTensorDescriptor_t;
typedef struct lwdnnColwolutionStruct*     lwdnnColwolutionDescriptor_t;
typedef struct lwdnnPoolingStruct*         lwdnnPoolingDescriptor_t;
typedef struct lwdnnFilterStruct*          lwdnnFilterDescriptor_t;
typedef struct lwdnnLRNStruct*             lwdnnLRNDescriptor_t;
typedef struct lwdnnActivationStruct*      lwdnnActivationDescriptor_t;
typedef struct lwdnnSpatialTransformerStruct* lwdnnSpatialTransformerDescriptor_t;
typedef struct lwdnnOpTensorStruct*        lwdnnOpTensorDescriptor_t;
/*
* LWDNN data type
*/
typedef enum
{
    LWDNN_DATA_FLOAT  = 0,
    LWDNN_DATA_DOUBLE = 1,
    LWDNN_DATA_HALF   = 2,
} lwdnnDataType_t;

/*
 * LWDNN propagate Nan
 */
typedef enum{
    LWDNN_NOT_PROPAGATE_NAN  = 0,
    LWDNN_PROPAGATE_NAN      = 1,
} lwdnnNanPropagation_t;

/* Maximum supported number of tensor dimensions */
typedef enum { LWDNN_DIM_MAX  = 8 }  lwdnnDimMaxFakeEnum;

/* Create an instance of a generic Tensor descriptor */
lwdnnStatus_t             lwdnnCreateTensorDescriptor(
                                lwdnnTensorDescriptor_t            *tensorDesc );

typedef enum
{
    LWDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    LWDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} lwdnnTensorFormat_t;

lwdnnStatus_t             lwdnnSetTensor4dDescriptor(
                                lwdnnTensorDescriptor_t             tensorDesc,
                                lwdnnTensorFormat_t                 format,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                int                                 n,        /* number of inputs (batch size)*/
                                int                                 c,        /* number of input feature maps*/
                                int                                 h,        /* height of input section*/
                                int                                 w );       /* width of input section*/


lwdnnStatus_t             lwdnnSetTensor4dDescriptorEx(
                                lwdnnTensorDescriptor_t             tensorDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                int                                 n,        /* number of inputs (batch size)*/
                                int                                 c,        /* number of input feature maps*/
                                int                                 h,        /* height of input section*/
                                int                                 w,        /* width of input section*/
                                int                                 nStride,
                                int                                 cStride,
                                int                                 hStride,
                                int                                 wStride );

lwdnnStatus_t             lwdnnGetTensor4dDescriptor(
                                const lwdnnTensorDescriptor_t       tensorDesc,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                int                                *n,        /* number of inputs (batch size)*/
                                int                                *c,        /* number of input feature maps*/
                                int                                *h,        /* height of input section*/
                                int                                *w,        /* width of input section*/
                                int                                *nStride,
                                int                                *cStride,
                                int                                *hStride,
                                int                                *wStride );

lwdnnStatus_t             lwdnnSetTensorNdDescriptor(
                                lwdnnTensorDescriptor_t             tensorDesc,
                                lwdnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] );

lwdnnStatus_t             lwdnnGetTensorNdDescriptor(
                                const lwdnnTensorDescriptor_t       tensorDesc,
                                int                                 nbDimsRequested,
                                lwdnnDataType_t                    *dataType,
                                int                                *nbDims,
                                int                                 dimA[],
                                int                                 strideA[] );

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/

/* Destroy an instance of Tensor4d descriptor */
lwdnnStatus_t             lwdnnDestroyTensorDescriptor(
                                lwdnnTensorDescriptor_t             tensorDesc );


/* Tensor layout colwersion helper (y = alpha * x + beta * y) */
lwdnnStatus_t             lwdnnTransformTensor(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );


/* Tensor Bias addition : C = alpha * A + beta * C  */
lwdnnStatus_t             lwdnnAddTensor(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       cDesc,
                                void                               *C );

/*
* LWDNN OpTensor op type
*/
typedef enum
{
    LWDNN_OP_TENSOR_ADD = 0,
    LWDNN_OP_TENSOR_MUL = 1,
    LWDNN_OP_TENSOR_MIN = 2,
    LWDNN_OP_TENSOR_MAX = 3,
} lwdnnOpTensorOp_t;

lwdnnStatus_t             lwdnnCreateOpTensorDescriptor(
                                lwdnnOpTensorDescriptor_t          *opTensorDesc );

lwdnnStatus_t             lwdnnSetOpTensorDescriptor(
                                lwdnnOpTensorDescriptor_t           opTensorDesc,
                                lwdnnOpTensorOp_t                   opTensorOp,
                                lwdnnDataType_t                     opTensorCompType,
                                lwdnnNanPropagation_t               opTensorNanOpt );

lwdnnStatus_t             lwdnnGetOpTensorDescriptor(
                                const lwdnnOpTensorDescriptor_t     opTensorDesc,
                                lwdnnOpTensorOp_t                  *opTensorOp,
                                lwdnnDataType_t                    *opTensorCompType,
                                lwdnnNanPropagation_t              *opTensorNanOpt );

lwdnnStatus_t             lwdnnDestroyOpTensorDescriptor(
                                lwdnnOpTensorDescriptor_t           opTensorDesc );

/* Tensor Bias operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
lwdnnStatus_t             lwdnnOpTensor(
                                lwdnnHandle_t                       handle,
                                const lwdnnOpTensorDescriptor_t     opTensorDesc,
                                const void                         *alpha1,
                                const lwdnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *alpha2,
                                const lwdnnTensorDescriptor_t       bDesc,
                                const void                         *B,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       cDesc,
                                void                               *C );

/* Set all values of a tensor to a given value : y[i] = value[0] */
lwdnnStatus_t             lwdnnSetTensor(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *valuePtr );

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
lwdnnStatus_t             lwdnnScaleTensor(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *alpha );

/*
 *  colwolution mode
 */
typedef enum
{
    LWDNN_COLWOLUTION       = 0,
    LWDNN_CROSS_CORRELATION = 1
} lwdnnColwolutionMode_t;


/* Create an instance of FilterStruct */
lwdnnStatus_t             lwdnnCreateFilterDescriptor(
                                lwdnnFilterDescriptor_t            *filterDesc );


lwdnnStatus_t             lwdnnSetFilter4dDescriptor(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                lwdnnTensorFormat_t                 format,
                                int                                 k,        /* number of output feature maps*/
                                int                                 c,        /* number of input feature maps*/
                                int                                 h,        /* height of each input filter*/
                                int                                 w );      /* width of  each input filter*/


lwdnnStatus_t             lwdnnGetFilter4dDescriptor(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                lwdnnTensorFormat_t                *format,
                                int                                *k,        /* number of output feature maps*/
                                int                                *c,        /* number of input feature maps*/
                                int                                *h,        /* height of each input filter*/
                                int                                *w );      /* width of  each input filter*/


lwdnnStatus_t             lwdnnSetFilterNdDescriptor(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                lwdnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] );

lwdnnStatus_t             lwdnnGetFilterNdDescriptor(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                lwdnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] );


lwdnnStatus_t             lwdnnDestroyFilterDescriptor(
                                lwdnnFilterDescriptor_t             filterDesc );

/* Create an instance of colwolution descriptor */
lwdnnStatus_t             lwdnnCreateColwolutionDescriptor(
                                lwdnnColwolutionDescriptor_t       *colwDesc );

lwdnnStatus_t             lwdnnSetColwolution2dDescriptor(
                                lwdnnColwolutionDescriptor_t        colwDesc,
                                int                                 pad_h,    /* zero-padding height*/
                                int                                 pad_w,    /* zero-padding width*/
                                int                                 u,        /* vertical filter stride*/
                                int                                 v,        /* horizontal filter stride*/
                                int                                 upscalex, /* upscale the input in x-direction*/
                                int                                 upscaley, /* upscale the input in y-direction*/
                                lwdnnColwolutionMode_t              mode );

lwdnnStatus_t             lwdnnSetColwolution2dDescriptor_v5( lwdnnColwolutionDescriptor_t colwDesc,
                                                             int pad_h,    /* zero-padding height*/
                                                             int pad_w,    /* zero-padding width*/
                                                             int u,   /* vertical filter stride*/
                                                             int v,   /* horizontal filter stride*/
                                                             int upscalex, /* upscale the input in x-direction*/
                                                             int upscaley, /* upscale the input in y-direction*/
                                                             lwdnnColwolutionMode_t mode,
                                                             lwdnnDataType_t dataType
                                                           );

lwdnnStatus_t             lwdnnGetColwolution2dDescriptor(
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                int                                *pad_h,    /* zero-padding height*/
                                int                                *pad_w,    /* zero-padding width*/
                                int                                *u,        /* vertical filter stride*/
                                int                                *v,        /* horizontal filter stride*/
                                int                                *upscalex, /* upscale the input in x-direction*/
                                int                                *upscaley, /* upscale the input in y-direction*/
                                lwdnnColwolutionMode_t             *mode );

lwdnnStatus_t             lwdnnGetColwolution2dDescriptor_v5(  const lwdnnColwolutionDescriptor_t colwDesc,
                                                            int* pad_h,    /* zero-padding height*/
                                                            int* pad_w,    /* zero-padding width*/
                                                            int* u,        /* vertical filter stride*/
                                                            int* v,        /* horizontal filter stride*/
                                                            int* upscalex, /* upscale the input in x-direction*/
                                                            int* upscaley, /* upscale the input in y-direction*/
                                                            lwdnnColwolutionMode_t* mode,
                                                            lwdnnDataType_t *dataType
                                                         );

/* Helper function to return the dimensions of the output tensor given a colwolution descriptor */
lwdnnStatus_t             lwdnnGetColwolution2dForwardOutputDim(
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       inputTensorDesc,
                                const lwdnnFilterDescriptor_t       filterDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


lwdnnStatus_t             lwdnnSetColwolutionNdDescriptor(
                                lwdnnColwolutionDescriptor_t        colwDesc,
                                int                                 arrayLength,             /* nbDims-2 size */
                                const int                           padA[],
                                const int                           filterStrideA[],
                                const int                           upscaleA[],
                                lwdnnColwolutionMode_t              mode,
                                lwdnnDataType_t                     dataType );  /* colwolution data type*/

lwdnnStatus_t             lwdnnGetColwolutionNdDescriptor(
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                int                                 arrayLengthRequested,
                                int                                *arrayLength,
                                int                                 padA[],
                                int                                 strideA[],
                                int                                 upscaleA[],
                                lwdnnColwolutionMode_t             *mode,
                                lwdnnDataType_t                    *dataType );   /* colwolution data type*/


/* Helper function to return the dimensions of the output tensor given a colwolution descriptor */
lwdnnStatus_t             lwdnnGetColwolutionNdForwardOutputDim(
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       inputTensorDesc,
                                const lwdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDims,
                                int                                 tensorOuputDimA[] );

/* Destroy an instance of colwolution descriptor */
lwdnnStatus_t             lwdnnDestroyColwolutionDescriptor(
                                lwdnnColwolutionDescriptor_t        colwDesc );


/* helper function to provide the colwolution algo that fit best the requirement */
typedef enum
{
    LWDNN_COLWOLUTION_FWD_NO_WORKSPACE            = 0,
    LWDNN_COLWOLUTION_FWD_PREFER_FASTEST          = 1,
    LWDNN_COLWOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} lwdnnColwolutionFwdPreference_t;


typedef enum
{
    LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    LWDNN_COLWOLUTION_FWD_ALGO_GEMM                  = 2,
    LWDNN_COLWOLUTION_FWD_ALGO_DIRECT                = 3,
    LWDNN_COLWOLUTION_FWD_ALGO_FFT                   = 4,
    LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING            = 5,
    LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD              = 6,
    LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7
} lwdnnColwolutionFwdAlgo_t;

typedef struct {
    lwdnnColwolutionFwdAlgo_t   algo;
    lwdnnStatus_t               status;
    float                       time;
    size_t                      memory;
} lwdnnColwolutionFwdAlgoPerf_t;

lwdnnStatus_t             lwdnnFindColwolutionForwardAlgorithm(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                lwdnnColwolutionFwdAlgoPerf_t      *perfResults );

lwdnnStatus_t             lwdnnFindColwolutionForwardAlgorithmEx(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                lwdnnColwolutionFwdAlgoPerf_t      *perfResults,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes );


lwdnnStatus_t             lwdnnGetColwolutionForwardAlgorithm(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       yDesc,
                                lwdnnColwolutionFwdPreference_t     preference,
                                size_t                              memoryLimitInBytes,
                                lwdnnColwolutionFwdAlgo_t          *algo );

/*
 *  colwolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the colwolution given an algo*/
lwdnnStatus_t             lwdnnGetColwolutionForwardWorkspaceSize(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       yDesc,
                                lwdnnColwolutionFwdAlgo_t           algo,
                                size_t                             *sizeInBytes );


/* Colwolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward pass for batch colwolution */
lwdnnStatus_t             lwdnnColwolutionForward(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                lwdnnColwolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to compute the bias gradient for batch colwolution */
lwdnnStatus_t             lwdnnColwolutionBackwardBias(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dbDesc,
                                void                               *db );


/* helper function to provide the colwolution algo that fit best the requirement */
typedef enum
{
    LWDNN_COLWOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    LWDNN_COLWOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    LWDNN_COLWOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} lwdnnColwolutionBwdFilterPreference_t;

typedef enum
{
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_0         = 0,  /* non-deterministic*/
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_1         = 1,
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT       = 2,
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_3         = 3,  /* non-deterministic, algo0 with workspace*/
    /* LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD  = 4, not implemented */
    LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5
} lwdnnColwolutionBwdFilterAlgo_t;


typedef struct {
    lwdnnColwolutionBwdFilterAlgo_t algo;
    lwdnnStatus_t status;
    float time;
    size_t memory;
} lwdnnColwolutionBwdFilterAlgoPerf_t;

lwdnnStatus_t             lwdnnFindColwolutionBackwardFilterAlgorithm(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnFilterDescriptor_t       dwDesc,
                                const int                           requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                lwdnnColwolutionBwdFilterAlgoPerf_t *perfResults );

lwdnnStatus_t             lwdnnFindColwolutionBackwardFilterAlgorithmEx(
                                lwdnnHandle_t                        handle,
                                const lwdnnTensorDescriptor_t        xDesc,
                                const void                          *x,
                                const lwdnnTensorDescriptor_t        dyDesc,
                                const void                          *y,
                                const lwdnnColwolutionDescriptor_t   colwDesc,
                                const lwdnnFilterDescriptor_t        dwDesc,
                                void                                *dw,
                                const int                            requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                lwdnnColwolutionBwdFilterAlgoPerf_t *perfResults,
                                void                                *workSpace,
                                size_t                               workSpaceSizeInBytes );

lwdnnStatus_t             lwdnnGetColwolutionBackwardFilterAlgorithm(
                                lwdnnHandle_t                         handle,
                                const lwdnnTensorDescriptor_t         xDesc,
                                const lwdnnTensorDescriptor_t         dyDesc,
                                const lwdnnColwolutionDescriptor_t    colwDesc,
                                const lwdnnFilterDescriptor_t         dwDesc,
                                lwdnnColwolutionBwdFilterPreference_t preference,
                                size_t                                memoryLimitInBytes,
                                lwdnnColwolutionBwdFilterAlgo_t      *algo );

/*
 *  colwolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the colwolution given an algo*/
lwdnnStatus_t             lwdnnGetColwolutionBackwardFilterWorkspaceSize(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnFilterDescriptor_t       gradDesc,
                                lwdnnColwolutionBwdFilterAlgo_t     algo,
                                size_t                             *sizeInBytes );

lwdnnStatus_t             lwdnnColwolutionBackwardFilter(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                lwdnnColwolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const lwdnnFilterDescriptor_t       dwDesc,
                                void                               *dw );

/*********************************************************/
/* helper function to provide the colwolution algo that fit best the requirement */
typedef enum
{
    LWDNN_COLWOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
    LWDNN_COLWOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
    LWDNN_COLWOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2,
} lwdnnColwolutionBwdDataPreference_t;

typedef enum
{
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_0          = 0, /* non-deterministic*/
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_1          = 1,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT        = 2,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD   = 4,
    LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
} lwdnnColwolutionBwdDataAlgo_t;

typedef struct {
    lwdnnColwolutionBwdDataAlgo_t   algo;
    lwdnnStatus_t                   status;
    float                           time;
    size_t                          memory;
} lwdnnColwolutionBwdDataAlgoPerf_t;


lwdnnStatus_t             lwdnnFindColwolutionBackwardDataAlgorithm(
                                lwdnnHandle_t                       handle,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                lwdnnColwolutionBwdDataAlgoPerf_t  *perfResults );

lwdnnStatus_t             lwdnnFindColwolutionBackwardDataAlgorithmEx(
                                lwdnnHandle_t                       handle,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                lwdnnColwolutionBwdDataAlgoPerf_t  *perfResults,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes );

lwdnnStatus_t             lwdnnGetColwolutionBackwardDataAlgorithm(
                                lwdnnHandle_t                       handle,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                lwdnnColwolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                lwdnnColwolutionBwdDataAlgo_t      *algo );

 /* Helper function to return the minimum size of the workspace to be passed to the colwolution given an algo*/
lwdnnStatus_t             lwdnnGetColwolutionBackwardDataWorkspaceSize(
                                lwdnnHandle_t                       handle,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                lwdnnColwolutionBwdDataAlgo_t       algo,
                                size_t                             *sizeInBytes );


lwdnnStatus_t             lwdnnColwolutionBackwardData(
                                lwdnnHandle_t                       handle,
                                const void                         *alpha,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                lwdnnColwolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );


lwdnnStatus_t             lwdnnIm2Col(
                                lwdnnHandle_t                       handle,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const lwdnnFilterDescriptor_t       wDesc,
                                const lwdnnColwolutionDescriptor_t  colwDesc,
                                void                               *colBuffer );


/*
 *  softmax algorithm
 */
typedef enum
{
    LWDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    LWDNN_SOFTMAX_ACLWRATE = 1,         /* subtract max from every point to avoid overflow */
    LWDNN_SOFTMAX_LOG      = 2
} lwdnnSoftmaxAlgorithm_t;

typedef enum
{
    LWDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    LWDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} lwdnnSoftmaxMode_t;

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
lwdnnStatus_t             lwdnnSoftmaxForward(
                                lwdnnHandle_t                       handle,
                                lwdnnSoftmaxAlgorithm_t             algo,
                                lwdnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward softmax */
lwdnnStatus_t             lwdnnSoftmaxBackward(
                                lwdnnHandle_t                       handle,
                                lwdnnSoftmaxAlgorithm_t             algo,
                                lwdnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 *  pooling mode
 */
typedef enum
{
    LWDNN_POOLING_MAX     = 0,
    LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, /* count for average includes padded values*/
    LWDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, /* count for average does not include padded values*/
    LWDNN_POOLING_AVERAGE = LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING // for backward compatibility
} lwdnnPoolingMode_t;

/* Create an instance of pooling descriptor */
lwdnnStatus_t             lwdnnCreatePoolingDescriptor(
                                lwdnnPoolingDescriptor_t           *poolingDesc );

lwdnnStatus_t             lwdnnSetPooling2dDescriptor(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                lwdnnPoolingMode_t                  mode,
                                lwdnnNanPropagation_t               maxpoolingNanOpt,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );

lwdnnStatus_t             lwdnnGetPooling2dDescriptor(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                lwdnnPoolingMode_t                 *mode,
                                lwdnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

lwdnnStatus_t             lwdnnSetPoolingNdDescriptor(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                const lwdnnPoolingMode_t            mode,
                                const lwdnnNanPropagation_t         maxpoolingNanOpt,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

lwdnnStatus_t             lwdnnGetPoolingNdDescriptor(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                int                                 nbDimsRequested,
                                lwdnnPoolingMode_t                 *mode,
                                lwdnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

lwdnnStatus_t             lwdnnGetPoolingNdForwardOutputDim(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                const lwdnnTensorDescriptor_t       inputTensorDesc,
                                int                                 nbDims,
                                int                                 outputTensorDimA[] );

lwdnnStatus_t             lwdnnGetPooling2dForwardOutputDim(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                const lwdnnTensorDescriptor_t       inputTensorDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


/* Destroy an instance of pooling descriptor */
lwdnnStatus_t             lwdnnDestroyPoolingDescriptor(
                                lwdnnPoolingDescriptor_t            poolingDesc );

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
lwdnnStatus_t             lwdnnPoolingForward(
                                lwdnnHandle_t                       handle,
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward pooling */
lwdnnStatus_t             lwdnnPoolingBackward(
                                lwdnnHandle_t                       handle,
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                const void                          *alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 * activation mode
 */
typedef enum
{
    LWDNN_ACTIVATION_SIGMOID      = 0,
    LWDNN_ACTIVATION_RELU         = 1,
    LWDNN_ACTIVATION_TANH         = 2,
    LWDNN_ACTIVATION_CLIPPED_RELU = 3
} lwdnnActivationMode_t;

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
lwdnnStatus_t             lwdnnCreateActivationDescriptor(
                                lwdnnActivationDescriptor_t        *activationDesc);

lwdnnStatus_t             lwdnnSetActivationDescriptor(
                                lwdnnActivationDescriptor_t         activationDesc,
                                lwdnnActivationMode_t               mode,
                                lwdnnNanPropagation_t               reluNanOpt,
                                double                              reluCeiling );

lwdnnStatus_t             lwdnnGetActivationDescriptor(
                                const lwdnnActivationDescriptor_t   activationDesc,
                                lwdnnActivationMode_t              *mode,
                                lwdnnNanPropagation_t              *reluNanOpt,
                                double*                             reluCeiling );

lwdnnStatus_t             lwdnnDestroyActivationDescriptor(
                                lwdnnActivationDescriptor_t activationDesc);

/* Function to perform forward activation  */
lwdnnStatus_t             lwdnnActivationForward(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward activation  */
lwdnnStatus_t             lwdnnActivationBackward(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
*/
lwdnnStatus_t             lwdnnCreateLRNDescriptor(
                                lwdnnLRNDescriptor_t               *normDesc );

typedef enum { LWDNN_LRN_MIN_N     = 1,        /*  minimum allowed lrnN */
               LWDNN_LRN_MAX_N     = 16 }      /*  maximum allowed lrnN */
  LRN_MinMaxFakeEnum;

/* static const float LWDNN_LRN_MIN_K  =   1e-5; */ /* minimum allowed lrnK*/
/* static const float LWDNN_LRN_MIN_BETA = 0.01; */   /* minimum allowed lrnBeta*/

/* LRN layer mode */
typedef enum
{
    LWDNN_LRN_CROSS_CHANNEL_DIM1 = 0,/* Normalize across tensor's dimA[1] dimension*/
} lwdnnLRNMode_t;

/*
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
*/
lwdnnStatus_t             lwdnnSetLRNDescriptor(
                                lwdnnLRNDescriptor_t                normDesc,
                                unsigned                            lrnN,
                                double                              lrnAlpha,
                                double                              lrnBeta,
                                double                              lrnK );
/*
* Retrieve the settings lwrrently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
*/
lwdnnStatus_t             lwdnnGetLRNDescriptor(
                                lwdnnLRNDescriptor_t                normDesc,
                                unsigned*                           lrnN,
                                double*                             lrnAlpha,
                                double*                             lrnBeta,
                                double*                             lrnK );

/* Destroy an instance of LRN descriptor */
lwdnnStatus_t             lwdnnDestroyLRNDescriptor( lwdnnLRNDescriptor_t lrnDesc );

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
lwdnnStatus_t             lwdnnLRNCrossChannelForward(
                                lwdnnHandle_t                       handle,
                                lwdnnLRNDescriptor_t                normDesc,
                                lwdnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
lwdnnStatus_t             lwdnnLRNCrossChannelBackward(
                                lwdnnHandle_t                       handle,
                                lwdnnLRNDescriptor_t                normDesc,
                                lwdnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx);

typedef enum
{
    LWDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} lwdnnDivNormMode_t;

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
lwdnnStatus_t             lwdnnDivisiveNormalizationForward(
                                lwdnnHandle_t                       handle,
                                lwdnnLRNDescriptor_t                normDesc,
                                lwdnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc, /* same desc for means, temp, temp2*/
                                const void                         *x,
                                const void                         *means, /* if NULL, means are assumed to be zero*/
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

lwdnnStatus_t             lwdnnDivisiveNormalizationBackward(
                                lwdnnHandle_t                       handle,
                                lwdnnLRNDescriptor_t                normDesc,
                                lwdnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc, /* same desc for x, means, dy, temp, temp2*/
                                const void                         *x,
                                const void                         *means, /* if NULL, means are assumed to be zero*/
                                const void                         *dy,
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dXdMeansDesc, /* same desc for dx, dMeans*/
                                void                               *dx, /* output x differential*/
                                void                               *dMeans ); /* output means differential, can be NULL*/

typedef enum
{
    /* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)*/
    LWDNN_BATCHNORM_PER_ACTIVATION = 0,

    /*bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)*/
    LWDNN_BATCHNORM_SPATIAL        = 1,
} lwdnnBatchNormMode_t;

/* static const float LWDNN_BN_MIN_EPSILON = 1e-5; */ /* Minimum epsilon allowed to be used in the Batch Normalization formula*/

/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, ilwVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMealwarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
*/
lwdnnStatus_t             lwdnnDeriveBNTensorDescriptor(
                                lwdnnTensorDescriptor_t             derivedBnDesc,
                                const lwdnnTensorDescriptor_t       xDesc,
                                lwdnnBatchNormMode_t                mode );

/* Computes y = BN(x). Also aclwmulates moving averages of mean and ilwerse variances */
lwdnnStatus_t             lwdnnBatchNormalizationForwardTraining(
                                lwdnnHandle_t                       handle,
                                lwdnnBatchNormMode_t                mode,

                                const void                         *alpha, /* alpha[0] = result blend factor*/
                                const void                         *beta,  /* beta[0] = dest layer blend factor*/

                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,     /* NxCxHxW*/
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y,     /* NxCxHxW*/

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
                                    (normalization is performed across N) */
                                const lwdnnTensorDescriptor_t       bnScaleBiasMealwarDesc,

                                /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation*/
                                const void                         *bnScale,
                                const void                         *bnBias,

                                /* MUST use factor=1 in the very first call of a complete training cycle.
                                   Use a factor=1/(1+n) at N-th call to the function to get
                                   Cumulative Moving Average (CMA) behavior
                                   CMA[n] = (x[1]+...+x[n])/n
                                   Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                   ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                                   CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
                                double                              exponentialAverageFactor,

                                /* Used in Training phase only.
                                   runningMean = newMean*factor + runningMean*(1-factor) */
                                void                               *resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of  variance[x] (factor is applied in the same way as for runningMean) */
                                void                               *resultRunningVariance,

                                /* Has to be >= LWDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double                              epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                void                               *resultSaveMean,
                                void                               *resultSaveIlwVariance );

/*
* Performs Batch Normalization during Inference:
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
* with bnScale, bnBias, runningMean, runningIlwVariance tensors indexed
* according to spatial or per-activation mode. Refer to lwdnnBatchNormalizationForwardTraining
* above for notes on function arguments.
*/
lwdnnStatus_t             lwdnnBatchNormalizationForwardInference(
                                lwdnnHandle_t                       handle,
                                lwdnnBatchNormMode_t                mode,
                                const void                         *alpha, /* alpha[0] = result blend factor*/
                                const void                         *beta,  /* beta[0] = dest layer blend factor*/
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,     /* NxCxHxW*/
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y,     /* NxCxHxW*/
                                const lwdnnTensorDescriptor_t       bnScaleBiasMealwarDesc,
                                const void                         *bnScale,
                                const void                         *bnBias,
                                const void                         *estimatedMean,
                                const void                         *estimatedVariance,
                                double                              epsilon );

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
lwdnnStatus_t             lwdnnBatchNormalizationBackward(
                                lwdnnHandle_t                       handle,
                                lwdnnBatchNormMode_t                mode,
                                const void                         *alphaDataDiff,
                                const void                         *betaDataDiff,
                                const void                         *alphaParamDiff,
                                const void                         *betaParamDiff,
                                const lwdnnTensorDescriptor_t       xDesc, /* same desc for x, dx, dy*/
                                const void                         *x,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const lwdnnTensorDescriptor_t       dBnScaleBiasDesc,
                                const void                         *bnScale, /* bnBias doesn't affect backpropagation*/
                                /* scale and bias diff are not backpropagated below this layer */
                                void                               *dBnScaleResult,
                                void                               *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double                              epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void                         *savedMean,
                                const void                         *savedIlwVariance );


/* APIs for spatial transformer network*/
typedef enum {
    LWDNN_SAMPLER_BILINEAR=0,
} lwdnnSamplerType_t;

lwdnnStatus_t             lwdnnCreateSpatialTransformerDescriptor(

                               lwdnnSpatialTransformerDescriptor_t        *stDesc);

lwdnnStatus_t             lwdnnSetSpatialTransformerNdDescriptor(
                                lwdnnSpatialTransformerDescriptor_t         stDesc,
                                lwdnnSamplerType_t                          samplerType,
                                lwdnnDataType_t                             dataType,
                                const int                                   nbDims,
                                const int                                   dimA[]);

lwdnnStatus_t             lwdnnDestroySpatialTransformerDescriptor(
                                 lwdnnSpatialTransformerDescriptor_t        stDesc);

lwdnnStatus_t             lwdnnSpatialTfGridGeneratorForward(
                                 lwdnnHandle_t                              handle,
                                 const lwdnnSpatialTransformerDescriptor_t  stDesc,
                                 const void                                *theta,
                                 void                                      *grid);

lwdnnStatus_t             lwdnnSpatialTfGridGeneratorBackward(
                                 lwdnnHandle_t                              handle,
                                 const lwdnnSpatialTransformerDescriptor_t  stDesc,
                                 const void                                *dgrid,
                                 void                                      *dtheta);

lwdnnStatus_t             lwdnnSpatialTfSamplerForward(
                                 lwdnnHandle_t                              handle,
                                 lwdnnSpatialTransformerDescriptor_t        stDesc,
                                 const void                                *alpha,
                                 const lwdnnTensorDescriptor_t              xDesc,
                                 const void                                *x,
                                 const void                                *grid,
                                 const void                                *beta,
                                 lwdnnTensorDescriptor_t                    yDesc,
                                 void                                      *y);

lwdnnStatus_t             lwdnnSpatialTfSamplerBackward(
                                 lwdnnHandle_t                              handle,
                                 lwdnnSpatialTransformerDescriptor_t        stDesc,
                                 const void                                *alpha,
                                 const lwdnnTensorDescriptor_t              xDesc,
                                 const void                                *x,
                                 const void                                *beta,
                                 const lwdnnTensorDescriptor_t              dxDesc,
                                 void                                      *dx,
                                 const void                                *alphaDgrid,
                                 const lwdnnTensorDescriptor_t              dyDesc,
                                 const void                                *dy,
                                 const void                                *grid,
                                 const void                                *betaDgrid,
                                 void                                      *dgrid);

typedef struct lwdnnDropoutStruct * lwdnnDropoutDescriptor_t;

lwdnnStatus_t             lwdnnCreateDropoutDescriptor(lwdnnDropoutDescriptor_t * dropoutDesc);

lwdnnStatus_t             lwdnnDestroyDropoutDescriptor(lwdnnDropoutDescriptor_t dropoutDesc);

/*helper function to determine size of the states to be passed to lwdnnSetDropoutDescriptor */
lwdnnStatus_t             lwdnnDropoutGetStatesSize(lwdnnHandle_t handle, size_t * sizeInBytes);

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
lwdnnStatus_t             lwdnnDropoutGetReserveSpaceSize(lwdnnTensorDescriptor_t xdesc, size_t * sizeInBytes);

lwdnnStatus_t             lwdnnSetDropoutDescriptor(lwdnnDropoutDescriptor_t dropoutDesc,
                                                    lwdnnHandle_t handle,
                                                    float dropout,
                                                    void * states,
                                                    size_t stateSizeInBytes,
                                                    unsigned long long seed);

lwdnnStatus_t             lwdnnDropoutForward(lwdnnHandle_t handle,
                                                      const lwdnnDropoutDescriptor_t dropoutDesc,
                                                      const lwdnnTensorDescriptor_t xdesc,
                                                      const void * x,
                                                      const lwdnnTensorDescriptor_t ydesc,
                                                      void * y,
                                                      void * reserveSpace,
                                                      size_t reserveSpaceSizeInBytes);

lwdnnStatus_t             lwdnnDropoutBackward(lwdnnHandle_t handle,
                                               const lwdnnDropoutDescriptor_t dropoutDesc,
                                               const lwdnnTensorDescriptor_t dydesc,
                                               const void * dy,
                                               const lwdnnTensorDescriptor_t dxdesc,
                                               void * dx,
                                               void * reserveSpace,
                                               size_t reserveSpaceSizeInBytes);

/* RNN API */
typedef enum
  {
    LWDNN_RNN_RELU = 0, /* Stock RNN with ReLu activation*/
    LWDNN_RNN_TANH = 1, /* Stock RNN with tanh activation*/
    LWDNN_LSTM = 2,     /* LSTM with no peephole connections*/
    LWDNN_GRU = 3       /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);*/
  } lwdnnRNNMode_t;

typedef enum
  {
   LWDNN_UNIDIRECTIONAL = 0,
   LWDNN_BIDIRECTIONAL = 1      /* Using output concatination at each step. Do we also want to support output sum?*/
  } lwdnnDirectionMode_t;

typedef enum
  {
   LWDNN_LINEAR_INPUT = 0,
   LWDNN_SKIP_INPUT = 1
  } lwdnnRNNInputMode_t;


struct lwdnnRNNStruct;
typedef struct lwdnnRNNStruct*        lwdnnRNNDescriptor_t;

lwdnnStatus_t             lwdnnCreateRNNDescriptor(lwdnnRNNDescriptor_t * rnnDesc);
lwdnnStatus_t             lwdnnDestroyRNNDescriptor(lwdnnRNNDescriptor_t rnnDesc);

lwdnnStatus_t             lwdnnSetRNNDescriptor(lwdnnRNNDescriptor_t rnnDesc,
                                                int hiddenSize,
                                                int numLayers,
                                                lwdnnDropoutDescriptor_t dropoutDesc,
                                                lwdnnRNNInputMode_t inputMode,
                                                lwdnnDirectionMode_t direction,
                                                lwdnnRNNMode_t mode,
                                                lwdnnDataType_t dataType);


// dataType in the RNN descriptor is used to determine math precision
// dataType in weight descriptors and input descriptors is used to describe storage

lwdnnStatus_t             lwdnnGetRNNWorkspaceSize( lwdnnHandle_t              handle,
                                                    const lwdnnRNNDescriptor_t rnnDesc,
                                                    const int seqLength,
                                                    const lwdnnTensorDescriptor_t    *xDesc,
                                                    size_t                     *sizeInBytes
                                                    );

lwdnnStatus_t             lwdnnGetRNNTrainingReserveSize( lwdnnHandle_t              handle,
                                                          const lwdnnRNNDescriptor_t rnnDesc,
                                                          const int seqLength,
                                                          const lwdnnTensorDescriptor_t    *xDesc,
                                                          size_t                     *sizeInBytes
                                                    );


lwdnnStatus_t             lwdnnGetRNNParamsSize( lwdnnHandle_t              handle,
                                                 const lwdnnRNNDescriptor_t rnnDesc,
                                                 const lwdnnTensorDescriptor_t    xDesc,
                                                 size_t                     *sizeInBytes,
                                                 lwdnnDataType_t dataType
                                                    );

lwdnnStatus_t             lwdnnGetRNNLinLayerMatrixParams( lwdnnHandle_t              handle,
                             const lwdnnRNNDescriptor_t rnnDesc,
                             const int layer,
                             const lwdnnTensorDescriptor_t xDesc,
                             const lwdnnFilterDescriptor_t wDesc,
                             const void * w,
                             const int linLayerID,
                             lwdnnFilterDescriptor_t linLayerMatDesc,
                             void ** linLayerMat
                             );

lwdnnStatus_t             lwdnnGetRNNLinLayerBiasParams( lwdnnHandle_t              handle,
                             const lwdnnRNNDescriptor_t rnnDesc,
                             const int layer,
                             const lwdnnTensorDescriptor_t xDesc,
                             const lwdnnFilterDescriptor_t wDesc,
                             const void * w,
                             const int linLayerID,
                             lwdnnFilterDescriptor_t linLayerBiasDesc,
                             void ** linLayerBias
                             );


lwdnnStatus_t             lwdnnRNNForwardInference( lwdnnHandle_t handle,
                                                    const lwdnnRNNDescriptor_t rnnDesc,
                                                    const int seqLength,
                                                    const lwdnnTensorDescriptor_t * xDesc,
                                                    const void * x,
                                                    const lwdnnTensorDescriptor_t hxDesc,
                                                    const void * hx,
                                                    const lwdnnTensorDescriptor_t cxDesc,
                                                    const void * cx,
                                                    const lwdnnFilterDescriptor_t wDesc,
                                                    const void * w,
                                                    const lwdnnTensorDescriptor_t *yDesc,
                                                    void * y,
                                                    const lwdnnTensorDescriptor_t hyDesc,
                                                    void * hy,
                                                    const lwdnnTensorDescriptor_t cyDesc,
                                                    void * cy,
                                                    void * workspace,
                                                    size_t workSpaceSizeInBytes);



lwdnnStatus_t             lwdnnRNNForwardTraining( lwdnnHandle_t handle,
                                                   const lwdnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const lwdnnTensorDescriptor_t *xDesc,
                                                   const void * x,
                                                   const lwdnnTensorDescriptor_t hxDesc,
                                                   const void * hx,
                                                   const lwdnnTensorDescriptor_t cxDesc,
                                                   const void * cx,
                                                   const lwdnnFilterDescriptor_t wDesc,
                                                   const void * w,
                                                   const lwdnnTensorDescriptor_t *yDesc,
                                                   void * y,
                                                   const lwdnnTensorDescriptor_t hyDesc,
                                                   void * hy,
                                                   const lwdnnTensorDescriptor_t cyDesc,
                                                   void * cy,
                                                   void * workspace,
                                                   size_t workSpaceSizeInBytes,
                                                   void * reserveSpace,
                                                   size_t reserveSpaceSizeInBytes);

lwdnnStatus_t             lwdnnRNNBackwardData( lwdnnHandle_t handle,
                                                const lwdnnRNNDescriptor_t rnnDesc,
                                                const int seqLength,
                                                const lwdnnTensorDescriptor_t * yDesc,
                                                const void * y,
                                                const lwdnnTensorDescriptor_t * dyDesc,
                                                const void * dy,
                                                const lwdnnTensorDescriptor_t dhyDesc,
                                                const void * dhy,
                                                const lwdnnTensorDescriptor_t dcyDesc,
                                                const void * dcy,
                                                const lwdnnFilterDescriptor_t wDesc,
                                                const void * w,
                                                const lwdnnTensorDescriptor_t hxDesc,
                                                const void * hx,
                                                const lwdnnTensorDescriptor_t cxDesc,
                                                const void * cx,
                                                const lwdnnTensorDescriptor_t * dxDesc,
                                                void * dx,
                                                const lwdnnTensorDescriptor_t dhxDesc,
                                                void * dhx,
                                                const lwdnnTensorDescriptor_t dcxDesc,
                                                void * dcx,
                                                void * workspace,
                                                size_t workSpaceSizeInBytes,
                                                const void * reserveSpace,
                                                size_t reserveSpaceSizeInBytes );


lwdnnStatus_t             lwdnnRNNBackwardWeights( lwdnnHandle_t handle,
                                                   const lwdnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const lwdnnTensorDescriptor_t * xDesc,
                                                   const void * x,
                                                   const lwdnnTensorDescriptor_t hxDesc,
                                                   const void * hx,
                                                   const lwdnnTensorDescriptor_t * yDesc,
                                                   const void * y,
                                                   const void * workspace,
                                                   size_t workSpaceSizeInBytes,
                                                   const lwdnnFilterDescriptor_t dwDesc,
                                                   void * dw,
                                                   const void * reserveSpace,
                                                   size_t reserveSpaceSizeInBytes );





/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v4 version)
   Routines with _v3 suffix has the functionality of the non-suffixed routines in the LWDNN V4
 */

lwdnnStatus_t             lwdnnSetFilter4dDescriptor_v3(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                int                                 k,        /* number of output feature maps*/
                                int                                 c,        /* number of input feature maps*/
                                int                                 h,        /* height of each input filter*/
                                int                                 w );      /* width of  each input filter*/

lwdnnStatus_t             lwdnnSetFilter4dDescriptor_v4(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                lwdnnTensorFormat_t                 format,
                                int                                 k,        /* number of output feature maps*/
                                int                                 c,        /* number of input feature maps*/
                                int                                 h,        /* height of each input filter*/
                                int                                 w );      /* width of  each input filter*/

lwdnnStatus_t             lwdnnGetFilter4dDescriptor_v3(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                int                                *k,        /* number of output feature maps*/
                                int                                *c,        /* number of input feature maps*/
                                int                                *h,        /* height of each input filter*/
                                int                                *w );      /* width of  each input filter*/

lwdnnStatus_t             lwdnnGetFilter4dDescriptor_v4(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                lwdnnTensorFormat_t                *format,
                                int                                *k,        /* number of output feature maps*/
                                int                                *c,        /* number of input feature maps*/
                                int                                *h,        /* height of each input filter*/
                                int                                *w );      /* width of  each input filter      */

lwdnnStatus_t             lwdnnSetFilterNdDescriptor_v3(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                int                                 nbDims,
                                const int                           filterDimA[] );


lwdnnStatus_t             lwdnnSetFilterNdDescriptor_v4(
                                lwdnnFilterDescriptor_t             filterDesc,
                                lwdnnDataType_t                     dataType, /* image data type*/
                                lwdnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] );

lwdnnStatus_t             lwdnnGetFilterNdDescriptor_v3(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                int                                *nbDims,
                                int                                 filterDimA[] );

lwdnnStatus_t             lwdnnGetFilterNdDescriptor_v4(
                                const lwdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                lwdnnDataType_t                    *dataType, /* image data type*/
                                lwdnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] );

lwdnnStatus_t             lwdnnSetPooling2dDescriptor_v3(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                lwdnnPoolingMode_t                  mode,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );

lwdnnStatus_t             lwdnnSetPooling2dDescriptor_v4(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                lwdnnPoolingMode_t                  mode,
                                lwdnnNanPropagation_t               maxpoolingNanOpt,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );
lwdnnStatus_t             lwdnnGetPooling2dDescriptor_v3(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                lwdnnPoolingMode_t                 *mode,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

lwdnnStatus_t             lwdnnGetPooling2dDescriptor_v4(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                lwdnnPoolingMode_t                 *mode,
                                lwdnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

lwdnnStatus_t             lwdnnSetPoolingNdDescriptor_v3(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                const lwdnnPoolingMode_t            mode,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

lwdnnStatus_t             lwdnnSetPoolingNdDescriptor_v4(
                                lwdnnPoolingDescriptor_t            poolingDesc,
                                const lwdnnPoolingMode_t            mode,
                                const lwdnnNanPropagation_t         maxpoolingNanOpt,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

lwdnnStatus_t             lwdnnGetPoolingNdDescriptor_v3(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                const int                           nbDimsRequested,
                                lwdnnPoolingMode_t                 *mode,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

lwdnnStatus_t             lwdnnGetPoolingNdDescriptor_v4(
                                const lwdnnPoolingDescriptor_t      poolingDesc,
                                int                                 nbDimsRequested,
                                lwdnnPoolingMode_t                 *mode,
                                lwdnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

lwdnnStatus_t             lwdnnActivationForward_v3(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationMode_t               mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

lwdnnStatus_t             lwdnnActivationForward_v4(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       yDesc,
                                void                               *y );

lwdnnStatus_t             lwdnnActivationBackward_v3(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationMode_t               mode,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

lwdnnStatus_t             lwdnnActivationBackward_v4(
                                lwdnnHandle_t                       handle,
                                lwdnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const lwdnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const lwdnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const lwdnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const lwdnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

]]

local LWDNN_PATH = os.getelw('LWDNN_PATH')
if LWDNN_PATH then
    print('Found Environment variable LWDNN_PATH = ' .. LWDNN_PATH)
    lwdnn.C = ffi.load(LWDNN_PATH)
else

    local libnames = {'liblwdnn.so.5', 'liblwdnn.5.dylib'}
    local ok = false
    for i=1,#libnames do
        ok = pcall(function () lwdnn.C = ffi.load(libnames[i]) end)
        if ok then break; end
    end

    if not ok then
        error([['liblwdnn (R5) not found in library path.
Please install LwDNN from https://developer.lwpu.com/lwDNN
Then make sure files named as liblwdnn.so.5 or liblwdnn.5.dylib are placed in
your library load path (for example /usr/local/lib , or manually add a path to LD_LIBRARY_PATH)

Alternatively, set the path to liblwdnn.so.5 or liblwdnn.5.dylib
to the environment variable LWDNN_PATH and rerun torch.
For example: export LWDNN_PATH = "/usr/local/lwca/lib64/liblwdnn.so.5"
]])
    end
end

-- check lwDNN version
lwdnn.version = tonumber(lwdnn.C.lwdnnGetVersion())
if lwdnn.version < 5005 then
  error('These bindings are for version 5005 or above, '
        .. 'while the loaded LwDNN is version: ' .. lwdnn.version
           .. '  \nAre you using an older version of LwDNN?')
end

-- check GPU driver version
local props = lwtorch.getDeviceProperties(lwtorch.getDevice())
if lwtorch.driverVersion and -- for backward compatiblity
     not(lwtorch.driverVersion >= 7050 -- desktop GPUs
       or (props.major == 5 and props.minor == 3 and lwtorch.driverVersion >= 7000) ) -- CheetAh X1
then
  error('Insufficient GPU driver version.')
end
