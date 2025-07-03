/// lwdaDataType_t is an enumeration of the types supported by LWCA libraries.
/// lwTENSOR supports real FP16, BF16, FP32 and FP64 as well as complex FP32 and FP64 input types.
typedef enum lwdaDataType_t
{
	LWDA_R_16F  =  2, ///< 16-bit real half precision floating-point type
//	LWDA_C_16F  =  6, /* complex as a pair of half numbers */
	LWDA_R_16BF = 14, ///< 16-bit real BF16 floating-point type
//	LWDA_C_16BF = 15, /* complex as a pair of lw_bfloat16 numbers */
	LWDA_R_32F  =  0, ///< 32-bit real single precision floating-point type
	LWDA_C_32F  =  4, ///< 32-bit complex single precision floating-point type (represented as pair of real and imaginary part)
	LWDA_R_64F  =  1, ///< 64-bit real double precision floating-point type
	LWDA_C_64F  =  5, ///< 32-bit complex double precision floating-point type (represented as pair of real and imaginary part)
} lwdaDataType;
