#ifndef js_cpucfg___
#define js_cpucfg___

#if (defined(linux) || defined(XP_MACOSX)) && defined(AMD64)
#include "jsautocfg.amd64.h"
#elif defined(linux) && defined(PPC64LE)
#include "jsautocfg.amd64.h"
#elif defined(linux) && defined(__arm__)
#include "jsautocfg.arm.h"
#elif defined(linux) && defined(__aarch64__)
#include "jsautocfg.amd64.h"
#else

/* AUTOMATICALLY GENERATED - DO NOT EDIT */

#define IS_LITTLE_ENDIAN 1
#undef  IS_BIG_ENDIAN

#define JS_BYTES_PER_BYTE   1L
#define JS_BYTES_PER_SHORT  2L
#define JS_BYTES_PER_INT    4L
#define JS_BYTES_PER_INT64  8L
#define JS_BYTES_PER_LONG   4L
#define JS_BYTES_PER_FLOAT  4L
#define JS_BYTES_PER_DOUBLE 8L
#define JS_BYTES_PER_WORD   4L
#define JS_BYTES_PER_DWORD  8L

#define JS_BITS_PER_BYTE    8L
#define JS_BITS_PER_SHORT   16L
#define JS_BITS_PER_INT     32L
#define JS_BITS_PER_INT64   64L
#define JS_BITS_PER_LONG    32L
#define JS_BITS_PER_FLOAT   32L
#define JS_BITS_PER_DOUBLE  64L
#define JS_BITS_PER_WORD    32L

#define JS_BITS_PER_BYTE_LOG2   3L
#define JS_BITS_PER_SHORT_LOG2  4L
#define JS_BITS_PER_INT_LOG2    5L
#define JS_BITS_PER_INT64_LOG2  6L
#define JS_BITS_PER_LONG_LOG2   5L
#define JS_BITS_PER_FLOAT_LOG2  5L
#define JS_BITS_PER_DOUBLE_LOG2 6L
#define JS_BITS_PER_WORD_LOG2   5L

#define JS_ALIGN_OF_SHORT   2L
#define JS_ALIGN_OF_INT     4L
#define JS_ALIGN_OF_LONG    4L
#define JS_ALIGN_OF_INT64   4L
#define JS_ALIGN_OF_FLOAT   4L
#define JS_ALIGN_OF_DOUBLE  4L
#define JS_ALIGN_OF_POINTER 4L
#define JS_ALIGN_OF_WORD    4L

#define JS_BYTES_PER_WORD_LOG2   2L
#define JS_BYTES_PER_DWORD_LOG2  3L
#define JS_WORDS_PER_DWORD_LOG2  1L

#define JS_STACK_GROWTH_DIRECTION (-1)

#endif /* !amd64 */

#endif /* js_cpucfg___ */
