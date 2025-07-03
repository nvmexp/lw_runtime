#ifndef __COMPILER_GNU_H_INCLUDED
#define __COMPILER_GNU_H_INCLUDED


#if defined(__X86__)
#define __INT64_ALIGN __attribute__((__aligned__(8)))
#else
#define __INT64_ALIGN
#endif

#if (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)) && !defined(_lint)
typedef int                         _GCC_ATTR_ALIGN_64t __attribute__((__mode__(__DI__)));
typedef unsigned int                _GCC_ATTR_ALIGN_u64t __attribute__((__mode__(__DI__)));
typedef _GCC_ATTR_ALIGN_64t         _Int64t __INT64_ALIGN;
typedef _GCC_ATTR_ALIGN_u64t        _Uint64t __INT64_ALIGN;
#else
typedef unsigned long long          _GCC_ATTR_ALIGN_u64t;
typedef signed long long            _GCC_ATTR_ALIGN_64t;
typedef _GCC_ATTR_ALIGN_u64t        _Uint64t __INT64_ALIGN;
typedef _GCC_ATTR_ALIGN_64t         _Int64t __INT64_ALIGN;
#endif

#if __INT_BITS__ == 32
typedef unsigned                    _GCC_ATTR_ALIGN_u32t;
typedef int                         _GCC_ATTR_ALIGN_32t;
typedef _GCC_ATTR_ALIGN_u32t        _Uint32t;
typedef _GCC_ATTR_ALIGN_32t            _Int32t;
#elif __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)
typedef int                         _GCC_ATTR_ALIGN_32t;
typedef unsigned int                _GCC_ATTR_ALIGN_u32t;
typedef _GCC_ATTR_ALIGN_32t         _Int32t;
typedef _GCC_ATTR_ALIGN_u32t        _Uint32t;
#else
typedef unsigned long                _GCC_ATTR_ALIGN_u32t;
typedef signed long                  _GCC_ATTR_ALIGN_32t;
typedef _GCC_ATTR_ALIGN_u32t        _Uint32t;
typedef _GCC_ATTR_ALIGN_32t         _Int32t;
#endif

#if __INT_BITS__ == 16
typedef int                         _GCC_ATTR_ALIGN_16t;
typedef unsigned                    _GCC_ATTR_ALIGN_u16t;
typedef _GCC_ATTR_ALIGN_u16t        _Uint16t;
typedef _GCC_ATTR_ALIGN_16t         _Int16t;
#elif (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)) && !defined(_lint)
typedef int                            _GCC_ATTR_ALIGN_16t __attribute__((__mode__(__HI__)));
typedef unsigned int                _GCC_ATTR_ALIGN_u16t __attribute__((__mode__(__HI__)));
typedef _GCC_ATTR_ALIGN_16t         _Int16t;
typedef _GCC_ATTR_ALIGN_u16t        _Uint16t;
#else
typedef signed short                _GCC_ATTR_ALIGN_16t;
typedef unsigned short              _GCC_ATTR_ALIGN_u16t;
typedef _GCC_ATTR_ALIGN_u16t        _Uint16t;
typedef _GCC_ATTR_ALIGN_16t         _Int16t;
#endif

#if (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 7)) && !defined(_lint)
typedef int                         _GCC_ATTR_ALIGN_8t __attribute__((__mode__(__QI__)));
typedef unsigned int                _GCC_ATTR_ALIGN_u8t __attribute__((__mode__(__QI__)));
typedef _GCC_ATTR_ALIGN_8t          _Int8t;
typedef _GCC_ATTR_ALIGN_u8t         _Uint8t;
#else
typedef signed char                  _GCC_ATTR_ALIGN_8t;
typedef unsigned char                _GCC_ATTR_ALIGN_u8t;
typedef _GCC_ATTR_ALIGN_u8t          _Uint8t;
typedef _GCC_ATTR_ALIGN_8t           _Int8t;
#endif

typedef _Uint32t                    _Size32t;
typedef _Uint64t                    _Size64t;
typedef _Int32t                     _Ssize32t;
typedef _Int64t                     _Ssize64t;
#if defined(__SIZEOF_SIZE_T__) && __SIZEOF_INT__+0 == __SIZEOF_SIZE_T__
typedef unsigned                    _Sizet;
typedef int                         _Ssizet;
#elif __SIZEOF_SIZE_T__+0 == 8
typedef _Uint64t                    _Sizet;
typedef _Int64t                     _Ssizet;
#elif !defined(__SIZEOF_SIZE_T__) || __SIZEOF_SIZE_T__+0 == 4
typedef _Uint32t                    _Sizet;
typedef _Int32t                     _Ssizet;
#else
#error Unable to declare size_t type
#endif


#endif /* __COMPILER_GNU_H_INCLUDED */
