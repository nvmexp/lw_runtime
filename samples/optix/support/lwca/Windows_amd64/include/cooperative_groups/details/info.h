 /* Copyright 1993-2016 LWPU Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to LWPU intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to LWPU and are being provided under the terms and
  * conditions of a form of LWPU software license agreement by and
  * between LWPU and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of LWPU is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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

#ifndef _CG_INFO_H_
#define _CG_INFO_H_
/*
** Define: _CG_VERSION
*/
#define _CG_VERSION 1000

/*
** Define: _CG_ABI_VERSION
*/
#ifndef _CG_ABI_VERSION
# define _CG_ABI_VERSION 1
#endif

/*
** Define: _CG_ABI_EXPERIMENTAL
** Desc: If enabled, sets all features enabled (ABI-breaking or experimental)
*/
#if defined(_CG_ABI_EXPERIMENTAL)
#endif

#define _CG_CONCAT_INNER(x, y) x ## y
#define _CG_CONCAT_OUTER(x, y) _CG_CONCAT_INNER(x, y)
#define _CG_NAMESPACE _CG_CONCAT_OUTER(__v, _CG_ABI_VERSION)

#define _CG_BEGIN_NAMESPACE \
    namespace cooperative_groups { namespace _CG_NAMESPACE {
#define _CG_END_NAMESPACE \
    }; using namespace _CG_NAMESPACE; };

#if !defined(_CG_STATIC_QUALIFIER)
# define _CG_STATIC_QUALIFIER static __forceinline__ __device__
#endif
#if !defined(_CG_QUALIFIER)
# define _CG_QUALIFIER __forceinline__ __device__
#endif

#if defined(_MSC_VER)
# define _CG_DEPRECATED __declspec(deprecated)
#else
# define _CG_DEPRECATED __attribute__((deprecated))
#endif

#if (__LWDA_ARCH__ >= 600) || !defined(__LWDA_ARCH__)
# define _CG_HAS_GRID_GROUP
#endif
#if (__LWDA_ARCH__ >= 600) || !defined(__LWDA_ARCH__)
# define _CG_HAS_MULTI_GRID_GROUP
#endif
#if (__LWDA_ARCH__ >= 700) || !defined(__LWDA_ARCH__)
# define _CG_HAS_MATCH_COLLECTIVE
#endif
// Has __half and __half2
// Only usable if you include the lwda_fp16.h extension, and
// _before_ including cooperative_groups.h
#ifdef __LWDA_FP16_TYPES_EXIST__
# define _CG_HAS_FP16_COLLECTIVE
#endif

#if (__LWDA_ARCH__ >= 800) || !defined(__LWDA_ARCH__)
# define _CG_HAS_OP_REDUX
#endif

#if (defined(__cplusplus) && (__cplusplus >= 201103L)) || (defined(_MSC_VER) && (_MSC_VER >= 1900))
# define _CG_CPP11_FEATURES
#endif

#if defined(CG_USE_LWDA_STL)
# define _CG_USE_LWDA_STL
#else
# define _CG_USE_OWN_TRAITS
#endif

#ifdef _CG_CPP11_FEATURES
// Use lwca::std:: for type_traits
# if defined(_CG_USE_LWDA_STL)
#  define _CG_STL_NAMESPACE lwca::std
#  include <lwca/std/type_traits>
// Use CG's implementation of type traits
# else
#  define _CG_STL_NAMESPACE cooperative_groups::details::templates
# endif
#endif

#ifdef _CG_CPP11_FEATURES
# define _CG_STATIC_CONST_DECL static constexpr
# define _CG_CONST_DECL constexpr
#else
# define _CG_STATIC_CONST_DECL static const
# define _CG_CONST_DECL const
#endif

#if (defined(_MSC_VER) && !defined(_WIN64)) || defined(__arm__)
# define _CG_ASM_PTR_CONSTRAINT "r"
#else
#  define _CG_ASM_PTR_CONSTRAINT "l"
#endif

/*
** Define: CG_DEBUG
** What: Enables various runtime safety checks
*/
#if defined(__LWDACC_DEBUG__) && defined(CG_DEBUG) && !defined(NDEBUG)
# define _CG_DEBUG
#endif

#if defined(_CG_DEBUG)
# include <assert.h>
# define _CG_ASSERT(x) assert((x));
# define _CG_ABORT() assert(0);
#else
# define _CG_ASSERT(x)
# define _CG_ABORT() __trap();
#endif

#if defined(_CG_CPP11_FEATURES) && !defined(_CG_USE_LWDA_STL)
_CG_BEGIN_NAMESPACE

namespace details {
namespace templates {

/**
 * Integral constants
 **/
template <typename Ty, Ty Val>
struct integral_constant {
    static constexpr Ty value = Val;
    typedef Ty type;

    _CG_QUALIFIER constexpr operator type() const noexcept { return value; }
    _CG_QUALIFIER constexpr type operator()() const noexcept { return value; }
};

typedef integral_constant<bool, true>  true_type;
typedef integral_constant<bool, false> false_type;

/**
 * CV Qualifiers
 **/
template <class Ty> struct is_lvalue_reference       : public details::templates::false_type {};
template <class Ty> struct is_lvalue_reference<Ty&>  : public details::templates::true_type {};

template <class Ty> struct remove_reference       {typedef Ty type;};
template <class Ty> struct remove_reference<Ty&>  {typedef Ty type;};
template <class Ty> struct remove_reference<Ty&&> {typedef Ty type;};

template <class Ty>
using remove_reference_t = typename details::templates::remove_reference<Ty>::type;

template <class Ty> struct remove_const           {typedef Ty type;};
template <class Ty> struct remove_const<const Ty> {typedef Ty type;};

template <class Ty> struct remove_volatile              {typedef Ty type;};
template <class Ty> struct remove_volatile<volatile Ty> {typedef Ty type;};

template <class Ty> struct remove_cv {typedef typename details::templates::remove_volatile<typename details::templates::remove_const<Ty>::type>::type type;};

template <class Ty>
using remove_cv_t = typename details::templates::remove_cv<Ty>::type;

template <class Ty>
_CG_QUALIFIER Ty&& forward(remove_reference_t<Ty> &t) noexcept {
    return static_cast<Ty&&>(t);
}

template <class Ty>
_CG_QUALIFIER Ty&& forward(remove_reference_t<Ty> &&t) noexcept {
    static_assert(!details::templates::is_lvalue_reference<Ty>::value, "Forwarding an rvalue as an lvalue is not allowed.");
    return static_cast<Ty&&>(t);
}

/**
 * is_integral
 **/
template <class Ty> struct _is_integral                     : public details::templates::false_type {};
template <>         struct _is_integral<bool>               : public details::templates::true_type {};
template <>         struct _is_integral<char>               : public details::templates::true_type {};
template <>         struct _is_integral<unsigned char>      : public details::templates::true_type {};
template <>         struct _is_integral<short>              : public details::templates::true_type {};
template <>         struct _is_integral<unsigned short>     : public details::templates::true_type {};
template <>         struct _is_integral<int>                : public details::templates::true_type {};
template <>         struct _is_integral<unsigned int>       : public details::templates::true_type {};
template <>         struct _is_integral<long>               : public details::templates::true_type {};
template <>         struct _is_integral<long long>          : public details::templates::true_type {};
template <>         struct _is_integral<unsigned long>      : public details::templates::true_type {};
template <>         struct _is_integral<unsigned long long> : public details::templates::true_type {};
//Vector type support?

template <typename Ty>
struct is_integral : public details::templates::_is_integral<typename details::templates::remove_cv<Ty>::type> {};

/**
 * is_floating_point
 **/
template <class Ty> struct _is_floating_point              : public details::templates::false_type {};
template <>         struct _is_floating_point<float>       : public details::templates::true_type {};
template <>         struct _is_floating_point<double>      : public details::templates::true_type {};
template <>         struct _is_floating_point<long double> : public details::templates::true_type {};
# ifdef __LWDA_FP16_TYPES_EXIST__
template <>         struct _is_floating_point<__half>      : public details::templates::true_type {};
template <>         struct _is_floating_point<__half2>     : public details::templates::true_type {};
# endif
//Vector type support?

template <typename Ty>
struct is_floating_point : public details::templates::_is_floating_point<typename details::templates::remove_cv<Ty>::type> {};

template <class T>
struct is_arithmetic : details::templates::integral_constant<
    bool,
    details::templates::is_integral<T>::value ||
    details::templates::is_floating_point<T>::value> {};

template <typename Ty, bool = details::templates::is_arithmetic<Ty>::value>
struct _is_unsigned : details::templates::integral_constant<bool, Ty(0) < Ty(-1)> {};

template <typename Ty>
struct _is_unsigned<Ty,false> : details::templates::false_type {};

template <typename Ty>
struct is_unsigned : _is_unsigned<typename details::templates::remove_cv<Ty>::type> {};

/**
 * programmatic type traits
 **/
template<bool B, class Ty = void>
struct enable_if {};

template<class Ty>
struct enable_if<true, Ty> { typedef Ty type; };

template<bool Cond, typename Ty = void>
using enable_if_t = typename details::templates::enable_if<Cond, Ty>::type;

template<class Ty1, class Ty2>
struct is_same : details::templates::false_type {};

template<class Ty>
struct is_same<Ty, Ty> : details::templates::true_type {};

} // templates
} // details
_CG_END_NAMESPACE

#endif // _CG_CPP11_FEATURES

#endif // _CG_INFO_H_
