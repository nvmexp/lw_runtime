//  (C) Copyright John Maddock 2001 - 2003.
//  (C) Copyright David Abrahams 2002 - 2003.
//  (C) Copyright Aleksey Gurtovoy 2002.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for most recent version.

//  CodeGear C++ compiler setup:

#if !defined( BOOST_WITH_CODEGEAR_WARNINGS )
// these warnings occur frequently in optimized template code
# pragma warn -8004 // var assigned value, but never used
# pragma warn -8008 // condition always true/false
# pragma warn -8066 // dead code can never execute
# pragma warn -8104 // static members with ctors not threadsafe
# pragma warn -8105 // reference member in class without ctors
#endif
//
// versions check:
// last known and checked version is 0x620
#if (__CODEGEARC__ > 0x620)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  else
#     pragma message( "Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif

// CodeGear C++ Builder 2009
#if (__CODEGEARC__ <= 0x613)
#  define BOOST_NO_INTEGRAL_INT64_T
#  define BOOST_NO_DEPENDENT_NESTED_DERIVATIONS
#  define BOOST_NO_PRIVATE_IN_AGGREGATE
#  define BOOST_NO_USING_DECLARATION_OVERLOADS_FROM_TYPENAME_BASE
   // we shouldn't really need this - but too many things choke
   // without it, this needs more ilwestigation:
#  define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  define BOOST_SP_NO_SP_COLWERTIBLE
#endif

// CodeGear C++ Builder 2010
#if (__CODEGEARC__ <= 0x620)
#  define BOOST_NO_TYPENAME_WITH_CTOR    // Cannot use typename keyword when making temporaries of a dependant type
#  define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_NESTED_FRIENDSHIP     // TC1 gives nested classes access rights as any other member
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
// Temporary hack, until specific MPL preprocessed headers are generated
#  define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#  ifdef NDEBUG
      // fix broken <cstring> so that Boost.test works:
#     include <cstring>
#     undef strcmp
#  endif
   // fix broken errno declaration:
#  include <errno.h>
#  ifndef errno
#     define errno errno
#  endif

#endif
//
// C++0x macros:
//
#define BOOST_HAS_CHAR16_T
#define BOOST_HAS_CHAR32_T
#define BOOST_HAS_LONG_LONG
// #define BOOST_HAS_ALIGNOF
#define BOOST_HAS_DECLTYPE
#define BOOST_HAS_EXPLICIT_COLWERSION_OPS
// #define BOOST_HAS_RVALUE_REFS
#define BOOST_HAS_SCOPED_ENUM
// #define BOOST_HAS_STATIC_ASSERT
#define BOOST_HAS_STD_TYPE_TRAITS

#define BOOST_NO_AUTO_DECLARATIONS
#define BOOST_NO_AUTO_MULTIDECLARATIONS
#define BOOST_NO_CONCEPTS
#define BOOST_NO_CONSTEXPR
#define BOOST_NO_DEFAULTED_FUNCTIONS
#define BOOST_NO_DELETED_FUNCTIONS
#define BOOST_NO_EXTERN_TEMPLATE
#define BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
#define BOOST_NO_INITIALIZER_LISTS
#define BOOST_NO_LAMBDAS
#define BOOST_NO_NULLPTR
#define BOOST_NO_RAW_LITERALS
#define BOOST_NO_RVALUE_REFERENCES
#define BOOST_NO_SFINAE_EXPR
#define BOOST_NO_STATIC_ASSERT
#define BOOST_NO_TEMPLATE_ALIASES
#define BOOST_NO_UNICODE_LITERALS
#define BOOST_NO_VARIADIC_TEMPLATES

//
// TR1 macros:
//
#define BOOST_HAS_TR1_HASH
#define BOOST_HAS_TR1_TYPE_TRAITS
#define BOOST_HAS_TR1_UNORDERED_MAP
#define BOOST_HAS_TR1_UNORDERED_SET

#define BOOST_HAS_MACRO_USE_FACET

#define BOOST_NO_INITIALIZER_LISTS

// On non-Win32 platforms let the platform config figure this out:
#ifdef _WIN32
#  define BOOST_HAS_STDINT_H
#endif

//
// __int64:
//
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_MS_INT64
#endif
//
// check for exception handling support:
//
#if !defined(_CPPUNWIND) && !defined(BOOST_CPPUNWIND) && !defined(__EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif
//
// all versions have a <dirent.h>:
//
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_DIRENT_H
#endif
//
// all versions support __declspec:
//
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_DECLSPEC
#endif
//
// ABI fixing headers:
//
#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/borland_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/borland_suffix.hpp"
#endif
//
// Disable Win32 support in ANSI mode:
//
#  pragma defineonoption BOOST_DISABLE_WIN32 -A
//
// MSVC compatibility mode does some nasty things:
// TODO: look up if this doesn't apply to the whole 12xx range
//
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_VOID_RETURNS
#endif

#define BOOST_COMPILER "CodeGear C++ version " BOOST_STRINGIZE(__CODEGEARC__)

