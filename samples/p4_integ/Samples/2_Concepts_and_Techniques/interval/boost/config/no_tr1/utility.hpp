//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// The aim of this header is just to include <utility> but to do
// so in a way that does not result in relwrsive inclusion of
// the Boost TR1 components if boost/tr1/tr1/utility is in the
// include search path.  We have to do this to avoid cirlwlar
// dependencies:
//

#ifndef BOOST_CONFIG_UTILITY
#  define BOOST_CONFIG_UTILITY

#  ifndef BOOST_TR1_NO_RELWRSION
#     define BOOST_TR1_NO_RELWRSION
#     define BOOST_CONFIG_NO_UTILITY_RELWRSION
#  endif

#  include <utility>

#  ifdef BOOST_CONFIG_NO_UTILITY_RELWRSION
#     undef BOOST_TR1_NO_RELWRSION
#     undef BOOST_CONFIG_NO_UTILITY_RELWRSION
#  endif

#endif
