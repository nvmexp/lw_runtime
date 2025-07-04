//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// The aim of this header is just to include <memory> but to do
// so in a way that does not result in relwrsive inclusion of
// the Boost TR1 components if boost/tr1/tr1/memory is in the
// include search path.  We have to do this to avoid cirlwlar
// dependencies:
//

#ifndef BOOST_CONFIG_MEMORY
#  define BOOST_CONFIG_MEMORY

#  ifndef BOOST_TR1_NO_RELWRSION
#     define BOOST_TR1_NO_RELWRSION
#     define BOOST_CONFIG_NO_MEMORY_RELWRSION
#  endif

#  include <memory>

#  ifdef BOOST_CONFIG_NO_MEMORY_RELWRSION
#     undef BOOST_TR1_NO_RELWRSION
#     undef BOOST_CONFIG_NO_MEMORY_RELWRSION
#  endif

#endif
