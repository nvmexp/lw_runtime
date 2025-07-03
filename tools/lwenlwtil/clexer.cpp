/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#include "pch.hpp"

#include <list>

#include <boost/wave/grammars/cpp_defined_grammar.hpp>
#include <boost/wave/grammars/cpp_grammar.hpp>

#include "clexer.h"

typedef std::list<
    LexToken
  , boost::fast_pool_allocator<LexToken>
  > TokenSequenceType;

template struct grammars::defined_grammar_gen<LexIterator>;
template struct grammars::cpp_grammar_gen<LexIterator, TokenSequenceType>;
