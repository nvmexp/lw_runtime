/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2014-2015 by LWPU Corporation. All rights reserved. All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation. Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifdef PRAGMA_ONCE_SUPPORTED
#pragma once
#endif

#ifndef VALIDATEKEYARG_H
#define VALIDATEKEYARG_H

#include <iostream>
#include <vector>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <boost/throw_exception.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <boost/algorithm/string.hpp>

#include "core/include/types.h"

#include "ctr64encryptor.h"

using namespace boost::xpressive;
namespace po = boost::program_options;

struct KeyPartUINT32
{
    UINT32 m_t;
    KeyPartUINT32(const UINT32 t) : m_t(t) {};
    KeyPartUINT32(const std::string &k)
    {
        m_t = 0;
        sregex isHexRegex = icase(as_xpr("0x") >> +boost::xpressive::set[_d | range('a', 'f')]);
        sregex checkFormat = (+_d) | isHexRegex;
        if (!regex_match(k, checkFormat))
        {
            boost::throw_exception(po::ilwalid_option_value(k));
        }
        std::stringstream ss;
        if (regex_match(k, isHexRegex))
        {
            ss << std::hex << (k.c_str() + 2);
        }
        else
        {
            ss << k;
        }
        ss >> m_t;
    };
    KeyPartUINT32() : m_t(0) {};
    KeyPartUINT32(const KeyPartUINT32 & t) : m_t(t.m_t) {}
    KeyPartUINT32 & operator=(const KeyPartUINT32 & rhs) { m_t = rhs.m_t; return *this; }
    KeyPartUINT32 & operator=(const UINT32 & rhs) { m_t = rhs; return *this; }
    operator const UINT32 & () const { return m_t; }
    operator UINT32 & () { return m_t; }
    UINT32 * operator &() { return &m_t; }
};

void validate(boost::any& v, const std::vector<std::string>& s, std::vector<KeyPartUINT32>*, int)
{
    using namespace po::validators;
    if (v.empty())
    {
        v = boost::any(std::vector<KeyPartUINT32>());
    }
    if (AES::DW != s.size())
    {
        boost::throw_exception(po::ilwalid_option_value(boost::algorithm::join(s, " ")));
    }
    std::vector<KeyPartUINT32>* tv = boost::any_cast<std::vector<KeyPartUINT32> >(&v);
    for (unsigned i = 0; i < s.size(); ++i)
    {
        tv->push_back(KeyPartUINT32(s[i]));
    }
}

#endif
