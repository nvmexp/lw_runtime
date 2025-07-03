/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

// Exception safe adapter for the boost json_parser class
// caught exceptions get translated into RX returns

#include "es_json_parser.h"

#include <exception>

#include <boost/property_tree/json_parser.hpp>

#include "core/include/tee.h"

RX es_json_parser::read_json(const std::string &filename,
                             es_ptree &pt,
                             const std::locale &loc)
{
    try
    {
        boost::property_tree::json_parser::read_json(filename, pt.m_Pt, loc);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::FILE_READ_ERROR, e.what());
    }
}

RX es_json_parser::write_json(const std::string &filename,
                              es_ptree &pt,
                              const std::locale &loc)
{
    try
    {
        boost::property_tree::json_parser::write_json(filename, pt.m_Pt, loc);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::FILE_WRITE_ERROR, e.what());
    }
}
