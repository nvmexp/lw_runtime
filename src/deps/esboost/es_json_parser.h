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

#pragma once
#ifndef INCLUDED_ES_JSON_PARSER_H
#define INCLUDED_ES_JSON_PARSER_H

#include <string>

#include "es_ptree.h"
#include "core/include/rx.h"

class es_json_parser
{
    public:

        static RX read_json
        (
            const std::string &filename,
            es_ptree &pt,
            const std::locale &loc = std::locale()
        );

        static RX write_json
        (
            const std::string &filename,
            es_ptree &pt,
            const std::locale &loc = std::locale()
        );
};

#endif
