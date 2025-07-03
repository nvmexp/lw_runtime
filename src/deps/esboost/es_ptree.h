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

// Exception safe wrapper around boost::property_tree::ptree
// ptree is typedef'ed to basic_ptree<string, string> and this is the
// assumed type for key/value considerations
// Note: Ptree is an extensive class, and thus this wrapper does not
// wrap the entirety of ptree functionality

// For future editors: if you need a funciton of ptree that is not implemented
// here, or a more complete implementation (e.g. templates) please add it.
// Do not use boost::ptree as it is not exception safe (unless MODS starts
// allowing exceptions in its builds, in which case, use the real boost::ptree).

#pragma once
#ifndef INCLUDED_ES_PTREE_H
#define INCLUDED_ES_PTREE_H

#include <string>
#include <locale>

#include <boost/property_tree/ptree.hpp>

#include "core/include/rx.h"
#include "core/include/tee.h"
#include "core/include/types.h"

using std::string;
using boost::property_tree::ptree;

class es_json_parser;
class es_ptree_iterator;

class es_ptree
{
    friend class es_json_parser;
    friend class es_ptree_iterator;
    private:
        ptree m_Pt;
        es_ptree(ptree pt);

    public:
        es_ptree();

        RX get(const string& path, string* rtn) const;
        RX get(const string& path, UINT32* rtn) const;

        RX get_child(const string& path, es_ptree* rtn) const;

        bool has_child(const string& path) const;

        RX put(const string& path, const string& value);
        RX put(const string& path, const UINT32& value);

        RX put_child(const string& path, const es_ptree& value);
        RX push_back(std::pair<string, es_ptree>& value);

        void sort();
        void clear();

        typedef es_ptree_iterator iterator;

        iterator begin() const;
        iterator end() const;

};

class es_ptree_iterator
{
    friend class es_ptree;
    private:
        ptree::const_iterator m_Pti;

    public:
        es_ptree_iterator();
        es_ptree_iterator(ptree::const_iterator iter);

        es_ptree_iterator& operator++(); //prefix increment
        es_ptree_iterator operator++(int); //postfix increment

        std::pair<string, es_ptree> operator*() const;

        bool operator==(const es_ptree_iterator&) const;
        bool operator!=(const es_ptree_iterator&) const;
};

#endif
