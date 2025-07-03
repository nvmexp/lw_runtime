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
// Note: This neither implements full ptree functionality,
// nor does it wrap the base template: basic_ptree<Key, Data, Compare>

// For future editors: if you need a funciton of ptree that is not
// implemented here, please add it. Do not use boost::ptree as it is
// not exception safe

#include "es_ptree.h"

es_ptree::es_ptree()
{}

es_ptree::es_ptree(ptree pt) :
    m_Pt(pt)
{}

RX es_ptree::get(const string& path, string* rtn) const
{
    try
    {
        *rtn = m_Pt.get<string>(path);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::ILWALID_FILE_FORMAT, e.what());
    }
}

RX es_ptree::get(const string& path, UINT32* rtn) const
{
    try
    {
        *rtn = m_Pt.get<UINT32>(path);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::ILWALID_FILE_FORMAT, e.what());
    }
}

RX es_ptree::get_child(const string& path, es_ptree* rtn) const
{
    try
    {
        rtn->m_Pt = m_Pt.get_child(path);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::ILWALID_FILE_FORMAT, e.what());
    }
}

bool es_ptree::has_child(const string& path) const
{
    // find() uses iterators and should not throw any exceptions
    return m_Pt.find(path) != m_Pt.not_found();
}

RX es_ptree::put(const string& path, const string& value)
{
    try
    {
        m_Pt.put(path, value);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::SOFTWARE_ERROR, e.what());
    }
}

RX es_ptree::put(const string& path, const UINT32& value)
{
    try
    {
        m_Pt.put(path, value);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::SOFTWARE_ERROR, e.what());
    }
}

RX es_ptree::put_child(const string& path, const es_ptree& value)
{
    try
    {
        m_Pt.put_child(path, value.m_Pt);
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::SOFTWARE_ERROR, e.what());
    }
}

RX es_ptree::push_back(std::pair<string, es_ptree>& value)
{
    try
    {
        m_Pt.push_back(std::make_pair(value.first, value.second.m_Pt));
        return RX::OK;
    }
    catch (std::exception& e)
    {
        return RX(RC::SOFTWARE_ERROR, e.what());
    }
}

void es_ptree::sort()
{
    // sort() should not be throwing any exceptions
    m_Pt.sort();
}

void es_ptree::clear()
{
    // clear() should not be throwing any exceptions
    m_Pt.clear();
}

// The iterator functions of ptree should be as exception safe as any
// other iterator. Don't bother with try-catches on them
es_ptree::iterator es_ptree::begin() const
{
    return iterator(m_Pt.begin());
}

es_ptree::iterator es_ptree::end() const
{
    return iterator(m_Pt.end());
}

es_ptree_iterator::es_ptree_iterator() :
    m_Pti(ptree::iterator())
{}

es_ptree_iterator::es_ptree_iterator(ptree::const_iterator iter) :
 m_Pti(iter)
{}

es_ptree_iterator& es_ptree_iterator::operator++()
{
    ++m_Pti;
    return (*this);
}

es_ptree_iterator es_ptree_iterator::operator++(int unused)
{
    es_ptree_iterator rtn = (*this);
    m_Pti++;
    return rtn;
}

std::pair<string, es_ptree> es_ptree_iterator::operator*() const
{
    return std::make_pair(m_Pti->first, es_ptree(m_Pti->second));
}

bool es_ptree_iterator::operator==(const es_ptree_iterator& it) const
{
    return m_Pti == it.m_Pti;
}
bool es_ptree_iterator::operator!=(const es_ptree_iterator& it) const
{
    return !((*this) == it);
}
