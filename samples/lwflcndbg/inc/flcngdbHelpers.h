/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbHelpers.h
 * @brief file has helper functions of flcngdb.
 *
 *  */
#ifndef _FLCNGDBHELPERS_H_
#define _FLCNGDBHELPERS_H_

#include <sstream>
#include <string>
#include <algorithm>

using namespace std;

const string delimiters = " \r\n\t,";

// right trim
inline void rtrim(string& line) {
    line.erase(line.find_last_not_of(delimiters)+1);
}
// left trim
inline void ltrim(string& line) {
    line.erase(0, line.find_first_not_of(delimiters));
}

// function that both left and right strips a line
inline void trim(string& line)
{
    rtrim(line);
    ltrim(line);
}

// function removes all spaces in a string
inline void removeSpace(string& line)
{
    line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
}

inline void replaceSeparators(string& str, const string& separators)
{
    int pos = (int) str.find_first_of(separators);

    while(pos != (int)string::npos)
    {
        if((str[pos+1] != '\\') && (str[pos+1] != '/'))
            str = str.replace(pos, 1, 1, ' ');
        pos = (int) str.find_first_of(separators, pos+1);
    }
}

// generates a <val, key> map (needs unique vals)
template <typename T, typename U>
map<U, T> reverseMap(const map<T, U>& src)
{
    typename map<T, U>::const_iterator it;
    map<U, T> dest;

    for(it = src.begin(); it != src.end(); ++it)
    {
        dest.insert(pair<U, T>(it->second, it->first));
    }

    return dest;
}

// get a struct from an arbitrary map
template <typename T, typename U>
bool getFromMap(const map<T, U>& mSrc, const T& key, U& ret)
{
    typename map<T, U>::const_iterator it;
    it = mSrc.find(key);

    if(it == mSrc.end())
        return false;

    ret = it->second;

    return true;
}

#endif /* _FLCNGDBHELPERS_H_ */



