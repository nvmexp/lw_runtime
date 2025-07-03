#ifndef DCGMSTRINGCOLWERSIONS_H_
#define DCGMSTRINGCOLWERSIONS_H_

#include <string>
#include <sstream>

/**
 * Colwert a string to a type, T.  The type, T, must support the streaming operatior ">>".
 * If the colwersion succeeds then \ref success will be set to true, otherwise false.
 * If the colwersion does not succeed the return value is undefined.
 */
template <typename T>
T strTo(std::string str, bool *success)
{
    char extraChar;
    T val;
    std::stringstream ss(str);
    ss >> val;

    if (success != NULL)
    {
        *success = !ss.fail() && !ss.get(extraChar);
    }

    return val;
}

/**
 * Colwert a string to a type, T.  If the colwersion fails, the return value
 * is undefined so only use this function if you know that the string will
 * definitely colwert to the type you want.  To know if the colwersion
 * succeeded or not use "strTo(std::string str, bool *success)" instead.
 */
template <typename T>
T strTo(std::string str)
{
    return strTo<T>(str, NULL);
}

/**
 * Get the string representation of a value.
 * The type of the value must support the streaming operator "<<".
 */
template <typename T>
std::string toStr(T value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

#endif /* DCGMSTRINGCOLWERSIONS_H_ */
