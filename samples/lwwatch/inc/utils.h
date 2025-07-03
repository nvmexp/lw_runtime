#ifndef __LWWATCH_UTILS_H
#define __LWWATCH_UTILS_H
#include <sstream>

// Splits the input at white spaces. 
inline void tokenize(const char * str, vector<string> & args)
{
    string input_str(str);
    string buf; 
    stringstream ss(input_str);

    args.clear();
    while(ss >> buf)
        args.push_back(buf);
}


inline bool wild_compare(const char *wild, const char *string) {
  const char *cp = NULL, *mp = NULL;

  while ((*string) && (*wild != '*')) {
    if ((*wild != *string) && (*wild != '?')) {
      return 0;
    }
    wild++;
    string++;
  }

  while (*string) {
    if (*wild == '*') {
      if (!*++wild) {
        return 1;
      }
      mp = wild;
      cp = string+1;
    } else if ((tolower(*wild) == tolower(*string)) || (*wild == '?')) {
      wild++;
      string++;
    } else {
      wild = mp;
      string = cp++;
    }
  }

  while (*wild == '*') {
    wild++;
  }
  return !*wild;
}

#endif
