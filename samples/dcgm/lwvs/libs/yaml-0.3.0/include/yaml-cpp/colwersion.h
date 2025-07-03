#ifndef COLWERSION_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define COLWERSION_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) || (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || (__GNUC__ >= 4)) // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif


#include "yaml-cpp/null.h"
#include "yaml-cpp/traits.h"
#include <limits>
#include <string>
#include <sstream>

namespace YAML
{
	// traits for colwersion
	
	template<typename T>
	struct is_scalar_colwertible { enum { value = is_numeric<T>::value }; };
	
	template<> struct is_scalar_colwertible<std::string> { enum { value = true }; };
	template<> struct is_scalar_colwertible<bool> { enum { value = true }; };
	template<> struct is_scalar_colwertible<_Null> { enum { value = true }; };

	// actual colwersion
	
	inline bool Colwert(const std::string& input, std::string& output) {
		output = input;
		return true;
	}
	
	YAML_CPP_API bool Colwert(const std::string& input, bool& output);
	YAML_CPP_API bool Colwert(const std::string& input, _Null& output);
	
	inline bool IsInfinity(const std::string& input) {
		return input == ".inf" || input == ".Inf" || input == ".INF" || input == "+.inf" || input == "+.Inf" || input == "+.INF";
	}
	
	inline bool IsNegativeInfinity(const std::string& input) {
		return input == "-.inf" || input == "-.Inf" || input == "-.INF";
	}
	
	inline bool IsNaN(const std::string& input) {
		return input == ".nan" || input == ".NaN" || input == ".NAN";
	}


	template <typename T> 
	inline bool Colwert(const std::string& input, T& output, typename enable_if<is_numeric<T> >::type * = 0) {
		std::stringstream stream(input);
		stream.unsetf(std::ios::dec);
        if((stream >> output) && (stream >> std::ws).eof())
            return true;
		
		if(std::numeric_limits<T>::has_infinity) {
			if(IsInfinity(input)) {
				output = std::numeric_limits<T>::infinity();
				return true;
			} else if(IsNegativeInfinity(input)) {
#ifdef _WINDOWS
#pragma warning(disable: 4146)
#endif
				output = -std::numeric_limits<T>::infinity();
#ifdef _WINDOWS
#pragma warning(default: 4146)
#endif
				return true;
			}
		}
		
		if(std::numeric_limits<T>::has_quiet_NaN && IsNaN(input)) {
			output = std::numeric_limits<T>::quiet_NaN();
			return true;
		}
		
		return false;
	}
}

#endif // COLWERSION_H_62B23520_7C8E_11DE_8A39_0800200C9A66
