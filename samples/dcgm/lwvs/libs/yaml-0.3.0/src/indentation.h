#ifndef INDENTATION_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define INDENTATION_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) || (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || (__GNUC__ >= 4)) // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif


#include "yaml-cpp/ostream.h"
#include <iostream>

namespace YAML
{
	struct Indentation {
		Indentation(unsigned n_): n(n_) {}
		unsigned n;
	};
	
	inline ostream& operator << (ostream& out, const Indentation& indent) {
		for(unsigned i=0;i<indent.n;i++)
			out << ' ';
		return out;
	}

	struct IndentTo {
		IndentTo(unsigned n_): n(n_) {}
		unsigned n;
	};
	
	inline ostream& operator << (ostream& out, const IndentTo& indent) {
		while(out.col() < indent.n)
			out << ' ';
		return out;
	}
}


#endif // INDENTATION_H_62B23520_7C8E_11DE_8A39_0800200C9A66
