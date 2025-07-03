#pragma once

#include "zlib.h"

int deflateInit_(z_stream *strm, int level, const char *version, int stream_size) { return 0; }
int inflateInit_(z_stream *strm, const char *version, int stream_size) { return 0; }
int deflateInit2_(z_stream *strm, int  level, int  method, int windowBits, int memLevel, int strategy, const char *version, int stream_size) { return 0; }

int deflate(z_stream *strm, int flush) { return 0; }
int deflateEnd(z_stream *strm) { return 0; }

uint32_t adler32_combine(uint32_t, uint32_t, z_off_t) { return 0; }
