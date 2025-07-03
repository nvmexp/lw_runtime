#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include <algorithm>
#include <string>

namespace dbg
{

//-----------------------------------------------------------------------------
// Compare count elements of buffer to reference. Prints (optional) msg and 
// for up to maxDiff differences. Returns true if buffer and reference are 
// the same.
template<typename T>
bool compare( const T* buffer, const T* reference, size_t count, const std::string& msg = "", int maxDiffs=1 )
{
  int numDiffs = 0;
  bool ret = true;
  for( size_t i=0; i < count; i++ )
  {
    if( buffer[i] != reference[i] )
    {
      if( !msg.empty() )
        std::cerr << msg << ": ";
      std::cerr << "Difference at index " << i << ": " << buffer[i] << "(ref:" << reference[i] << ")" << std::endl;
      ret = false;
      if( ++numDiffs >= maxDiffs && maxDiffs > 0 )
        break;
    }
  }
  return ret;
}

//-----------------------------------------------------------------------------
// Byte-level comparisons for count elements of buffer and reference
template<typename T>
bool compareBytes( const T* buffer, const T* reference, size_t count, const std::string& msg = "", int maxDiffs=1 )
{
  return compare( (char*)buffer, (char*)reference, count*sizeof(T), msg, maxDiffs );
}

//-----------------------------------------------------------------------------
// Byte-level comparisons between buffer and reference vectors
template<typename T>
bool compareBytes( const std::vector<T>& buffer, const std::vector<T>& reference, const std::string& msg = "", int maxDiffs=1 )
{
  if( buffer.size() != reference.size() )
  {
    std::cerr << msg << ": sizes do not match\n";
    return false;
  }

  if( buffer.size() > 0 )
    return compareBytes( (char*)buffer.data(), (char*)reference.data(), buffer.size()*sizeof(T), msg, maxDiffs );
  else
    return true;
}

//------------------------------------------------------------------------------
// Dump a buffer to a file. Records the size of the buffer in the first size_t.
inline void dump( FILE* file, const void* buffer, size_t size )
{
  fwrite( &size, sizeof(size), 1, file );
  fwrite( buffer, 1, size, file );
  fflush( file );
}

//-----------------------------------------------------------------------------
// Dump a buffer to a files with the given filename. Records the size of the buffer in the first size_t.
inline void dump( const std::string& filename, const void* buffer, size_t size )
{
  FILE* file = NULL;
  if( (file = fopen( filename.c_str(), "wb" )) == NULL )
  {
    std::cerr << "Could not open file " << filename << std::endl;
    return;
  }

  dump(file, buffer, size);

  fclose(file);
}

//------------------------------------------------------------------------------
// Load a buffer from a file. 
inline void load( FILE* file, void* buffer, size_t maxSize )
{
  size_t size = 0;
  if( fread( &size, sizeof(size), 1, file ) != 1 )
  {
    std::cerr << "Could not read size" << std::endl;
    return;
  }

  if( size > maxSize )
  {
    std::cerr << "Size exceeds buffer size" << std::endl;
    return;
  }

  if( fread( buffer, 1, size, file ) != size )
  {
    std::cerr << "Could not read buffer" << std::endl;
  }
}

//-----------------------------------------------------------------------------
// Load a buffer from a file with a given filename
inline void load( const std::string& filename, void* buffer, size_t maxSize )
{
  FILE* file = NULL;
  if( (file = fopen( filename.c_str(), "rb" )) == NULL )
  {
    std::cerr << "Could not open file " << filename << std::endl;
    return;
  }

  load(file, buffer, maxSize);

  fclose(file);
}


//------------------------------------------------------------------------------
// Load a buffer vector from a file.
inline void load( FILE* file, std::vector<char> &buffer )
{
  buffer.resize(0);
  
  size_t size = 0;
  if( fread( &size, sizeof(size), 1, file ) != 1 )
  {
    std::cerr << "Could not read size" << std::endl;
    return;
  }

  buffer.resize( size );
  if( size && fread( buffer.data(), 1, size, file ) != size )
  {
    std::cerr << "Could not read buffer" << std::endl;
  }
}

//-----------------------------------------------------------------------------
// Load a buffer vector from a file with a given filename.
inline void load( const std::string& filename, std::vector<char>& buffer )
{
  FILE* file = NULL;
  if( (file = fopen( filename.c_str(), "rb" )) == NULL )
  {
    std::cerr << "Could not open file " << filename << std::endl;
    return;
  }

  load(file, buffer);

  fclose(file);
}

///////////////////////////////////////////////////////////////////////////////
//
// compLog()
//
// The following functions are useful for comparison between two programs that
// should produce the same data but do not. compLog() can provide data checkpoints
// at different parts of the algorithm. Here is a typical usage scenario:
//
//   dbg::openGlobalLog("c:/temp/compLog.bin", dbg::OP_WRITE);
//   ...
//   dbg::compLog("buffer1", buffer1, size1);
//   dbg::compLog("foo-before-bar", foo, fooSize );
//   bar();
//   dbg::compLog("foo-after-bar", foo, fooSize);
//   ...
//   dbg::closeGlobalLog();
//
// A global log file is opened in OP_WRITE mode. Each call to compLog() writes the
// specified buffers to the log. The second program would specify the same calls
// but would call openGlobalLog() with OP_COMPARE. Then each call to compLog()
// compares the values in the buffer to what is in the log, report differences
// in size and value. This can help track down where differences are arising.
// If the sizes of the buffers match but the contents do not, it can sometimes
// be useful to open the log in OP_READ mode. This replaces the contents of the 
// buffer with those in the log. This will verify that the program will work
// if the correct buffer values can be computed. To be able to operate on 
// the reference values in the log, compLog() can take a buffer into which 
// the values will be written. This makes it easier to do more detailed 
// comparisons, say in a debugger.
//
// To make it easy to change the compLog mode, you can set an environment
// variable DBG_COMP_OP, and use elwCompOp() to retrieve it.
//
// If compLog() is used without first calling openGlobalLog(), the buffer is 
// associated with a file based on the name passed to compLog(), i.e. 
// "c:/temp/compLog-<counter>-<name>.bin".
//
// Specify DEVICE for the location parameter to compLog() for using LWCA 
// buffers.
//
///////////////////////////////////////////////////////////////////////////////


//------------------------------------------------------------------------------
enum Location
{
  HOST=0,
  DEVICE,
};

//------------------------------------------------------------------------------
enum CompOp
{
  OP_IGNORE=0,      // Do nothing. Use this to turn compLog() calls into no-ops
  OP_READ,          // Replace the data in buffer with data from the log
  OP_WRITE,         // Write data to log
  OP_COMPARE,       // Compare data in buffer with data in log

  OP_USE_GLOBAL     // Use the global compOp
};

//------------------------------------------------------------------------------
// Opens the given file for logging with the given comparison mode. The mode
// can be retrieved later with globalCompOp().
void openGlobalLog( const char* path, CompOp op );

//------------------------------------------------------------------------------
void closeGlobalLog();

//------------------------------------------------------------------------------
// Return the global compOp
CompOp globalCompOp();

//------------------------------------------------------------------------------
void setGlobalCompOp( CompOp op );

//------------------------------------------------------------------------------
// Returns the comparison operator set from the environment variable DBG_COMP_OP 
CompOp elwCompOp();

//------------------------------------------------------------------------------
void compLogCpu( const std::string& name, const void* buffer, size_t size, void* outRef = nullptr, CompOp op = OP_USE_GLOBAL );

//------------------------------------------------------------------------------
void compLogDev( const std::string& name, const void* buffer, size_t size, void* outRef = nullptr, CompOp op = OP_USE_GLOBAL );

//------------------------------------------------------------------------------
inline void compLog( const std::string& name, const void* buffer, size_t size, int location = HOST, void* outRef = nullptr, CompOp op = OP_USE_GLOBAL )
{
  if( location == DEVICE )
    compLogDev( name, (void*)buffer, size, outRef, op );
  else
    compLogCpu( name, (void*)buffer, size, outRef, op );
}

//------------------------------------------------------------------------------
// compLog for a buffer of count elements of type T with optional output of reference
// data in outRef vector. outRef is automatically  resized to the specified count.
template<typename T>
inline void compLogT( const std::string& name, const T* buffer, size_t count=1, int location = HOST, std::vector<T>* outRef = nullptr, CompOp op = OP_USE_GLOBAL )
{
  T* outRefPtr = 0;
  if( outRef )
  {
    outRef->resize(count);
    if( count > 0 )
      outRefPtr = outRef->data();
  }
  compLog( name, (void*)buffer, count*sizeof(T),  location, outRefPtr, op );
}

//------------------------------------------------------------------------------
// compLog for a vector. 
template<typename T>
inline void compLogT( const std::string& name, const std::vector<T>& buffer, std::vector<T>* outRef = nullptr, CompOp op = OP_USE_GLOBAL )
{
  const T* bufferPtr = 0;
  if( buffer.size() > 0 )
    bufferPtr = buffer.data();
  compLog( name, bufferPtr, buffer.size(), outRef, HOST, op );
}

//------------------------------------------------------------------------------
} // namespace dbg
