#include "DbgComp.h"

#include <corelib/misc/String.h>

#include <lwda_runtime_api.h>

#include <vector>
#include <string>
#include <stdlib.h>

using namespace dbg;

namespace dbg
{
FILE* g_logFile = NULL;
int g_fileCounter=0;
std::string g_logFileAccess;
std::string g_logFilePath;
CompOp g_compOp = elwCompOp();


//------------------------------------------------------------------------------
std::string getFilenameForObject( const std::string& name )
{
  return corelib::stringf("c:/temp/complog_%04d_%s.bin", g_fileCounter++, name.c_str() );
}

//------------------------------------------------------------------------------
void getFile( const std::string& name, CompOp op, FILE** file, bool* close )
{
  const char* access = (op == OP_WRITE) ? "wb" : "rb";
  *file = NULL;
  *close = false;

  if( g_logFile )
  {
    if( access != g_logFileAccess )
      std::cerr << "Incompatible access to global log file" << std::endl;
    else
      *file = g_logFile;
  }
  else
  {
    *close = true;
    const std::string filename = getFilenameForObject( name );
    if( (*file = fopen( filename.c_str(), access )) == NULL )
      std::cerr << "Could not open file " << filename << std::endl;
  }
}

//------------------------------------------------------------------------------
void openGlobalLog( const std::string& path, CompOp op )
{
  setGlobalCompOp( op );
  if( op == OP_IGNORE )
    return;

  const char* access = (op == OP_WRITE) ? "wb" : "rb";
  
  if( g_logFile != NULL && g_logFilePath == path && g_logFileAccess == access )
    return;

  closeGlobalLog();
  
  if( (g_logFile = fopen( path.c_str(), access )) == NULL )
  {
    std::cerr << "Could not open file " << path << std::endl;  
    return;
  }
  g_logFilePath = path;
  g_logFileAccess = access;
}

//------------------------------------------------------------------------------
void closeGlobalLog()
{
  if( g_logFile )
  {
    fclose( g_logFile );
    g_logFile = NULL;
    g_logFilePath = "";
  }
}

//------------------------------------------------------------------------------
void setGlobalCompOp(CompOp op)
{
  g_compOp = op;
}

//------------------------------------------------------------------------------
CompOp elwCompOp()
{
  if( getelw("DBG_COMP_OP") )
  {
    std::string opStr = getelw("DBG_COMP_OP");
    if( opStr == "WRITE" )
      return OP_WRITE;
    else if( opStr == "COMPARE" )
      return OP_COMPARE;
    else if( opStr == "READ" )
      return OP_READ;
    else
      return OP_IGNORE;
  }
  else 
    return OP_IGNORE;
}

//------------------------------------------------------------------------------
void compLogDev( const std::string& name, const void* buffer, size_t size, void* outRef, CompOp op )
{
  if( op == OP_USE_GLOBAL )
    op = g_compOp;

  if( op == OP_IGNORE )
    return;

  // download the buffer
  std::vector<char> hostTemp(size);
  lwdaError_t err = lwdaMemcpy( hostTemp.data(), buffer, size, lwdaMemcpyDeviceToHost );
  if( err != lwdaSuccess )
    std::cerr << "Error copying from device\n";
  compLogCpu( name, (void*)hostTemp.data(), size, outRef, op );
  if( op == OP_READ && size > 0 )
  {
    err = lwdaMemcpy( (void*)buffer, hostTemp.data(), size, lwdaMemcpyHostToDevice );
    if( err != lwdaSuccess )
      std::cerr << "Error copying to device\n";
  }
}

//------------------------------------------------------------------------------
void compLogCpu( const std::string& name, const void* buffer, size_t size, void* outRef, CompOp op )
{
  if( op == OP_USE_GLOBAL )
    op = g_compOp;

  if( op == OP_IGNORE )
    return;

  FILE* file = NULL;
  bool close;
  getFile( name, op, &file, &close );
  if( file == NULL )
    return;

  if( op == OP_WRITE )
  {
    dump( file, buffer, size );
    if( outRef && size > 0 )
      memcpy( outRef, buffer, size );
  }
  else 
  {
    // load the object
    std::vector<char> ref;
    load( file, ref );

    size_t validSize = std::min(size,ref.size());
    if( outRef && ref.size() > 0 )
      memcpy( outRef, ref.data(), validSize );

    if( op == OP_COMPARE || op == OP_READ )
    {
      if( ref.size() != size )
        std::cerr << name << ": Size differs " << size << "(" << ref.size() << ")" << std::endl;
      if( validSize > 0 )
      {
        if( op == OP_COMPARE )
          compare( (char*)buffer, ref.data(), validSize, name );
        else if( op == OP_READ )
          memcpy( (void*)buffer, ref.data(), validSize );
      }
    }
  }

  if( close )
    fclose( file );
}

} // namespace dbg
