#pragma once

#include <vector>
#include <lwda_runtime_api.h>

namespace dbg
{

template<typename T>
class LwdaBuffer
{
public:
  LwdaBuffer( const void* devPtr, size_t count=1 )
  {
    set( (const T*)devPtr, count );
  }

  void set( const T* devPtr, size_t count=1 )
  {
    // download the data to the host
    lwdaError_t err = lwdaDeviceSynchronize();
    if( err != lwdaSuccess )
      printf("LwdaBuffer(): Error before intialization: %s\n", lwdaGetErrorString(err));

    if( devPtr == NULL )
      printf("LwdaBuffer(): devPtr is NULL\n");

    m_data.resize(count);
    if( count > 0 && devPtr != NULL )
    {
      err = lwdaMemcpy(m_data.data(), devPtr, count*sizeof(T), lwdaMemcpyDeviceToHost );
      if( err != lwdaSuccess )
        printf("LwdaBuffer(): Error copying: %s\n", lwdaGetErrorString(err));
    }
  }

  size_t sizeInBytes() { return m_data.size() * sizeof(T); }

  T& operator[]( size_t idx )            { return m_data[idx]; }
  const T& operator[](size_t idx ) const { return m_data[idx]; }

  std::vector<T> m_data;
};

} // namespace dbg