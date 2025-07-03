#pragma once

// The unit test uses this class to isolate the public header files from internal header files. Including
// optix_ptx_encryption.h (which includes optixu/optixpp_namespace.h) and Context.h at the same time does not work.

typedef struct RTcontext_api* RTcontext;

#include <memory>
#include <string>

class EncryptionHelperImpl;

class EncryptionHelper
{
  public:
    EncryptionHelper( RTcontext context, const char* publicVendorKey, const char* secretVendorKey );

    EncryptionHelper( RTcontext context, const char* publicVendorKey, const char* secretVendorKey, const char* optixSalt, const char* vendorSalt );

    ~EncryptionHelper();

    std::string encrypt( const std::string& input ) const;

  private:
    std::shared_ptr<EncryptionHelperImpl> m_pimpl;
};
