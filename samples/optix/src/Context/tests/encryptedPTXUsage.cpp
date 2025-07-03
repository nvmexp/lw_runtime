#include "encryptedPTXUsage.h"

#include <optix_ptx_encryption.h>

class EncryptionHelperImpl
{
  public:
    EncryptionHelperImpl( RTcontext context, const char* publicVendorKey, const char* secretVendorKey )
        : m_ptxEncryption( context, publicVendorKey, secretVendorKey )
    {
    }

    EncryptionHelperImpl( RTcontext context, const char* publicVendorKey, const char* secretVendorKey, const char* optixSalt, const char* vendorSalt )
        : m_ptxEncryption( context, publicVendorKey, secretVendorKey, optixSalt, vendorSalt )
    {
    }

    std::string encrypt( const std::string& input ) const { return m_ptxEncryption.encrypt( input ); }

    optix::PtxEncryption m_ptxEncryption;
};


EncryptionHelper::EncryptionHelper( RTcontext context, const char* publicVendorKey, const char* secretVendorKey )
    : m_pimpl( new EncryptionHelperImpl( context, publicVendorKey, secretVendorKey ) )
{
}

EncryptionHelper::EncryptionHelper( RTcontext   context,
                                    const char* publicVendorKey,
                                    const char* secretVendorKey,
                                    const char* optixSalt,
                                    const char* vendorSalt )
    : m_pimpl( new EncryptionHelperImpl( context, publicVendorKey, secretVendorKey, optixSalt, vendorSalt ) )
{
}

EncryptionHelper::~EncryptionHelper()
{
}

std::string EncryptionHelper::encrypt( const std::string& input ) const
{
    return m_pimpl->encrypt( input );
}
