#pragma once

// PSD
#include "Psd/Psd.h"
#include "Psd/PsdMallocAllocator.h"
#include "Psd/PsdNativeFile.h"

#include <string>

namespace psd
{
    struct Document;
    struct ImageResourcesSection;
    struct LayerMaskSection;
    struct ImageDataSection;
}

class PhotoShopImportDoc {
public:
    ~PhotoShopImportDoc();
    bool Open(std::wstring &filename);
    void Read();

    psd::LayerMaskSection* GetLayerMaskSection() const;
    psd::Document* GetDolwment() const;
    psd::MallocAllocator GetMallocAllocator() const;
    psd::NativeFile GetNativeFile() const;

private:
    psd::Document *m_dolwment = nullptr;
    psd::ImageResourcesSection *m_imageResourcesSection = nullptr;
    psd::LayerMaskSection *m_layerMaskSection = nullptr;
    psd::ImageDataSection *m_imageDataSection = nullptr;
    psd::MallocAllocator m_allocator;
    psd::NativeFile m_file = psd::NativeFile(&m_allocator);

    void Destroy();
    psd::ImageResourcesSection* GetImageResourcesSection() const;
    psd::ImageDataSection* GetImageDataSection() const;
};
