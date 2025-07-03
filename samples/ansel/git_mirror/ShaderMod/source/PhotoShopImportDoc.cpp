#include "PhotoShopImportDoc.h"
#include "Psd/PsdDolwment.h"
#include "Psd/PsdParseImageResourcesSection.h"
#include "Psd/PsdParseLayerMaskSection.h"
#include "Psd/PsdParseImageDataSection.h"
#include "Psd/PsdParseDolwment.h"
#include "Psd/PsdColorMode.h"
#include "Log.h"

PhotoShopImportDoc::~PhotoShopImportDoc()
{
    Destroy();
}

bool PhotoShopImportDoc::Open(std::wstring &filename)
{
    // Try opening the file, return true on success
    if (!m_file.OpenRead(filename.c_str()))
    {
        LOG_ERROR("Cannot find file \"%s\" for PSD import.\n", filename.c_str());
        return false;
    }

    m_dolwment = CreateDolwment(&m_file, &m_allocator);
    if (!m_dolwment)
    {
        LOG_ERROR("Could not create document for PSD import.\n");
        m_file.Close();
        return false;
    }

    // Psd must be RGB (for now)
    if (m_dolwment->colorMode != psd::colorMode::RGB)
    {
        OutputDebugStringA("Document is not in RGB color mode.\n");
        DestroyDolwment(m_dolwment, &m_allocator);
        m_file.Close();
        return false;
    }

    return true;
}

void PhotoShopImportDoc::Read()
{
    // Acquire Image Resources Section
    m_imageResourcesSection = ParseImageResourcesSection(m_dolwment, &m_file, &m_allocator);
    if (!m_imageResourcesSection)
    {
        LOG_ERROR("Could not acquire Image Resource Section of psd file.");
    }

    // Acquire Layer Mask Section
    m_layerMaskSection = ParseLayerMaskSection(m_dolwment, &m_file, &m_allocator);
    if (!m_layerMaskSection)
    {
        LOG_ERROR("Could not acquire Layer Mask Section of psd file.");
    }

    // Acquire Image Data Section
    if (m_dolwment->imageDataSection.length != 0)
    {
        m_imageDataSection = ParseImageDataSection(m_dolwment, &m_file, &m_allocator);
        if (!m_imageDataSection)
        {
            LOG_ERROR("Could not acquire Image Data Section of psd file.");
        }
    }
    else
    {
        LOG_WARN("Selected psd file does not have an Image Data Section!");
    }
}

void PhotoShopImportDoc::Destroy()
{
    if (m_imageResourcesSection)
    {
        DestroyImageResourcesSection(m_imageResourcesSection, &m_allocator);
    }

    if (m_layerMaskSection)
    {
        DestroyLayerMaskSection(m_layerMaskSection, &m_allocator);
    }

    if (m_imageDataSection)
    {
        DestroyImageDataSection(m_imageDataSection, &m_allocator);
    }

    DestroyDolwment(m_dolwment, &m_allocator);
    m_file.Close();
}

psd::ImageResourcesSection* PhotoShopImportDoc::GetImageResourcesSection() const
{
    return m_imageResourcesSection;
}

psd::LayerMaskSection* PhotoShopImportDoc::GetLayerMaskSection() const
{
    return m_layerMaskSection;
}

psd::ImageDataSection* PhotoShopImportDoc::GetImageDataSection() const
{
    return m_imageDataSection;
}

psd::Document* PhotoShopImportDoc::GetDolwment() const
{
    return m_dolwment;
}

psd::MallocAllocator PhotoShopImportDoc::GetMallocAllocator() const
{
    return m_allocator;
}

psd::NativeFile PhotoShopImportDoc::GetNativeFile() const
{
    return m_file;
}
