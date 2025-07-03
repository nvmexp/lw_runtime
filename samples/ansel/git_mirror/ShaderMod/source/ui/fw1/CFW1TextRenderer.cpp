// CFW1TextRenderer.cpp

#include "FW1Precompiled.h"

#include "CFW1TextRenderer.h"

#define SAFE_RELEASE(pObject) { if(pObject) { (pObject)->Release(); (pObject) = NULL; } }


namespace FW1FontWrapper {


// Construct
CFW1TextRenderer::CFW1TextRenderer() :
    m_pGlyphProvider(NULL),
    
    m_lwrrentFlags(0),
    m_lwrrentColor(0xff000000),
    
    m_cachedGlyphMap(0),
    m_pCachedGlyphMapFontFace(NULL),
    m_cachedGlyphMapFontSize(0),
    
    m_pDWriteTextRendererProxy(0)
{
}


// Destruct
CFW1TextRenderer::~CFW1TextRenderer() {
    SAFE_RELEASE(m_pGlyphProvider);
    
    delete m_pDWriteTextRendererProxy;
}


// Init
HRESULT CFW1TextRenderer::initTextRenderer(
    IFW1Factory *pFW1Factory,
    IFW1GlyphProvider *pGlyphProvider
) {
    HRESULT hResult = initBaseObject(pFW1Factory);
    if(FAILED(hResult))
        return hResult;
    
    if(pGlyphProvider == NULL)
        return E_ILWALIDARG;
    
    pGlyphProvider->AddRef();
    m_pGlyphProvider = pGlyphProvider;
    
    m_pDWriteTextRendererProxy = new CDWriteTextRendererProxy(this);
    
    return S_OK;
}

}// namespace FW1FontWrapper
