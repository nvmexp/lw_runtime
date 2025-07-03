////////////////////////////////////////////////////////////////////////////////
// Filename: d3dclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _D3DCLASS_H_
#define _D3DCLASS_H_


/////////////
// LINKING //
/////////////
#pragma comment(lib, "d3d10_1.lib")
#pragma comment(lib, "d3dx10.lib")
#pragma comment(lib, "dxgi.lib")


//////////////
// INCLUDES //
//////////////
#include <d3d10_1.h>
//#include <d3d10.h>
#include <d3dx10.h>
#include <D3DX10Tex.h>

#include "lwapi.h"

////////////////////////////////////////////////////////////////////////////////
// Class name: D3DClass
////////////////////////////////////////////////////////////////////////////////
class D3DClass
{
public:
    D3DClass();
    D3DClass(const D3DClass&);
    ~D3DClass();

    bool Initialize(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
                          float screenDepth, float screenNear, DWORD presentModel);
    void Shutdown();
    ID3D10ShaderResourceView** getTextureSRVArray();
    ID3D10Texture2D** getTextureArray();
    
    void TexelCopy(unsigned int source,unsigned int x, unsigned int y);
    bool VerifyTextureCopy();
    bool ColwertTexturesToFiles();
    void BeginScene();
    void EndScene();

    ID3D10Device* GetDevice();

    void GetProjectionMatrix(D3DXMATRIX&);
    void GetWorldMatrix(D3DXMATRIX&);
    void GetOrthoMatrix(D3DXMATRIX&);

private:
    bool m_vsync_enabled;
    DWORD presentModel; 
    ID3D10Device1* m_device;
    IDXGISwapChain* m_swapChain;
    ID3D10RenderTargetView* m_renderTargetView;
    ID3D10Texture2D* m_depthStencilBuffer;
    ID3D10DepthStencilState* m_depthStencilState;
    ID3D10DepthStencilView* m_depthStencilView;
    ID3D10RasterizerState* m_rasterState;
    D3DXMATRIX m_projectionMatrix;
    D3DXMATRIX m_worldMatrix;
    D3DXMATRIX m_orthoMatrix;
    ID3D10ShaderResourceView* m_textureSRV[3];
    ID3D10Texture2D* m_textures[3];
    D3D10_TEXTURE2D_DESC m_desc;
};

#endif