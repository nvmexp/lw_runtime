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
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////
// Class name: D3DClass
////////////////////////////////////////////////////////////////////////////////
class D3DClass
{
public:
    D3DClass();
    D3DClass(const D3DClass&);
    ~D3DClass();

    bool Initialize(int, int, bool, HWND, bool, float, float, DWORD, DWORD dxgiFormat = 28, bool bWindowedFullScreenTransition = false, bool bDXTLAutomationTesting = false);
	void Shutdown();
	void ShutdownD3DDeviceResources();
	void ShutdownSwapChain();
    void BeginScene();
    void EndScene();
	IDXGISwapChain* getDXGISwapChain() {return m_swapChain;}
    ID3D10Device* GetDevice();

    void GetProjectionMatrix(D3DXMATRIX&);
    void GetWorldMatrix(D3DXMATRIX&);
    void GetOrthoMatrix(D3DXMATRIX&);
	bool CreateSwapChain(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
                          float screenDepth, float screenNear, DWORD presentModel, DWORD dxgiFormat);
	bool InitializeD3D(int, int, float, float);
private:
    bool m_vsync_enabled;
	bool m_bWindowedFullScreenTransition;
	bool m_FullScreen;
	bool m_bDXTLAutomationTesting;
	DWORD m_StartTime;
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
	FILE* m_fpError;
};

#endif