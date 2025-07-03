#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#include <string.h>
#include <stdio.h>
#include "lwapi.h"

#pragma comment(linker,"\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
HWND                            g_hWnd                  = NULL;
LwPhysicalGpuHandle				g_hPhysicalGpu = NULL;
LwPhysicalGpuHandle *			g_hPhysicalGpuArrayPtr = NULL;
LwU32							g_gpuCount = 0;

// Direct Write related variable
IDWriteFactory*                 g_pDWriteFactory        = NULL;
IDWriteTextFormat*              g_pTextFormat           = NULL;
ID2D1Factory*                   g_pD2DFactory           = NULL;
ID2D1HwndRenderTarget*          g_pRT                   = NULL;
ID2D1SolidColorBrush*           g_pBlackBrush           = NULL;

LRESULT WINAPI MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );

//-----------------------------------------------------------------------------
// Formatting defines
//-----------------------------------------------------------------------------
#define							WINDOW_WIDTH			640
#define							WINDOW_HEIGHT			560
#define                         FONT_TYPE               (L"Courier")
#define                         FONT_SIZE               (14.0f)
#define                         X_OFFSET                20      // X Offset for Texts
#define                         Y_OFFSET                50      // Y Offset for First element
#define                         FIELD_WIDTH             240     // Each UI element width
#define                         FIELD_HEIGHT            70      // Each UI element height
#define                         EDITBOX_WIDTH           300     // Each UI element width
#define                         EDITBOX_HEIGHT          25      // Each UI element height
#define                         F2F_X_OFFSET            25      // Field to Field Y offset
#define                         F2F_Y_OFFSET            70      // Field to Field Y offset

#define                         MAX_TEXT_LENGTH         (64)  // Adding one more character for the NULL terminator
#define                         SIGNED_BUFFER_SIZE_1K   (128)
#define                         SIGNED_BUFFER_SIZE_2K   (256)
#define                         OUTPUT_BUFFER_SIZE      1024

//------------------------------------------------------------------------------
// Control Identifiers
//------------------------------------------------------------------------------
#define                         IDC_BUTTONBASE          100
#define                         IDC_BUTTON_GENKEY       101
#define                         IDC_BUTTON_CLOSE        102
#define                         IDC_BUTTON_BROWSE       103

#define                         IDC_EDITBOXBASE         200
#define                         IDC_EDITBOX_NONCE       201
#define                         IDC_EDITBOX_PROGID      202
#define                         IDC_EDITBOX_SESSIONID   203

#define                         IDC_COMBOBOXBASE        300
#define                         IDC_COMBOBOX_ALGO       304
#define                         IDC_COMBOBOX_GPU        305

// SafeRelease inline function.
template <class T> inline void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}

typedef enum
{
    FIELD_EDIT = 0,
    FIELD_BUTTON,
    FIELD_COMBOBOX,
} FIELD_TYPE;


typedef struct _DialogElements
{
    wchar_t     *pFieldName;
    FIELD_TYPE  enFieldtype;
}DialogElements;

const DialogElements TextFields[] = {
    { L"Random Number 16 Bytes:", FIELD_EDIT },
    { L"Program ID 4 Bytes:", FIELD_EDIT },
    { L"Session ID 4 Bytes:", FIELD_EDIT },
    { L"Signing Algo:", FIELD_COMBOBOX },
	{ L"GPU:", FIELD_COMBOBOX },
    { L"Encrypted Private Key Binary File:", FIELD_BUTTON },	
};

const int NUMFIELDS = sizeof(TextFields)/sizeof(DialogElements);

const int NUMEDITBOX = 3;
const int NUMCOMBOBOX = 2;

const UINT32 EditBoxSize[NUMEDITBOX] = { 32, 8, 8};

const wchar_t* ButtonNames[] = {
    L"Generate Key",
    L"Close",
};

const wchar_t* ErrorMessages[] = {
    L"Below are the list of input errors: \n\n\n",
    L"   *Random Nonce can't be empty or more than 16 Bytes.\n",
    L"   *Random Nonce can only accept Hexadecimal values.\n",
    L"   *Program ID can't be empty or more than 4 Bytes.\n",
    L"   *Program ID can only accept Hexadecimal values.\n",
    L"   *Session Id can't be empty or more than 4 Bytes.\n",
    L"   *Session Id can only accept Hexadecimal values.\n",
    L"   *Select a Signing Algo.\n",
    L"",
	L"   *Select a GPU.\n",
	L"",
    L"   *Private Key buffer is not valid.\n",
    L"   *Private Key buffer size cannot be zero.\n",
};

wchar_t ErrorBuffer[1024];
wchar_t DisplayBuffer[4096];

unsigned char* pszPrivateKey;

unsigned int g_ErrorMask = 0x0;

const int NUMBUTTONS = sizeof(ButtonNames)/sizeof(wchar_t*);

// Functions in POPFILE.C
void PopFileInitialize (HWND) ;
BOOL PopFileOpenDlg (HWND, PTSTR, PTSTR) ;
BOOL PopFileRead (HWND, PTSTR, unsigned char**, UINT32*);

//-----------------------------------------------------------------------------------------------
// LWAPI structures
//-----------------------------------------------------------------------------------------------
LW_SIGN_GPUID_INPUT stInput = {0};
LW_ENCRYPTED_KEY_INFO stEncryptKey = {0};

HRESULT CreateDeviceIndependentResources()
{
    HRESULT hr;

    // Create Direct2D factory.
    hr = D2D1CreateFactory(
        D2D1_FACTORY_TYPE_SINGLE_THREADED,
        &g_pD2DFactory
        );

    // Create a shared DirectWrite factory.
    if (SUCCEEDED(hr))
    {
        hr = DWriteCreateFactory(
            DWRITE_FACTORY_TYPE_SHARED,
            __uuidof(IDWriteFactory),
            reinterpret_cast<IUnknown**>(&g_pDWriteFactory)
            );
    }

    // Create a text format using Gabriola with a font size of 72.
    // This sets the default font, weight, stretch, style, and locale.
    if (SUCCEEDED(hr))
    {
        hr = g_pDWriteFactory->CreateTextFormat(
            FONT_TYPE,                  // Font family name.
            NULL,                       // Font collection (NULL sets it to use the system font collection).
            DWRITE_FONT_WEIGHT_DEMI_BOLD,
            DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL,
            FONT_SIZE,
            L"en-us",
            &g_pTextFormat
            );
    }

    // Center align (horizontally) the text.
    if (SUCCEEDED(hr))
    {
        hr = g_pTextFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING);
    }

    if (SUCCEEDED(hr))
    {
        hr = g_pTextFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    }

    return hr;
}

HRESULT CreateDeviceResources()
{
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect(g_hWnd, &rc);

    D2D1_SIZE_U size = D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top);

    if (!g_pRT)
    {
        // Create a Direct2D render target.
        hr = g_pD2DFactory->CreateHwndRenderTarget(
                D2D1::RenderTargetProperties(),
                D2D1::HwndRenderTargetProperties(
                    g_hWnd,
                    size
                    ),
                &g_pRT
                );

        // Create a black brush.
        if (SUCCEEDED(hr))
        {
            hr = g_pRT->CreateSolidColorBrush(
                D2D1::ColorF(D2D1::ColorF::Black),
                &g_pBlackBrush
                );
        }
    }

    return hr;
}

void DiscardDeviceResources()
{
    SafeRelease(&g_pRT);
    SafeRelease(&g_pBlackBrush);
}

//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HINSTANCE hinstance)
{
    HRESULT hr = S_OK;
    ATOM atom;
    WNDCLASSEX wcx;
    static TCHAR szAppName[] = TEXT("RSASign");

    // Call CoInitialize first
    if (FAILED(CoInitialize(NULL)))
    {
        return E_FAIL;
    }
 
    // Fill in the window class structure with parameters 
    // that describe the main window. 
 
    wcx.cbSize = sizeof(wcx);          // size of structure 
    wcx.style = CS_HREDRAW | 
        CS_VREDRAW;                    // redraw if size changes 
    wcx.lpfnWndProc = MsgProc;         // points to window procedure 
    wcx.cbClsExtra = 0;                // no extra class memory 
    wcx.cbWndExtra = 0;                // no extra window memory 
    wcx.hInstance = hinstance;         // handle to instance 
    wcx.hIcon = LoadIcon(NULL, 
        IDI_APPLICATION);              // predefined app. icon 
    wcx.hLwrsor = LoadLwrsor(NULL, 
        IDC_ARROW);                    // predefined arrow 
    wcx.hbrBackground = (HBRUSH)
        GetStockObject(WHITE_BRUSH);   // white background brush 
    wcx.lpszMenuName =  NULL;          // name of menu resource 
    wcx.lpszClassName = szAppName;     // name of window class 
    wcx.hIconSm = NULL;                // small class icon
 
    // Register the window class. 

    // Register the window class
    atom = RegisterClassEx(&wcx);

    hr = atom ? S_OK : E_FAIL;

    if(SUCCEEDED(hr))
    {
        // Create the application's window
        g_hWnd = CreateWindow(
                szAppName,
                TEXT("RSA Signing Unit Test"),
                WS_OVERLAPPED | WS_SYSMENU | WS_MINIMIZEBOX, 
                CW_USEDEFAULT, 
                CW_USEDEFAULT,
				WINDOW_WIDTH,
				WINDOW_HEIGHT,
                NULL, 
                NULL,
                hinstance,
                NULL);
    }

    if (SUCCEEDED(hr))
    {
        hr = g_hWnd ? S_OK : E_FAIL;
    }
    
    if (SUCCEEDED(hr))
    {
        hr = CreateDeviceIndependentResources();
    }

    if (SUCCEEDED(hr))
    {
        // Show the window
        ShowWindow(g_hWnd,SW_SHOWNORMAL);
        UpdateWindow(g_hWnd);
    }

    return S_OK;
}

HRESULT DrawHwndText(const wchar_t* wszText, UINT32 cTextLength, RECT& rc)
{
    // Create a D2D rect that is the same size as the window.
    D2D1_RECT_F layoutRect = D2D1::RectF(
        static_cast<FLOAT>(rc.left) / 1.0f,
        static_cast<FLOAT>(rc.top) / 1.0f,
        static_cast<FLOAT>(rc.right - rc.left) / 1.0f,
        static_cast<FLOAT>(rc.bottom - rc.top) / 1.0f
        );

    // Use the DrawText method of the D2D render target interface to draw.
    g_pRT->DrawText(
        wszText,        // The string to render.
        cTextLength,    // The string's length.
        g_pTextFormat,   // The text format.
        layoutRect,      // The region of the window where the text will be rendered.
        g_pBlackBrush    // The brush used to draw the text.
        );

    return S_OK;
}

HRESULT DrawWindowElements()
{
    HRESULT hr = S_OK;

    hr = CreateDeviceResources();

    if (!(g_pRT->CheckWindowState() & D2D1_WINDOW_STATE_OCCLUDED))
    {
        g_pRT->BeginDraw();

        g_pRT->SetTransform(D2D1::IdentityMatrix());

        g_pRT->Clear(D2D1::ColorF(D2D1::ColorF::White));

        if (SUCCEEDED(hr))
        {
            RECT rc;
            for(int i = 0; i < NUMFIELDS; i++)
            {
                // Display program Id
                rc.top = Y_OFFSET + (i * (F2F_Y_OFFSET + FIELD_HEIGHT));
                rc.left = X_OFFSET;
                rc.bottom = rc.top + FIELD_HEIGHT;
                rc.right = rc.left + FIELD_WIDTH;

                // Call the DrawText method of this class.
                DrawHwndText(TextFields[i].pFieldName, static_cast<UINT32>(wcslen(TextFields[i].pFieldName)), rc);

                // Display program Id
                rc.left = rc.right + F2F_X_OFFSET - 2;
                rc.right = rc.left + F2F_Y_OFFSET;

                if(TextFields[i].enFieldtype == FIELD_EDIT)
                    DrawHwndText(L"0x", static_cast<UINT32>(wcslen(L"0x")), rc);
            }
        }

        if (SUCCEEDED(hr))
        {
            hr = g_pRT->EndDraw();
        }
    }

    if (FAILED(hr))
    {
        DiscardDeviceResources();
    }

    return hr;
}

//-----------------------------------------------------------------------------
// ValidateInputData
//-----------------------------------------------------------------------------
bool ValidateInputData(char* szInBuffer, UINT32 cszBufferSize)
{
	bool retValue = true;

	// Assuming this input is multibyte string, colwert to hex value.
	for (UINT32 i = 0; i < cszBufferSize; i++)
	{
		switch(szInBuffer[((cszBufferSize - 1) * 2) - (i * 2)])
		{
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
		case 'a':
		case 'A':
		case 'b':
		case 'B':
		case 'c':
		case 'C':
		case 'd':
		case 'D':
		case 'e':
		case 'E':
		case 'f':
		case 'F':
			continue;

		default:
			return false;
		};
	}

	return retValue;
}

//-----------------------------------------------------------------------------
// ColwertStringtoHex
//-----------------------------------------------------------------------------
UINT64 ColwertWStringtoHex(char* szInBuffer, UINT32 cszBufferSize)
{
    UINT64 retValue = 0;

    // Assuming this input is multibyte string, colwert to hex value.
    for(UINT32 i = 0; i < cszBufferSize; i++)
    {
        switch(szInBuffer[((cszBufferSize - 1) * 2) - (i * 2)])
        {
            case '0': retValue |= ((0x0ULL) << (i*4)); break;
            case '1': retValue |= ((0x1ULL) << (i*4)); break;
            case '2': retValue |= ((0x2ULL) << (i*4)); break;
            case '3': retValue |= ((0x3ULL) << (i*4)); break;
            case '4': retValue |= ((0x4ULL) << (i*4)); break;
            case '5': retValue |= ((0x5ULL) << (i*4)); break;
            case '6': retValue |= ((0x6ULL) << (i*4)); break;
            case '7': retValue |= ((0x7ULL) << (i*4)); break;
            case '8': retValue |= ((0x8ULL) << (i*4)); break;
            case '9': retValue |= ((0x9ULL) << (i*4)); break;
            case 'a':
            case 'A': retValue |= ((0xaULL) << (i*4)); break;
            case 'b':
            case 'B': retValue |= ((0xbULL) << (i*4)); break;
            case 'c':
            case 'C': retValue |= ((0xlwLL) << (i*4)); break;
            case 'd':
            case 'D': retValue |= ((0xdULL) << (i*4)); break;
            case 'e':
            case 'E': retValue |= ((0xeULL) << (i*4)); break;
            case 'f':
            case 'F': retValue |= ((0xfULL) << (i*4)); break;
            default:
                break;
        };
    }

    return retValue;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
VOID Cleanup()
{
    UnregisterClass( TEXT("RSA Sign"), GetModuleHandle(NULL));
    CoUninitialize();
}

BOOL ReadAndValidateInputData(HWND hWnd, HWND* hEditBox)
{
    char szTextBuffer[MAX_TEXT_LENGTH];
    UINT64 uiValueHi = 0;
    UINT64 uiValueLo = 0;
    BOOL retValue = TRUE;
    DWORD dwTextSize = 0;

    if(!hEditBox)
        return FALSE;

    // First validate if the text size is within limit and it contains only hex 
    // values.
    for(int i = 0; i < NUMFIELDS; ++i)
    {
        if(TextFields[i].enFieldtype == FIELD_EDIT)
        {
            dwTextSize = (DWORD) SendMessage(hEditBox[i], EM_LINELENGTH, 0, 0);

            // Check if the text length is valid. If the length is zero just set the value to zero.
            if(dwTextSize > EditBoxSize[i] || !dwTextSize)
            {
                //Set the correct error mask and error flag
                g_ErrorMask |= (0x1 << (i * 2));
            }

            if(dwTextSize)
            {
                // Get the data and validate if it's only hex values
                *((DWORD *)szTextBuffer) = dwTextSize;
                if(SendMessage(hEditBox[i], EM_GETLINE, 0, (LPARAM) szTextBuffer))
                {
                    if(!ValidateInputData(szTextBuffer, dwTextSize))
                    {
                        //Set the correct error mask and error flag
                        g_ErrorMask |= (0x1 << ((i * 2) + 1));
                    }
                }
                else
                {
                    MessageBox(hWnd,TEXT("Failed to get Edit box data !!!\n"), TEXT("ERROR"), MB_OK | MB_ICONERROR | MB_APPLMODAL);
                    g_ErrorMask = -1;
                }
            }

            if(!g_ErrorMask)
            {
                // Retrieve the data from the text box.
                if(dwTextSize > 0)
                {
                    // Colwert from the string to hex value.
                    // Store this inside the structure and call LwAPI
                    if(dwTextSize > 16)
                    {
                        uiValueHi = ColwertWStringtoHex(szTextBuffer, (dwTextSize - 16));
                        uiValueLo = ColwertWStringtoHex(&szTextBuffer[(dwTextSize - 16) * 2], 16);
                    }
                    else if(dwTextSize <= 16)
                    {
                        uiValueLo = ColwertWStringtoHex(szTextBuffer, dwTextSize);
                    }

                    // Store the value in the LWAPI input structure
                    switch(IDC_EDITBOXBASE + i + 1)
                    {
                        case IDC_EDITBOX_NONCE:
                            memcpy(stInput.Nonce, &uiValueLo, sizeof(uiValueLo));
                            memcpy(&stInput.Nonce[8], &uiValueHi, sizeof(uiValueHi));
                            break;

                        case IDC_EDITBOX_PROGID:
                            uiValueLo &= 0xFFFFFFFF;
                            memcpy(&stInput.ProgramId, &uiValueLo, sizeof(UINT32));
                            break;

                        case IDC_EDITBOX_SESSIONID:
                            uiValueLo &= 0xFFFFFFFF;
                            memcpy(&stInput.SessionId, &uiValueLo, sizeof(UINT32));
                            break;

                        default:
                            MessageBox(hWnd,TEXT("Invalid Edit Box ID!!!\n"), TEXT("ERROR"), MB_OK | MB_ICONERROR | MB_APPLMODAL);
                            retValue = FALSE;
                            break;
                    }
                }
            }
        }
		else if(TextFields[i].enFieldtype == FIELD_COMBOBOX)
        {
			// Store the value in the LWAPI input structure
			switch (IDC_COMBOBOXBASE + i + 1)
			{
			case IDC_COMBOBOX_ALGO:
			{ // intentional scoping for LwU32 declaration
				LwU32 algo = (LwU32)stInput.SignAlgo;
				if (!algo)
					g_ErrorMask |= (0x1 << (i * 2));
				break;
			}
			case IDC_COMBOBOX_GPU:
				if (!g_hPhysicalGpu)
				{
					g_ErrorMask |= (0x1 << (i * 2));
				}
				break;
			default:
				MessageBox(hWnd, TEXT("Invalid Combo Box!!!\n"), TEXT("ERROR"), MB_OK | MB_ICONERROR | MB_APPLMODAL);
				retValue = FALSE;
				break;
			}
        }
        else if(TextFields[i].enFieldtype == FIELD_BUTTON)
        {
            if(!stEncryptKey.encryptedKey)
                g_ErrorMask |= (0x1 << (i * 2));
            else if(!stEncryptKey.encryptedKeySize)
                g_ErrorMask |= (0x1 << ((i * 2) + 1));
        }                        

        retValue = g_ErrorMask?FALSE:TRUE;
    }

    return retValue;
}

void PrintErrorMessage(HWND hWnd)
{
    //Build the error Message.
    wchar_t* lpszbuffer = ErrorBuffer;
    UINT32 maxErrorCnt = (sizeof(ErrorMessages)/sizeof(wchar_t*));

    wsprintfW(lpszbuffer, ErrorMessages[0]);
    lpszbuffer += wcslen(ErrorMessages[0]);
    for(UINT32 i = 1; i < maxErrorCnt; i++)
    {
        if(g_ErrorMask)
        {
            if(g_ErrorMask & (0x1 << (i-1)))
            {
                wsprintfW(lpszbuffer, ErrorMessages[i]);
                lpszbuffer += wcslen(ErrorMessages[i]);
                // Clear the error bit
                g_ErrorMask &= ~(0x1 << (i-1));
            }
        }
        else
        {
            // Make sure you null terminate the string.
            lpszbuffer[0] = '\0';
            break; // There are no more errors then break out of loop
        }
    }

    // Display the error message
    MessageBox(hWnd, ErrorBuffer, TEXT("Input Errors"), MB_APPLMODAL | MB_ICONERROR | MB_OK);
}

bool DumpOutput(unsigned char* pOutputBuffer, DWORD dwBufSize)
{
    DWORD dwBytesWritten;
    TCHAR szDirectory[MAX_PATH];
    TCHAR szFileName[MAX_PATH];
    DWORD dwBytesRead = 0;
    HANDLE hFile;
    LW_SIGNED_GPUID_V2* pOutputData = reinterpret_cast<LW_SIGNED_GPUID_V2*>(pOutputBuffer);

    // Validate the input parameters.
    if(!pOutputBuffer || dwBufSize == 0)
        return false;

    //Get the current directory
    dwBytesRead = GetLwrrentDirectory(MAX_PATH, szDirectory);

    if(dwBytesRead)
    {
        // Dump the whole output data.
        wsprintf(szFileName,TEXT("%ls\\Complete_Output_Data.bin"),szDirectory);

        // Open the file.
        if (ILWALID_HANDLE_VALUE == 
                (hFile = CreateFile (szFileName, GENERIC_WRITE, 0,
                        NULL, CREATE_ALWAYS, 0, NULL)))
            return FALSE ;

        // Write the whole data out as binary blob.
        WriteFile(hFile, pOutputBuffer, dwBufSize, &dwBytesWritten, NULL);
        CloseHandle(hFile);

        // Dump just the signing payload. This is SHA256(Nonce, Program Id, Signing Algo, SHA256(RAW ECID), device ID, uCode Version)
        wsprintf(szFileName,TEXT("%ls\\RSA_Payload.bin"),szDirectory);

        // Open the file.
        if (ILWALID_HANDLE_VALUE == 
                (hFile = CreateFile (szFileName, GENERIC_WRITE, 0,
                        NULL, CREATE_ALWAYS, 0, NULL)))
            return FALSE ;

        // Note session ID is not included for now.
        WriteFile(hFile, pOutputData->input.Nonce, sizeof(pOutputData->input.Nonce), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->input.ProgramId, sizeof(pOutputData->input.ProgramId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->input.SignAlgo, sizeof(pOutputData->input.SignAlgo), &dwBytesWritten, NULL);
        WriteFile(hFile, pOutputData->ecid.ecidSha2Hash, sizeof(pOutputData->ecid.ecidSha2Hash), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.vendorId, sizeof(pOutputData->deviceInfo.vendorId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.deviceId, sizeof(pOutputData->deviceInfo.deviceId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.subSystemId, sizeof(pOutputData->deviceInfo.subSystemId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.subVendorId, sizeof(pOutputData->deviceInfo.subVendorId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.revisionId, sizeof(pOutputData->deviceInfo.revisionId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->deviceInfo.chipId, sizeof(pOutputData->deviceInfo.chipId), &dwBytesWritten, NULL);
        WriteFile(hFile, &pOutputData->uCodeVersion, sizeof(pOutputData->uCodeVersion), &dwBytesWritten, NULL);

        CloseHandle(hFile);

        // Write the ECID data into a separate file;
        wsprintf(szFileName,TEXT("%ls\\ECID_SHA256_Hash.bin"),szDirectory);

         // Open the file.
        if (ILWALID_HANDLE_VALUE == 
                (hFile = CreateFile (szFileName, GENERIC_WRITE, 0,
                        NULL, CREATE_ALWAYS, 0, NULL)))
            return FALSE ;
        
        // Write the ECID SHA2 Hash.
        WriteFile(hFile, pOutputData->ecid.ecidSha2Hash, LW_ECID_HASH_SIZE_IN_BYTES, &dwBytesWritten, NULL);
        CloseHandle(hFile);

        // Write the signature as a separate blob.
        // Write the ECID data into a separate file;
        wsprintf(szFileName,TEXT("%ls\\RSA_Signature.bin"),szDirectory);

         // Open the file.
        if (ILWALID_HANDLE_VALUE == 
                (hFile = CreateFile (szFileName, GENERIC_WRITE, 0,
                        NULL, CREATE_ALWAYS, 0, NULL)))
            return FALSE ;

        // Write the RSA signature.
        switch(pOutputData->input.SignAlgo)
        {
            case 1:
                WriteFile(hFile, (pOutputBuffer + sizeof(LW_SIGNED_GPUID_V2) - LW_SIGNED_ECID_RSA_2048_BUF_SIZE), SIGNED_BUFFER_SIZE_1K, &dwBytesWritten, NULL);
                break;

            case 2:
                WriteFile(hFile, (pOutputBuffer + sizeof(LW_SIGNED_GPUID_V2) - LW_SIGNED_ECID_RSA_2048_BUF_SIZE), SIGNED_BUFFER_SIZE_2K, &dwBytesWritten, NULL);
                break;

            default:
                break;
        }
        CloseHandle(hFile);

        return true;
    }
    else
    {
        return false;
    }
}

BOOL GetSignedGPUID(HWND hWnd)
{
    LwAPI_Status status = LWAPI_OK;
    unsigned char* pOutputBuffer;
    LwU32 uSignedBufferSize;
    LwU32 uOutputBufferSize;
    unsigned char* signedBuf = NULL;

    LW_SIGNED_GPUID_V2* pOutputData = new LW_SIGNED_GPUID_V2;
    memset(pOutputData, 0, sizeof(LW_SIGNED_GPUID_V2));
    pOutputData->version = LW_SIGNED_GPUID_VER2;

    // Copy the input param into the newly created structure
    memcpy(&pOutputData->input, &stInput, sizeof(LW_SIGN_GPUID_INPUT));

    switch(stInput.SignAlgo)
    {
        case LW_ECID_SIGNALGO_RSA_1024:
            // Create the output buffer.
            uOutputBufferSize = sizeof(LW_SIGNED_GPUID_V2) - sizeof(pOutputData->signedPayload) + LW_SIGNED_ECID_RSA_1024_BUF_SIZE;
            uSignedBufferSize = LW_SIGNED_ECID_RSA_1024_BUF_SIZE;
            signedBuf = pOutputData->signedPayload.RSA_1024_BUF;
            break;

        case LW_ECID_SIGNALGO_RSA_2048:
            // Create the output buffer.
            uOutputBufferSize = sizeof(LW_SIGNED_GPUID_V2) - sizeof(pOutputData->signedPayload) + LW_SIGNED_ECID_RSA_2048_BUF_SIZE;
            uSignedBufferSize = LW_SIGNED_ECID_RSA_2048_BUF_SIZE;
            signedBuf = pOutputData->signedPayload.RSA_2048_BUF;
            break;

        default:
            break;
    }

    // Display the input parameters to validate they are right
    wchar_t* lpszInputBuffer = DisplayBuffer;
    wsprintfW(lpszInputBuffer,
              TEXT("*Random Nonce : 0x%016I64X%016I64X\n    *Program ID : 0x%08X \n    *Session Id : 0x%08X \n    *Signed Algo : 0x%08x"),
              *((LwU64 *)&stInput.Nonce[8]), *((LwU64 *)&stInput.Nonce),
              stInput.ProgramId,
              stInput.SessionId,
              stInput.SignAlgo);

    MessageBox(hWnd, DisplayBuffer, TEXT("Input Data"), MB_OK | MB_APPLMODAL);
    ZeroMemory(&DisplayBuffer, sizeof(DisplayBuffer));
    
    // Make the LwApi Call to generate the key
    status = LwAPI_GPU_GetSignedGPUID(g_hPhysicalGpu, &stEncryptKey, pOutputData);
    if (status != LWAPI_OK)
    {
        wchar_t* lpszbuffer = ErrorBuffer;
        wsprintfW(lpszbuffer, TEXT("LWAPI call failed. Status is %d"), status);
        // Display the error message
        MessageBox(hWnd, lpszbuffer, TEXT("Input Errors"), MB_APPLMODAL | MB_ICONERROR | MB_OK);
        return FALSE;
    }
    else
    {
        // Check if a valid buffer and size were returned.
        if(pOutputData)
        {
            pOutputBuffer = reinterpret_cast<unsigned char*>(pOutputData);

            TCHAR szDirectory[MAX_PATH];
            //Build the display Message.
            wchar_t* lpszbuffer = DisplayBuffer;

            //Dump the output data into binary files.
            DumpOutput(pOutputBuffer, uOutputBufferSize);

            wsprintfW(lpszbuffer,
                TEXT("Results of signing:\n    *Random Nonce : 0x%016I64X%016I64X\n    *Program ID : 0x%08X \n    *Session Id : 0x%08X \n    *Signed Buffer size : %d \n    *Signed Algo : 0x%08x \n    *Microcode Version : %d \n    *Return Status : %d \nHashed Ecid: 0x"),
                        *((LwU64 *)&pOutputData->input.Nonce[8]), *((LwU64 *)&pOutputData->input.Nonce),
                        pOutputData->input.ProgramId,
                        pOutputData->input.SessionId,
                        uSignedBufferSize,
                        pOutputData->input.SignAlgo,
                        pOutputData->uCodeVersion, 
                        status);

            lpszbuffer = DisplayBuffer + lstrlenW(DisplayBuffer);
            for(int i = 0; i < LW_ECID_HASH_SIZE_IN_BYTES; i++)
            {
                wsprintfW(lpszbuffer,TEXT("%02x"), pOutputData->ecid.ecidSha2Hash[i]);
                lpszbuffer += 2;
            }

            wsprintfW(lpszbuffer,
                      TEXT("\nDevice Info:\n    *Vendor ID: 0x%04X\n    *Device ID: 0x%04X\n    *Subsys ID: 0x%04X\n    *Subvendor ID: 0x%04X\n     *Revision ID: 0x%04X\n     *Chip ID: 0x%04X"),
                      pOutputData->deviceInfo.vendorId,
                      pOutputData->deviceInfo.deviceId,
                      pOutputData->deviceInfo.subSystemId,
                      pOutputData->deviceInfo.subVendorId,
                      pOutputData->deviceInfo.revisionId,
                      pOutputData->deviceInfo.chipId);
            lpszbuffer = DisplayBuffer + lstrlenW(DisplayBuffer);

            wsprintfW(lpszbuffer,TEXT("\nSigned Buffer: 0x"));
            lpszbuffer = DisplayBuffer + lstrlenW(DisplayBuffer);
            for(unsigned int i = 0; i < uSignedBufferSize; i++)
            {
                wsprintfW(lpszbuffer,TEXT("%02x"), signedBuf[i]);
                lpszbuffer += 2;
            }

            //Get the current directory
            GetLwrrentDirectory(MAX_PATH, szDirectory);
            wsprintfW(lpszbuffer,TEXT("\nBinary file dump directory:\n %ls"), szDirectory);
            lpszbuffer = DisplayBuffer + lstrlenW(DisplayBuffer);

            // Null terminate the string.
            *lpszbuffer = 0;

            // Display the generated key and input parameters.
            MessageBox(hWnd,DisplayBuffer,TEXT("Output Data"), MB_OK | MB_APPLMODAL);
        }
    }

    // NOTE: Delete the output data
    delete pOutputData;

    return TRUE;
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
LRESULT WINAPI MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
    static HWND hButtons[NUMBUTTONS];
    static HWND hEditBox[NUMEDITBOX];
    static HWND hBrowse, hComboBox;
    static TCHAR szFileName[MAX_PATH], szTitleName[MAX_PATH] ;
    DWORD dwTextSize = 0;
    BOOL bInputSuccess = true;

    switch( msg )
    {
        case WM_CREATE:
            for(int i = 0; i < NUMBUTTONS; ++i)
            {
                hButtons[i] = CreateWindow( 
                        L"BUTTON",  // Predefined class; Unicode assumed 
                        ButtonNames[i],      // Button text 
                        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles 
                        (90 + (i * (210 + 75))),        // x position 
						(WINDOW_HEIGHT - F2F_Y_OFFSET - 30),      // y position 
                        150,      // Button width
                        25,       // Button height
                        hWnd,     // Parent window
                        (HMENU)(IDC_BUTTONBASE + i + 1), // Identifier
                        ((LPCREATESTRUCT) lParam)->hInstance, 
                        NULL);      // Pointer not needed.
            }

            // Create the edit box controls and make sure the size of each edit box is restricted
            // to the allowed maximum characters.
            for(int i = 0; i < NUMFIELDS; ++i)
            {
                switch (TextFields[i].enFieldtype)
                {
                    case FIELD_BUTTON:
                        hBrowse = CreateWindow( 
                            L"BUTTON",  // Predefined class; Unicode assumed 
                            L"Browse...",      // Button text 
                            WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles 
                            X_OFFSET + FIELD_WIDTH + F2F_X_OFFSET,            // x position 
                            Y_OFFSET + (i * (F2F_Y_OFFSET)),   // y position 
                            125,      // Button width
                            25,       // Button height
                            hWnd,     // Parent window
                            (HMENU)IDC_BUTTON_BROWSE, // Identifier
                            ((LPCREATESTRUCT) lParam)->hInstance,
                            NULL);      // Pointer not needed.
                        break;

                    case FIELD_EDIT:
                        hEditBox[i] = CreateWindow( 
                                L"EDIT",  // Predefined class; Unicode assumed 
                                NULL,     // Button text 
                                WS_TABSTOP | WS_BORDER | WS_VISIBLE | WS_CHILD,  // Styles 
                                X_OFFSET + FIELD_WIDTH + F2F_X_OFFSET,            // x position 
                                Y_OFFSET + (i * (F2F_Y_OFFSET)),   // y position 
                                EDITBOX_WIDTH,    // Editbox width
                                EDITBOX_HEIGHT,   // Editbox height
                                hWnd,     // Parent window
                                (HMENU)(IDC_EDITBOXBASE + i + 1), // Identifier
                                ((LPCREATESTRUCT) lParam)->hInstance, 
                                NULL);      // Pointer not needed.
                
                        SendMessage(hEditBox[i], EM_SETLIMITTEXT, EditBoxSize[i], 0);
                        break;

                    case FIELD_COMBOBOX:
                        hComboBox = CreateWindow(
                                L"COMBOBOX",  // Predefined class; Unicode assumed 
                                NULL,     // Button text 
                                CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VISIBLE | WS_CHILD,  // Styles 
                                X_OFFSET + FIELD_WIDTH + F2F_X_OFFSET,            // x position 
                                Y_OFFSET + (i * (F2F_Y_OFFSET)),   // y position 
                                EDITBOX_WIDTH,    // Editbox width
                                EDITBOX_HEIGHT,   // Editbox height
                                hWnd,     // Parent window
								(HMENU)(IDC_COMBOBOXBASE + i + 1), // Identifier
                                ((LPCREATESTRUCT) lParam)->hInstance, 
                                NULL);      // Pointer not needed.
						break;

                    default:
                        break;
                }
            }

			hComboBox = GetDlgItem(hWnd, IDC_COMBOBOX_ALGO);
			if (!!hComboBox)
			{
				SendMessage(hComboBox, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)L"RSA 1024");
				SendMessage(hComboBox, (UINT)CB_ADDSTRING, (WPARAM)1, (LPARAM)L"RSA 2048");
			}
			else
			{
				MessageBox(hWnd, TEXT("Failed to populate Signing Algo options"), TEXT("Output Data"), MB_OK | MB_APPLMODAL);
				return FALSE;
			}

			hComboBox = GetDlgItem(hWnd, IDC_COMBOBOX_GPU);
			if (!!hComboBox)
			{
				for (LwU32 i = 0; i < g_gpuCount; i++)
				{
					LwPhysicalGpuHandle gpuHandle = g_hPhysicalGpuArrayPtr[i];
					LwAPI_ShortString gpuFullName;
					LwAPI_Status status = LwAPI_GPU_GetFullName(gpuHandle, gpuFullName);
					if (status != LWAPI_OK)
					{
						wchar_t* lpszbuffer = ErrorBuffer;
						wsprintfW(lpszbuffer, TEXT("LWAPI call to get GPU name failed. Status is %d"), status);
						MessageBox(hWnd, lpszbuffer, TEXT("GPU Lookup Error"), MB_APPLMODAL | MB_ICONERROR | MB_OK);
						return FALSE;
					}

					LwU32 busId = 0;
					status = LwAPI_GPU_GetBusId(gpuHandle, &busId);
					if (status != LWAPI_OK)
					{
						wchar_t* lpszbuffer = ErrorBuffer;
						wsprintfW(lpszbuffer, TEXT("LWAPI call to get GPU bus ID failed. Status is %d"), status);
						MessageBox(hWnd, lpszbuffer, TEXT("GPU Lookup Errors"), MB_APPLMODAL | MB_ICONERROR | MB_OK);
						return FALSE;
					}

					wchar_t gpuFullNameWide[64] = { 0 };
					if (MultiByteToWideChar(CP_UTF8, 0, reinterpret_cast<PCSTR>(gpuFullName), static_cast<int>(sizeof(gpuFullName)), gpuFullNameWide, sizeof(gpuFullNameWide)) == 0)
					{
						wchar_t* lpszbuffer = ErrorBuffer;
						wsprintfW(lpszbuffer, TEXT("Failed to colwert GPU Name"), status);
						MessageBox(hWnd, lpszbuffer, TEXT("GPU Lookup Errors"), MB_APPLMODAL | MB_ICONERROR | MB_OK);
						return FALSE;
					}


					wchar_t gpuNameIdBuffer[128] = { 0 };
					wsprintfW(gpuNameIdBuffer, TEXT("%s:%d"), gpuFullNameWide, busId);
					SendMessage(hComboBox, static_cast<UINT>(CB_ADDSTRING), static_cast<WPARAM>(0), reinterpret_cast<LPARAM>(gpuNameIdBuffer));
				}
			}
			else
			{
				MessageBox(hWnd, TEXT("Failed to populate GPUs"), TEXT("Output Data"), MB_OK | MB_APPLMODAL);
				return FALSE;
			}
			

            // Reset the input buffer during creation.
            ZeroMemory(&stInput, sizeof(stInput));

            PopFileInitialize(hWnd);
            break;

        case WM_COMMAND:
            switch(LOWORD(wParam))
            {
                case IDC_BUTTON_CLOSE:
                    PostQuitMessage(0);    
                    return 0;

                case IDC_BUTTON_GENKEY:
                {
                    // Clear the output buffer
                    memset(DisplayBuffer,0,1024);
                    bInputSuccess = ReadAndValidateInputData(hWnd, hEditBox);

                    // If an error had oclwrred then build an errorMessage and throw an error.
                    if(!bInputSuccess)
                    {
                        PrintErrorMessage(hWnd);
                    }
                    else
                    {
                        GetSignedGPUID(hWnd);
                    }
                    break;
                }

                case IDC_BUTTON_BROWSE:
                    ZeroMemory(&stEncryptKey, sizeof(stEncryptKey));
                    if (PopFileOpenDlg (hWnd, szFileName, szTitleName))
                    {
                        if (!PopFileRead (g_hWnd, szFileName, &stEncryptKey.encryptedKey, (UINT32 *)&stEncryptKey.encryptedKeySize))
                        {
                            MessageBox(hWnd, TEXT ("Could not read file %s!"), szTitleName, MB_OK | MB_APPLMODAL) ;
                            szFileName[0] = '\0';
                            szTitleName[0] = '\0';
                        }
                    }
                    break;

				case IDC_COMBOBOX_ALGO:
                    if(HIWORD(wParam) == CBN_SELCHANGE)
                    // If the user makes a selection from the list:
                    //   Send CB_GETLWRSEL message to get the index of the selected list item.
                    { 
                            stInput.SignAlgo = (LW_ECID_SIGNALGO) ( SendMessage((HWND) lParam, (UINT) CB_GETLWRSEL, (WPARAM) 0, (LPARAM) 0) + 1);
                    }
                    break;

				case IDC_COMBOBOX_GPU:
					if (HIWORD(wParam) == CBN_SELCHANGE)
					{
						LwU32 selection = static_cast<LwU32>(SendMessage(reinterpret_cast<HWND>(lParam), 
																		 static_cast<UINT>(CB_GETLWRSEL), 
																		 static_cast<WPARAM>(0), static_cast<LPARAM>(0) + 1));
						g_hPhysicalGpu = g_hPhysicalGpuArrayPtr[selection];
					}
					break;

                default:
                    break;
            }
            break;

        case WM_PAINT:
            DrawWindowElements();
            break;

        case WM_CLOSE:
            Cleanup();
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }

    return DefWindowProc( hWnd, msg, wParam, lParam );
}

//-----------------------------------------------------------------------------
// Name: RunTest()
// Desc: Runs the test
//-----------------------------------------------------------------------------
HRESULT InitLwAPI()
{
    HRESULT hr;

    if (FAILED(hr = LwAPI_Initialize()))
    {
        printf("LwAPI_Initialize failed %08x\n", hr);
        return E_FAIL;
    }

	g_hPhysicalGpuArrayPtr = new LwPhysicalGpuHandle[LWAPI_MAX_PHYSICAL_GPUS];
	LwAPI_Status status = LwAPI_EnumPhysicalGPUs(g_hPhysicalGpuArrayPtr, &g_gpuCount);
	if (status != LWAPI_OK)
	{
		printf("Failed to enumerate LW GPUs %d\n", status);
		return status;
	}


    return S_OK;
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                     LPSTR nCmdLine, int nCmdShow)
{
    MSG msg;
    BOOL fGotMessage;

    // Must initialize LWAPI before app window and controls as the GPU combobox makes use of LWAPI
    InitLwAPI();

	InitD3D(hInstance);

    while ((fGotMessage = GetMessage(&msg, (HWND) NULL, 0, 0)) != 0 && fGotMessage != -1) 
    { 
        if (!IsDialogMessage(g_hWnd, &msg))
        {
            TranslateMessage(&msg); 
            DispatchMessage(&msg); 
        }
    } 
	delete[] g_hPhysicalGpuArrayPtr;
    return msg.wParam; 
}
