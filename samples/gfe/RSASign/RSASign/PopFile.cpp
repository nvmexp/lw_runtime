/*------------------------------------------
 POPFILE.C -- Popup Editor File Functions
 ------------------------------------------*/
#include <windows.h>
#include <commdlg.h>

static OPENFILENAME ofn ;

void PopFileInitialize (HWND hwnd)
{
     static TCHAR szFilter[] = TEXT ("Binary Files (*.BIN)\0*.bin\0") \
     TEXT ("All Files (*.*)\0*.*\0\0") ;
 
     ofn.lStructSize = sizeof (OPENFILENAME) ;
     ofn.hwndOwner = hwnd ;
     ofn.hInstance = NULL ;
     ofn.lpstrFilter = szFilter ;
     ofn.lpstrLwstomFilter = NULL ;
     ofn.nMaxLwstFilter = 0 ;
     ofn.nFilterIndex = 0 ;
     ofn.lpstrFile = NULL ; // Set in Open and Close functions
     ofn.nMaxFile = MAX_PATH ;
     ofn.lpstrFileTitle = NULL ; // Set in Open and Close functions
     ofn.nMaxFileTitle = MAX_PATH ;
     ofn.lpstrInitialDir = NULL ;
     ofn.lpstrTitle = NULL ;
     ofn.Flags = 0 ; // Set in Open and Close functions
     ofn.nFileOffset = 0 ;
     ofn.nFileExtension = 0 ;
     ofn.lpstrDefExt = TEXT ("txt") ;
     ofn.lLwstData = 0L ;
     ofn.lpfnHook = NULL ;
     ofn.lpTemplateName = NULL ;
}

BOOL PopFileOpenDlg (HWND hwnd, PTSTR pstrFileName, PTSTR pstrTitleName)
{
    ofn.hwndOwner = hwnd ;
    ofn.lpstrFile = pstrFileName ;
    ofn.lpstrFileTitle = pstrTitleName ;
    ofn.Flags = OFN_HIDEREADONLY | OFN_CREATEPROMPT ;
 
    return GetOpenFileName (&ofn) ;
}

BOOL PopFileRead (HWND hwndEdit, PTSTR pstrFileName, BYTE** pOutBuffer, UINT32* pBufferSize)
{
    DWORD dwBytesRead ;
    HANDLE hFile ;
    int iFileLength;
    PBYTE pBuffer;

    // Validate the input parameters.
    if(!pOutBuffer || !pBufferSize)
        return false;

    // Open the file.
    if (ILWALID_HANDLE_VALUE == 
            (hFile = CreateFile (pstrFileName, GENERIC_READ, FILE_SHARE_READ,
                    NULL, OPEN_EXISTING, 0, NULL)))
        return FALSE ;

    // Get file size in bytes and allocate memory for read.
    iFileLength = GetFileSize (hFile, NULL); 
    pBuffer = (BYTE *)malloc (iFileLength);

    if(pBuffer)
    {
        // Read file and put terminating zeros at end. 
        ReadFile (hFile, pBuffer, iFileLength, &dwBytesRead, NULL);
        CloseHandle (hFile);

        *pOutBuffer = pBuffer;
        *pBufferSize = iFileLength;

        return TRUE ;
    }
    else
    {
        *pBufferSize = 0;
        return FALSE;
    }
}