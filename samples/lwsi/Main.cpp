// Main.cpp

#include "stdafx.h"
#include "LwsiZipFile.h"
#include "GPUDataSegment.h"
#include "lwsi.pb.h" 

CWinApp theApp;

int _tmain(int argc, TCHAR* argv[], TCHAR* elwp[])
{
    CLwsiZipFile zipfile;

    try
    {
	    // initialize MFC and print and error on failure
	    if (!AfxWinInit(::GetModuleHandle(NULL), NULL, ::GetCommandLine(), 0))
	    {
		    throw CLwsiException("Error: MFC initialization failed");
	    }

        GOOGLE_PROTOBUF_VERIFY_VERSION; 


        CGPUDataSegment *pGpuDS = new CGPUDataSegment("GPU.bin"); // GPU Data Segment - holds GPU, System + Misc Info

        pGpuDS->RunCmds("default");

        pGpuDS->RawViewData();
        
        CString tmpGPUDataSegment;
        pGpuDS->SaveCatpuredData(tmpGPUDataSegment);


        
        printf("LWSI. Woot.\n");

        zipfile.OpenZip("c:\\testlwsi.lwsi");
        zipfile.AddFile(tmpGPUDataSegment,"GPU.bin");
        zipfile.AddFile("c:\\readme.txt","readme.txt");
        zipfile.CloseZip();

//        _unlink(tmpGPUDataSegment);

        delete pGpuDS;

    } 
    catch ( CLwsiException e ) {
        LwsiErrorMsg("%s\n",e.m_msg);
        LwsiErrorMsg("Aborting...\n");

        return 1;
    }
    catch ( ... ) {
        throw; // Rethrow any other exceptions (for now)
    }
    return 0;
}


#if 0
    if (argc == 1)
    {
        CFileDialog dlg(FALSE,"lwsi","LWPU.lwsi");
        if (dlg.DoModal() != IDOK)
        {
            return 1;
        }
        printf("Save to file %s\n",dlg.GetPathName());
    }
#endif
