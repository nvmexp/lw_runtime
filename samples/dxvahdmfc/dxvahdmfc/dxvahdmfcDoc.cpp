// dxvahdmfcDoc.cpp : implementation of the CdxvahdmfcDoc class
//

#include "stdafx.h"
#include "dxvahdmfc.h"
#include "dxvahdmfcDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CdxvahdmfcDoc

IMPLEMENT_DYNCREATE(CdxvahdmfcDoc, CDolwment)

BEGIN_MESSAGE_MAP(CdxvahdmfcDoc, CDolwment)
END_MESSAGE_MAP()



// CdxvahdmfcDoc construction/destruction

CdxvahdmfcDoc::CdxvahdmfcDoc()
{
	// TODO: add one-time construction code here

}

CdxvahdmfcDoc::~CdxvahdmfcDoc()
{
}

BOOL CdxvahdmfcDoc::OnNewDolwment()
{
	if (!CDolwment::OnNewDolwment())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI dolwments will reuse this document)

	return TRUE;
}




// CdxvahdmfcDoc serialization

void CdxvahdmfcDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CdxvahdmfcDoc diagnostics

#ifdef _DEBUG
void CdxvahdmfcDoc::AssertValid() const
{
	CDolwment::AssertValid();
}

void CdxvahdmfcDoc::Dump(CDumpContext& dc) const
{
	CDolwment::Dump(dc);
}
#endif //_DEBUG


// CdxvahdmfcDoc commands
