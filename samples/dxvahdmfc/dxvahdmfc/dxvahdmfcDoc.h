// dxvahdmfcDoc.h : interface of the CdxvahdmfcDoc class
//


#pragma once


class CdxvahdmfcDoc : public CDolwment
{
protected: // create from serialization only
	CdxvahdmfcDoc();
	DECLARE_DYNCREATE(CdxvahdmfcDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDolwment();
	virtual void Serialize(CArchive& ar);

// Implementation
public:
	virtual ~CdxvahdmfcDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


