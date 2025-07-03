// dxvahdmfcView.h : interface of the CdxvahdmfcView class
//


#pragma once


class CdxvahdmfcView : public CView
{
protected: // create from serialization only
	CdxvahdmfcView();
	DECLARE_DYNCREATE(CdxvahdmfcView)

// Attributes
public:
	CdxvahdmfcDoc* GetDolwment() const;
	int FrameRate;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CdxvahdmfcView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
public:
	afx_msg void OnTimer(UINT_PTR nIDEvent);
public:
	afx_msg void OnEditConfig();
public:
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
};

#ifndef _DEBUG  // debug version in dxvahdmfcView.cpp
inline CdxvahdmfcDoc* CdxvahdmfcView::GetDolwment() const
   { return reinterpret_cast<CdxvahdmfcDoc*>(m_pDolwment); }
#endif

