#pragma once

class CDataSegment
{
public:
    CDataSegment(const char *szFilename);
    ~CDataSegment(void);

public:
    CString m_filename; // Name of the file in .lwsi container which corresponds to this data

    virtual void Capture() = 0;
    virtual void SaveCatpuredData(CString &tempFilename) = 0;
    virtual void LoadCatpuredData() = 0;
    virtual void RawViewData() = 0;
    virtual void FormattedViewData() = 0;
    virtual void RunCmds(const char *szCmds) = 0;
};
