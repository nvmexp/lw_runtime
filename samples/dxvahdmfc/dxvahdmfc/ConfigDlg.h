#pragma once


// CConfigDlg dialog

class CConfigDlg : public CDialog
{
	DECLARE_DYNAMIC(CConfigDlg)

public:
	CConfigDlg(CWnd* pParent = NULL);   // standard constructor
	virtual ~CConfigDlg();

// Dialog Data
	enum { IDD = IDD_DIALOG_CONFIG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
public:
	int GraphicsAlphaLevel;
public:
public:
	afx_msg void OnBnClickedBgAlphaenable();
public:
	afx_msg void OnNMReleasedcaptureBgAlphalevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgAnamorphicscalingenable();
public:
	afx_msg void OnNMReleasedcaptureBgAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgBrightnessenable();
public:
	afx_msg void OnNMReleasedcaptureBgBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgContrastenable();
public:
	afx_msg void OnNMReleasedcaptureBgContrastlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgEdgeenhancementenable();
public:
	afx_msg void OnNMReleasedcaptureBgEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgFrameformat();
public:
	afx_msg void OnBnClickedBgHueenable();
public:
	afx_msg void OnNMReleasedcaptureBgHuelevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnEnChangeBgLumakeyLower();
public:
	afx_msg void OnEnChangeBgLumakeyUpper();
public:
	afx_msg void OnBnClickedBgLumakeyenable();
public:
	afx_msg void OnBnClickedBgNoisereductionenable();
public:
	afx_msg void OnNMReleasedcaptureBgNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedBgSaturationenable();
public:
	afx_msg void OnNMReleasedcaptureBgSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrAlphaenable();
public:
	afx_msg void OnNMReleasedcaptureGrAlphalevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrAnamorphicscalingenable();
public:
	afx_msg void OnNMReleasedcaptureGrAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrBrightnessenable();
public:
	afx_msg void OnNMReleasedcaptureGrBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrContrastenable();
public:
	afx_msg void OnNMReleasedcaptureGrContrastlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrEdgeenhancementenable();
public:
	afx_msg void OnNMReleasedcaptureGrEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedGrFrameformat();
public:
	afx_msg void OnBnClickedGrHueenable();
public:
	afx_msg void OnNMReleasedcaptureGrHuelevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnEnChangeGrLumakeyLower();
public:
	afx_msg void OnEnChangeGrLumakeyUpper();
public:
	afx_msg void OnBnClickedGrLumakeyenable();
public:
	afx_msg void OnBnClickedGrNoisereductionenable();
public:
	afx_msg void OnBnClickedGrSaturationenable();
public:
	afx_msg void OnNMReleasedcaptureGrNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnNMReleasedcaptureSvSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvSaturationenable();
public:
	afx_msg void OnNMReleasedcaptureGrSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvAlphaenable();
public:
	afx_msg void OnNMReleasedcaptureMvAlphalevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvAnamorphicscalingenable();
public:
	afx_msg void OnNMReleasedcaptureMvAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvBrightnessenable();
public:
	afx_msg void OnNMReleasedcaptureMvBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvContrastenable();
public:
	afx_msg void OnNMReleasedcaptureMvContrastlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvEdgeenhancementenable();
public:
	afx_msg void OnNMReleasedcaptureMvEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvFrameformat();
public:
	afx_msg void OnBnClickedMvHueenable();
public:
	afx_msg void OnNMReleasedcaptureMvHuelevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnEnChangeMvLumakeyLower();
public:
	afx_msg void OnEnChangeMvLumakeyUpper();
public:
	afx_msg void OnBnClickedMvLumakeyenable();
public:
	afx_msg void OnBnClickedMvNoisereductionenable();
public:
	afx_msg void OnNMReleasedcaptureMvNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedMvSaturationenable();
public:
	afx_msg void OnNMReleasedcaptureMvSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStAlphaenable();
public:
	afx_msg void OnNMReleasedcaptureStAlphalevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStAnamorphicscalingenable();
public:
	afx_msg void OnNMReleasedcaptureStAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStBrightnessenable();
public:
	afx_msg void OnNMReleasedcaptureStBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStContrastenable();
public:
	afx_msg void OnNMReleasedcaptureStContrastlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStEdgeenhancementenable();
public:
	afx_msg void OnNMReleasedcaptureStEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStFrameformat();
public:
	afx_msg void OnBnClickedStHueenable();
public:
	afx_msg void OnNMReleasedcaptureStHuelevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnEnChangeStLumakeyLower();
public:
	afx_msg void OnEnChangeStLumakeyUpper();
public:
	afx_msg void OnBnClickedStLumakeyenable();
public:
	afx_msg void OnBnClickedStNoisereductionenable();
public:
	afx_msg void OnNMReleasedcaptureStNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedStSaturationenable();
public:
	afx_msg void OnNMReleasedcaptureStSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvAlphaenable();
public:
	afx_msg void OnNMReleasedcaptureSvAlphalevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvAnamorphicscalingenable();
public:
	afx_msg void OnNMReleasedcaptureSvAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvBrightnessenable();
public:
	afx_msg void OnNMReleasedcaptureSvBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvContrastenable();
public:
	afx_msg void OnNMReleasedcaptureSvContrastlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvEdgeenhancementenable();
public:
	afx_msg void OnNMReleasedcaptureSvEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnBnClickedSvFrameformat();
public:
	afx_msg void OnBnClickedSvHueenable();
public:
	afx_msg void OnNMReleasedcaptureSvHuelevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	afx_msg void OnEnChangeSvLumakeyLower();
public:
	afx_msg void OnEnChangeSvLumakeyUpper();
public:
	afx_msg void OnBnClickedSvLumakeyenable();
public:
	afx_msg void OnBnClickedSvNoisereductionenable();
public:
	afx_msg void OnNMReleasedcaptureSvNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult);
public:
	BOOL MVFrameFormat;
public:
	BOOL MVAlphaEnable;
public:
	BOOL MVHueEnable;
public:
	int MVAlphaLevel;
public:
	int MVHueLevel;
public:
	BOOL MVBrightnessEnable;
public:
	BOOL MVSaturationEnable;
public:
	int MVBrightnessLevel;
public:
	int MVSaturationLevel;
public:
	BOOL MVContrastEnable;
public:
	BOOL MVNoiseReductionEnable;
public:
	int MVContrastLevel;
public:
	int MVNoiseReductionLevel;
public:
	BOOL MVEdgeEnhancementEnable;
public:
	int MVEdgeEnhancementLevel;
public:
	BOOL MVAnamorphicScalingEnable;
public:
	int MVAnamorphicScalingLevel;
public:
	BOOL SVFrameFormat;
public:
	BOOL SVAlphaEnable;
public:
	int SVAlphaLevel;
public:
	BOOL SVBrightnessEnable;
public:
	int SVBrightnessLevel;
public:
	BOOL SVContrastEnable;
public:
	int SVContrastLevel;
public:
	BOOL SVLumaKeyEnable;
public:
	BOOL SVHueEnable;
public:
	int SVHueLevel;
public:
	BOOL SVSaturationEnable;
public:
	int SVSaturationlevel;
public:
	BOOL SVNoiseReductionEnable;
public:
	int SVNoiseReductionLevel;
public:
	BYTE SVLumaKeyUpper;
public:
	BYTE MVLumakeyUpper;
public:
	BYTE MVLumaKeyLower;
public:
	BYTE SVLumaKeyLower;
public:
	BOOL SVEdgeEnhancementEnable;
public:
	int SVEdgeENhancementLevel;
public:
	BOOL SVAnamorphicScalingEnable;
public:
	int SVAnamorphicScalingLevel;
public:
	BOOL GRAlphaEnable;
public:
	int GRAlphaLevel;
public:
	BOOL GRBrightnessEnable;
public:
	int GRBrightnessLevel;
public:
	BOOL GRContrastEnable;
public:
	int GRContrastLevel;
public:
	BOOL GRHueEnable;
public:
	int GRHueLevel;
public:
	BOOL GRSaturationEnable;
public:
	int GRSaturationLevel;
public:
	BOOL GRNoiseReductionEnable;
public:
	int GRNoiseReductionLevel;
public:
	BOOL GREdgeEnhancementEnable;
public:
	int GREdgeEnhancementLevel;
public:
	BOOL GRAnamorphicScalingEnable;
public:
	int GRAnamorphicScalingLevel;
public:
	BOOL MVLumaKeyEnable;
public:
	BOOL BltDownSampleEnable;
public:
	int BltDownSampleLevel;
public:
	afx_msg void OnBnClickedBltDownsample();
public:
	afx_msg void OnNMReleasedcaptureBltDownsamplelevel(NMHDR *pNMHDR, LRESULT *pResult);
};
