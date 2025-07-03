// ConfigDlg.cpp : implementation file
//

#include "stdafx.h"
#include <windows.h>    /* required for all Windows applications */
#include <windowsx.h>
#include <mmsystem.h>

#include <stdio.h>
#include <io.h>
#include <fcntl.h>

#include <initguid.h>
#include <d3d9.h>
#include <d3d9types.h>
#include <dxva2api.h>
#include "dxvahdapi.h"

#include "dxvahdmfc.h"
#include "ConfigDlg.h"


// CConfigDlg dialog

IMPLEMENT_DYNAMIC(CConfigDlg, CDialog)

extern CVideoData gVideoData;

CConfigDlg::CConfigDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CConfigDlg::IDD, pParent)
	, MVFrameFormat(FALSE)
	, MVAlphaEnable(FALSE)
	, MVHueEnable(FALSE)
	, MVAlphaLevel(0)
	, MVHueLevel(0)
	, MVBrightnessEnable(FALSE)
	, MVSaturationEnable(FALSE)
	, MVBrightnessLevel(0)
	, MVSaturationLevel(0)
	, MVContrastEnable(FALSE)
	, MVNoiseReductionEnable(FALSE)
	, MVContrastLevel(0)
	, MVNoiseReductionLevel(0)
	, MVEdgeEnhancementEnable(FALSE)
	, MVEdgeEnhancementLevel(0)
	, MVAnamorphicScalingEnable(FALSE)
	, MVAnamorphicScalingLevel(0)
	, SVFrameFormat(FALSE)
	, SVAlphaEnable(FALSE)
	, SVAlphaLevel(0)
	, SVBrightnessEnable(FALSE)
	, SVBrightnessLevel(0)
	, SVContrastEnable(FALSE)
	, SVContrastLevel(0)
	, SVLumaKeyEnable(FALSE)
	, SVHueEnable(FALSE)
	, SVHueLevel(0)
	, SVSaturationEnable(FALSE)
	, SVSaturationlevel(0)
	, SVNoiseReductionEnable(FALSE)
	, SVNoiseReductionLevel(0)
	, SVLumaKeyUpper(0)
	, MVLumakeyUpper(0)
	, MVLumaKeyLower(0)
	, SVLumaKeyLower(0)
	, SVEdgeEnhancementEnable(FALSE)
	, SVEdgeENhancementLevel(0)
	, SVAnamorphicScalingEnable(FALSE)
	, SVAnamorphicScalingLevel(0)
	, GRAlphaEnable(FALSE)
	, GRAlphaLevel(0)
	, GRBrightnessEnable(FALSE)
	, GRBrightnessLevel(0)
	, GRContrastEnable(FALSE)
	, GRContrastLevel(0)
	, GRHueEnable(FALSE)
	, GRHueLevel(0)
	, GRSaturationEnable(FALSE)
	, GRSaturationLevel(0)
	, GRNoiseReductionEnable(FALSE)
	, GRNoiseReductionLevel(0)
	, GREdgeEnhancementEnable(FALSE)
	, GREdgeEnhancementLevel(0)
	, GRAnamorphicScalingEnable(FALSE)
	, GRAnamorphicScalingLevel(0)
	, MVLumaKeyEnable(FALSE)
	, BltDownSampleEnable(FALSE)
	, BltDownSampleLevel(0)
{

}

CConfigDlg::~CConfigDlg()
{
}

void CConfigDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_MV_FRAMEFORMAT, MVFrameFormat);
	DDX_Check(pDX, IDC_MV_ALPHAENABLE, MVAlphaEnable);
	DDX_Check(pDX, IDC_MV_HUEENABLE, MVHueEnable);
	DDX_Slider(pDX, IDC_MV_ALPHALEVEL, MVAlphaLevel);
	DDX_Slider(pDX, IDC_MV_HUELEVEL, MVHueLevel);
	DDX_Check(pDX, IDC_MV_BRIGHTNESSENABLE, MVBrightnessEnable);
	DDX_Check(pDX, IDC_MV_SATURATIONENABLE, MVSaturationEnable);
	DDX_Slider(pDX, IDC_MV_BRIGHTNESSLEVEL, MVBrightnessLevel);
	DDX_Slider(pDX, IDC_MV_SATURATIONLEVEL, MVSaturationLevel);
	DDX_Check(pDX, IDC_MV_CONTRASTENABLE, MVContrastEnable);
	DDX_Check(pDX, IDC_MV_NOISEREDUCTIONENABLE, MVNoiseReductionEnable);
	DDX_Slider(pDX, IDC_MV_CONTRASTLEVEL, MVContrastLevel);
	DDX_Slider(pDX, IDC_MV_NOISEREDUCTIONLEVEL, MVNoiseReductionLevel);
	DDX_Check(pDX, IDC_MV_EDGEENHANCEMENTENABLE, MVEdgeEnhancementEnable);
	DDX_Slider(pDX, IDC_MV_EDGEENHANCEMENTLEVEL, MVEdgeEnhancementLevel);
	DDX_Check(pDX, IDC_MV_ANAMORPHICSCALINGENABLE, MVAnamorphicScalingEnable);
	DDX_Slider(pDX, IDC_MV_ANAMORPHICSCALINGLEVEL, MVAnamorphicScalingLevel);
	DDX_Check(pDX, IDC_SV_FRAMEFORMAT, SVFrameFormat);
	DDX_Check(pDX, IDC_SV_ALPHAENABLE, SVAlphaEnable);
	DDX_Slider(pDX, IDC_SV_ALPHALEVEL, SVAlphaLevel);
	DDX_Check(pDX, IDC_SV_BRIGHTNESSENABLE, SVBrightnessEnable);
	DDX_Slider(pDX, IDC_SV_BRIGHTNESSLEVEL, SVBrightnessLevel);
	DDX_Check(pDX, IDC_SV_CONTRASTENABLE, SVContrastEnable);
	DDX_Slider(pDX, IDC_SV_CONTRASTLEVEL, SVContrastLevel);
	DDX_Check(pDX, IDC_SV_LUMAKEYENABLE, SVLumaKeyEnable);
	DDX_Check(pDX, IDC_SV_HUEENABLE, SVHueEnable);
	DDX_Slider(pDX, IDC_SV_HUELEVEL, SVHueLevel);
	DDX_Check(pDX, IDC_SV_SATURATIONENABLE, SVSaturationEnable);
	DDX_Slider(pDX, IDC_SV_SATURATIONLEVEL, SVSaturationlevel);
	DDX_Check(pDX, IDC_SV_NOISEREDUCTIONENABLE, SVNoiseReductionEnable);
	DDX_Slider(pDX, IDC_SV_NOISEREDUCTIONLEVEL, SVNoiseReductionLevel);
	DDX_Text(pDX, IDC_SV_LUMAKEY_UPPER, SVLumaKeyUpper);
	DDX_Text(pDX, IDC_MV_LUMAKEY_UPPER, MVLumakeyUpper);
	DDX_Text(pDX, IDC_MV_LUMAKEY_LOWER, MVLumaKeyLower);
	DDX_Text(pDX, IDC_SV_LUMAKEY_LOWER, SVLumaKeyLower);
	DDX_Check(pDX, IDC_SV_EDGEENHANCEMENTENABLE, SVEdgeEnhancementEnable);
	DDX_Slider(pDX, IDC_SV_EDGEENHANCEMENTLEVEL, SVEdgeENhancementLevel);
	DDX_Check(pDX, IDC_SV_ANAMORPHICSCALINGENABLE, SVAnamorphicScalingEnable);
	DDX_Slider(pDX, IDC_SV_ANAMORPHICSCALINGLEVEL, SVAnamorphicScalingLevel);
	DDX_Check(pDX, IDC_GR_ALPHAENABLE, GRAlphaEnable);
	DDX_Slider(pDX, IDC_GR_ALPHALEVEL, GRAlphaLevel);
	DDX_Check(pDX, IDC_GR_BRIGHTNESSENABLE, GRBrightnessEnable);
	DDX_Slider(pDX, IDC_GR_BRIGHTNESSLEVEL, GRBrightnessLevel);
	DDX_Check(pDX, IDC_GR_CONTRASTENABLE, GRContrastEnable);
	DDX_Slider(pDX, IDC_GR_CONTRASTLEVEL, GRContrastLevel);
	DDX_Check(pDX, IDC_GR_HUEENABLE, GRHueEnable);
	DDX_Slider(pDX, IDC_GR_HUELEVEL, GRHueLevel);
	DDX_Check(pDX, IDC_GR_SATURATIONENABLE, GRSaturationEnable);
	DDX_Slider(pDX, IDC_GR_SATURATIONLEVEL, GRSaturationLevel);
	DDX_Check(pDX, IDC_GR_NOISEREDUCTIONENABLE, GRNoiseReductionEnable);
	DDX_Slider(pDX, IDC_GR_NOISEREDUCTIONLEVEL, GRNoiseReductionLevel);
	DDX_Check(pDX, IDC_GR_EDGEENHANCEMENTENABLE, GREdgeEnhancementEnable);
	DDX_Slider(pDX, IDC_GR_EDGEENHANCEMENTLEVEL, GREdgeEnhancementLevel);
	DDX_Check(pDX, IDC_GR_ANAMORPHICSCALINGENABLE, GRAnamorphicScalingEnable);
	DDX_Slider(pDX, IDC_GR_ANAMORPHICSCALINGLEVEL, GRAnamorphicScalingLevel);
	DDX_Check(pDX, IDC_MV_LUMAKEYENABLE, MVLumaKeyEnable);
	DDX_Check(pDX, IDC_BLT_DOWNSAMPLE, BltDownSampleEnable);
	DDX_Slider(pDX, IDC_BLT_DOWNSAMPLELEVEL, BltDownSampleLevel);
}


BEGIN_MESSAGE_MAP(CConfigDlg, CDialog)
	ON_BN_CLICKED(IDC_BG_ALPHAENABLE, &CConfigDlg::OnBnClickedBgAlphaenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_ALPHALEVEL, &CConfigDlg::OnNMReleasedcaptureBgAlphalevel)
	ON_BN_CLICKED(IDC_BG_ANAMORPHICSCALINGENABLE, &CConfigDlg::OnBnClickedBgAnamorphicscalingenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_ANAMORPHICSCALINGLEVEL, &CConfigDlg::OnNMReleasedcaptureBgAnamorphicscalinglevel)
	ON_BN_CLICKED(IDC_BG_BRIGHTNESSENABLE, &CConfigDlg::OnBnClickedBgBrightnessenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_BRIGHTNESSLEVEL, &CConfigDlg::OnNMReleasedcaptureBgBrightnesslevel)
	ON_BN_CLICKED(IDC_BG_CONTRASTENABLE, &CConfigDlg::OnBnClickedBgContrastenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_CONTRASTLEVEL, &CConfigDlg::OnNMReleasedcaptureBgContrastlevel)
	ON_BN_CLICKED(IDC_BG_EDGEENHANCEMENTENABLE, &CConfigDlg::OnBnClickedBgEdgeenhancementenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_EDGEENHANCEMENTLEVEL, &CConfigDlg::OnNMReleasedcaptureBgEdgeenhancementlevel)
	ON_BN_CLICKED(IDC_BG_FRAMEFORMAT, &CConfigDlg::OnBnClickedBgFrameformat)
	ON_BN_CLICKED(IDC_BG_HUEENABLE, &CConfigDlg::OnBnClickedBgHueenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_HUELEVEL, &CConfigDlg::OnNMReleasedcaptureBgHuelevel)
	ON_EN_CHANGE(IDC_BG_LUMAKEY_LOWER, &CConfigDlg::OnEnChangeBgLumakeyLower)
	ON_EN_CHANGE(IDC_BG_LUMAKEY_UPPER, &CConfigDlg::OnEnChangeBgLumakeyUpper)
	ON_BN_CLICKED(IDC_BG_LUMAKEYENABLE, &CConfigDlg::OnBnClickedBgLumakeyenable)
	ON_BN_CLICKED(IDC_BG_NOISEREDUCTIONENABLE, &CConfigDlg::OnBnClickedBgNoisereductionenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_NOISEREDUCTIONLEVEL, &CConfigDlg::OnNMReleasedcaptureBgNoisereductionlevel)
	ON_BN_CLICKED(IDC_BG_SATURATIONENABLE, &CConfigDlg::OnBnClickedBgSaturationenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BG_SATURATIONLEVEL, &CConfigDlg::OnNMReleasedcaptureBgSaturationlevel)
	ON_BN_CLICKED(IDC_GR_ALPHAENABLE, &CConfigDlg::OnBnClickedGrAlphaenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_ALPHALEVEL, &CConfigDlg::OnNMReleasedcaptureGrAlphalevel)
	ON_BN_CLICKED(IDC_GR_ANAMORPHICSCALINGENABLE, &CConfigDlg::OnBnClickedGrAnamorphicscalingenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_ANAMORPHICSCALINGLEVEL, &CConfigDlg::OnNMReleasedcaptureGrAnamorphicscalinglevel)
	ON_BN_CLICKED(IDC_GR_BRIGHTNESSENABLE, &CConfigDlg::OnBnClickedGrBrightnessenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_BRIGHTNESSLEVEL, &CConfigDlg::OnNMReleasedcaptureGrBrightnesslevel)
	ON_BN_CLICKED(IDC_GR_CONTRASTENABLE, &CConfigDlg::OnBnClickedGrContrastenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_CONTRASTLEVEL, &CConfigDlg::OnNMReleasedcaptureGrContrastlevel)
	ON_BN_CLICKED(IDC_GR_EDGEENHANCEMENTENABLE, &CConfigDlg::OnBnClickedGrEdgeenhancementenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_EDGEENHANCEMENTLEVEL, &CConfigDlg::OnNMReleasedcaptureGrEdgeenhancementlevel)
	ON_BN_CLICKED(IDC_GR_FRAMEFORMAT, &CConfigDlg::OnBnClickedGrFrameformat)
	ON_BN_CLICKED(IDC_GR_HUEENABLE, &CConfigDlg::OnBnClickedGrHueenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_HUELEVEL, &CConfigDlg::OnNMReleasedcaptureGrHuelevel)
	ON_EN_CHANGE(IDC_GR_LUMAKEY_LOWER, &CConfigDlg::OnEnChangeGrLumakeyLower)
	ON_EN_CHANGE(IDC_GR_LUMAKEY_UPPER, &CConfigDlg::OnEnChangeGrLumakeyUpper)
	ON_BN_CLICKED(IDC_GR_LUMAKEYENABLE, &CConfigDlg::OnBnClickedGrLumakeyenable)
	ON_BN_CLICKED(IDC_GR_NOISEREDUCTIONENABLE, &CConfigDlg::OnBnClickedGrNoisereductionenable)
	ON_BN_CLICKED(IDC_GR_SATURATIONENABLE, &CConfigDlg::OnBnClickedGrSaturationenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_NOISEREDUCTIONLEVEL, &CConfigDlg::OnNMReleasedcaptureGrNoisereductionlevel)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_SATURATIONLEVEL, &CConfigDlg::OnNMReleasedcaptureSvSaturationlevel)
	ON_BN_CLICKED(IDC_SV_SATURATIONENABLE, &CConfigDlg::OnBnClickedSvSaturationenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_GR_SATURATIONLEVEL, &CConfigDlg::OnNMReleasedcaptureGrSaturationlevel)
	ON_BN_CLICKED(IDC_MV_ALPHAENABLE, &CConfigDlg::OnBnClickedMvAlphaenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_ALPHALEVEL, &CConfigDlg::OnNMReleasedcaptureMvAlphalevel)
	ON_BN_CLICKED(IDC_MV_ANAMORPHICSCALINGENABLE, &CConfigDlg::OnBnClickedMvAnamorphicscalingenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_ANAMORPHICSCALINGLEVEL, &CConfigDlg::OnNMReleasedcaptureMvAnamorphicscalinglevel)
	ON_BN_CLICKED(IDC_MV_BRIGHTNESSENABLE, &CConfigDlg::OnBnClickedMvBrightnessenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_BRIGHTNESSLEVEL, &CConfigDlg::OnNMReleasedcaptureMvBrightnesslevel)
	ON_BN_CLICKED(IDC_MV_CONTRASTENABLE, &CConfigDlg::OnBnClickedMvContrastenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_CONTRASTLEVEL, &CConfigDlg::OnNMReleasedcaptureMvContrastlevel)
	ON_BN_CLICKED(IDC_MV_EDGEENHANCEMENTENABLE, &CConfigDlg::OnBnClickedMvEdgeenhancementenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_EDGEENHANCEMENTLEVEL, &CConfigDlg::OnNMReleasedcaptureMvEdgeenhancementlevel)
	ON_BN_CLICKED(IDC_MV_FRAMEFORMAT, &CConfigDlg::OnBnClickedMvFrameformat)
	ON_BN_CLICKED(IDC_MV_HUEENABLE, &CConfigDlg::OnBnClickedMvHueenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_HUELEVEL, &CConfigDlg::OnNMReleasedcaptureMvHuelevel)
	ON_EN_CHANGE(IDC_MV_LUMAKEY_LOWER, &CConfigDlg::OnEnChangeMvLumakeyLower)
	ON_EN_CHANGE(IDC_MV_LUMAKEY_UPPER, &CConfigDlg::OnEnChangeMvLumakeyUpper)
	ON_BN_CLICKED(IDC_MV_LUMAKEYENABLE, &CConfigDlg::OnBnClickedMvLumakeyenable)
	ON_BN_CLICKED(IDC_MV_NOISEREDUCTIONENABLE, &CConfigDlg::OnBnClickedMvNoisereductionenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_NOISEREDUCTIONLEVEL, &CConfigDlg::OnNMReleasedcaptureMvNoisereductionlevel)
	ON_BN_CLICKED(IDC_MV_SATURATIONENABLE, &CConfigDlg::OnBnClickedMvSaturationenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_MV_SATURATIONLEVEL, &CConfigDlg::OnNMReleasedcaptureMvSaturationlevel)
	ON_BN_CLICKED(IDC_ST_ALPHAENABLE, &CConfigDlg::OnBnClickedStAlphaenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_ALPHALEVEL, &CConfigDlg::OnNMReleasedcaptureStAlphalevel)
	ON_BN_CLICKED(IDC_ST_ANAMORPHICSCALINGENABLE, &CConfigDlg::OnBnClickedStAnamorphicscalingenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_ANAMORPHICSCALINGLEVEL, &CConfigDlg::OnNMReleasedcaptureStAnamorphicscalinglevel)
	ON_BN_CLICKED(IDC_ST_BRIGHTNESSENABLE, &CConfigDlg::OnBnClickedStBrightnessenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_BRIGHTNESSLEVEL, &CConfigDlg::OnNMReleasedcaptureStBrightnesslevel)
	ON_BN_CLICKED(IDC_ST_CONTRASTENABLE, &CConfigDlg::OnBnClickedStContrastenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_CONTRASTLEVEL, &CConfigDlg::OnNMReleasedcaptureStContrastlevel)
	ON_BN_CLICKED(IDC_ST_EDGEENHANCEMENTENABLE, &CConfigDlg::OnBnClickedStEdgeenhancementenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_EDGEENHANCEMENTLEVEL, &CConfigDlg::OnNMReleasedcaptureStEdgeenhancementlevel)
	ON_BN_CLICKED(IDC_ST_FRAMEFORMAT, &CConfigDlg::OnBnClickedStFrameformat)
	ON_BN_CLICKED(IDC_ST_HUEENABLE, &CConfigDlg::OnBnClickedStHueenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_HUELEVEL, &CConfigDlg::OnNMReleasedcaptureStHuelevel)
	ON_EN_CHANGE(IDC_ST_LUMAKEY_LOWER, &CConfigDlg::OnEnChangeStLumakeyLower)
	ON_EN_CHANGE(IDC_ST_LUMAKEY_UPPER, &CConfigDlg::OnEnChangeStLumakeyUpper)
	ON_BN_CLICKED(IDC_ST_LUMAKEYENABLE, &CConfigDlg::OnBnClickedStLumakeyenable)
	ON_BN_CLICKED(IDC_ST_NOISEREDUCTIONENABLE, &CConfigDlg::OnBnClickedStNoisereductionenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_NOISEREDUCTIONLEVEL, &CConfigDlg::OnNMReleasedcaptureStNoisereductionlevel)
	ON_BN_CLICKED(IDC_ST_SATURATIONENABLE, &CConfigDlg::OnBnClickedStSaturationenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_ST_SATURATIONLEVEL, &CConfigDlg::OnNMReleasedcaptureStSaturationlevel)
	ON_BN_CLICKED(IDC_SV_ALPHAENABLE, &CConfigDlg::OnBnClickedSvAlphaenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_ALPHALEVEL, &CConfigDlg::OnNMReleasedcaptureSvAlphalevel)
	ON_BN_CLICKED(IDC_SV_ANAMORPHICSCALINGENABLE, &CConfigDlg::OnBnClickedSvAnamorphicscalingenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_ANAMORPHICSCALINGLEVEL, &CConfigDlg::OnNMReleasedcaptureSvAnamorphicscalinglevel)
	ON_BN_CLICKED(IDC_SV_BRIGHTNESSENABLE, &CConfigDlg::OnBnClickedSvBrightnessenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_BRIGHTNESSLEVEL, &CConfigDlg::OnNMReleasedcaptureSvBrightnesslevel)
	ON_BN_CLICKED(IDC_SV_CONTRASTENABLE, &CConfigDlg::OnBnClickedSvContrastenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_CONTRASTLEVEL, &CConfigDlg::OnNMReleasedcaptureSvContrastlevel)
	ON_BN_CLICKED(IDC_SV_EDGEENHANCEMENTENABLE, &CConfigDlg::OnBnClickedSvEdgeenhancementenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_EDGEENHANCEMENTLEVEL, &CConfigDlg::OnNMReleasedcaptureSvEdgeenhancementlevel)
	ON_BN_CLICKED(IDC_SV_FRAMEFORMAT, &CConfigDlg::OnBnClickedSvFrameformat)
	ON_BN_CLICKED(IDC_SV_HUEENABLE, &CConfigDlg::OnBnClickedSvHueenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_HUELEVEL, &CConfigDlg::OnNMReleasedcaptureSvHuelevel)
	ON_EN_CHANGE(IDC_SV_LUMAKEY_LOWER, &CConfigDlg::OnEnChangeSvLumakeyLower)
	ON_EN_CHANGE(IDC_SV_LUMAKEY_UPPER, &CConfigDlg::OnEnChangeSvLumakeyUpper)
	ON_BN_CLICKED(IDC_SV_LUMAKEYENABLE, &CConfigDlg::OnBnClickedSvLumakeyenable)
	ON_BN_CLICKED(IDC_SV_NOISEREDUCTIONENABLE, &CConfigDlg::OnBnClickedSvNoisereductionenable)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_SV_NOISEREDUCTIONLEVEL, &CConfigDlg::OnNMReleasedcaptureSvNoisereductionlevel)
	ON_BN_CLICKED(IDC_BLT_DOWNSAMPLE, &CConfigDlg::OnBnClickedBltDownsample)
	ON_NOTIFY(NM_RELEASEDCAPTURE, IDC_BLT_DOWNSAMPLELEVEL, &CConfigDlg::OnNMReleasedcaptureBltDownsamplelevel)
	END_MESSAGE_MAP()


// CConfigDlg message handlers


void CConfigDlg::OnBnClickedBgAlphaenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgAlphalevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgAnamorphicscalingenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgBrightnessenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgContrastenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgContrastlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgEdgeenhancementenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgFrameformat()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedBgHueenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgHuelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnEnChangeBgLumakeyLower()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnEnChangeBgLumakeyUpper()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnBnClickedBgLumakeyenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedBgNoisereductionenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBgSaturationenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnNMReleasedcaptureBgSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrAlphaenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaGraphics.Enable = TRUE;
	else
		alphaGraphics.Enable = FALSE;
	alphaGraphics.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_GR_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrAlphalevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaGraphics.Enable = TRUE;
	else
		alphaGraphics.Enable = FALSE;
	alphaGraphics.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_GR_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaGraphics));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrAnamorphicscalingenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingGraphics.Enable = TRUE;
	else
		anamorphicScalingGraphics.Enable = FALSE;
	anamorphicScalingGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_ANAMORPHICSCALINGLEVEL))->GetPos()) * 256 / 100;
	if (anamorphicScalingGraphics.Level <= 16)
		anamorphicScalingGraphics.Level = 16;
	if (anamorphicScalingGraphics.Level >= 64)
		anamorphicScalingGraphics.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingGraphics.Enable = TRUE;
	else
		anamorphicScalingGraphics.Enable = FALSE;
	anamorphicScalingGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_ANAMORPHICSCALINGLEVEL))->GetPos()) * 256 / 100;
	if (anamorphicScalingGraphics.Level <= 16)
		anamorphicScalingGraphics.Level = 16;
	if (anamorphicScalingGraphics.Level >= 64)
		anamorphicScalingGraphics.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingGraphics));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrBrightnessenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessGraphics.Enable = TRUE;
	else
		brightnessGraphics.Enable = FALSE;
	brightnessGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessGraphics.Enable = TRUE;
	else
		brightnessGraphics.Enable = FALSE;
	brightnessGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessGraphics));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrContrastenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastGraphics.Enable = TRUE;
	else
		contrastGraphics.Enable = FALSE;
	contrastGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrContrastlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastGraphics.Enable = TRUE;
	else
		contrastGraphics.Enable = FALSE;
	contrastGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastGraphics));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrEdgeenhancementenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementGraphics.Enable = TRUE;
	else
		edgeEnhancementGraphics.Enable = FALSE;
	edgeEnhancementGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementGraphics.Enable = TRUE;
	else
		edgeEnhancementGraphics.Enable = FALSE;
	edgeEnhancementGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementGraphics));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedGrFrameformat()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedGrHueenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueGraphics.Enable = TRUE;
	else
		hueGraphics.Enable = FALSE;
	hueGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrHuelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueGraphics.Enable = TRUE;
	else
		hueGraphics.Enable = FALSE;
	hueGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueGraphics));
	*pResult = 0;
}

void CConfigDlg::OnEnChangeGrLumakeyLower()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnEnChangeGrLumakeyUpper()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnBnClickedGrLumakeyenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedGrNoisereductionenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionGraphics.Enable = TRUE;
	else
		noiseReductionGraphics.Enable = FALSE;
	noiseReductionGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionGraphics));
}

void CConfigDlg::OnBnClickedGrSaturationenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationGraphics.Enable = TRUE;
	else
		saturationGraphics.Enable = FALSE;
	saturationGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationGraphics));
}

void CConfigDlg::OnNMReleasedcaptureGrNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionGraphics.Enable = TRUE;
	else
		noiseReductionGraphics.Enable = FALSE;
	noiseReductionGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionGraphics));

	*pResult = 0;
}

void CConfigDlg::OnNMReleasedcaptureSvSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationSubVideo.Enable = TRUE;
	else
		saturationSubVideo.Enable = FALSE;
	saturationSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvSaturationenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationSubVideo.Enable = TRUE;
	else
		saturationSubVideo.Enable = FALSE;
	saturationSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureGrSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationGraphics;
	if (((CButton *)GetDlgItem(IDC_GR_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationGraphics.Enable = TRUE;
	else
		saturationGraphics.Enable = FALSE;
	saturationGraphics.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_GR_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		2,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationGraphics));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvAlphaenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaMailwideo.Enable = TRUE;
	else
		alphaMailwideo.Enable = FALSE;
	alphaMailwideo.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_MV_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvAlphalevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaMailwideo.Enable = TRUE;
	else
		alphaMailwideo.Enable = FALSE;
	alphaMailwideo.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_MV_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvAnamorphicscalingenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingMailwideo.Enable = TRUE;
	else
		anamorphicScalingMailwideo.Enable = FALSE;
	anamorphicScalingMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_ANAMORPHICSCALINGLEVEL))->GetPos()) * 255 / 100;
	if (anamorphicScalingMailwideo.Level <= 16)
		anamorphicScalingMailwideo.Level = 16;
	if (anamorphicScalingMailwideo.Level >= 64)
		anamorphicScalingMailwideo.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingMailwideo.Enable = TRUE;
	else
		anamorphicScalingMailwideo.Enable = FALSE;
	anamorphicScalingMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_ANAMORPHICSCALINGLEVEL))->GetPos()) * 255 / 100;
	if (anamorphicScalingMailwideo.Level <= 16)
		anamorphicScalingMailwideo.Level = 16;
	if (anamorphicScalingMailwideo.Level >= 64)
		anamorphicScalingMailwideo.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvBrightnessenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessMailwideo.Enable = TRUE;
	else
		brightnessMailwideo.Enable = FALSE;
	brightnessMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessMailwideo.Enable = TRUE;
	else
		brightnessMailwideo.Enable = FALSE;
	brightnessMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvContrastenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastMailwideo.Enable = TRUE;
	else
		contrastMailwideo.Enable = FALSE;
	contrastMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvContrastlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastMailwideo.Enable = TRUE;
	else
		contrastMailwideo.Enable = FALSE;
	contrastMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvEdgeenhancementenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementMailwideo.Enable = TRUE;
	else
		edgeEnhancementMailwideo.Enable = FALSE;
	edgeEnhancementMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementMailwideo.Enable = TRUE;
	else
		edgeEnhancementMailwideo.Enable = FALSE;
	edgeEnhancementMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvFrameformat()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedMvHueenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueMailwideo.Enable = TRUE;
	else
		hueMailwideo.Enable = FALSE;
	hueMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvHuelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueMailwideo.Enable = TRUE;
	else
		hueMailwideo.Enable = FALSE;
	hueMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnEnChangeMvLumakeyLower()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnEnChangeMvLumakeyUpper()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnBnClickedMvLumakeyenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedMvNoisereductionenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionMailwideo.Enable = TRUE;
	else
		noiseReductionMailwideo.Enable = FALSE;
	noiseReductionMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionMailwideo.Enable = TRUE;
	else
		noiseReductionMailwideo.Enable = FALSE;
	noiseReductionMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedMvSaturationenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationMailwideo.Enable = TRUE;
	else
		saturationMailwideo.Enable = FALSE;
	saturationMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationMailwideo));
}

void CConfigDlg::OnNMReleasedcaptureMvSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationMailwideo;
	if (((CButton *)GetDlgItem(IDC_MV_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationMailwideo.Enable = TRUE;
	else
		saturationMailwideo.Enable = FALSE;
	saturationMailwideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_MV_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationMailwideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedStAlphaenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaSubtitle.Enable = TRUE;
	else
		alphaSubtitle.Enable = FALSE;
	alphaSubtitle.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_ST_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStAlphalevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaSubtitle.Enable = TRUE;
	else
		alphaSubtitle.Enable = FALSE;
	alphaSubtitle.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_ST_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedStAnamorphicscalingenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingSubtitle.Enable = TRUE;
	else
		anamorphicScalingSubtitle.Enable = FALSE;
	anamorphicScalingSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_ANAMORPHICSCALINGLEVEL))->GetPos()) * 256 / 100;
	if (anamorphicScalingSubtitle.Level <= 16)
		anamorphicScalingSubtitle.Level = 16;
	if (anamorphicScalingSubtitle.Level >= 64)
		anamorphicScalingSubtitle.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingSubtitle.Enable = TRUE;
	else
		anamorphicScalingSubtitle.Enable = FALSE;
	anamorphicScalingSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_ANAMORPHICSCALINGLEVEL))->GetPos()) * 256 / 100;
	if (anamorphicScalingSubtitle.Level <= 16)
		anamorphicScalingSubtitle.Level = 16;
	if (anamorphicScalingSubtitle.Level >= 64)
		anamorphicScalingSubtitle.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedStBrightnessenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessSubtitle.Enable = TRUE;
	else
		brightnessSubtitle.Enable = FALSE;
	brightnessSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	*pResult = 0;
}

void CConfigDlg::OnBnClickedStContrastenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastSubtitle.Enable = TRUE;
	else
		contrastSubtitle.Enable = FALSE;
	contrastSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStContrastlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastSubtitle.Enable = TRUE;
	else
		contrastSubtitle.Enable = FALSE;
	contrastSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedStEdgeenhancementenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementSubtitle.Enable = TRUE;
	else
		edgeEnhancementSubtitle.Enable = FALSE;
	edgeEnhancementSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementSubtitle.Enable = TRUE;
	else
		edgeEnhancementSubtitle.Enable = FALSE;
	edgeEnhancementSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedStFrameformat()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedStHueenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueSubtitle.Enable = TRUE;
	else
		hueSubtitle.Enable = FALSE;
	hueSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStHuelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueSubtitle.Enable = TRUE;
	else
		hueSubtitle.Enable = FALSE;
	hueSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnEnChangeStLumakeyLower()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnEnChangeStLumakeyUpper()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnBnClickedStLumakeyenable()
{
	// TODO: Add your control notification handler code here
}

void CConfigDlg::OnBnClickedStNoisereductionenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionSubtitle.Enable = TRUE;
	else
		noiseReductionSubtitle.Enable = FALSE;
	noiseReductionSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionSubtitle.Enable = TRUE;
	else
		noiseReductionSubtitle.Enable = FALSE;
	noiseReductionSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedStSaturationenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationSubtitle.Enable = TRUE;
	else
		saturationSubtitle.Enable = FALSE;
	saturationSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubtitle));
}

void CConfigDlg::OnNMReleasedcaptureStSaturationlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubtitle;
	if (((CButton *)GetDlgItem(IDC_ST_SATURATIONENABLE))->GetCheck() == BST_CHECKED)
		saturationSubtitle.Enable = TRUE;
	else
		saturationSubtitle.Enable = FALSE;
	saturationSubtitle.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_ST_SATURATIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		3,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubtitle));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvAlphaenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaSubVideo.Enable = TRUE;
	else
		alphaSubVideo.Enable = FALSE;
	alphaSubVideo.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_SV_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvAlphalevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_ALPHAENABLE))->GetCheck()	== BST_CHECKED)
		alphaSubVideo.Enable = TRUE;
	else
		alphaSubVideo.Enable = FALSE;
	alphaSubVideo.Alpha = (float)(((CSliderCtrl *)GetDlgItem(IDC_SV_ALPHALEVEL))->GetPos()) / 100.;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubVideo));

	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvAnamorphicscalingenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingSubVideo.Enable = TRUE;
	else
		anamorphicScalingSubVideo.Enable = FALSE;
	anamorphicScalingSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_ANAMORPHICSCALINGLEVEL))->GetPos()) * 256 / 100;
	if (anamorphicScalingSubVideo.Level <= 16)
		anamorphicScalingSubVideo.Level = 16;
	if (anamorphicScalingSubVideo.Level >= 64)
		anamorphicScalingSubVideo.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvAnamorphicscalinglevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_ANAMORPHICSCALINGENABLE))->GetCheck()	== BST_CHECKED)
		anamorphicScalingSubVideo.Enable = TRUE;
	else
		anamorphicScalingSubVideo.Enable = FALSE;
	anamorphicScalingSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_ANAMORPHICSCALINGLEVEL))->GetPos()) * 255 / 100;
	if (anamorphicScalingSubVideo.Level <= 16)
		anamorphicScalingSubVideo.Level = 16;
	if (anamorphicScalingSubVideo.Level >= 64)
		anamorphicScalingSubVideo.Level = 64;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvBrightnessenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessSubVideo.Enable = TRUE;
	else
		brightnessSubVideo.Enable = FALSE;
	brightnessSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvBrightnesslevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_BRIGHTNESSENABLE))->GetCheck() == BST_CHECKED)
		brightnessSubVideo.Enable = TRUE;
	else
		brightnessSubVideo.Enable = FALSE;
	brightnessSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_BRIGHTNESSLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvContrastenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastSubVideo.Enable = TRUE;
	else
		contrastSubVideo.Enable = FALSE;
	contrastSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvContrastlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_CONTRASTENABLE))->GetCheck() == BST_CHECKED)
		contrastSubVideo.Enable = TRUE;
	else
		contrastSubVideo.Enable = FALSE;
	contrastSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_CONTRASTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvEdgeenhancementenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementSubVideo.Enable = TRUE;
	else
		edgeEnhancementSubVideo.Enable = FALSE;
	edgeEnhancementSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvEdgeenhancementlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_EDGEENHANCEMENTENABLE))->GetCheck() == BST_CHECKED)
		edgeEnhancementSubVideo.Enable = TRUE;
	else
		edgeEnhancementSubVideo.Enable = FALSE;
	edgeEnhancementSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_EDGEENHANCEMENTLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedSvFrameformat()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_FRAMEFORMAT))->GetCheck() == BST_CHECKED)
		frameFormatSubVideo.FrameFormat = DXVAHD_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
	else
		frameFormatSubVideo.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatSubVideo));
}

void CConfigDlg::OnBnClickedSvHueenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueSubVideo.Enable = TRUE;
	else
		hueSubVideo.Enable = FALSE;
	hueSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvHuelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA hueSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_HUEENABLE))->GetCheck() == BST_CHECKED)
		hueSubVideo.Enable = TRUE;
	else
		hueSubVideo.Enable = FALSE;
	hueSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_HUELEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnEnChangeSvLumakeyLower()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnEnChangeSvLumakeyUpper()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}

void CConfigDlg::OnBnClickedSvLumakeyenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeySubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_LUMAKEYENABLE))->GetCheck() == BST_CHECKED)
		lumaKeySubVideo.Enable = TRUE;
	else
		lumaKeySubVideo.Enable = FALSE;

	lumaKeySubVideo.Lower = 0;
	lumaKeySubVideo.Upper = 0;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeySubVideo));
}

void CConfigDlg::OnBnClickedSvNoisereductionenable()
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionSubVideo.Enable = TRUE;
	else
		noiseReductionSubVideo.Enable = FALSE;
	noiseReductionSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubVideo));
}

void CConfigDlg::OnNMReleasedcaptureSvNoisereductionlevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubVideo;
	if (((CButton *)GetDlgItem(IDC_SV_NOISEREDUCTIONENABLE))->GetCheck() == BST_CHECKED)
		noiseReductionSubVideo.Enable = TRUE;
	else
		noiseReductionSubVideo.Enable = FALSE;
	noiseReductionSubVideo.Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_SV_NOISEREDUCTIONLEVEL))->GetPos()) * 255 / 100;
	gVideoData.m_pHDVP->SetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubVideo));
	*pResult = 0;
}

void CConfigDlg::OnBnClickedBltDownsample()
{
	// TODO: Add your control notification handler code here
	DXVAHD_BLT_STATE_DOWNSAMPLE_DATA downSample;
	INT Level;
	DXVAHD_BLT_STATE_TARGET_RECT_DATA TargetRect;
	gVideoData.m_pHDVP->GetVideoProcessBltState(
				DXVAHD_BLT_STATE_TARGET_RECT,
				sizeof(DXVAHD_BLT_STATE_TARGET_RECT_DATA),
				(void *)(&TargetRect));

	if (((CButton *)GetDlgItem(IDC_BLT_DOWNSAMPLEENABLE))->GetCheck() == BST_CHECKED)
		downSample.Enable = TRUE;
	else
		downSample.Enable = FALSE;

	Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_BLT_DOWNSAMPLELEVEL))->GetPos());

	downSample.Size.cx = (TargetRect.TargetRect.right - TargetRect.TargetRect.left) * Level / 100;
	downSample.Size.cy = (TargetRect.TargetRect.bottom - TargetRect.TargetRect.top) * Level / 100;;

	gVideoData.m_pHDVP->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_DOWNSAMPLE,
				sizeof(_DXVAHD_BLT_STATE_DOWNSAMPLE_DATA),
				(void *)(&downSample));
}

void CConfigDlg::OnNMReleasedcaptureBltDownsamplelevel(NMHDR *pNMHDR, LRESULT *pResult)
{
	// TODO: Add your control notification handler code here
	DXVAHD_BLT_STATE_DOWNSAMPLE_DATA downSample;
	INT Level;
	DXVAHD_BLT_STATE_TARGET_RECT_DATA TargetRect;
	gVideoData.m_pHDVP->GetVideoProcessBltState(
				DXVAHD_BLT_STATE_TARGET_RECT,
				sizeof(DXVAHD_BLT_STATE_TARGET_RECT_DATA),
				(void *)(&TargetRect));

	if (((CButton *)GetDlgItem(IDC_BLT_DOWNSAMPLEENABLE))->GetCheck() == BST_CHECKED)
		downSample.Enable = TRUE;
	else
		downSample.Enable = FALSE;

	Level = (INT)(((CSliderCtrl *)GetDlgItem(IDC_BLT_DOWNSAMPLELEVEL))->GetPos());

	downSample.Size.cx = (TargetRect.TargetRect.right - TargetRect.TargetRect.left) * Level / 100;
	downSample.Size.cy = (TargetRect.TargetRect.bottom - TargetRect.TargetRect.top) * Level / 100;;

	gVideoData.m_pHDVP->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_DOWNSAMPLE,
				sizeof(_DXVAHD_BLT_STATE_DOWNSAMPLE_DATA),
				(void *)(&downSample));
	*pResult = 0;
}
