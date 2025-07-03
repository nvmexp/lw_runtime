#define NOMINMAX
#include "UI.h"

#include "anselutils/Session.h"
#include "CommonTools.h"
#include "Utils.h"
#include "Log.h"
#include "darkroom/ImageLoader.h"
#include "i18n/LocalizedStringHelper.h"

#include <d3dcompiler.h>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h> 

#define UI_SLIDER_BOLD_CAPTIONS     0

#define SAFE_DELETE(x) if (x) { delete x; x = nullptr; }

#define UI_ATLAS_SIZE_X     2048.0f
#define UI_ATLAS_SIZE_Y     160.0f

inline float getUIAtlasCoordU(int pixelCoordX)
{
    return pixelCoordX / UI_ATLAS_SIZE_X;
}
inline float getUIAtlasCoordV(int pixelCoordY)
{
    return pixelCoordY / UI_ATLAS_SIZE_Y;
}
inline float getUIAtlasSizeU(int pixelSizeX)
{
    return pixelSizeX / UI_ATLAS_SIZE_X;
}
inline float getUIAtlasSizeV(int pixelSizeY)
{
    return pixelSizeY / UI_ATLAS_SIZE_Y;
}

namespace
{
    template<typename T> T clamp(T x, T min, T max) { return x > max ? max : x < min ? min : x; }
}

struct UIShaderConstBuf
{
    float cr, cg, cb, ca;
    float posX, posY;
    float sizeX, sizeY;
};

#if (DBG_USE_OUTLINE == 1)
struct FontOutlineShaderConstBuf
{
    float cr, cg, cb, ca;
    float outlineWidth;
};
#endif

bool AnselUI::isControlEnabled(ui::ControlContainer * control)
{
    ui::ControlContainer * cointainerTraverse = control;
    while (cointainerTraverse->m_parent)
    {
        if (!cointainerTraverse->isEnabled)
            return false;
        cointainerTraverse = cointainerTraverse->m_parent;
    }

    if (areControlsInteractionsDisabled)
    {
        // All controls are disabled
        if (!excludeContainer)
            return false;

        // Containers excluded
        // Check if excludedcontainer is among the parents of checked container
        bool containerFound = false;
        ui::ControlContainer * cointainerTraverse = control;
        while (cointainerTraverse->m_parent)
        {
            if (cointainerTraverse == excludeContainer)
            {
                containerFound = true;
                break;
            }
            cointainerTraverse = cointainerTraverse->m_parent;
        }

        if (!containerFound)
            return false;
    }

    return true;
}

void AnselUI::onSessionStarted(bool isCameraWorksSessionActive)
{
    setIsEnabled(true);
    setIsVisible(true);

    m_isCameraWorksSessionActive = isCameraWorksSessionActive;

    // Setting 'camera' icon visible (along with other controls) if CW integration is discovered and enabled
    setCWControlsVisibility(isCameraWorksSessionActive);

    // No need to hide roll slider for hybrid controller anymore, it is just treated differently
    // Hiding roll slider because it doesn't make sense for hybrid camera controller:
    //if (lwanselutils::isHybridCameraEnabled)
    //    m_components.sldRoll->isVisible = false;

    if (m_useHybridController)
    {
    }
    else
    {
        m_components.btnResetRoll->isVisible = false;
    }

    // "Kind" slider needs to be reset, as non-CW ansel can only work with regular screenshots now
    ui::ControlSliderKind * sldKind = static_cast<ui::ControlSliderKind *>(m_components.sldKind);
    sldKind->setShotPermissions(m_shotTypeEnabled);
    m_components.sldKind->setSelected(0);

    m_components.btnSnap->isEnabled = true;

    m_components.sldKind->isEnabled = m_shotTypeEnabled[(int)ShotType::k360] || m_shotTypeEnabled[(int)ShotType::k360Stereo] ||
        m_shotTypeEnabled[(int)ShotType::kHighRes] || m_shotTypeEnabled[(int)ShotType::kStereo];

#if (DISABLE_STATIC_FILTERS == 1)
    filterIDs.resize(0);
    //filterIDs[0] = L"";
#else
    filterIDs.resize(3);
    filterIDs[0] = L"";
    filterIDs[1] = m_defaultEffectPath + L"Adjustments.yaml";
    filterIDs[2] = m_defaultEffectPath + L"SpecialFX.yaml";
#endif

    m_usedGamepadForUIDuringTheSession = false;
}

void AnselUI::onSessionStopped()
{
    setIsEnabled(false);
    setIsVisible(false);

    // Disabling extended CW controls
    setCWControlsVisibility(false);

#if 0
    // This will potentially be required when we'll enable modding

    m_shotTypeEnabled[(int)ShotType::kRegular] = true;
    m_shotTypeEnabled[(int)ShotType::kRegularHDR] = true;
    m_shotTypeEnabled[(int)ShotType::kRegularUI] = true;
    m_shotTypeEnabled[(int)ShotType::k360] = false;
    m_shotTypeEnabled[(int)ShotType::k360Stereo] = false;
    m_shotTypeEnabled[(int)ShotType::kHighRes] = false;
    m_shotTypeEnabled[(int)ShotType::kStereo] = false;
#endif

    // "Kind" slider needs to be reset, as non-CW ansel can only work with regular screenshots now
    ui::ControlSliderKind * sldKind = static_cast<ui::ControlSliderKind *>(m_components.sldKind);
    sldKind->setShotPermissions(m_shotTypeEnabled);
    m_components.sldKind->setSelected(0);
}

void AnselUI::repopulateEffectsList(const std::vector<std::wstring> & filterIds, const std::vector<std::wstring> & filterNames)
{
    m_isEffectListRequested = false;

    ui::ControlFlyoutToggleShared::LabelsStorage * labelsStorage = &m_components.flySpecialFXLabels;
    size_t fxSliderPositions = filterIds.size() + 1;
    labelsStorage->labels.resize(fxSliderPositions);
    labelsStorage->ids.resize(fxSliderPositions);
    
    labelsStorage->labels[0] = m_textFilterNone;
    labelsStorage->ids[0] = shadermod::Tools::wstrNone;

    for (size_t resCnt = 0, resCntEnd = filterIds.size(); resCnt < resCntEnd; ++resCnt)
    {
        labelsStorage->ids[resCnt+1] = filterIds[resCnt];
        labelsStorage->labels[resCnt+1] = filterNames[resCnt].c_str();
    }

    // TODO avoroshilov stacking: push this for each flyout control
    ui::ControlFlyoutToggleShared * fxFlyout = static_cast<ui::ControlFlyoutToggleShared * >(m_components.flySpecialFX);
    if (size_t(fxFlyout->getSelected()) >= fxSliderPositions)
    {
        fxFlyout->setSelected(0, shadermod::Tools::wstrNone.c_str());
    }
    if (fxFlyout->needsChangeOnCWToggle())
        fxFlyout->onChange();
}

size_t AnselUI::getLwrrentFilterNum() const
{
#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        return filterIDs.size() + m_components.m_dynamicFilterContainers.size();
    }
    else
    {
        return filterIDs.size();
    }
#else
    return filterIDs.size();
#endif
}

#ifdef ENABLE_STYLETRANSFER
void AnselUI::repopulateStylesList(const std::vector<std::wstring>& styleIds, const std::vector<std::wstring> &styleNames, const std::vector<std::wstring> &stylePaths)
{
    m_isStylesListRequested = false;

    if (stylePaths.size() != styleNames.size())
    {
        assert(0);
        return;
    }

    ui::ControlFlyoutStylesToggle * flyStyles = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyles);
    size_t fxSliderPositions = styleIds.size();
    flyStyles->labelsStorage->labels.resize(fxSliderPositions);
    flyStyles->labelsStorage->ids.resize(fxSliderPositions);
    flyStyles->paths.resize(fxSliderPositions);

    if (styleIds.size() == 0)
    {
        flyStyles->labelsStorage->labels.push_back(m_textFilterNone);
        flyStyles->labelsStorage->ids.push_back(shadermod::Tools::wstrNone);
        flyStyles->paths.push_back(L"");
    }
    else
    {
        const size_t maxLabelLength = 18;
        for (size_t resCnt = 0, resCntEnd = styleIds.size(); resCnt < resCntEnd; ++resCnt)
        {
            std::wstring label = styleNames[resCnt];
            if (label.length() > maxLabelLength)
                label = label.substr(0, maxLabelLength) + L"\u2026";    // \u2026 - triple dots (...) or ellipsis

            flyStyles->labelsStorage->ids[resCnt] = styleIds[resCnt];
            flyStyles->labelsStorage->labels[resCnt] = label;
            flyStyles->paths[resCnt] = stylePaths[resCnt].c_str();
        }
    }

    int selectionId = flyStyles->getSelected() < (int)fxSliderPositions ? flyStyles->getSelected() : 0;
    flyStyles->setSelected(selectionId, flyStyles->getLabel(selectionId));

    if (flyStyles->needsChangeOnCWToggle())
        flyStyles->onChange();
}

void AnselUI::repopulateStyleNetworksList(const std::vector<std::wstring>& netIds, const std::vector<std::wstring> &netNames)
{
    m_isStyleNetworksListRequested = false;

    ui::ControlFlyoutStylesToggle * flyStyleNetworks = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyleNetworks);
    size_t fxSliderPositions = netIds.size();
    flyStyleNetworks->labelsStorage->labels.resize(fxSliderPositions);
    flyStyleNetworks->labelsStorage->ids.resize(fxSliderPositions);

    if (netIds.size() == 0)
    {
        flyStyleNetworks->labelsStorage->labels.push_back(m_textFilterNone);
        flyStyleNetworks->labelsStorage->ids.push_back(shadermod::Tools::wstrNone);
    }
    else
    {
        const size_t maxLabelLength = 18;
        for (size_t resCnt = 0, resCntEnd = netIds.size(); resCnt < resCntEnd; ++resCnt)
        {
            std::wstring label = netNames[resCnt];

            // Acquire translation
            if (label == L"Low")
            {
                label = m_textStyleNetLow;
            }
            else if (label == L"High")
            {
                label = m_textStyleNetHigh;
            }

            if (label.length() > maxLabelLength)
                label = label.substr(0, maxLabelLength) + L"\u2026";    // \u2026 - triple dots (...) or ellipsis

            flyStyleNetworks->labelsStorage->ids[resCnt] = netIds[resCnt];
            flyStyleNetworks->labelsStorage->labels[resCnt] = label;
        }
    }

    int selectionId = flyStyleNetworks->getSelected() < (int)fxSliderPositions ? flyStyleNetworks->getSelected() : 0;
    flyStyleNetworks->setSelected(selectionId, flyStyleNetworks->getSelectedLabel());

    if (flyStyleNetworks->needsChangeOnCWToggle())
        flyStyleNetworks->onChange();
}
#endif

const std::wstring AnselUI::getLwrrentFilter(size_t effectStackIdx) const
{
    if (effectStackIdx == TEMP_SELECTABLE_FILTER_ID)
    {
        ui::ControlFlyoutToggleShared * flyoutFX = static_cast<ui::ControlFlyoutToggleShared * >(m_components.flySpecialFX);
        if (flyoutFX->labelsStorage->ids.size() > 0)
            return flyoutFX->labelsStorage->ids[flyoutFX->getSelected()];
    }
    else if (effectStackIdx < filterIDs.size())
    {
        // Special case for Adjustments and SpecialFX
        return filterIDs[effectStackIdx];
    }
#if (DBG_STACKING_PROTO == 1)
    else if (effectStackIdx - filterIDs.size() < (int)m_components.m_dynamicFilterContainers.size())
    {
        if (m_allowDynamicFilterStacking)
        {
            ui::ControlDynamicFilterContainer * lwrFilterContainer = m_components.m_dynamicFilterContainers[effectStackIdx - filterIDs.size()];
            ui::ControlFlyoutToggleShared * flyoutFX = lwrFilterContainer->m_filterToggle;
            if (flyoutFX->labelsStorage->ids.size() > 0)
                return flyoutFX->labelsStorage->ids[flyoutFX->getSelected()];
        }
        else
        {
            assert(false);
            return shadermod::Tools::wstrNone;
        }
    }
#endif
    return shadermod::Tools::wstrNone;
}

#ifdef ENABLE_STYLETRANSFER
void AnselUI::mainSwitchStyleTransfer(bool status)
{
    m_isStyleTransferAllowed = status;
}

const std::wstring & AnselUI::getLwrrentStylePath() const
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle * flyStyles = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyles);
    if (flyStyles->paths.size() > 0)
        return flyStyles->paths[flyStyles->getSelected()];
#endif
    return shadermod::Tools::wstrNone;
}

const std::wstring & AnselUI::getLwrrentStyle() const
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle * flyStyles = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyles);
    if (flyStyles->labelsStorage->ids.size() > 0)
        return flyStyles->labelsStorage->ids[flyStyles->getSelected()];
#endif
    return shadermod::Tools::wstrNone;
}

const std::wstring & AnselUI::getLwrrentStyleNetwork() const
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle * flyStyleNetworks = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyleNetworks);
    if (flyStyleNetworks->labelsStorage->ids.size() > 0)
        return flyStyleNetworks->labelsStorage->ids[flyStyleNetworks->getSelected()];
#endif
    return shadermod::Tools::wstrNone;
}
void AnselUI::setLwrrentStyleNetwork(const std::wstring & netid)
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle * flyStyleNetworks = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyleNetworks);
    for (size_t i = 0, iend = flyStyleNetworks->labelsStorage->ids.size(); i < iend; ++i)
    {
        if (flyStyleNetworks->labelsStorage->ids[i] == netid)
        {
            flyStyleNetworks->setSelected((int)i, flyStyleNetworks->getLabel((int)i));
            ui::ControlFlyoutContainer * dstFlyoutContainer = m_components.flyoutPane;

            // Need to rebuild flyout pane if Style Networks flyout is open at the moment
            if (dstFlyoutContainer->srcToggle == flyStyleNetworks && dstFlyoutContainer->isVisible)
            {
                m_flyoutRebuildRequest.isValid = true;
                m_flyoutRebuildRequest.dstFlyoutContainer = dstFlyoutContainer;
                m_flyoutRebuildRequest.srcFlyoutToggle = flyStyleNetworks;
            }
        }
    }
#endif
}

void AnselUI::setStyleTransferStatus(bool isStyleTransferEnabled)
{
    m_components.chkEnableStyleTransfer->isChecked = isStyleTransferEnabled;
}

void AnselUI::showRestyleDownloadConfirmation()
{
    m_components.dlgDownloadRestyle->isVisible = true;
    areControlsInteractionsDisabled = true;
    areCameraInteractionsDisabled = true;
    excludeContainer = m_components.dlgDownloadRestyle;
}

AnselUIBase::RestyleDownloadStatus AnselUI::getRestyleDownloadConfirmationStatus()
{
    return m_restyleDownloadConfirmationStatus;
}

void AnselUI::clearRestyleDownloadConfirmationStatus()
{
    m_restyleDownloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kNone;
}

void AnselUI::toggleRestyleProgressBar(bool status)
{
    if (status == true)
    {
        m_restyleProgressBarVisGoal = 1.0f;
        m_restyleProgressBarNeedsResizing = true;
        m_components.chkEnableStyleTransfer->isEnabled = false;
    }
    else
    {
        m_restyleProgressBarVisGoal = 0.0f;
        m_restyleProgressBarNeedsResizing = true;
        m_components.chkEnableStyleTransfer->isEnabled = true;
    }
}
void AnselUI::formatRestyleProgressLabel()
{
#ifdef ENABLE_STYLETRANSFER
    if (m_restyleProgressState == RestyleProgressState::kDownloadProgress)
    {
        swprintf_s(m_components.stylePBLabel, m_components.stylePBLabelSize, m_textStyleProgress.c_str(), (int)(m_components.pbRestyleProgress->progress * 100.0f + 0.5f));
        m_components.lblRestyleProgress->caption = m_components.stylePBLabel;
    }
    else
    {
        swprintf_s(m_components.stylePBLabel, m_components.stylePBLabelSize, L"%s", m_textStyleInstalling.c_str());
        m_components.lblRestyleProgress->caption = m_components.stylePBLabel;
    }
#endif
}
void AnselUI::setRestyleProgressBarValue(float value)
{
#ifdef ENABLE_STYLETRANSFER
    if (value > 1.0f)
        value = 1.0f;
    else if (value < 0.0f)
        value = 0.0f;

    m_components.pbRestyleProgress->progress = value;
    formatRestyleProgressLabel();
#endif
}
void AnselUI::setRestyleProgressState(RestyleProgressState progressState)
{
#ifdef ENABLE_STYLETRANSFER
    m_restyleProgressState = progressState;
    formatRestyleProgressLabel();
#endif
}

void AnselUI::toggleRestyleProgressIndicator(bool status)
{
#ifdef ENABLE_STYLETRANSFER
    m_components.lblRestyleProgressIndicator->isVisible = status;
#endif
}
#endif

int AnselUI::getLwrrentFilterOldStackIdx(size_t effectStackIdx) const
{
    if (effectStackIdx < filterIDs.size())
    {
        // If static filter, then its stack index is unchanged
        return (int)effectStackIdx;
    }
#if (DBG_STACKING_PROTO == 1)
    else if (effectStackIdx - filterIDs.size() < (int)m_components.m_dynamicFilterContainers.size())
    {
        if (m_allowDynamicFilterStacking)
        {
            ui::ControlDynamicFilterContainer * lwrFilterContainer = m_components.m_dynamicFilterContainers[effectStackIdx - filterIDs.size()];
            return lwrFilterContainer->stackIdx;
        }
        else
        {
            assert(false);
            return -1;
        }
    }
#endif
    return -1;
}
void AnselUI::updateLwrrentFilterStackIndices()
{
#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        for (size_t ci = 0, ciEnd = m_components.m_dynamicFilterContainers.size(); ci < ciEnd; ++ci)
        {
            // Dynamic filter idx + static filter num
            m_components.m_dynamicFilterContainers[ci]->stackIdx = (int)(ci + filterIDs.size());
        }
    }
#endif
}

void AnselUI::clampStringNumChars(std::wstring * str, const size_t maxStrLength) const
{
    if (!str)
        return;

    if (str->length() > maxStrLength)
        *str = str->substr(0, maxStrLength) + L"\u2026";    // \u2026 - triple dots (...) or ellipsis
};

void AnselUI::clampStringFloatSize(std::wstring * str, float maxSize) const
{
    if (!str)
        return;

    float designSpaceTextW, designSpaceTextH;
    const wchar_t * rawStr = str->c_str();
    const size_t strLenTotal = str->length();

    if (strLenTotal == 0)
        return;

    // Starting guess, typically strings of length 18 (on average) are on the edge of fitting
    const size_t avgFittingStringLen = 18;
    size_t strLen = avgFittingStringLen;
    if (strLen > strLenTotal)
        strLen = strLenTotal;

    // Will show if the original guess was under- or overestimation (+1 or -1)
    enum class SearchDir
    {
        kUninit = 1,
        kIncreasing = 1,
        kDecreasing = 2
    };
    SearchDir searchDir = SearchDir::kUninit;
    while (true)
    {
        getTextRect(rawStr, strLen, true, &designSpaceTextW, &designSpaceTextH);
        if (designSpaceTextW > maxSize)
        {
            // If that's first entry then we need to decrease size to fit the label
            if (searchDir == SearchDir::kUninit)
                searchDir = SearchDir::kDecreasing;

            // The estimated text doesn't fit, we need to roll back 1 character
            --strLen;

            // If the original search direction was +ve, then we've found correct size
            if (searchDir == SearchDir::kIncreasing)
                break;

            if (strLen == 0)
                break;              // Something went wrong, and none of the label could fit
        }
        else
        {
            // If that's first entry then we need to increase size to fit the label
            if (searchDir == SearchDir::kUninit)
                searchDir = SearchDir::kIncreasing;

            // The estimated text just fits, se we need to report length prior to increasing
            // If the original search direction was -ve, then we've found correct size
            if (searchDir == SearchDir::kDecreasing)
                break;

            ++strLen;

            if (strLen >= strLenTotal)
                break;              // Whole label could fit into the desired maxSize
        }
    }
    clampStringNumChars(str, strLen);
};

void AnselUI::removeDynamicElement(ui::ControlContainer * element, ui::ControlContainer ** selectedControl, bool * needToReselectControl)
{
    if (*selectedControl != element)
    {
        *needToReselectControl = true;
    }
    else
    {
        // If one of the sliders was selected, we need to drop the selection
        //  otherwise selection issues might happen if controls are updated when mouse is over one of them
        containerHelper.setSelectedControl(m_components.btnSnap, *selectedControl);
        *selectedControl = m_components.btnSnap;
    }

    ui::ControlContainer * parentContainer = element->m_parent;
    releaseDynamicControl(element);
    parentContainer->removeControlFast(element);
}

void AnselUI::removeDynamicElementsInContainer(ui::ControlContainer * neededContainer, ui::ControlContainer ** selectedControl, bool * needToReselectControl)
{
    assert(selectedControl && needToReselectControl);

    // Fast deletion, swap with last element and delete

    ui::ControlContainer * lwrContainer;
    containerHelper.startSearchHierarchical(&mainContainer);

    // The needToReselectControl is needed in case control was selected in a container that is being cleaned up
    //  in this case, selection indices become outdated, and we need to update them y reselecting the same control
    while (lwrContainer = containerHelper.getNextControlHierarchical())
    {
        bool containerStays = true;

        ui::ControlContainer * controlToCheck = lwrContainer;
        while (controlToCheck)
        {
            // If control belongs to the required container, and also is dynamic
            if (controlToCheck->m_parent == neededContainer && (!controlToCheck->isStatic))
            {
                containerStays = false;
                break;
            }
            controlToCheck = controlToCheck->m_parent;
        }

        if (containerStays)
        {
            if (*selectedControl == lwrContainer)
            {
                *needToReselectControl = true;
            }
            continue;
        }

        // If nested containers are selected
        controlToCheck = *selectedControl;
        while (controlToCheck)
        {
            // If control belongs to the required (selected) container
            if (controlToCheck == lwrContainer)
            {
                // If one of the sliders was selected, we need to drop the selection
                //  otherwise selection issues might happen if controls are updated when mouse is over one of them
                containerHelper.setSelectedControl(m_components.btnSnap, *selectedControl);
                *selectedControl = m_components.btnSnap;
                break;
            }
            controlToCheck = controlToCheck->m_parent;
        }

        // It should be safe to remove control at this point, since the list is already built
        if (lwrContainer->getControlsNum() != 0)
            containerHelper.skipChildrenHierarchical();
        containerHelper.reportElementDeletedHierarchical();
        ui::ControlContainer * parentContainer = lwrContainer->m_parent;
        releaseDynamicControl(lwrContainer);
        parentContainer->removeControlFast(lwrContainer);
    }
    containerHelper.stopSearchHierarchical();
}

void AnselUI::updateContainerControls(size_t effectStackIdx, ui::ControlContainer * neededContainer, int neededBlockID, EffectPropertiesDescription * effectDesc)
{
    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

    bool needReselectControl = false;
    removeDynamicElementsInContainer(neededContainer, &selectedControl, &needReselectControl);
    if (needReselectControl)
    {
        // Reselecting the same control to update selection indices
        containerHelper.setSelectedControl(selectedControl, selectedControl);
    }

    // TODO avoroshilov UI
    //  we need to rebuild controls list here
    containerHelper.rebuildControlsArray(&mainContainer);

    std::wstring tmpString;
    for (unsigned int uccnt = 0, ucend = effectDesc ? ((unsigned int)effectDesc->attributes.size()) : 0; uccnt < ucend; ++uccnt)
    {
        const EffectPropertiesDescription::EffectAttributes & lwrAttrib = effectDesc->attributes[uccnt];

        bool shouldBeVisible = true;

        if (effectStackIdx != GameSpecificStackIdx)
        {
            if (effectStackIdx < filterIDs.size())
                shouldBeVisible = (filterIDs[effectStackIdx] != shadermod::Tools::wstrNone);
#if (DBG_STACKING_PROTO == 1)
            else if (m_allowDynamicFilterStacking)
            {
                int dynamicFilterId = (int)effectStackIdx - (int)filterIDs.size();
                assert(dynamicFilterId >= 0);

                const std::wstring & selectedId = m_components.m_dynamicFilterContainers[dynamicFilterId]->m_filterToggle->getSelectedIdStr();
                shouldBeVisible = (selectedId != shadermod::Tools::wstrNone);
            }
#endif
        }

        if (lwrAttrib.controlType == AnselUIBase::ControlType::kSlider)
        {
            tmpString = lwrAttrib.displayName;
            clampStringFloatSize(&tmpString, m_storedSizes.defaultSliderSizeX * 0.8f);

            if (lwrAttrib.dataType == AnselUIBase::DataType::kFloat)
            {
                ui::ControlSliderEffectTweak * lwrSlider = getNewDynamicControlMem<ui::ControlSliderEffectTweak>();
                new (lwrSlider) ui::ControlSliderEffectTweak(&m_onSliderEffectTweakChange);
                lwrSlider->isStatic = false;

                lwrSlider->dataType = lwrAttrib.dataType;

                lwrSlider->filterId = effectDesc->filterId;
                lwrSlider->controlIdx = lwrAttrib.controlId;
                lwrSlider->stackIdx = (int)effectStackIdx;                  // TODO: remove this field, infer the value instead

                lwrSlider->setTitle(tmpString.c_str(), tmpString.length());

                float milwalue[MAX_GROUPED_VARIABLE_DIMENSION], maxValue[MAX_GROUPED_VARIABLE_DIMENSION];
                lwrAttrib.milwalue.get(milwalue, MAX_GROUPED_VARIABLE_DIMENSION);
                lwrAttrib.maxValue.get(maxValue, MAX_GROUPED_VARIABLE_DIMENSION);

                lwrSlider->baseValue = 0.5f;

                float stickyValue, stickyRegion;
                float uiMilwalue[MAX_GROUPED_VARIABLE_DIMENSION], uiMaxValue[MAX_GROUPED_VARIABLE_DIMENSION], stepSizeUI[MAX_GROUPED_VARIABLE_DIMENSION];

                stickyValue = lwrAttrib.stickyValue;
                stickyRegion = lwrAttrib.stickyRegion;

                // TODO: jingham Fix this for grouped values. Will be fixed for standalone UI support of the new UI elements, which is not immediately critical.
                lwrAttrib.uiMilwalue.get(uiMilwalue, MAX_GROUPED_VARIABLE_DIMENSION);
                lwrAttrib.uiMaxValue.get(uiMaxValue, MAX_GROUPED_VARIABLE_DIMENSION);
                lwrAttrib.stepSizeUI.get(stepSizeUI, MAX_GROUPED_VARIABLE_DIMENSION);
                float stepSize = lwrAttrib.getStepSize(stepSizeUI[0], milwalue[0], maxValue[0], uiMilwalue[0], uiMaxValue[0]);

                lwrSlider->stickyValue = stickyValue;
                lwrSlider->stickyRegion = stickyRegion;
                lwrSlider->step = stepSize / (maxValue - milwalue);

                lwrSlider->milwalue = milwalue[0];
                lwrSlider->maxValue = maxValue[0];

                lwrSlider->uiMilwalue = uiMilwalue[0];
                lwrSlider->uiMaxValue = uiMaxValue[0];

                lwrSlider->uiMeasurementUnit = lwrAttrib.uiMeasurementUnit;

                float defaultValue[MAX_GROUPED_VARIABLE_DIMENSION], lwrrentValue[MAX_GROUPED_VARIABLE_DIMENSION];
                lwrAttrib.defaultValue.get(defaultValue, MAX_GROUPED_VARIABLE_DIMENSION);
                lwrAttrib.lwrrentValue.get(lwrrentValue, MAX_GROUPED_VARIABLE_DIMENSION);

                lwrSlider->defaultValue = ui::colwertValueToPercentage(defaultValue[0], 
                    0.5f, milwalue[0], maxValue[0]);
                lwrSlider->percentage = ui::colwertValueToPercentage(lwrrentValue[0],
                    0.5f, milwalue[0], maxValue[0]);

                lwrSlider->setValue(lwrSlider->percentage);

                // TODO avoroshilov: move init into a separate function
                lwrSlider->state = UI_CONTROL_ORDINARY;
                // So we want to save selectedEffect in order to not rebuild it if user switches between the two: None/Current
                //  this way we want to keep selectedEffect, and it can load up sliders while shader mod is actually not active
                //  thus this check
                //  (however this should go away once we'll introduce proper resources caching system and decrease effects building times)

                // TODO avoroshilov UIA
                //  double-check this by dragging filter slider violently

                lwrSlider->isVisible = shouldBeVisible;
                lwrSlider->pIndexBuf = pRectIndexBuf;
                lwrSlider->pVertexBuf = pRectVertexBuf;

                lwrSlider->blockID = neededBlockID;

                lwrSlider->tabStop = TABSTOP_INIT * 2;

                // TODO avoroshilov: add tab stopping here as well
                neededContainer->addControl(lwrSlider);
            }
            else if (lwrAttrib.dataType == AnselUIBase::DataType::kInt || lwrAttrib.dataType == AnselUIBase::DataType::kBool)
            {
                ui::ControlSliderIntEffectTweak * lwrSlider = getNewDynamicControlMem<ui::ControlSliderIntEffectTweak>();
                new (lwrSlider) ui::ControlSliderIntEffectTweak(&m_onSliderIntEffectTweakChange);
                lwrSlider->isStatic = false;

                lwrSlider->dataType = lwrAttrib.dataType;

                lwrSlider->filterId = effectDesc->filterId;
                lwrSlider->controlIdx = lwrAttrib.controlId;
                lwrSlider->stackIdx = (int)effectStackIdx;          // TODO: remove this field, infer the value instead

                lwrSlider->setTitle(tmpString.c_str(), tmpString.length());

                int defaultValue, lwrrentValue;
                if (lwrAttrib.dataType == AnselUIBase::DataType::kInt)
                {
                    int milwalue, maxValue;
                    lwrAttrib.milwalue.get(&milwalue, 1);
                    lwrAttrib.maxValue.get(&maxValue, 1);

                    lwrSlider->m_milwal = milwalue;
                    lwrSlider->m_maxVal = maxValue;

                    lwrSlider->m_step = 1;// TODO: jingham fix this: lwrAttrib.getStepSize(); Will be fixed for standalone UI support of the new UI elements, which is not immediately critical.
                    if (lwrSlider->m_step < 1)
                        lwrSlider->m_step = 1;

                    lwrAttrib.defaultValue.get(&defaultValue, 1);
                    lwrAttrib.lwrrentValue.get(&lwrrentValue, 1);
                }
                else
                {
                    lwrSlider->m_milwal = 0;
                    lwrSlider->m_maxVal = 1;
                    lwrSlider->m_step = 1;

                    bool defaultValueBool, lwrrentValueBool;
                    lwrAttrib.defaultValue.get(&defaultValueBool, 1);
                    lwrAttrib.lwrrentValue.get(&lwrrentValueBool, 1);

                    defaultValue = defaultValueBool ? 1 : 0;
                    lwrrentValue = lwrrentValueBool ? 1 : 0;
                }
                lwrSlider->defaultValue = defaultValue;
                lwrSlider->calcSelectedFromInt(lwrrentValue);
                lwrSlider->setValue(lwrSlider->calcIntFromSelected());

                lwrSlider->uiMeasurementUnit = lwrAttrib.uiMeasurementUnit;

                // TODO avoroshilov: move init into a separate function
                lwrSlider->state = UI_CONTROL_ORDINARY;
                // So we want to save selectedEffect in order to not rebuild it if user switches between the two: None/Current
                //  this way we want to keep selectedEffect, and it can load up sliders while shader mod is actually not active
                //  thus this check
                //  (however this should go away once we'll introduce proper resources caching system and decrease effects building times)

                // TODO avoroshilov UIA
                //  double-check this by dragging filter slider violently

                lwrSlider->isVisible = shouldBeVisible;
                lwrSlider->pIndexBuf = pRectIndexBuf;
                lwrSlider->pVertexBuf = pRectVertexBuf;

                lwrSlider->blockID = neededBlockID;

                lwrSlider->tabStop = TABSTOP_INIT * 2;

                // TODO avoroshilov: add tab stopping here as well
                neededContainer->addControl(lwrSlider);
            }
        }
        else if (lwrAttrib.controlType == AnselUIBase::ControlType::kCheckbox)
        {
            ui::ControlCheckboxEffectTweak * lwrCheckbox = getNewDynamicControlMem<ui::ControlCheckboxEffectTweak>();
            new (lwrCheckbox) ui::ControlCheckboxEffectTweak(&m_onCheckboxEffectTweakChange);
            lwrCheckbox->isStatic = false;

            tmpString = lwrAttrib.displayName;

            lwrCheckbox->dataType = lwrAttrib.dataType;

            lwrCheckbox->filterId = effectDesc->filterId;
            lwrCheckbox->controlIdx = lwrAttrib.controlId;
            lwrCheckbox->stackIdx = (int)effectStackIdx;                    // TODO: remove this field, infer the value instead

            clampStringFloatSize(&tmpString, m_storedSizes.defaultCheckboxSizeX * 0.9f);
            lwrCheckbox->setTitle(tmpString.c_str(), tmpString.length());

            lwrCheckbox->milwalue = lwrAttrib.milwalue;
            lwrCheckbox->maxValue = lwrAttrib.maxValue;

            lwrCheckbox->isChecked = lwrCheckbox->checkLwrrentValue(lwrAttrib.lwrrentValue);
            lwrCheckbox->setValue();

            // TODO avoroshilov: move init into a separate function
            lwrCheckbox->state = UI_CONTROL_ORDINARY;
            // So we want to save selectedEffect in order to not rebuild it if user switches between the two: None/Current
            //  this way we want to keep selectedEffect, and it can load up sliders while shader mod is actually not active
            //  thus this check
            //  (however this should go away once we'll introduce proper resources caching system and decrease effects building times)

            // TODO avoroshilov UIA
            //  double-check this by dragging filter slider violently

            lwrCheckbox->isVisible = shouldBeVisible;

            lwrCheckbox->blockID = neededBlockID;

            lwrCheckbox->tabStop = TABSTOP_INIT * 2;

            // TODO avoroshilov: add tab stopping here as well
            neededContainer->addControl(lwrCheckbox);
        }
        else if (lwrAttrib.controlType == AnselUIBase::ControlType::kColorPicker)
        {
            ui::ControlColorPickerEffectTweak * lwrColorPicker = getNewDynamicControlMem<ui::ControlColorPickerEffectTweak>();

            unsigned int numChannels = lwrAttrib.defaultValue.getDimensionality();
            new (lwrColorPicker) ui::ControlColorPickerEffectTweak(&m_onColorPickerEffectTweakChange, numChannels);
            lwrColorPicker->isStatic = false;

            tmpString = lwrAttrib.displayName;

            lwrColorPicker->dataType = lwrAttrib.dataType;

            lwrColorPicker->filterId = effectDesc->filterId;
            lwrColorPicker->controlIdx = lwrAttrib.controlId;
            lwrColorPicker->stackIdx = (int)effectStackIdx;                 // TODO: remove this field, infer the value instead

            float milwalue[4], maxValue[4];
            lwrAttrib.milwalue.get(milwalue, 4);
            lwrAttrib.maxValue.get(maxValue, 4);

            float defaultValue[4], lwrrentValue[4];
            lwrAttrib.defaultValue.get(defaultValue, 4);
            lwrAttrib.lwrrentValue.get(lwrrentValue, 4);

            ui::ControlColorPickerEffectTweak::SliderTweak * sliders[4];
            for (unsigned int ch = 0; ch < numChannels; ++ch)
            {
                sliders[ch] = getNewDynamicControlMem<ui::ControlColorPickerEffectTweak::SliderTweak>();
                new (sliders[ch]) ui::ControlColorPickerEffectTweak::SliderTweak();
                
                lwrColorPicker->m_sliders[ch] = sliders[ch];
            }
            lwrColorPicker->initSliders();

            for (unsigned int ch = 0; ch < numChannels; ++ch)
            {
                sliders[ch]->baseValue = 0.5f;

                sliders[ch]->defaultValue = ui::colwertValueToPercentage(defaultValue[ch], 
                    0.5f, milwalue[ch], maxValue[ch]);
                sliders[ch]->percentage = ui::colwertValueToPercentage(lwrrentValue[ch],
                    0.5f, milwalue[ch], maxValue[ch]);

                sliders[ch]->stickyValue = defaultValue[ch];
                sliders[ch]->stickyRegion = 0.01f;
                sliders[ch]->step = 0.5f / (maxValue - milwalue);

                sliders[ch]->milwalue = milwalue[ch];
                sliders[ch]->maxValue = maxValue[ch];

                sliders[ch]->uiMilwalue = milwalue[ch];
                sliders[ch]->uiMaxValue = maxValue[ch];

                sliders[ch]->uiMeasurementUnit = L"";

                sliders[ch]->setValue(sliders[ch]->percentage);

                sliders[ch]->tabStop = TABSTOP_INIT * (ch + 1);
                
                const wchar_t* channelLiteralsRGBA[] = { L".R", L".G", L".B", L".A" };
                const wchar_t* channelLiteralsXY[] = { L".X", L".Y" };
                std::wstring sliderTitle = tmpString;

                if (numChannels > 2)
                    sliderTitle += channelLiteralsRGBA[ch];
                else
                    sliderTitle += channelLiteralsXY[ch];

                clampStringFloatSize(&sliderTitle, m_storedSizes.defaultSliderSizeX * 0.8f);
                sliders[ch]->setTitle(sliderTitle.c_str(), sliderTitle.length());
            }

            // TODO avoroshilov: move init into a separate function
            lwrColorPicker->state = UI_CONTROL_ORDINARY;
            // So we want to save selectedEffect in order to not rebuild it if user switches between the two: None/Current
            //  this way we want to keep selectedEffect, and it can load up sliders while shader mod is actually not active
            //  thus this check
            //  (however this should go away once we'll introduce proper resources caching system and decrease effects building times)

            // TODO avoroshilov UIA
            //  double-check this by dragging filter slider violently

            lwrColorPicker->isVisible = shouldBeVisible;
            lwrColorPicker->setBuffers(pRectIndexBuf, pRectVertexBuf);

            lwrColorPicker->setBlockID(neededBlockID);

            //lwrColorPicker->setTitle(tmpString.c_str(), tmpString.length());

            lwrColorPicker->tabStop = TABSTOP_INIT * 2;

            // TODO avoroshilov: add tab stopping here as well
            neededContainer->addControl(lwrColorPicker);
        }
        else if (lwrAttrib.controlType == AnselUIBase::ControlType::kFlyout || lwrAttrib.controlType == AnselUIBase::ControlType::kRadioButton)
        {
            tmpString = lwrAttrib.displayName;
            // Lean slider - top label is of full width
            clampStringFloatSize(&tmpString, m_storedSizes.defaultSliderSizeX * 0.95f);

            ui::ControlSliderListEffectTweak * lwrSliderList = getNewDynamicControlMem<ui::ControlSliderListEffectTweak>();
            new (lwrSliderList) ui::ControlSliderListEffectTweak(&m_onSliderListEffectTweakChange);
            lwrSliderList->isStatic = false;

            lwrSliderList->isLeanStyle = false;

            lwrSliderList->dataType = lwrAttrib.dataType;

            lwrSliderList->filterId = effectDesc->filterId;
            lwrSliderList->controlIdx = lwrAttrib.controlId;
            lwrSliderList->stackIdx = (int)effectStackIdx;              // TODO: remove this field, infer the value instead

            lwrSliderList->setTitle(tmpString.c_str(), tmpString.length());

            int defaultOption = lwrAttrib.userConstant->getDefaultListOption();
            uint32_t optionsNum = lwrAttrib.userConstant->getNumListOptions();

            // Conservative, there will be no less than this amount of characters
            size_t numCharacters = 0;
            for (uint32_t oi = 0; oi < optionsNum; ++oi)
            {
                numCharacters += lwrAttrib.userConstant->getListOptionNameLocalized(oi, m_langID).size() + 1;
            }

            lwrSliderList->allocLabels(numCharacters);
            wchar_t * labelsMem = lwrSliderList->labelsMem;
            for (uint32_t oi = 0; oi < optionsNum; ++oi)
            {
                const std::string & lwrLabelUTF8 = lwrAttrib.userConstant->getListOptionNameLocalized(oi, m_langID);
                std::wstring lwrLabel = darkroom::getWstrFromUtf8(lwrLabelUTF8);

                clampStringFloatSize(&lwrLabel, m_storedSizes.defaultSliderSizeX);

                size_t lwrLabelLen = lwrLabel.length() + 1;

                lwrSliderList->data.push_back(lwrAttrib.userConstant->getListOption(oi));
                lwrSliderList->labels.push_back(labelsMem);
                swprintf_s(labelsMem, lwrLabelLen, L"%s", lwrLabel.c_str());

                labelsMem += lwrLabelLen;
            }

            // Includes SetValue
            lwrSliderList->setSelected(defaultOption);

            // TODO avoroshilov: move init into a separate function
            lwrSliderList->state = UI_CONTROL_ORDINARY;
            // So we want to save selectedEffect in order to not rebuild it if user switches between the two: None/Current
            //  this way we want to keep selectedEffect, and it can load up sliders while shader mod is actually not active
            //  thus this check
            //  (however this should go away once we'll introduce proper resources caching system and decrease effects building times)

            // TODO avoroshilov UIA
            //  double-check this by dragging filter slider violently

            lwrSliderList->isVisible = shouldBeVisible;
            lwrSliderList->pIndexBuf = pRectIndexBuf;
            lwrSliderList->pVertexBuf = pRectVertexBuf;

            lwrSliderList->blockID = neededBlockID;

            lwrSliderList->tabStop = TABSTOP_INIT * 2;

            // TODO avoroshilov: add tab stopping here as well
            neededContainer->addControl(lwrSliderList);
        }
    }

    applyTabStop();
    containerHelper.rebuildControlsArray(&mainContainer);

    // Update freshly created sliders positions
    float aspect = m_width / (float)m_height;
    recallwlateUILayout(aspect);

    // Reselecting the same control to update selection indices (since freshly added elements could makle indices obsolete)
    containerHelper.setSelectedControl(selectedControl, selectedControl);
}

void AnselUI::updateEffectControls(size_t effectStackIdx, EffectPropertiesDescription * effectDesc)
{
    ui::ControlContainer * neededContainer = m_components.cntFilter;
    // We don't really need blockID anymore, we can operate on parents instead
    int neededBlockID = TEMP_DYNAMIC_BLOCKID;
#if (DISABLE_STATIC_FILTERS == 1)
    // Nothing to do here
#else
    if (effectStackIdx == TEMP_SELECTABLE_FILTER_ID)
    {
        neededBlockID = TEMP_DYNAMIC_BLOCKID;
        neededContainer = m_components.cntFilter;

        if (effectDesc == nullptr || effectDesc->filterId == shadermod::Tools::wstrNone)
        {
            // If the pointer is NULL, then error happened
            m_components.flySpecialFX->setSelected(0, m_components.flySpecialFX->getLabel(0));
        }
    }
    else if (effectStackIdx == TEMP_ADJUSTMENTS_ID)
    {
        neededContainer = m_components.cntAdjustments;
        neededBlockID = ADJUSTMENTS_BLOCKID;
    }
    else if (effectStackIdx == TEMP_FX_ID)
    {
        neededContainer = m_components.cntFX;
        neededBlockID = FX_BLOCKID;
    }
    else
#endif
#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        int dynamicFilterIdx = (int)effectStackIdx - (int)filterIDs.size();
        neededBlockID = DYNAMIC_FILTERS_BLOCKID + dynamicFilterIdx;
        neededContainer = m_components.m_dynamicFilterContainers[dynamicFilterIdx];

        if (effectDesc == nullptr || effectDesc->filterId == shadermod::Tools::wstrNone)
        {
            ui::ControlFlyoutToggleShared * flyFilterType = m_components.m_dynamicFilterContainers[dynamicFilterIdx]->m_filterToggle;
            flyFilterType->setSelected(0, flyFilterType->getLabel(0));
        }
    }
    else
    {
        return;
    }
#endif

    if (effectStackIdx < filterIDs.size() && effectDesc != nullptr)
    {
        filterIDs[effectStackIdx] = effectDesc->filterId;
    }

    updateContainerControls(effectStackIdx, neededContainer, neededBlockID, effectDesc);
}

void AnselUI::updateGameSpecificControls(EffectPropertiesDescription * effectDesc)
{
    if (effectDesc == nullptr)
    {
        // This shouldn't really happen
        isGameSettingsPanelDisplayed = false;
        m_components.btnToggleGameSpecific->isVisible = false;
        m_components.cntGameSpecific->isVisible = false;

        // Update freshly created sliders positions
        float aspect = m_width / (float)m_height;
        recallwlateUILayout(aspect);

        LOG_WARN("Game settings description pointer is null");

        return;
    }

    if (effectDesc->attributes.size() == 0)
    {
        isGameSettingsPanelDisplayed = false;
        m_components.btnToggleGameSpecific->isVisible = false;
    }
    else
    {
        isGameSettingsPanelDisplayed = true;
        m_components.btnToggleGameSpecific->isVisible = true;
    }

    updateContainerControls(TEMP_GAMESPECIFIC_ID, m_components.cntGameSpecific, GAMESPEC_BLOCKID, effectDesc);
}

void AnselUI::setFovControlEnabled(bool enabled)
{
    m_fovChangeAllowed = enabled;
}

void AnselUI::setFOVLimitsDegrees(double lo, double hi)
{
    static_cast<ui::ControlSliderFOV *>(m_components.sldFOV)->setFOVLimits(lo, hi);
}

void AnselUI::setRollLimitsDegrees(double lo, double hi)
{
    static_cast<ui::ControlSliderRoll *>(m_components.sldRoll)->setRollRangeDegrees(lo, hi);
}

void AnselUI::set360WidthLimits(uint64_t lo, uint64_t hi)
{
    ui::ControlSlider360Quality * sld360Quality = static_cast<ui::ControlSlider360Quality * >(m_components.sldSphereFOV);
    sld360Quality->setResolutionLimits(lo, hi);
    sld360Quality->onChange();
}

bool AnselUI::isGridOfThirdsEnabled() const
{
    return m_components.chkGridOfThirds->isChecked;
}

double AnselUI::getFOVDegrees() const
{
    return static_cast<ui::ControlSliderFOV *>(m_components.sldFOV)->getFOV();
}
double AnselUI::getRollDegrees() const
{
    return (m_components.sldRoll->percentage - 0.5f) * 360.0f;
}
bool AnselUI::processFOVChange()
{
    return m_components.sldFOV->processChange();
}
bool AnselUI::processRollChange()
{
    return m_components.sldRoll->processChange();
}
bool AnselUI::getCameraDragActive()
{
    return isCameraDragActive;
}
bool AnselUI::isCameraInteractive()
{
    return !areCameraInteractionsDisabled;
}

void AnselUI::setFOVDegrees(double fov)
{
    static_cast<ui::ControlSliderFOV *>(m_components.sldFOV)->setFOV((float)fov);
}
void AnselUI::setRollDegrees(double roll)
{
    static_cast<ui::ControlSliderRoll *>(m_components.sldRoll)->setRollDegrees((float)roll);
}

void AnselUI::advanceState()
{
    m_stateLwrrent = (m_stateLwrrent + 1) % kStateCount;

    switch (m_stateLwrrent)
    {
    case kStateTransitionToSession:
    {
#if 0
        bool ok = startSession();
        if (!ok)
        {
            m_stateRequested = kStateDeactive;
            m_stateLwrrent = kStateDeactive;
            m_stateDuration = 0.0f;
            setFadeValue(0.0f);
        }
        else
#else
        m_isAnselStartRequested = true;
        m_isEffectListRequested = true;
#ifdef ENABLE_STYLETRANSFER
        m_isStylesListRequested = true;
        m_isStyleNetworksListRequested = true;
#endif
#endif
        {
            // start fading in Ansel
            m_stateDuration = m_fadeEnabled ? FADE_IN_DURATION_IN_SECONDS : 0.0f;
            m_stateTime = 0.0f;

            //initRegistryDependentPathsAndOptions();
            m_inputstate.init();
        }
    }
    break;
    case kStateSessionActive:
        m_stateDuration = 0.0f;
        setFadeValue(0.0f);
        break;
    case kStateTransitionFromSession:
        // start fading out Ansel
        m_stateDuration = m_fadeEnabled ? FADE_OUT_DURATION_IN_SECONDS : 0.0f;
        m_stateTime = 0.0f;
        break;
    case kStateTransitionToDeactive:
        m_stateDuration = 0.0f;

#if 0
        stopSession();
#else
        m_isAnselStopRequested = true;
#endif
        hideFloatingContainers();

        // disable input stealing
        m_inputstate.deinit();
        break;
        // we need one frame of black to hide previous ansel frame,
        // session is only really restored on the next frame
        setFadeValue(1.0f);
    case kStateDeactive:
        m_isAnselPoststopRequested = true;
        setFadeValue(0.0f);
        m_stateDuration = 0.0f;
        break;
    }

}

void AnselUI::updateState(double dt)
{
    if (m_stateLwrrent != m_stateRequested)
    {
        // first see if advance is gated by timer:
        if (m_stateDuration)
        {
            m_stateTime += float(dt / 1000.0);
            if (m_stateTime >= m_stateDuration)
            {
                advanceState();
            }
        }
        else
            advanceState();

        if (m_stateDuration)
        {
            float x = m_stateTime / m_stateDuration;

            if (m_stateLwrrent == kStateTransitionToSession)
                x = 1.0f - x;

            x /= 0.75f; // 25% of the time we stay at black

#define M_PI_UI 3.1415926536f
            float fade;
            if (x > 1.0f)
                fade = 1.0f;
            else
                fade = 0.5f - 0.5f*cosf(x*float(M_PI_UI));
#undef M_PI_UI

            setFadeValue(fade);
        }
    }
}

void AnselUI::forceImmediateStateChange()
{
    while (m_stateLwrrent != m_stateRequested)
        advanceState();
}

void AnselUI::update(double dt)
{
    updateState(dt);
}

void AnselUI::camWorksToggle()
{
    // change state
    m_stateRequested = isEnabled() ? kStateDeactive : kStateSessionActive;
    if (m_stateRequested == kStateSessionActive)
    {
        m_isAnselPrestartRequested = true;
    }
}
void AnselUI::checkToggleCamWorks()
{
    if (m_needToggleCamWorks)
    {
        camWorksToggle();
        m_needToggleCamWorks = false;
    }
}

bool AnselUI::isAnselPrestartRequested()    { return m_isAnselPrestartRequested; }
bool AnselUI::isAnselStartRequested()       { return m_isAnselStartRequested; }
// right now standalone UI doesn't have a shortlwt to enter a mode 
// where the Ansel session is started, but the game is not on pause
// even if that's Ansel SDK integrated game
bool AnselUI::isAnselSDKSessionRequested()  { return true; }
bool AnselUI::isAnselStopRequested()        { return m_isAnselStopRequested; }
bool AnselUI::isAnselPoststopRequested()    { return m_isAnselPoststopRequested; }
bool AnselUI::isAnselFeatureSetRequested() { return false; }

void AnselUI::anselPrestartDone(AnselUIBase::Status status, bool isSDKDetected, bool requireSDK)
{
    m_isSDKDetected = isSDKDetected;
    m_isAnselPrestartRequested = false;

    if (requireSDK && !m_isSDKDetected)
    {
        m_stateRequested = kStateDeactive;
        forceImmediateStateChange();
        return;
    }
}
void AnselUI::anselStartDone(AnselUIBase::Status status)
{
    m_isAnselStartRequested = false;

    // In case Ansel didn't want the interface to be loaded
    //  and we need to send tear down commands in order to keep proper state
    if (status == AnselUIBase::Status::kDeclined || status == AnselUIBase::Status::kUnknown)
    {
        m_stateRequested = kStateDeactive;
        forceImmediateStateChange();
        return;
    }
}
void AnselUI::anselStopDone(AnselUIBase::Status status)
{
    m_isAnselStopRequested = false;
}
void AnselUI::anselPoststopDone(AnselUIBase::Status status)
{
    RegistrySettings & anselRegistrySettings = m_pAnselServer->getRegistrySettings();
    anselRegistrySettings.setValue(anselRegistrySettings.registryPathAnselWrite(), L"EnableEnhancedHighres", m_components.chkEnhanceHiRes->isChecked);
    m_isAnselPoststopRequested = false;
}

void AnselUI::forceEnableUI()
{
    if (!isEnabled())
    {
        m_needToggleCamWorks = true;
    }
}
void AnselUI::forceDisableUI()
{
    if (isEnabled())
    {
        m_needToggleCamWorks = true;
    }
}

bool AnselUI::isResetRollNeeded() const
{
    return m_needToResetRoll;
}
void AnselUI::resetRollDone()
{
    m_needToResetRoll = false;
}
void AnselUI::setResetRollStatus(bool isAvailable)
{
    if (isAvailable)
    {
        m_components.btnResetRoll->isEnabled = true;
        m_components.btnResetRoll->color = m_doneDoneColor;
    }
    else
    {
        m_components.btnResetRoll->isEnabled = false;
        checkSelectedControl();
        m_components.btnResetRoll->color = m_doneDoneColor;
        m_components.btnResetRoll->color.val[3] = 0.3f;
    }
}

bool AnselUI::isSDKCaptureAbortRequested()
{
    return m_isSDKCaptureAbortRequested;
}
void AnselUI::sdkCaptureAbortDone(int status)
{
    m_isSDKCaptureAbortRequested = false;
}

bool AnselUI::queryIsModdingEnabled()
{
    return true;
    // do nothing in Standalone
}

bool AnselUI::isModdingEnabled()
{
    // This is where we enable/disable modding in Standalone
    // For now we're enabling it for internal purposes. The DLL still can be unloaded
    // by looking at IPCenabled setting
    return m_isStandaloneModdingEnabled;
}

bool AnselUI::isModdingAllowed()
{
    if (m_isStandaloneModdingEnabled && m_components.chkAllowModding)
    {
        ui::ControlCheckbox * chkAllowModding = static_cast<ui::ControlCheckbox *>(m_components.chkAllowModding);
        return chkAllowModding->isChecked;
    }
    else
        return false;
}

void AnselUI::processAnselStatus(const AnselSDKState & sdkState, bool * needToSkipRestFrame, int * anselSessionStateChangeRequest, bool * forceNotSkipRestFrame)
{
    *needToSkipRestFrame = false;
    *anselSessionStateChangeRequest = -1;

    bool isWindowForeground = true;
    {
        HWND hGameWnd = nullptr;
        if (sdkState.isDetected() && sdkState.isConfigured())
            hGameWnd = static_cast<HWND>(sdkState.getConfiguration().gameWindowHandle);

        isWindowForeground = !hGameWnd || (hGameWnd == GetForegroundWindow());
    }

#if 0
    bool hotkeyModifierDown = (0 != (0x8000 & GetAsyncKeyState(m_pAnselServer->m_hotkeyModifier)));
#else
    bool hotkeyModifierDown = true;
    if (m_pAnselServer->m_toggleHotkeyModCtrl && (0 == (0x8000 & GetAsyncKeyState(VK_CONTROL))))
        hotkeyModifierDown = false;
    if (m_pAnselServer->m_toggleHotkeyModShift && (0 == (0x8000 & GetAsyncKeyState(VK_SHIFT))))
        hotkeyModifierDown = false;
    if (m_pAnselServer->m_toggleHotkeyModAlt && (0 == (0x8000 & GetAsyncKeyState(VK_MENU))))
        hotkeyModifierDown = false;
#endif
    bool altKeyDown = (0 != (0x8000 & GetAsyncKeyState(VK_MENU)));
    bool ctrlKeyDown = (0 != (0x8000 & GetAsyncKeyState(VK_CONTROL)));
    bool toggleKeyDown = (0 != (0x8000 & GetAsyncKeyState(m_pAnselServer->m_toggleAnselHotkey)));
    bool f12KeyDown = (0 != (0x8000 & GetAsyncKeyState(VK_F12)));

    bool toggleKeyChanged = (hotkeyModifierDown != m_hotkeyModifierDown) || (toggleKeyDown != m_toggleKeyDown);
    bool toggleKeyPressed = (hotkeyModifierDown && toggleKeyDown);

    bool forceToggleAnsel = false;

    // This part of code is about DISABLING (stopping session and hiding UI until next enable) Ansel CW/UI
    if (forceToggleAnsel || (isWindowForeground && toggleKeyChanged && toggleKeyPressed))
    {
        // TODO avoroshilov: if we just enabled Ansel, we need to skip second check for "isOtherOverlayActive"
        //  so that Ansel doesn't turn on and then immediately shuts down, this can lead to wrong behavior
        if (!isOtherOverlayActive())
        {
            // This variable is needed, because we want to toggle CamWorks slightly later
            //  so we get all the new surface parameters (width/height) from the shim, before we enable CW
            //  (see below)
            m_needToggleCamWorks = true;
            *forceNotSkipRestFrame = true;
        }
    }
    else if (!isWindowForeground && toggleKeyChanged && toggleKeyPressed)
    {
        LOG_WARN("Ansel toggle hotkey combination registered while the Ansel-enabled app is out of focus");
    }


    bool disableKeyChanged = (altKeyDown != m_altKeyDown) || (f12KeyDown != m_f12KeyDown) || (ctrlKeyDown != m_ctrlKeyDown);
    bool disableKeyPressed = (altKeyDown && ctrlKeyDown && f12KeyDown);

    // This part of code is about DEACTIVATING Ansel
    //  i.e. when user presses CTRL+ALT+F12 and ExelwtePostProcessing should be omitted
    //  in the future, this will differ from DISABLING Ansel CW/UI - because we will be able
    //  to apply post processing even when UI and/or CW is not enabled
    if (isWindowForeground && disableKeyChanged && disableKeyPressed)
    {
        m_isAnselActive = !m_isAnselActive;
        if (!m_isAnselActive)
        {
            if (isActive())
            {
                // We need to disable session if there was any
                //camWorksToggle();
                anselSessionStateChangeRequest = 0;
            }
            //return S_OK;
            *needToSkipRestFrame = true;
            return;
        }
    }

    m_hotkeyModifierDown = hotkeyModifierDown;
    m_altKeyDown = altKeyDown;
    m_ctrlKeyDown = ctrlKeyDown;
    m_toggleKeyDown = toggleKeyDown;
    m_f12KeyDown = f12KeyDown;
}

void AnselUI::dormantUpdate(const AnselSDKState & sdkState, bool * needToAbortCapture, bool * needToSkipRestFrame)
{
    *needToSkipRestFrame = false;
    if (isOtherOverlayActive())
    {
        // Dormant Ansel
        if (sdkState.isDetected() && isEnabled())
        {
            if (sdkState.getCaptureState() != CAPTURE_NOT_STARTED)
            {
                //abortCapture();
                *needToAbortCapture = true;
                //m_isSDKCaptureAbortRequested = true;
            }
        }

        m_inputstate.dormantUpdate();

        //return S_OK;
        *needToSkipRestFrame = true;
        return;
    }
}

void AnselUI::highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries)
{
    m_isHighResolutionRecalcRequested = false;

    int32_t maxMultiplier = (int32_t)highResEntries.size();
    if ((int)m_components.sldHiResMult->labels.size() != maxMultiplier)
    {
        if (m_components.sldHiResMult->labels.size() > 0)
            delete[] m_components.sldHiResMult->labels[0];

        wchar_t * stringBufs = new wchar_t[m_components.sldHiResMultStringBufSize * maxMultiplier];

        m_components.sldHiResMult->labels.resize(maxMultiplier);

        for (int resCnt = 0; resCnt < maxMultiplier; ++resCnt)
        {
            m_components.sldHiResMult->labels[resCnt] = stringBufs + resCnt*m_components.sldHiResMultStringBufSize;
        }
    }

    for (int resCnt = 0; resCnt < maxMultiplier; ++resCnt)
    {
        const HighResolutionEntry & hrEntry = highResEntries[resCnt];
        double totalFileSizeGB = hrEntry.byteSize / 1024. / 1024. / 1024.;

        const size_t strDimBufSize = 32;
        wchar_t strDimX[strDimBufSize];
        wchar_t strDimY[strDimBufSize];

        lwanselutils::buildSplitStringFromNumber(hrEntry.width, strDimX, strDimBufSize);
        lwanselutils::buildSplitStringFromNumber(hrEntry.height, strDimY, strDimBufSize);

        swprintf_s(m_components.sldHiResMult->labels[resCnt], m_components.sldHiResMultStringBufSize, L"%s \u00D7 %s\n(\u00D7%d)  %.1f%s", strDimX, strDimY, resCnt + 2, totalFileSizeGB, m_textGB.c_str());
    }
}

void AnselUI::setShotType(ShotType type)
{
    m_shotToTake = type;
}

bool AnselUI::isVisible() const
{
    return m_isVisible;
}
void AnselUI::setIsVisible(bool isVisible)
{
    m_isVisible = isVisible;
}

bool AnselUI::isEnabled() const
{
    return m_isEnabled;
}
void AnselUI::setIsEnabled(bool isEnabled)
{
    m_isEnabled = isEnabled;
}

void AnselUI::setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize)
{
    if (shotTypeEnabled != NULL)
    {
        memcpy(m_shotTypeEnabled, shotTypeEnabled, (int)ShotType::kNumEntries * sizeof(bool));
    }

    if (m_shotHDREnabled != shotHDREnabled)
    {
        LOG_DEBUG(" allowedHDR set from %s to %s", m_shotHDREnabled ? "true" : "false", shotHDREnabled ? "true" : "false");
        m_shotHDREnabled = shotHDREnabled;
        m_components.chkHDR->isEnabled = m_shotHDREnabled;
    }
}

void AnselUI::setScreenSize(uint32_t w, uint32_t h)
{
    m_width = w;
    m_height = h;

    if (resolutionBaseWidth != w || resolutionBaseHeight != h)
    {
        ui::ControlBase * lwrContainer;
        containerHelper.startSearch();
        while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kIcon, false))
        {
            ui::ControlIcon * lwrIcon = static_cast<ui::ControlIcon *>(lwrContainer);
            lwrIcon->resize(w, h);
        }
        containerHelper.stopSearch();

        float aspect = w / (float)h;
        recallwlateUILayout(aspect);

        m_isHighResolutionRecalcRequested = true;
    }
}

// Whether UI receives messages and processes them
bool AnselUI::isActive() const
{
    return isEnabled();
}

bool AnselUI::doBlockMessageTransfer() const
{
    // TODO:
    //return isActive() && ((mouseStates.mode == UI_MOUSEMODE_COMBINED) || (AnselSDK.DLLfound && AnselSDK.pCWSession));
    return false;
}

bool AnselUI::isControlInteractiveHierarchical(ui::ControlContainer * controlToCheck)
{
    bool isSelectedInteractive = true;
    while (controlToCheck)
    {
        if (!controlToCheck->isInteractive() || !isControlEnabled(controlToCheck))
        {
            isSelectedInteractive = false;
            break;
        }
        controlToCheck = controlToCheck->m_parent;
    }

    return isSelectedInteractive;
}

void AnselUI::selectNextElement()
{
    // TODO avoroshilov UI
    //  this does't look good, 

    // First, get the selected control
    ui::ControlContainer * oldSelectedControl = containerHelper.getSelectedControl(&mainContainer);
    containerHelper.startSearch(containerHelper.findLinearIndex(oldSelectedControl));
    ui::ControlContainer * lwrContainer = nullptr;
    ui::ControlContainer * newSelectedControl = nullptr;

    while (true)
    {
        lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL & (~(int)ui::ControlType::kContainer), true);
        if (!lwrContainer)
            break;
        
        newSelectedControl = lwrContainer;
        // If selected control state didn't change to inactive, first call to getNextControl
        //  will return previously selected control, in this case we need to select another one
        if (newSelectedControl == oldSelectedControl)
            newSelectedControl = containerHelper.getNextControl((int)ui::ControlType::kALL & (~(int)ui::ControlType::kContainer), true);

        // If new found control is non-interactive, we skip it
        if (!isControlInteractiveHierarchical(newSelectedControl))
        {
            newSelectedControl = nullptr;
            continue;
        }

        // Found control passed all the checks
        break;
    }
    containerHelper.stopSearch();

    // Then, set it as new selected control
    if (newSelectedControl)
        containerHelper.setSelectedControl(newSelectedControl, oldSelectedControl);
}

void AnselUI::selectPrevElement()
{
    ui::ControlContainer * oldSelectedControl = containerHelper.getSelectedControl(&mainContainer);
    containerHelper.startSearch(containerHelper.findLinearIndex(oldSelectedControl));
    ui::ControlContainer * lwrContainer = nullptr;
    ui::ControlContainer * newSelectedControl = nullptr;

    while (true)
    {
        lwrContainer = containerHelper.getPrevControl((int)ui::ControlType::kALL & (~(int)ui::ControlType::kContainer), true);
        if (!lwrContainer)
            break;

        newSelectedControl = lwrContainer;
        if (newSelectedControl == oldSelectedControl)
            newSelectedControl = containerHelper.getPrevControl((int)ui::ControlType::kALL & (~(int)ui::ControlType::kContainer), true);

        if (!isControlInteractiveHierarchical(newSelectedControl))
        {
            newSelectedControl = nullptr;
            continue;
        }

        break;
    }
    containerHelper.stopSearch();

    if (newSelectedControl)
        containerHelper.setSelectedControl(newSelectedControl, oldSelectedControl);
}

void AnselUI::checkSelectedControl()
{
    // TODO avoroshilov UI
    //  probably makes sense to have a variable storing selected control

    // Determine the right container
    ui::ControlContainer * selectedContainer = containerHelper.getSelectedControl(&mainContainer);
    ui::ControlContainer * lwrContainer = selectedContainer;

    if (!isControlInteractiveHierarchical(lwrContainer))
    {
        if (containerHelper.m_searchInProgress)
        {
            // We're in the middle fo container traverse, so we cannot initiate second search
            //  just select any control that is available

            if (isControlInteractiveHierarchical(m_components.btnSnap))
            {
                containerHelper.setSelectedControl(m_components.btnSnap, selectedContainer);
            }
            else if (excludeContainer)
            {
                ui::ControlContainer * newSelectedContainer = excludeContainer;
                while (newSelectedContainer->getType() == ui::ControlType::kContainer)
                {
                    if (newSelectedContainer->getControlsNum() > 0)
                    {
                        newSelectedContainer = newSelectedContainer->getControl(0);
                    }
                    else
                    {
                        break;
                    }
                }
                if (newSelectedContainer)
                    containerHelper.setSelectedControl(newSelectedContainer, selectedContainer);
            }
            else
            {
                // Container reselection failed
                //  but that might be ok if every control is disabled
                //assert(false);
            }
        }
        else
        {
            selectNextElement();
        }
    }
}

void AnselUI::setCWControlsVisibility(bool visibility)
{
    if (m_components.icoCamera)
        m_components.icoCamera->isVisible = visibility;
    m_components.sldFOV->isVisible = visibility;
    m_components.sldFOV->isEnabled = m_fovChangeAllowed;
    m_components.sldRoll->isVisible = visibility;

    m_components.chkHDR->isEnabled = m_shotHDREnabled;

    // Slider "Kind" couldn't be set to anything but regular screenshot upon CW integration toggle
    m_components.sldHiResMult->isVisible = false;
    m_components.sldHiResMult->isEnabled = false;
    m_components.sldSphereFOV->isVisible = false;
    m_components.sldSphereFOV->isEnabled = false;

    m_components.chkEnhanceHiRes->isVisible = false;
    m_components.chkEnhanceHiRes->isEnabled = false;

    checkSelectedControl();
}

void AnselUI::setAnselSDKDetected(bool detected)
{
    m_enableAnselSDKUI = detected;
}

void AnselUI::onSliderKindChange(void * object)
{
    auto sldKind = static_cast<ui::ControlSliderKind*>(m_components.sldKind);

    auto isResolutionSliderVisible = [&sldKind]()
    {
        return sldKind->getSelected() == UI_DIRCETORSTATE_HIGHRES;
    };

    auto is360QualitySliderVisible = [&sldKind]()
    {
        return (sldKind->getSelected() == UI_DIRCETORSTATE_360) || (sldKind->getSelected() == UI_DIRCETORSTATE_STEREO_360);
    };

    if (!m_pAnselServer->m_anselSDK.isDetected())
    {
        sldKind->setSelectedRaw(0);
    }
    else
    {
        if (!sldKind->isShotTypeEnabled(sldKind->getSelected()))
        {
            sldKind->setSelectedRaw(sldKind->isShotTypeEnabled(sldKind->getPrevSelected()) ? sldKind->getPrevSelected() : 0);
        }
    }

    m_components.sldHiResMult->isEnabled = false;
    m_components.sldSphereFOV->isEnabled = false;
    m_components.sldHiResMult->isVisible = false;
    m_components.sldSphereFOV->isVisible = false;
    m_components.chkEnhanceHiRes->isVisible = false;
    m_components.chkEnhanceHiRes->isEnabled = false;

    if (isResolutionSliderVisible())
    {
        m_components.sldHiResMult->onChange();
        m_components.sldHiResMult->isVisible = true;
        m_components.sldHiResMult->isEnabled = true;
        m_components.chkEnhanceHiRes->isVisible = true;
        m_components.chkEnhanceHiRes->isEnabled = true;
        m_components.sldSphereFOV->isVisible = false;
    }

    if (is360QualitySliderVisible())
    {
        m_components.sldSphereFOV->onChange();
        m_components.sldSphereFOV->isVisible = true;
        m_components.sldSphereFOV->isEnabled = true;
        m_components.sldHiResMult->isVisible = false;
        m_components.chkEnhanceHiRes->isVisible = false;
    }

    checkSelectedControl();

    m_needToRecallwILayout = true;
}

void AnselUI::onFlyoutSpecialFXChange(void * object)
{
    // TODO avoroshilov UI
    //  we can limit search to a sub-tree here

    // Function that hides in teh parent every control except given
    // TODO avoroshilov UI: use isStatic field instead of "flyout" pointer
    auto setTweakingSlidersState = [&](bool state, ui::ControlContainer * flyout, ui::ControlContainer * parent)
    {
        containerHelper.startSearch();
        ui::ControlContainer * lwrContainer;
        while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
        {
            if (lwrContainer != flyout && lwrContainer->m_parent == parent)
            {
                lwrContainer->isEnabled = state;
                lwrContainer->isVisible = state;
            }
        }
        containerHelper.stopSearch();
    };

    ui::ControlFlyoutToggleShared * flyoutFX = reinterpret_cast<ui::ControlFlyoutToggleShared *>(object);

    // None
    if (flyoutFX->getSelected() == 0)
    {
        setTweakingSlidersState(false, flyoutFX, flyoutFX->m_parent);
        flyoutFX->lastSelectedLabel = L'\0';
        return;
    }

    setTweakingSlidersState(true, flyoutFX, flyoutFX->m_parent);
    flyoutFX->lastSelectedLabel = flyoutFX->labelsStorage->labels[flyoutFX->getSelected()];
}

#if (DBG_STACKING_PROTO == 1)
void AnselUI::onFlyoutSpecialFXChangeDynamic(void * object)
{
    assert(m_allowDynamicFilterStacking);

    onFlyoutSpecialFXChange(object);

    ui::ControlFlyoutToggleShared * flyoutFX = reinterpret_cast<ui::ControlFlyoutToggleShared *>(object);
    assert(flyoutFX->m_parent);
    if (!flyoutFX->m_parent)
        return;

    ui::ControlDynamicFilterContainer * dynamicFilterContainer = static_cast<ui::ControlDynamicFilterContainer *>(flyoutFX->m_parent);
    std::wstring filterName = flyoutFX->getSelectedLabel();
    // GlyphWidth takes part in the toggle size too, but it is hardcoded constant at the moment, hence hand-tuned multiplier
    clampStringFloatSize(&filterName, (m_storedSizes.defaultDynamicToggleSizeX - dynamicFilterContainer->m_toggleButton->m_glyphMargin) * 0.82f);
    swprintf_s(dynamicFilterContainer->filterName, dynamicFilterContainer->maxFilterNameSize, L"%s", filterName.c_str());
    if (dynamicFilterContainer->filterName == m_textFilterNone)
    {
        dynamicFilterContainer->m_toggleButton->caption = m_textFilter.c_str();
    }
    else
    {
        dynamicFilterContainer->m_toggleButton->caption = dynamicFilterContainer->filterName;
    }

    checkUpDownButtons();
}
#endif

#ifdef ENABLE_STYLETRANSFER
void AnselUI::onFlyoutStylesChange(void * object)
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle* flyStyles = static_cast<ui::ControlFlyoutStylesToggle*>(m_components.flyStyles);

    // None
    if (flyStyles->getSelected() == 0)
    {
        //setTweakingSlidersState(false, TEMP_DYNAMIC_BLOCKID);
        flyStyles->lastSelectedLabel = L'\0';
        return;
    }

    //setTweakingSlidersState(true, TEMP_DYNAMIC_BLOCKID);
    flyStyles->lastSelectedLabel = flyStyles->labelsStorage->labels[flyStyles->getSelected()];
#endif
}

void AnselUI::onFlyoutStyleNetworksChange(void * object)
{
#ifdef ENABLE_STYLETRANSFER
    ui::ControlFlyoutStylesToggle * flyStyles = static_cast<ui::ControlFlyoutStylesToggle*>(m_components.flyStyles);

    // None
    if (flyStyles->getSelected() == 0)
    {
        //setTweakingSlidersState(false, TEMP_DYNAMIC_BLOCKID);
        flyStyles->lastSelectedLabel = L'\0';
        return;
    }

    //setTweakingSlidersState(true, TEMP_DYNAMIC_BLOCKID);
    flyStyles->lastSelectedLabel = flyStyles->labelsStorage->labels[flyStyles->getSelected()];
#endif
}

bool AnselUI::isUIInteractionActive()
{
    return isUIInteractActive;
}
#endif

void AnselUI::onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
    const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
    const input::FolwsChecker& folwsChecker, const input::MouseTrapper& mouseTrapper)
{
    bool ignoreMouse = !mouseTrapper.isMouseInClientArea();
        
    input::EDPadDirection::Enum         nextControlDir = input::EDPadDirection::kDown,
        prevControlDir = input::EDPadDirection::kUp,
        increaseDir = input::EDPadDirection::kRight,
        decreaseDir = input::EDPadDirection::kLeft;

    
    //////////////////////
    // Process gamepad
    input::EDPadDirection::Enum dpadDir = gpadSt.getDpadDirection();

    if (isEnabled())
    {
        bool elementChanged = false;
        ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

        if (gpadSt.isDpadDirectionChanged())
        {
            if (dpadDir == prevControlDir)
            {
                selectPrevElement();
                elementChanged = true;
                gamepadFilterTime = UI_GAMEPAD_DPAD_FIRST_TIME_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == nextControlDir)
            {
                selectNextElement();
                elementChanged = true;
                gamepadFilterTime = UI_GAMEPAD_DPAD_FIRST_TIME_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == increaseDir)
            {
                selectedControl->onIncrease();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FIRST_TIME_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == decreaseDir)
            {
                selectedControl->onDecrease();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FIRST_TIME_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
        }

        // TODO avoroshilov UI
        //  do we really need this? in case of simultaneous press, maybe we don't care
        if (elementChanged)
        {
            selectedControl = containerHelper.getSelectedControl(&mainContainer);
        }

        if (gpadSt.isButtonStateChangedToDown(input::EGamepadButton::kA) ||
            gpadSt.isButtonStateChangedToDown(input::EGamepadButton::kB) ||
            gpadSt.isButtonStateChangedToDown(input::EGamepadButton::kX) ||
            gpadSt.isButtonStateChangedToDown(input::EGamepadButton::kY))
        {
            selectedControl->onClick();
            m_usedGamepadForUIDuringTheSession = true;
        }
    }

    //Process mouse

    if (!ignoreMouse)
    {
        uint32_t width = m_pAnselServer->getWidth();
        uint32_t height = m_pAnselServer->getHeight();

        float dmz = 0.0f;

        if (!isCameraDragActive && ((mouseSt.getLastCoordX() || mouseSt.getLastCoordY() || mouseSt.getLastCoordWheel()) && width && height))
        {
            // * 2.0f is needed since normalized space is -1..1
            mouseCoordsAbsX += (float)mouseSt.getLastCoordX() / (float)width * mouseSensititvityUI * 2.0f;
            mouseCoordsAbsY += -(float)mouseSt.getLastCoordY() / (float)height * mouseSensititvityUI * 2.0f;

            dmz = (float)mouseSt.getLastCoordWheel() / mouseWheelSensitivityUI;

            if (mouseCoordsAbsX > 1.0f)
                mouseCoordsAbsX = 1.0f;
            if (mouseCoordsAbsX < -1.0f)
                mouseCoordsAbsX = -1.0f;

            if (mouseCoordsAbsY > 1.0f)
                mouseCoordsAbsY = 1.0f;
            if (mouseCoordsAbsY < -1.0f)
                mouseCoordsAbsY = -1.0f;
        }

        mouseLwrsor = UI_MOUSELWRSOR_DONTCARE;

        // Broadcast mouseMove and mouseUp events
        input::EMouseButton::Enum primaryMouseButton = input::EMouseButton::kLButton;
        if (m_areMouseButtonsSwapped)
        {
            primaryMouseButton = input::EMouseButton::kRButton;
        }
        bool mouseDownThisTime = mouseSt.isButtonStateChangedToDown(primaryMouseButton);
        bool mouseUpThisTime = mouseSt.isButtonStateChangedToUp(primaryMouseButton);

        if (mouseUpThisTime)
        {
            isCameraDragActive = false;
#ifdef ENABLE_STYLETRANSFER
            isUIInteractActive = false;
#endif
        }

        if (isEnabled())
        {
            ui::ControlContainer * newSelectedControl = nullptr;
            ui::ControlContainer * oldSelectedControl = containerHelper.getSelectedControl(&mainContainer);

            containersStack.resize(0);
            isMouseOverStack.resize(0);
            std::vector<int> & offsetsStack = containerHelper.offsetsStack;
            containersStack.push_back(&mainContainer);
            isMouseOverStack.push_back(mainContainer.isMouseOver(mouseCoordsAbsX - mainContainer.absPosX, mouseCoordsAbsY - mainContainer.absPosY) != 0);

            containerHelper.startSearchHierarchical(&mainContainer);
            // This basically is fused jumpToNextControlHierarchical/getNextControlHierarchical
            while (true)
            {
                if (containersStack.size() == 0)
                    break;
                ui::ControlContainer * lwrContainerHierarchical = containersStack.back();
                bool isMouseOver = isMouseOverStack.back();

                int offsetIdx = (int)offsetsStack.size() - 1;
                int offset = offsetsStack.back();

                if (offset == -1)
                {
                    ++offsetsStack.back();
                    continue;
                }

                if (!lwrContainerHierarchical->isVisible)
                {
                    if (offset == 0)
                    {
                        // We're looking at parent container, to skip it hierarchically, we can just set offsets
                        offsetsStack.back() = (int)lwrContainerHierarchical->getControlsNum();
                        offset = offsetsStack.back();
                    }
                }

                float scrollValueY = 0.0f;

                {
                    ui::ControlContainer * cointainerTraverse = lwrContainerHierarchical;
                    while (cointainerTraverse->m_parent)
                    {
                        scrollValueY += cointainerTraverse->m_parent->m_scrollValueY;
                        cointainerTraverse = cointainerTraverse->m_parent;
                    }
                }

                if (offset < int(lwrContainerHierarchical->getControlsNum()))
                {
                    ++offsetsStack.back();
                    ui::ControlContainer * childContainer = lwrContainerHierarchical->getControl(offset);

                    if ((childContainer->isInteractive() && isControlEnabled(childContainer)) && childContainer->onMouseMove(mouseCoordsAbsX - childContainer->absPosX, mouseCoordsAbsY - (childContainer->absPosY + scrollValueY + lwrContainerHierarchical->m_scrollValueY), dmz))
                    {
                        // Selection shouldn't happen while mouse is down
                        if (isMouseOver && !mouseSt.isButtonDown(primaryMouseButton))
                        {
                            if (!childContainer->isBasicContainer())
                            {
                                newSelectedControl = childContainer;

                                int preferredLwrsor = newSelectedControl->getPreferredLwrsor();
                                if (preferredLwrsor != UI_MOUSELWRSOR_DONTCARE)
                                {
                                    mouseLwrsor = preferredLwrsor;
                                }
                            }
                        }
                    }

                    if (mouseUpThisTime)
                        childContainer->onMouseUp(mouseCoordsAbsX - childContainer->absPosX, mouseCoordsAbsY - (childContainer->absPosY + scrollValueY + lwrContainerHierarchical->m_scrollValueY));

                    if (lwrContainerHierarchical->isChildBasicContainer(offset))
                    {
                        offsetsStack.push_back(-1);
                        containersStack.push_back(childContainer);
                        isMouseOverStack.push_back(isMouseOver && childContainer->isMouseOver(mouseCoordsAbsX - childContainer->absPosX, mouseCoordsAbsY - (childContainer->absPosY + scrollValueY + lwrContainerHierarchical->m_scrollValueY)));
                        continue;
                    }
                    else
                    {
                        continue;
                    }
                }

                // We're at the container, skip everything that it contains
                offsetsStack.pop_back();
                containersStack.pop_back();
                isMouseOverStack.pop_back();
            }
            containerHelper.stopSearchHierarchical();

            if (newSelectedControl && (newSelectedControl != oldSelectedControl))
            {
                containerHelper.setSelectedControl(newSelectedControl, oldSelectedControl);
            }
            else
            {
                // Set newSelectedControl to the old value for further reuse
                newSelectedControl = oldSelectedControl; 
            }

            if (mouseDownThisTime)
            {
                bool isControlHit = false;
                ui::ControlContainer * hitContainer = nullptr;

                containersStack.resize(0);
                isMouseOverStack.resize(0);
                containersStack.push_back(&mainContainer);
                isMouseOverStack.push_back(mainContainer.isMouseOver(mouseCoordsAbsX - mainContainer.absPosX, mouseCoordsAbsY - mainContainer.absPosY) != 0);
                containerHelper.startSearchHierarchical(&mainContainer);

                // This basically is fused jumpToNextControlHierarchical/getNextControlHierarchical
                while (true)
                {
                    if (containersStack.size() == 0)
                        break;
                    ui::ControlContainer * lwrContainerHierarchical = containersStack.back();
                    bool isMouseOver = isMouseOverStack.back();

                    int offsetIdx = (int)offsetsStack.size() - 1;
                    int offset = offsetsStack.back();

                    if (offset == -1)
                    {
                        ++offsetsStack.back();
                        continue;
                    }

                    if (!lwrContainerHierarchical->isVisible)
                    {
                        if (offset == 0)
                        {
                            // We're looking at parent container, to skip it hierarchically, we can just set offsets
                            offset = offsetsStack.back() = (int)lwrContainerHierarchical->getControlsNum();
                        }
                    }

                    float scrollValueY = 0.0f;

                    {
                        ui::ControlContainer * cointainerTraverse = lwrContainerHierarchical;
                        while (cointainerTraverse->m_parent)
                        {
                            scrollValueY += cointainerTraverse->m_parent->m_scrollValueY;
                            cointainerTraverse = cointainerTraverse->m_parent;
                        }
                    }

                    if (offset < int(lwrContainerHierarchical->getControlsNum()))
                    {
                        ++offsetsStack.back();
                        ui::ControlContainer * childContainer = lwrContainerHierarchical->getControl(offset);
                        if (lwrContainerHierarchical->isChildBasicContainer(offset))
                        {
                            offsetsStack.push_back(-1);
                            containersStack.push_back(childContainer);
                            isMouseOverStack.push_back(isMouseOver && childContainer->isMouseOver(mouseCoordsAbsX - childContainer->absPosX, mouseCoordsAbsY - (childContainer->absPosY + scrollValueY + lwrContainerHierarchical->m_scrollValueY)));
                            continue;
                        }
                        else
                        {
                            if ((childContainer->isInteractive() && isControlEnabled(childContainer)) && isMouseOver && childContainer->onMouseDown(mouseCoordsAbsX - childContainer->absPosX, mouseCoordsAbsY - (childContainer->absPosY + scrollValueY + lwrContainerHierarchical->m_scrollValueY)))
                            {
                                isControlHit = true;
                                hitContainer = childContainer;
                                break;
                            }
                            continue;
                        }
                    }

                    if ((lwrContainerHierarchical->isInteractive() && isControlEnabled(lwrContainerHierarchical)) && isMouseOver && lwrContainerHierarchical->onMouseDown(mouseCoordsAbsX - lwrContainerHierarchical->absPosX, mouseCoordsAbsY - (lwrContainerHierarchical->absPosY + scrollValueY)))
                    {
                        isControlHit = true;
                        hitContainer = lwrContainerHierarchical;
                        break;
                    }
                    // We're at the container, skip everything that it contains
                    offsetsStack.pop_back();
                    containersStack.pop_back();
                    isMouseOverStack.pop_back();
                }
                containerHelper.stopSearchHierarchical();

                if (!isControlHit)
                {
                    // Enable mouse movement for the camera
                    isCameraDragActive = true;
                    hideFloatingContainers();
                }
                else
                {
                    // Check if floating container is among the parents of hit container. If it is, we don't need to forcefully hide it.
                    for (size_t fci = 0, fciEnd = floatingContainers.size(); fci < fciEnd; ++fci)
                    {
                        bool containerFound = false;
                        ui::ControlContainer * floatingContainer = floatingContainers[fci].first;
                        ui::ControlContainer * cointainerTraverse = hitContainer;
                        while (cointainerTraverse->m_parent)
                        {
                            // Making use of the related control (stored as "second") - do not hide if the governing control is hit
                            if (cointainerTraverse == floatingContainer || cointainerTraverse == floatingContainers[fci].second)
                            {
                                containerFound = true;
                                break;
                            }
                            cointainerTraverse = cointainerTraverse->m_parent;
                        }

                        if (!containerFound)
                        {
                            hideFloatingContainer(floatingContainer);

                            // Theoretically, no need to check selection, since if other control was clicked
                            // it should be selected
                        }
                    }
                    
#ifdef ENABLE_STYLETRANSFER
                    ui::ControlContainer * lwrContainer = containerHelper.getSelectedControl(&mainContainer);
                    if (lwrContainer && (lwrContainer->getType() == ui::ControlType::kSliderCont) &&
                        (lwrContainer == m_components.sldFOV || lwrContainer == m_components.sldRoll))
                        isUIInteractActive = true;
                    else
                        isUIInteractActive = false;
#endif
                }
            }
        }
    }// if (!ignoreMouse)
        
    ui::ControlContainer * lwrContainer = containerHelper.getSelectedControl(&mainContainer);

    if (lwrContainer->getType() == ui::ControlType::kSliderCont)
    {
        ui::ControlSliderCont * lwrSlider = static_cast<ui::ControlSliderCont *>(lwrContainer);
        lwrSlider->fineTuning = kbdSt.isKeyDown(VK_CONTROL);
    }
    
    if (isEnabled())
    {
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
        if (m_selectingHotkey)
        {
            unsigned int vkey = kbdSt.getNextKeyDown(input::KeyboardState::StartIterating);
            
            auto isVKeyAllowed = [&kbdSt](unsigned int vkey) -> bool
            {
                // We need to skip Shift/Ctrl/Alt, they are allowed to be modifiers, not the key itself
                switch (vkey)
                {
                    case VK_SHIFT:
                    case VK_CONTROL:
                    case VK_MENU:
                    case VK_LSHIFT:
                    case VK_LCONTROL:
                    case VK_LMENU:
                    case VK_RSHIFT:
                    case VK_RCONTROL:
                    case VK_RMENU:
                    {
                        return false;
                    }
                }
                // Not allowed:
                // Alt+F4 (app close), Alt+Enter (enter fullscreen), Alt+Tab (task switcher/flip)
                if (kbdSt.isKeyDown(VK_MENU))
                {
                    switch (vkey)
                    {
                        case VK_RETURN:
                        case VK_TAB:
                        case VK_F4:
                        {
                            return false;
                        }
                    }
                }
                // Not allowed:
                // Ctrl+F4 (child window close)
                if (kbdSt.isKeyDown(VK_CONTROL))
                {
                    switch (vkey)
                    {
                        case VK_F4:
                        {
                            return false;
                        }
                    }
                }
                return true;
            };

            while (!isVKeyAllowed(vkey))
                vkey = kbdSt.getNextKeyDown(vkey);

            if (vkey != input::KeyboardState::DoneIterating)
            {
                m_selectingHotkey = false;
                m_hotkeyModifierShift = m_selectingHotkeyShift;
                m_hotkeyModifierCtrl = m_selectingHotkeyCtrl;
                m_hotkeyModifierAlt = m_selectingHotkeyAlt;
                m_hotkeyModifierVKey = vkey;
            }
        }
        else
#else
#endif
        {
            if (kbdSt.isKeyStateChangedToDown(VK_UP) || (kbdSt.isKeyStateChangedToDown(VK_TAB) && kbdSt.isKeyDown(VK_SHIFT)))
            {
                selectPrevElement();
            }
            else if (kbdSt.isKeyStateChangedToDown(VK_DOWN) || (kbdSt.isKeyStateChangedToDown(VK_TAB) && !kbdSt.isKeyDown(VK_SHIFT)))
            {
                selectNextElement();
            }

            for (unsigned int it = kbdSt.getNextKeyStateChangedToDown(input::KeyboardState::StartIterating); it != input::KeyboardState::DoneIterating;
                    it = kbdSt.getNextKeyStateChangedToDown(it))
            {
                ui::ControlContainer * lwrContainer = containerHelper.getSelectedControl(&mainContainer);
                if (isControlEnabled(lwrContainer))
                    lwrContainer->onKeyPress(it);
            }

            AnselSDKState& SDKState = m_pAnselServer->m_anselSDK;

            if (kbdSt.isKeyStateChangedToDown(VK_F9))
            {
                if (m_pAnselServer->m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED)
                {
                    m_isSDKCaptureAbortRequested = true;
                }
            }
#if DBG_ADDITIONAL_HOTKEYS
            else if (kbdSt.isKeyStateChangedToDown(VK_F5))
                m_pAnselServer->m_bNextFrameDepthBufferUsed = !m_pAnselServer->m_bNextFrameDepthBufferUsed;
            else if (kbdSt.isKeyStateChangedToDown(VK_F6))
                m_pAnselServer->m_bNextFrameRenderDepthAsRGB = !m_pAnselServer->m_bNextFrameRenderDepthAsRGB;
#if DBG_HARDCODED_EFFECT_BW_ENABLED
            else if (kbdSt.isKeyStateChangedToDown(VK_F8))
                m_pAnselServer->m_bNextFrameEnableBlackAndWhite = !m_pAnselServer->m_bNextFrameEnableBlackAndWhite;
#endif
            else if (kbdSt.isKeyStateChangedToDown(VK_F10))
                m_pAnselServer->m_bNextFramePrevEffect = true;
#endif
            else if (kbdSt.isKeyStateChangedToDown(VK_F11))
            {
                //m_pAnselServer->m_bNextFrameNextEffect = true;
                //m_pAnselServer->m_bNextFrameRebuildYAML = true;
                m_pAnselServer->m_bNextFrameRefreshEffectStack = true;
            }
            else if (kbdSt.isKeyStateChangedToDown(VK_INSERT))
            {
                if (isEnabled())
                {
                    setIsVisible(!isVisible());
                }
            }
            else if (kbdSt.isKeyStateChangedToDown(VK_F12))
            {
                m_shotToTake = ShotType::kRegularUI;
            }
            else if (kbdSt.isKeyStateChangedToDown(VK_ESCAPE))
            {
                // [ESC] key works exactly like pressing Done button. It stops capture in progress
                // or exits Ansel mode if capture was not in progress:
                onButtonHideClick(nullptr);
            }
#if DBG_ADDITIONAL_HOTKEYS
            else if (kbdSt.isKeyStateChangedToDown(VK_OEM_4))
                SDKState.setSettleLatency(SDKState.getSettleLatency() + 1);
            else if (kbdSt.isKeyStateChangedToDown(VK_OEM_6))
            {// dec settle
                if (SDKState.getSettleLatency() > 0)
                    SDKState.setSettleLatency(SDKState.getSettleLatency() - 1);
            }
            else if (kbdSt.isKeyStateChangedToDown(VK_OEM_1))
                // inc capture
                SDKState.setCaptureLatency(SDKState.getCaptureLatency() + 1);
            else if (kbdSt.isKeyStateChangedToDown(VK_OEM_7))
            {
                // dec capture
                if (SDKState.getCaptureLatency() > 0)
                    SDKState.setCaptureLatency(SDKState.getCaptureLatency() - 1);
            }
#endif
        }
    }
    
    return;
}

void AnselUI::processInputState(const AnselSDKState & sdkState, float dt)
{
    // update input capture
    m_inputstate.update(sdkState, this);

    input::EDPadDirection::Enum         nextControlDir = input::EDPadDirection::kDown,
        prevControlDir = input::EDPadDirection::kUp,
        increaseDir = input::EDPadDirection::kRight,
        decreaseDir = input::EDPadDirection::kLeft;


    const input::GamepadState& gpadSt = m_inputstate.getGamepadState();
    input::EDPadDirection::Enum dpadDir = gpadSt.getDpadDirection();

    if (isEnabled())
    {
        if (gamepadFilterTime <= 0.0f)
        {
            if (dpadDir == prevControlDir)
            {
                selectPrevElement();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == nextControlDir)
            {
                selectNextElement();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == increaseDir)
            {
                ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
                selectedControl->onIncrease();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
            else if (dpadDir == decreaseDir)
            {
                ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
                selectedControl->onDecrease();
                gamepadFilterTime = UI_GAMEPAD_DPAD_FILTERTIME;
                m_usedGamepadForUIDuringTheSession = true;
            }
        }
    }
}

void AnselUI::setFilterControlsEnabled(bool isEnabled)
{
    // TODO avoroshilov stacking: push this for each flyout control
    m_components.flySpecialFX->isEnabled = isEnabled;
    m_components.btnSnap->isEnabled = isEnabled;

    for (size_t i = 0, iEnd = m_components.m_dynamicFilterContainers.size(); i < iEnd; ++i)
    {
        ui::ControlDynamicFilterContainer * dynFilterContainer = m_components.m_dynamicFilterContainers[i];
        dynFilterContainer->isEnabled = isEnabled;
        // No need to disable toggles - they do not affect the picture
        //  if (dynFilterContainer->m_toggleButton)
        //  dynFilterContainer->m_toggleButton->isEnabled = isEnabled;
        if (dynFilterContainer->m_btnRemove)
            dynFilterContainer->m_btnRemove->isEnabled = isEnabled;
        if (dynFilterContainer->m_btnUp)
            dynFilterContainer->m_btnUp->isEnabled = isEnabled;
        if (dynFilterContainer->m_btnDown)
            dynFilterContainer->m_btnDown->isEnabled = isEnabled;
    }

    m_components.btnAddFilter->isEnabled = isEnabled;

    if (isEnabled)
    {
        // We only need to do this is controls are enabled
        //  if they are disabled, we're ok to keep disabled them all
        checkUpDownButtons();
    }
}

void AnselUI::onCaptureStarted(int numShotsTotal)
{
    m_progressInfo_shotIdx = 0;
    m_progressInfo_inProgress = true;
    m_progressInfo_shotsTotal = numShotsTotal;

    setFilterControlsEnabled(false);

    // TODO avoroshilov UI
    //  we can do this for specific containers instead

    containerHelper.startSearch();
    ui::ControlBase * lwrContainer;
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, true))
    {
        // Lwrrently we have static blockID (0), everything else - dynamic - belongs to filters
        if (lwrContainer->blockID == STATIC_BLOCKID)
            continue;

        lwrContainer->isEnabled = false;
    }
    containerHelper.stopSearch();

    // Selecting "Cancel" right after pressing snap seems logical
    ui::ControlContainer * oldSelectedControl = containerHelper.getSelectedControl(&mainContainer);
    if (oldSelectedControl != m_components.btnDone)
        containerHelper.setSelectedControl(m_components.btnDone, oldSelectedControl);

    checkSelectedControl();
}

void AnselUI::onCaptureTaken(int shotIdx)
{
    m_progressInfo_shotIdx = shotIdx;
}

void AnselUI::onCaptureStopped(AnselUIBase::MessageType status)
{
    m_progressInfo_inProgress = false;

    setFilterControlsEnabled(true);

    containerHelper.startSearch();
    ui::ControlBase * lwrContainer;
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        // Lwrrently we have static blockID (0), everything else - dynamic - belongs to filters
        if (lwrContainer->blockID == STATIC_BLOCKID)
            continue;

        lwrContainer->isEnabled = true;
    }
    containerHelper.stopSearch();

    // We need to select "Snap" again, if "Cancel" didn't loose selection
    ui::ControlContainer * oldSelectedControl = containerHelper.getSelectedControl(&mainContainer);
    if (oldSelectedControl == m_components.btnDone)
        containerHelper.setSelectedControl(m_components.btnSnap, oldSelectedControl);

    checkSelectedControl();
}

void AnselUI::onSliderEffectTweakChange(void * object)
{
    ui::ControlSliderEffectTweak * sliderTweak = (ui::ControlSliderEffectTweak *)(object);

    AnselUIBase::EffectChange userConstChange;
    userConstChange.controlId = sliderTweak->controlIdx;
    userConstChange.filterId = sliderTweak->filterId;
    userConstChange.stackIdx = sliderTweak->stackIdx;
    
    userConstChange.value = ui::colwertPercentageToValue(sliderTweak->percentage, sliderTweak->baseValue, sliderTweak->milwalue, sliderTweak->maxValue);

    m_controlChangeQueue.push_back(userConstChange);
}

void AnselUI::onSliderListEffectTweakChange(void * object)
{
    ui::ControlSliderListEffectTweak * sliderListTweak = (ui::ControlSliderListEffectTweak *)(object);

    AnselUIBase::EffectChange userConstChange;
    userConstChange.controlId = sliderListTweak->controlIdx;
    userConstChange.filterId = sliderListTweak->filterId;
    userConstChange.stackIdx = sliderListTweak->stackIdx;

    userConstChange.value = sliderListTweak->getData(sliderListTweak->getSelected());

    m_controlChangeQueue.push_back(userConstChange);
}

void AnselUI::onSliderIntEffectTweakChange(void * object)
{
    ui::ControlSliderIntEffectTweak * sliderIntTweak = (ui::ControlSliderIntEffectTweak *)(object);

    AnselUIBase::EffectChange userConstChange;
    userConstChange.controlId = sliderIntTweak->controlIdx;
    userConstChange.filterId = sliderIntTweak->filterId;
    userConstChange.stackIdx = sliderIntTweak->stackIdx;

    if (sliderIntTweak->dataType == AnselUIBase::DataType::kInt)
        userConstChange.value = sliderIntTweak->calcIntFromSelected();
    else
        userConstChange.value = (sliderIntTweak->calcIntFromSelected() != 0);

    m_controlChangeQueue.push_back(userConstChange);
}

void AnselUI::onColorPickerEffectTweakChange(void * object)
{
    ui::ControlColorPickerEffectTweak * colorPickerTweak = (ui::ControlColorPickerEffectTweak *)(object);

    AnselUIBase::EffectChange userConstChange;
    userConstChange.controlId = colorPickerTweak->controlIdx;
    userConstChange.filterId = colorPickerTweak->filterId;
    userConstChange.stackIdx = colorPickerTweak->stackIdx;

    float values[4];

    for (unsigned int ch = 0; ch < colorPickerTweak->m_numChannels; ++ch)
    {
        values[ch] = ui::colwertPercentageToValue(colorPickerTweak->m_sliders[ch]->percentage, colorPickerTweak->m_sliders[ch]->baseValue, colorPickerTweak->m_sliders[ch]->milwalue, colorPickerTweak->m_sliders[ch]->maxValue);
    }

    userConstChange.value = shadermod::ir::TypelessVariable(values, colorPickerTweak->m_numChannels);

    m_controlChangeQueue.push_back(userConstChange);
}

void AnselUI::onCheckboxEffectTweakChange(void * object)
{
    ui::ControlCheckboxEffectTweak * checkboxTweak = (ui::ControlCheckboxEffectTweak *)(object);

    AnselUIBase::EffectChange userConstChange;
    userConstChange.controlId = checkboxTweak->controlIdx;
    userConstChange.filterId = checkboxTweak->filterId;
    userConstChange.stackIdx = checkboxTweak->stackIdx;

    bool isValueValid = true;
    if (checkboxTweak->dataType == AnselUIBase::DataType::kFloat)
    {
        userConstChange.value = checkboxTweak->isChecked ? checkboxTweak->maxValue : checkboxTweak->milwalue;
    }
    else if (checkboxTweak->dataType == AnselUIBase::DataType::kBool)
    {
        userConstChange.value = checkboxTweak->isChecked;
    }
    else if (checkboxTweak->dataType == AnselUIBase::DataType::kInt)
    {
        userConstChange.value = checkboxTweak->isChecked ? checkboxTweak->maxValue : checkboxTweak->milwalue;
    }
    else
    {
        isValueValid = false;
        LOG_WARN("Standalone UI: checkbox effect tweak failure - unsupported type");
    }

    if (isValueValid)
        m_controlChangeQueue.push_back(userConstChange);
}

void AnselUI::onButtonToggleClick(void * object)
{
    ui::ControlButtonToggle * toggleButton = (ui::ControlButtonToggle *)(object);
    if (toggleButton->m_containerToggle)
        toggleButton->m_containerToggle->isVisible = !toggleButton->m_containerToggle->isVisible;
}


void AnselUI::hideFloatingContainers()
{
    // Hide all of the floating containers
    for (size_t fci = 0, fciEnd = floatingContainers.size(); fci < fciEnd; ++fci)
    {
        floatingContainers[fci].first->isVisible = false;

        if (containerHelper.checkIfPresentInSelectionChain(&mainContainer, floatingContainers[fci].first))
        {
            ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
            if (floatingContainers[fci].second != nullptr)
            {
                containerHelper.setSelectedControl(floatingContainers[fci].second, selectedControl);
            }
            else
            {
                containerHelper.setSelectedControl(m_components.btnSnap, selectedControl);
            }
        }
    }

    floatingContainers.clear();
}

void AnselUI::onFlyoutHidePane(void * object)
{
    ui::ControlFlyoutSelector * lwrSelector = (ui::ControlFlyoutSelector *)(object);
    ui::ControlFlyoutToggleShared * flyoutToggle = lwrSelector->srcToggle;

    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
    containerHelper.setSelectedControl(flyoutToggle, selectedControl);

    hideFloatingContainer(lwrSelector->dstContainer);
}

void AnselUI::onFlyoutSelectorClick(void * object)
{
    ui::ControlFlyoutSelector * lwrSelector = (ui::ControlFlyoutSelector *)(object);
    ui::ControlFlyoutToggleShared * flyoutToggle = lwrSelector->srcToggle;

    std::wstring label = flyoutToggle->getLabel(lwrSelector->id);
    clampStringFloatSize(&label, m_storedSizes.defaultSliderSizeX);
    flyoutToggle->setSelected(lwrSelector->id, label.c_str());
    flyoutToggle->onChange();

    // Do not switch focus in the model where controls remains visible after selection
    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
    containerHelper.setSelectedControl(flyoutToggle, selectedControl);

    // Highlight chosen selector
    for (size_t i = 0, iEnd = lwrSelector->dstContainer->getControlsNum(); i < iEnd; ++i)
    {
        if (lwrSelector->dstContainer->getControl((int)i)->getType() != ui::ControlType::kButton)
        {
            continue;
        }

        ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrSelector->dstContainer->getControl((int)i));
        if (lwrButton->renderType != ui::ControlButton::RenderType::kSelector)
        {
            continue;
        }

        ui::ControlFlyoutSelector * lwrSelector = static_cast<ui::ControlFlyoutSelector *>(lwrButton);
        if (lwrSelector->id == flyoutToggle->getSelected())
        {
            lwrSelector->color = ui::ColorF4(0.196f, 0.212f, 0.231f, 0.800f);
        }
        else
        {
            lwrSelector->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    hideFloatingContainer(lwrSelector->dstContainer);
}

ui::ControlContainer * AnselUI::clearFlyout(FlyoutRebuildRequest * request)
{
    // Removing lwrrently existing selectors
    ui::ControlContainer * lwrContainer = nullptr;
    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

    bool needReselectControl = false;
    containerHelper.startSearchHierarchical(&mainContainer);
    while (lwrContainer = containerHelper.getNextControlHierarchical())
    {
        if (lwrContainer->m_parent != request->dstFlyoutContainer)
        {
            if (selectedControl == lwrContainer)
            {
                needReselectControl = true;
            }
            continue;
        }

        if (selectedControl == lwrContainer)
        {
            // If one of the buttons was selected, we need to drop the selection
            //  otherwise selection issues might happen if controls are updated when mouse is over one of them
            containerHelper.setSelectedControl(m_components.btnSnap, selectedControl);
            selectedControl = m_components.btnSnap;
        }

        // It should be safe to remove control at this point, since the list is already built
        containerHelper.reportElementDeletedHierarchical();
        if (lwrContainer->getControlsNum() != 0)
            containerHelper.skipChildrenHierarchical();
        ui::ControlContainer * parentContainer = lwrContainer->m_parent;
        releaseDynamicControl(lwrContainer);
        parentContainer->removeControlFast(lwrContainer);
    }
    containerHelper.stopSearchHierarchical();

    if (needReselectControl)
    {
        // Reselecting the same control to update selection indices
        containerHelper.setSelectedControl(selectedControl, selectedControl);
    }

    containerHelper.rebuildControlsArray(&mainContainer);

    return selectedControl;
}

void AnselUI::rebuildFlyout(FlyoutRebuildRequest * request)
{
    // Remove all the dynamic selectors first, update selected control if needed
    ui::ControlContainer * selectedControl = clearFlyout(request);

    // Adding new controls
    ui::ControlContainer * newSelectedControl = selectedControl;
    ui::ControlFlyoutSelector * prevSelector = nullptr;
    for (size_t i = 0, iEnd = request->srcFlyoutToggle->labelsStorage->labels.size(); i < iEnd; ++i)
    {
        ui::ControlFlyoutSelector * lwrSelector = getNewDynamicControlMem<ui::ControlFlyoutSelector>();
        new (lwrSelector) ui::ControlFlyoutSelector(&m_onFlyoutSelectorClick, &m_onFlyoutHidePane);

        lwrSelector->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
        lwrSelector->isBold = false;
        lwrSelector->hlColor = ui::ColorF4(0x76 / 255.f, 0xb9 / 255.f, 0.0f, 1.0f);

        lwrSelector->state = UI_CONTROL_ORDINARY;
        if (i == request->srcFlyoutToggle->getSelected())
        {
            lwrSelector->fontColor = ui::ColorF4(0x76 / 255.f, 0xb9 / 255.f, 0.0f, 1.0f);
            lwrSelector->isBold = true;
            newSelectedControl = lwrSelector;
        }

        lwrSelector->needsAutosize = true;
        
        lwrSelector->renderType = ui::ControlButton::RenderType::kSelector;

        lwrSelector->tabStop = 1000 - 5 * (int)i;

        lwrSelector->renderBufsShared = true;       // Shared idx and vertex buffers
        lwrSelector->renderBufsAuxShared = true;    // HL/DN buffers shared
        lwrSelector->pIndexBuf = pRectIndexBuf;
        lwrSelector->pVertexBuf = pRectVertexBuf;
        lwrSelector->pVertexBufHl = pRectVertexBuf;
        lwrSelector->pVertexBufDn = pRectVertexBuf;

        lwrSelector->srcToggle = request->srcFlyoutToggle;
        lwrSelector->dstContainer = request->dstFlyoutContainer;

        //lwrSelector->caption = request->srcFlyoutToggle->labels[i].c_str();
        std::wstring dynCaption = request->srcFlyoutToggle->labelsStorage->labels[i];
        clampStringFloatSize(&dynCaption, m_storedSizes.defaultSelectorSizeX * 0.9f);
        swprintf_s(lwrSelector->dynamicCaption, lwrSelector->dynamicCaptionSize, L"%s", dynCaption.c_str());
        lwrSelector->id = (int)i;

        // TODO avoroshilov: add tab stopping here as well
        request->dstFlyoutContainer->addControl(lwrSelector);

        prevSelector = lwrSelector;
    }

    containerHelper.setSelectedControl(newSelectedControl, selectedControl);

    applyTabStop();
    containerHelper.rebuildControlsArray(&mainContainer);

    m_components.flyoutPane->srcToggle = request->srcFlyoutToggle;
    request->isValid = false;
}

void AnselUI::onFlyoutClick(void * object)
{
    ui::ControlFlyoutToggleShared * srcFlyoutToggle = (ui::ControlFlyoutToggleShared *)(object);

    if (m_components.flyoutPane->srcToggle != srcFlyoutToggle || !m_components.flyoutPane->isVisible)
    {
        m_needToRecallwILayout = true;
        ui::ControlContainer * dstFlyoutContainer = m_components.flyoutPane;

        // Callwlate toggle on-screen position (hierarchy.abs+scrolling)
        float controlAbsPosY = srcFlyoutToggle->absPosY;
        {
            ui::ControlContainer * cointainerTraverse = srcFlyoutToggle;
            while (cointainerTraverse->m_parent)
            {
                controlAbsPosY += cointainerTraverse->m_parent->m_scrollValueY;
                cointainerTraverse = cointainerTraverse->m_parent;
            }
        }

        dstFlyoutContainer->posY = 1.0f - controlAbsPosY - srcFlyoutToggle->sizeY;

        // Check if container with callwlated position could fit outside the screen
        const float screenMargin = 0.01f;
        if (dstFlyoutContainer->posY + dstFlyoutContainer->sizeYMax > 2.0f - screenMargin)
        {
            dstFlyoutContainer->posY = 2.0f - screenMargin - dstFlyoutContainer->sizeYMax;
        }
        if (dstFlyoutContainer->posY < 0.0f)
        {
            dstFlyoutContainer->posY = 0.0f;
        }

        dstFlyoutContainer->isVisible = true;

        addFloatingContainer(dstFlyoutContainer, srcFlyoutToggle);

        m_flyoutRebuildRequest.isValid = true;
        m_flyoutRebuildRequest.dstFlyoutContainer = dstFlyoutContainer;
        m_flyoutRebuildRequest.srcFlyoutToggle = srcFlyoutToggle;
    }
    else
    {
        hideFloatingContainer(m_components.flyoutPane);
    }
}

void AnselUI::onButtonSnapClick(void * object)
{
    // TODO: get access to AnselSDK here
    if (/*AnselSDK.pCWDirector*/1)
    {
        switch (m_components.sldKind->getSelected())
        {
        case UI_DIRCETORSTATE_360:
        {
            // Spherical ("360")
            setShotType(ShotType::k360);
            break;
        }
        case UI_DIRCETORSTATE_HIGHRES:
        {
            // Highres
            setShotType(ShotType::kHighRes);
            break;
        }
#if (ENABLE_STEREO_SHOTS == 1)
        case UI_DIRCETORSTATE_STEREO:
        {
            // Stereo
            setShotType(ShotType::kStereo);
            break;
        }
        case UI_DIRCETORSTATE_STEREO_360:
        {
            // Stereo
            setShotType(ShotType::k360Stereo);
            break;
        }
#endif
        default:
        case UI_DIRCETORSTATE_NONE:
        {
            // Basic screenshot
            setShotType(ShotType::kRegular);
            break;
        }
        }
    }
    else
    {
        // Basic screenshot
        setShotType(ShotType::kRegular);
    }

    //telemetry 
    {
        AnselServer::AnselStateForTelemetry state;
        HRESULT telemetryStatus = m_pAnselServer->makeStateSnapshotforTelemetry(state);
        if (telemetryStatus == S_OK)
            m_pAnselServer->sendTelemetryMakeSnapshotEvent(state);
    }
}

void AnselUI::onButtonHideClick(void * object)
{
    if (m_pAnselServer->m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED)
    {
        m_isSDKCaptureAbortRequested = true;
    }
    else
        camWorksToggle();
}

void AnselUI::onButtonResetRollClick(void * object)
{
    m_needToResetRoll = true;
}

#ifdef ENABLE_STYLETRANSFER
void AnselUI::onButtonDownloadRestyleConfirmClick(void * object)
{
    m_restyleDownloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kConfirmed;
    m_components.dlgDownloadRestyle->isVisible = false;
    areControlsInteractionsDisabled = false;
    areCameraInteractionsDisabled = false;
    excludeContainer = nullptr;
}
void AnselUI::onButtonDownloadRestyleCancelClick(void * object)
{
    m_restyleDownloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kRejected;
    m_components.dlgDownloadRestyle->isVisible = false;
    areControlsInteractionsDisabled = false;
    areCameraInteractionsDisabled = false;
    excludeContainer = nullptr;
}
#endif

#if (DBG_STACKING_PROTO == 1)
void AnselUI::addDynamicFilter()
{
    if (!m_allowDynamicFilterStacking)
        return;

    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

    ui::ControlButtonToggle * newFilterToggle = getNewDynamicControlMem<ui::ControlButtonToggle>();
    new (newFilterToggle) ui::ControlButtonToggle(&m_onButtonToggleClick);

    ui::ControlContainer * oldLastAnchor = m_components.btnAddFilter->m_anchorY;
    newFilterToggle->m_anchorY = oldLastAnchor;
    newFilterToggle->tabStop = oldLastAnchor->tabStop - 1;

    newFilterToggle->state = UI_CONTROL_ORDINARY;

    newFilterToggle->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    newFilterToggle->caption = m_textFilter.c_str();//L"Toggle filter";
    newFilterToggle->isBold = true;


    // Up button
    ui::ControlButtonClickable * newButtonUp = getNewDynamicControlMem<ui::ControlButtonClickable>();
    new (newButtonUp) ui::ControlButtonClickable(&m_onButtonMoveFilterUpClick);

    newButtonUp->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    newButtonUp->renderType = ui::ControlButton::RenderType::kBasic;
    //newButtonUp->caption = L"\u21d1"; // Double arrow
    //newButtonUp->caption = L"\u290a"; // Tripple arrow (~)
    //newButtonUp->caption = L"\u21e7"; // Upwards white arrow (bad)
    //newButtonUp->caption = L"\u2b06"; // Upwards black arrow
    //newButtonUp->caption = L"\u2912"; // Upwards arrow to bar (bad)
    newButtonUp->caption = L"\u25b2";    // Black triangle up
    newButtonUp->isBold = false;
    newButtonUp->hlType = ui::ControlButton::HighlightType::kFont;
    newButtonUp->hlColor = m_lwGreenColor;

    newButtonUp->state = UI_CONTROL_ORDINARY;

    newButtonUp->renderBufsShared = true;       // Shared idx and vertex buffers
    newButtonUp->renderBufsAuxShared = true;    // HL/DN buffers shared
    newButtonUp->pVertexBuf = pRectVertexBuf;
    newButtonUp->pVertexBufHl = pRectVertexBuf;
    newButtonUp->pVertexBufDn = pRectVertexBuf;
    newButtonUp->pIndexBuf = pRectIndexBuf;

    newButtonUp->tabStop = newFilterToggle->tabStop - 2;

    newButtonUp->m_anchorY = newFilterToggle;

    // Down button
    ui::ControlButtonClickable * newButtonDown = getNewDynamicControlMem<ui::ControlButtonClickable>();
    new (newButtonDown) ui::ControlButtonClickable(&m_onButtonMoveFilterDownClick);

    newButtonDown->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    newButtonDown->renderType = ui::ControlButton::RenderType::kBasic;
    //newButtonDown->caption = L"\u21d3";   // Double arrow
    //newButtonDown->caption = L"\u290b";   // Tripple arrow (~)
    //newButtonDown->caption = L"\u21e9";   // Downwards white arrow (bad)
    //newButtonDown->caption = L"\u2b07";   // Downwards black arrow
    //newButtonDown->caption = L"\u2913";   // Downwards arrow to bar
    newButtonDown->caption = L"\u25bc";    // Black triangle down
    newButtonDown->isBold = false;
    newButtonDown->hlType = ui::ControlButton::HighlightType::kFont;
    newButtonDown->hlColor = m_lwGreenColor;

    newButtonDown->state = UI_CONTROL_ORDINARY;

    newButtonDown->renderBufsShared = true;     // Shared idx and vertex buffers
    newButtonDown->renderBufsAuxShared = true;  // HL/DN buffers shared
    newButtonDown->pVertexBuf = pRectVertexBuf;
    newButtonDown->pVertexBufHl = pRectVertexBuf;
    newButtonDown->pVertexBufDn = pRectVertexBuf;
    newButtonDown->pIndexBuf = pRectIndexBuf;

    newButtonDown->tabStop = newFilterToggle->tabStop - 3;

    newButtonDown->m_anchorY = newFilterToggle;

    // Remove button
    ui::ControlButtonClickable * newButtonRemove = getNewDynamicControlMem<ui::ControlButtonClickable>();
    new (newButtonRemove) ui::ControlButtonClickable(&m_onButtonRemoveFilterClick);

    newButtonRemove->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    newButtonRemove->fontColor = m_doneAbortColor;
    newButtonRemove->renderType = ui::ControlButton::RenderType::kBasic;
    //newButtonRemove->caption = L"\u232b"; // Backspace
    newButtonRemove->caption = L"\u2715";   // Multiplication X
    //newButtonRemove->caption = L"\u00D7"; // Times X
    //newButtonRemove->caption = L"\u2716"; // Heavy Multiplication X
    newButtonRemove->isBold = true;
    newButtonRemove->hlType = ui::ControlButton::HighlightType::kFont;
    newButtonRemove->hlColor = ui::ColorF4(1.0f, 0.5f, 0.5f, 1.0f);

    newButtonRemove->state = UI_CONTROL_ORDINARY;

    newButtonRemove->renderBufsShared = true;       // Shared idx and vertex buffers
    newButtonRemove->renderBufsAuxShared = true;    // HL/DN buffers shared
    newButtonRemove->pVertexBuf = pRectVertexBuf;
    newButtonRemove->pVertexBufHl = pRectVertexBuf;
    newButtonRemove->pVertexBufDn = pRectVertexBuf;
    newButtonRemove->pIndexBuf = pRectIndexBuf;

    newButtonRemove->tabStop = newFilterToggle->tabStop - 4;

    newButtonRemove->m_anchorY = newFilterToggle;



    // Filter container
    ui::ControlDynamicFilterContainer * newFilterContainer = getNewDynamicControlMem<ui::ControlDynamicFilterContainer>();
    new (newFilterContainer) ui::ControlDynamicFilterContainer;
    // Real stack index is dynamic stack index + number of static filters
    newFilterContainer->stackIdx = (int)(m_components.m_dynamicFilterContainers.size() + filterIDs.size());
    m_components.m_dynamicFilterContainers.push_back(newFilterContainer);

    m_components.btnAddFilter->m_anchorY = newFilterContainer;
    newFilterContainer->m_anchorY = newButtonRemove;

    newFilterContainer->posX = 0.0f;
    newFilterContainer->posY = 0.0f;
    newFilterContainer->sizeX = m_components.cntControls->sizeX;
    newFilterContainer->sizeY = 0.0f;

    newFilterContainer->state = UI_CONTROL_ORDINARY;

    newFilterContainer->tabStop = newFilterToggle->tabStop - 5;
    newFilterContainer->m_needsContentAutoPlacement = true;
    newFilterContainer->m_renderSideVLine = true;

    m_components.cntControls->addControl(newFilterToggle);
    m_components.cntControls->addControl(newButtonRemove);
    m_components.cntControls->addControl(newButtonUp);
    m_components.cntControls->addControl(newButtonDown);
    m_components.cntControls->addControl(newFilterContainer);

    // Connect toggle button and the new container
    newFilterToggle->m_containerToggle = newFilterContainer;

    newFilterContainer->m_toggleButton = newFilterToggle;
    newFilterContainer->m_btnRemove = newButtonRemove;
    newFilterContainer->m_btnUp = newButtonUp;
    newFilterContainer->m_btnDown = newButtonDown;

    // Child controls
    //////////////////////////////////////////////////////////////////////////////////////////

    ui::ControlFlyoutToggleShared * newFilterTypeFlyout = getNewDynamicControlMem<ui::ControlFlyoutToggleShared>();
    new (newFilterTypeFlyout) ui::ControlFlyoutToggleShared(&m_onFlyoutClick, &m_onFlyoutSpecialFXChangeDynamic);

    newFilterTypeFlyout->labelsStorage = &m_components.flySpecialFXLabels;
    newFilterTypeFlyout->setSelected(0, newFilterTypeFlyout->getLabel(0));

    newFilterTypeFlyout->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    newFilterTypeFlyout->renderType = ui::ControlButton::RenderType::kFlyoutToggle;
    newFilterTypeFlyout->caption = m_textFilterType.c_str();
    newFilterTypeFlyout->isBold = false;
    
    newFilterTypeFlyout->state = UI_CONTROL_ORDINARY;

    newFilterTypeFlyout->renderBufsShared = true;       // Shared idx and vertex buffers
    newFilterTypeFlyout->renderBufsAuxShared = true;    // HL/DN buffers shared
    newFilterTypeFlyout->pVertexBuf = pRectVertexBuf;
    newFilterTypeFlyout->pVertexBufHl = pRectVertexBuf;
    newFilterTypeFlyout->pVertexBufDn = pRectVertexBuf;
    newFilterTypeFlyout->pIndexBuf = pRectIndexBuf;

    newFilterTypeFlyout->tabStop = TABSTOP_INIT * 100; // Biggest tabstop, since this one should be at the top always

    newFilterTypeFlyout->m_anchorY = nullptr;

    newFilterContainer->addControl(newFilterTypeFlyout);
    newFilterContainer->m_filterToggle = newFilterTypeFlyout;

    //////////////////////////////////////////////////////////////////////////////////////////

    applyTabStop();
    containerHelper.rebuildControlsArray(&mainContainer);

    // Reselecting the same control to update selection indices
    containerHelper.setSelectedControl(selectedControl, selectedControl);

    float aspect = m_width / (float)m_height;
    recallwlateUILayout(aspect);

    checkUpDownButtons();

    m_needToAddDynamicFilter = false;
}

void AnselUI::removeDynamicFilter()
{
    if ((m_dynamicFilterIdxToRemove < 0) || (m_dynamicFilterIdxToRemove >= (int)m_components.m_dynamicFilterContainers.size()))
    {
        m_dynamicFilterIdxToRemove = -1;
        return;
    }

    ui::ControlDynamicFilterContainer * filterContainer = m_components.m_dynamicFilterContainers[m_dynamicFilterIdxToRemove];

    // Relink containers if needed
    for (size_t ci = 0, ciEnd = m_components.m_dynamicFilterContainers.size(); ci < ciEnd; ++ci)
    {
        ui::ControlDynamicFilterContainer * lwrFilterContainer = m_components.m_dynamicFilterContainers[ci];
        if (lwrFilterContainer->m_toggleButton->m_anchorY == filterContainer)
            lwrFilterContainer->m_toggleButton->m_anchorY = filterContainer->m_toggleButton->m_anchorY;
    }
    // Relink button add if needed
    if (m_components.btnAddFilter->m_anchorY == filterContainer)
        m_components.btnAddFilter->m_anchorY = filterContainer->m_toggleButton->m_anchorY;

    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

    bool needReselectControl = false;
    removeDynamicElementsInContainer(filterContainer, &selectedControl, &needReselectControl);

    removeDynamicElement(filterContainer->m_btnRemove, &selectedControl, &needReselectControl);
    removeDynamicElement(filterContainer->m_btnUp, &selectedControl, &needReselectControl);
    removeDynamicElement(filterContainer->m_btnDown, &selectedControl, &needReselectControl);
    removeDynamicElement(filterContainer->m_toggleButton, &selectedControl, &needReselectControl);
    removeDynamicElement(filterContainer->m_filterToggle, &selectedControl, &needReselectControl);
    removeDynamicElement(filterContainer, &selectedControl, &needReselectControl);

    if (needReselectControl)
    {
        // Reselecting the same control to update selection indices
        containerHelper.setSelectedControl(selectedControl, selectedControl);
    }

    m_components.m_dynamicFilterContainers.erase(m_components.m_dynamicFilterContainers.begin() + m_dynamicFilterIdxToRemove);

    applyTabStop();
    containerHelper.rebuildControlsArray(&mainContainer);

    // Update freshly created sliders positions
    float aspect = m_width / (float)m_height;
    recallwlateUILayout(aspect);

    // Reselecting the same control to update selection indices (since freshly added elements could makle indices obsolete)
    containerHelper.setSelectedControl(selectedControl, selectedControl);

    checkUpDownButtons();

    m_dynamicFilterIdxToRemove = -1;
}

void AnselUI::onButtonRemoveFilterClick(void * object)
{
    // Search for the filter index
    ui::ControlButtonClickable * btnRemove = reinterpret_cast<ui::ControlButtonClickable *>(object);

    for (size_t i = 0, iEnd = m_components.m_dynamicFilterContainers.size(); i < iEnd; ++i)
    {
        ui::ControlDynamicFilterContainer * filterContainer = m_components.m_dynamicFilterContainers[i];
        if (filterContainer->m_btnRemove == btnRemove)
        {
            m_dynamicFilterIdxToRemove = (int)i;
            break;
        }
    }
}
void AnselUI::dynamicFilterExchangeTabStops(ui::ControlDynamicFilterContainer * first, ui::ControlDynamicFilterContainer * second)
{
    auto swapFn = [](int & tabStop1, int & tabStop2)
    {
        int temp;
        temp = tabStop1;
        tabStop1 = tabStop2;
        tabStop2 = temp;
    };
    swapFn(first->m_toggleButton->tabStop, second->m_toggleButton->tabStop);
    swapFn(first->m_btnRemove->tabStop, second->m_btnRemove->tabStop);
    swapFn(first->m_btnUp->tabStop, second->m_btnUp->tabStop);
    swapFn(first->m_btnDown->tabStop, second->m_btnDown->tabStop);
    swapFn(first->tabStop, second->tabStop);
}
void AnselUI::checkUpDownButtons()
{
    ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);

    size_t numDynamicFilterContainers = m_components.m_dynamicFilterContainers.size();
    if (numDynamicFilterContainers == 0)
        return;

    if (numDynamicFilterContainers == 1)
    {
        if (m_components.m_dynamicFilterContainers[0]->m_toggleButton->caption == m_textFilter ||
            m_components.m_dynamicFilterContainers[0]->m_toggleButton->caption == m_textFilterNone)
        {
            m_components.m_dynamicFilterContainers[0]->m_btnRemove->isEnabled = false;
        }
        else
        {
            m_components.m_dynamicFilterContainers[0]->m_btnRemove->isEnabled = true;
        }
    }
    else if (numDynamicFilterContainers > 1)
    {
        if (!m_components.m_dynamicFilterContainers[0]->m_btnRemove->isEnabled)
        {
            m_components.m_dynamicFilterContainers[0]->m_btnRemove->isEnabled = true;
        }
    }

    for (size_t i = 0, iEnd = numDynamicFilterContainers; i < iEnd; ++i)
    {
        m_components.m_dynamicFilterContainers[i]->m_btnUp->isEnabled = true;
        m_components.m_dynamicFilterContainers[i]->m_btnDown->isEnabled = true;
    }

    m_components.m_dynamicFilterContainers[0]->m_btnUp->isEnabled = false;
    m_components.m_dynamicFilterContainers[numDynamicFilterContainers-1]->m_btnDown->isEnabled = false;

    // In case button that is now being disabled was selected
    if (selectedControl == m_components.m_dynamicFilterContainers[0]->m_btnUp)
    {
        containerHelper.setSelectedControl(m_components.m_dynamicFilterContainers[0]->m_toggleButton, selectedControl);
    }
    else if (selectedControl == m_components.m_dynamicFilterContainers[numDynamicFilterContainers-1]->m_btnDown)
    {
        containerHelper.setSelectedControl(m_components.m_dynamicFilterContainers[numDynamicFilterContainers-1]->m_toggleButton, selectedControl);
    }
}
void AnselUI::onButtonMoveFilterUpClick(void * object)
{
    int dynamicFilterIdxToMove = -1;
    ui::ControlDynamicFilterContainer * filterContainer = nullptr;
    ui::ControlButtonClickable * btnUp = reinterpret_cast<ui::ControlButtonClickable *>(object);
    for (size_t i = 0, iEnd = m_components.m_dynamicFilterContainers.size(); i < iEnd; ++i)
    {
        filterContainer = m_components.m_dynamicFilterContainers[i];
        if (filterContainer->m_btnUp == btnUp)
        {
            dynamicFilterIdxToMove = (int)i;
            break;
        }
    }

    // We cannot move dynamic filter 0 any higher, or filter not found
    if (dynamicFilterIdxToMove <= 0)
        return;

    ui::ControlDynamicFilterContainer * prevFilterContainer = m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove - 1];
    
    // Relink two filter containers
    filterContainer->m_toggleButton->m_anchorY = prevFilterContainer->m_toggleButton->m_anchorY;
    prevFilterContainer->m_toggleButton->m_anchorY = filterContainer;

    // Relink control that is outside of pair in question
    ui::ControlContainer * linkedControl = m_components.btnAddFilter;
    // If new prev control position isn't the last one
    if (dynamicFilterIdxToMove < (int)m_components.m_dynamicFilterContainers.size() - 1)
        linkedControl = m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove + 1]->m_toggleButton;

    linkedControl->m_anchorY = prevFilterContainer;

    dynamicFilterExchangeTabStops(filterContainer, prevFilterContainer);
    m_needToApplyTabStop = true;
    m_needToRecallwILayout = true;

    m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove] = prevFilterContainer;
    m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove - 1] = filterContainer;

    checkUpDownButtons();
}
void AnselUI::onButtonMoveFilterDownClick(void * object)
{
    int dynamicFilterIdxToMove = -1;
    ui::ControlDynamicFilterContainer * filterContainer = nullptr;
    ui::ControlButtonClickable * btnDown = reinterpret_cast<ui::ControlButtonClickable *>(object);
    for (size_t i = 0, iEnd = m_components.m_dynamicFilterContainers.size(); i < iEnd; ++i)
    {
        filterContainer = m_components.m_dynamicFilterContainers[i];
        if (filterContainer->m_btnDown == btnDown)
        {
            dynamicFilterIdxToMove = (int)i;
            break;
        }
    }

    // We cannot move last dynamic filter any lower, or filter not found
    if (dynamicFilterIdxToMove < 0 || dynamicFilterIdxToMove >= (int)m_components.m_dynamicFilterContainers.size() - 1)
        return;

    ui::ControlDynamicFilterContainer * nextFilterContainer = m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove + 1];

    // Relink two filter containers
    nextFilterContainer->m_toggleButton->m_anchorY = filterContainer->m_toggleButton->m_anchorY;
    filterContainer->m_toggleButton->m_anchorY = nextFilterContainer;

    // Relink control that is outside of pair in question
    ui::ControlContainer * linkedControl = m_components.btnAddFilter;
    // If new position isn't the last one
    if (dynamicFilterIdxToMove + 1 < (int)m_components.m_dynamicFilterContainers.size() - 1)
        linkedControl = m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove + 2]->m_toggleButton;

    linkedControl->m_anchorY = filterContainer;

    dynamicFilterExchangeTabStops(filterContainer, nextFilterContainer);
    m_needToApplyTabStop = true;
    m_needToRecallwILayout = true;

    m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove] = nextFilterContainer;
    m_components.m_dynamicFilterContainers[dynamicFilterIdxToMove + 1] = filterContainer;

    checkUpDownButtons();
}
void AnselUI::onButtonAddFilterClick(void * object)
{
    m_needToAddDynamicFilter = true;
}
#endif

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
void AnselUI::onButtonHotkeySetupClick(void * object)
{
    m_selectingHotkey = !m_selectingHotkey;
}
#endif

class AnselTabStopSortComparator
{
public:

    inline bool operator() (ui::ControlContainer * & element1, ui::ControlContainer * & element2)
    {
        // Sorting desc
        return (element1->tabStop > element2->tabStop);
    }
};

void AnselUI::applyTabStop()
{
    // TODO avoroshilov UI
    //  init does linear array rebuilding which is not required here (just go through controls hierarchically)
    containerHelper.init(&mainContainer);
    ui::ControlContainer * lwrParentContainer;
    while (lwrParentContainer = containerHelper.getNextControl((int)ui::ControlType::kContainer, false))
    {
        ui::ControlContainer * lwrChildContainer = static_cast<ui::ControlContainer *>(lwrParentContainer);
        std::sort(lwrChildContainer->getControlsRaw().begin(), lwrChildContainer->getControlsRaw().end(), AnselTabStopSortComparator());
    }
    containerHelper.stopSearch();
}

namespace standaloneui_base_vs40
{
#include "shaders/include/standaloneui_base.vs_40.h"
}
namespace standaloneui_base_ps40
{
#include "shaders/include/standaloneui_base.ps_40.h"
}
namespace standaloneui_text_ps40
{
#include "shaders/include/standaloneui_text.ps_40.h"
}


HRESULT AnselUI::init(HMODULE hResourceModule, ID3D11Device* d3dDevice, AnselServer* pAnselServer, const std::wstring& installationFolderPath)
{
    m_pAnselServer = pAnselServer;
    m_errorManager = &pAnselServer->m_errorManager;

    m_shotToTake = ShotType::kNone;
    memset(m_shotTypeEnabled, 0, (int)ShotType::kNumEntries * sizeof(bool));
    m_shotTypeEnabled[(int)ShotType::kRegular] = true;

    HRESULT status = S_OK;

    m_gameplayOverlayNotifications.init(10);

#define CAPTION_STRING_SIZE 32

#if 1
    if (m_forcedLocale != 0)
    {
        m_langID = LANGIDFROMLCID(m_forcedLocale);
    }
    //m_langID = GetUserDefaultUILanguage();
    //m_langID = MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL);
    //m_langID = MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US);
#else
    // Forced localization (debug purposes)
    //m_langID = MAKELANGID(LANG_GERMAN, SUBLANG_GERMAN);
    //m_langID = MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH);
    //m_langID = MAKELANGID(LANG_SPANISH, SUBLANG_SPANISH_MEXICAN);
    //m_langID = MAKELANGID(LANG_FRENCH, SUBLANG_FRENCH);
    //m_langID = MAKELANGID(LANG_ITALIAN, SUBLANG_ITALIAN);
    //m_langID = MAKELANGID(LANG_RUSSIAN, SUBLANG_RUSSIAN_RUSSIA);
    m_langID = MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL);
    //m_langID = MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED);
#endif

#if 1
    // DEBUG
    BYTE primary = PRIMARYLANGID(m_langID);
    BYTE sublang = SUBLANGID(m_langID);
#endif

    PFND3DCREATEBLOBFUNC pfnD3DCreateBlob = pAnselServer->getD3DCompiler().getD3DCreateBlobFunc();

    ID3D11VertexShader      *pVS = NULL;
    ID3D11PixelShader       *pPS = NULL;

    // UI ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Vertex Shader
    const BYTE * vsBaseByteCode = standaloneui_base_vs40::g_main;
    size_t vsBaseByteCodeSize = sizeof(standaloneui_base_vs40::g_main)/sizeof(BYTE);

    if (!SUCCEEDED(status = d3dDevice->CreateVertexShader(vsBaseByteCode, vsBaseByteCodeSize, NULL, &pVS)))
    {
        HandleFailure();
    }

    D3D11_INPUT_ELEMENT_DESC inputLayoutDesc[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    if (!SUCCEEDED(status = d3dDevice->CreateInputLayout(inputLayoutDesc, 2, vsBaseByteCode, vsBaseByteCodeSize, &pInputLayout)))
    {
        HandleFailure();
    }

    // Pixel Shader
    const BYTE * psBaseByteCode = standaloneui_base_ps40::g_main;
    size_t psBaseByteCodeSize = sizeof(standaloneui_base_ps40::g_main)/sizeof(BYTE);
    if (!SUCCEEDED(status = d3dDevice->CreatePixelShader(psBaseByteCode, psBaseByteCodeSize, NULL, &pPS)))
    {
        HandleFailure();
    }

    D3D11_BUFFER_DESC constBufDesc;
    // Init
    {
        ZeroMemory(&constBufDesc, sizeof(constBufDesc));
        constBufDesc.Usage = D3D11_USAGE_DYNAMIC;
        constBufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        constBufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        constBufDesc.MiscFlags = 0;
        constBufDesc.StructureByteStride = 0;
    }

    D3D11_TEXTURE2D_DESC textureDesc;
    // Init
    {
        ZeroMemory(&textureDesc, sizeof(textureDesc));
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.CPUAccessFlags = 0;
        textureDesc.MiscFlags = 0;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    // Init
    {
        ZeroMemory(&shaderResourceViewDesc, sizeof(shaderResourceViewDesc));
        shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
        shaderResourceViewDesc.Texture2D.MipLevels = 1;
    }


    std::wstring imageFolderPath = installationFolderPath + L"ui.tga";

    unsigned int w, h;

    auto atlasImageData = darkroom::loadImage(&imageFolderPath[0], w, h, darkroom::BufferFormat::RGBA8);

    //darkroom::Capture capt;
    //capt.width = w;
    //capt.height = h;
    //capt.colorBuffer = atlasImageData;
    //capt.colorBufferFormat = darkroom::BufferFormat::RGBA8;
    //capt.name = "d:\\ui_resaved.bmp";
    //darkroom::submitCapture(capt);

    if (atlasImageData.empty())
    {
        status = E_FAIL;
        HandleFailureWMessage("atlasImagedata not found at %s", darkroom::getUtf8FromWstr(imageFolderPath.c_str()).c_str());
    }


    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.Width = w;
    textureDesc.Height = h;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initialData;
    initialData.pSysMem = atlasImageData.data();
    initialData.SysMemPitch = w * 4 * sizeof(unsigned char);
    initialData.SysMemSlicePitch = 0;

    if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(d3dDevice, &textureDesc, &initialData, &pUIAtlasTexture)))
    {
        HandleFailureWMessage("%d", status);
    }
    // release memory allocated for atlas image
    atlasImageData = decltype(atlasImageData)();

    shaderResourceViewDesc.Format = textureDesc.Format;

    if (!SUCCEEDED(status = d3dDevice->CreateShaderResourceView(pUIAtlasTexture, &shaderResourceViewDesc, &pUIAtlasSRV)))
    {
        HandleFailureWMessage("%d", status);
    }

    // Create offsets constant buffers
    constBufDesc.ByteWidth = (sizeof(UIShaderConstBuf) + 15) & ~15;
    UIShaderConstBuf controlInitialData =
    {
        1.0f, 1.0f, 1.0f, 1.0f, // White color
        0.0f, 0.0f, 1.0f, 1.0f  // Zero offsets & unity scaling
    };
    initialData.pSysMem = &controlInitialData;
    initialData.SysMemPitch = 0;
    initialData.SysMemSlicePitch = 0;
    if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&constBufDesc, &initialData, &pZeroOffsetsBuffer)))
    {
        HandleFailureWMessage("%d", status);
    }

    if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&constBufDesc, 0, &pVariableOffsetsBuffer)))
    {
        HandleFailure();
    }

#if (DBG_USE_OUTLINE == 1)
    if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&constBufDesc, 0, &pFontOutlineBuffer)))
    {
        HandleFailure();
    }
#endif


    // Can reuse pSamplerState

    D3D11_DEPTH_STENCIL_DESC dsStateDesc;
    memset(&dsStateDesc, 0, sizeof(dsStateDesc));
    dsStateDesc.DepthEnable = FALSE;
    dsStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsStateDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.StencilEnable = FALSE;
    dsStateDesc.StencilReadMask = 0xFF;
    dsStateDesc.StencilWriteMask = 0xFF;
    dsStateDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

    if (!SUCCEEDED(status = d3dDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        HandleFailure();
    }

    D3D11_BLEND_DESC blendStateDesc;
    memset(&blendStateDesc, 0, sizeof(blendStateDesc));
    blendStateDesc.AlphaToCoverageEnable = FALSE;
    blendStateDesc.IndependentBlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    blendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    if (!SUCCEEDED(status = d3dDevice->CreateBlendState(&blendStateDesc, &pBlendState)))
    {
        HandleFailure();
    }

    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_NONE,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        TRUE,                      //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    if (!SUCCEEDED(status = d3dDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        HandleFailure();
    }

    // Create layout
    ID3D11Buffer * pVertexBuf = 0, *pVertexBufHl = 0, *pVertexBufDn = 0;
    ID3D11Buffer * pIndexBuf = 0;

    struct VSInput
    {
        float position[4];
        float texcoord[2];
    };

    vertexStrideUI = sizeof(VSInput);

    // Common buffers for untextured rectangles
    const float solidFillTCX = 0.0f, solidFillTCY = 0.0f;
    {
        const unsigned int vertsPerButton = 4;
        const unsigned int indsPerButton = 6;

        unsigned int inds[indsPerButton] = { 0, 1, 2, 2, 3, 0 };

        D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData, indexData;

        indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerButton;
        indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        indexBufferDesc.CPUAccessFlags = 0;
        indexBufferDesc.MiscFlags = 0;
        indexBufferDesc.StructureByteStride = 0;

        indexData.pSysMem = inds;
        indexData.SysMemPitch = 0;
        indexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pRectIndexBuf)))
        {
            HandleFailure();
        }

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        VSInput verts[vertsPerButton] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { 1.0f, 1.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 3
            {
                { 0.0f, 1.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pRectVertexBuf)))
        {
            HandleFailure();
        }
    }
    {
        const unsigned int vertsNumTotal = 3;
        const unsigned int indsNumTotal = 3;

        unsigned int inds[indsNumTotal] = { 0, 1, 2 };

        D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData, indexData;

        indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsNumTotal;
        indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        indexBufferDesc.CPUAccessFlags = 0;
        indexBufferDesc.MiscFlags = 0;
        indexBufferDesc.StructureByteStride = 0;

        indexData.pSysMem = inds;
        indexData.SysMemPitch = 0;
        indexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pTriUpIndexBuf)))
        {
            HandleFailure();
        }

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsNumTotal;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        VSInput verts[vertsNumTotal] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { 0.5f, 1.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pTriUpVertexBuf)))
        {
            HandleFailure();
        }
    }
    {
        const unsigned int indsNum = 12;

        unsigned int inds[indsNum] = { 0, 1, 5, 1, 4, 5, 1, 2, 3, 1, 3, 4 };

        D3D11_BUFFER_DESC indexBufferDesc;
        D3D11_SUBRESOURCE_DATA indexData;

        indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsNum;
        indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        indexBufferDesc.CPUAccessFlags = 0;
        indexBufferDesc.MiscFlags = 0;
        indexBufferDesc.StructureByteStride = 0;

        indexData.pSysMem = inds;
        indexData.SysMemPitch = 0;
        indexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pArrowIndexBuf)))
        {
            HandleFailure();
        }
    }
    // Arrow down
    {
        const unsigned int vertsNum = 6;

        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsNum;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float arrowOffsetX = -0.5f;
        const float arrowThickness = 2 * 1 / 10.0f; // Glyph size is 10x10, 1/10 is one pixel
        const float arrowHeight = 0.5f;
        VSInput verts[vertsNum] =
        {
            // 0
            {
                { arrowOffsetX + 0.0f, arrowHeight, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { arrowOffsetX + 0.5f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { arrowOffsetX + 1.0f, arrowHeight, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 3
            {
                { arrowOffsetX + 1.0f - arrowThickness, arrowHeight, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 4
            {
                { arrowOffsetX + 0.5f, arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 5
            {
                { arrowOffsetX + arrowThickness, arrowHeight, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pArrowDowlwertexBuf)))
        {
            HandleFailure();
        }
    }
    // Arrow up
    {
        const unsigned int vertsNum = 6;

        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsNum;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float arrowOffsetX = -0.5f;
        const float arrowThickness = 2 * 1 / 10.0f; // Glyph size is 10x10, 1/10 is one pixel
        const float arrowHeight = 0.5f;
        VSInput verts[vertsNum] =
        {
            // 0
            {
                { arrowOffsetX + 1.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { arrowOffsetX + 0.5f, arrowHeight, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { arrowOffsetX + 0.0f, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 3
            {
                { arrowOffsetX + arrowThickness, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 4
            {
                { arrowOffsetX + 0.5f, arrowHeight - arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 5
            {
                { arrowOffsetX + 1.0f - arrowThickness, 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pArrowUpVertexBuf)))
        {
            HandleFailure();
        }
    }
    // Arrow right
    {
        const unsigned int vertsNum = 6;

        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsNum;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float arrowOffsetX = -0.5f;
        const float arrowThickness = 2 * 1 / 10.0f; // Glyph size is 10x10, 1/10 is one pixel
        const float arrowHeight = 0.5f;
        VSInput verts[vertsNum] =
        {
            // 0
            {
                { 0.0f, arrowOffsetX + 1.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { arrowHeight, arrowOffsetX + 0.5f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { 0.0f, arrowOffsetX + 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 3
            {
                { 0.0f, arrowOffsetX + arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 4
            {
                { arrowHeight - arrowThickness, arrowOffsetX + 0.5f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 5
            {
                { 0.0f, arrowOffsetX + 1.0f - arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pArrowRightVertexBuf)))
        {
            HandleFailure();
        }
    }
    // Arrow down
    {
        const unsigned int vertsNum = 6;

        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsNum;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float arrowOffsetX = -0.5f;
        const float arrowThickness = 2 * 1 / 10.0f; // Glyph size is 10x10, 1/10 is one pixel
        const float arrowHeight = 0.5f;
        VSInput verts[vertsNum] =
        {
            // 0
            {
                { arrowHeight, arrowOffsetX + 0.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 1
            {
                { 0.0f, arrowOffsetX + 0.5f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 2
            {
                { arrowHeight, arrowOffsetX + 1.0f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 3
            {
                { arrowHeight, arrowOffsetX + 1.0f - arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 4
            {
                { arrowThickness, arrowOffsetX + 0.5f, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            },

            // 5
            {
                { arrowHeight, arrowOffsetX + arrowThickness, 0.0f, 1.0f },{ solidFillTCX, solidFillTCY }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pArrowLeftVertexBuf)))
        {
            HandleFailure();
        }
    }

    // Camera icon buffer (uses pRectIndexBuf)
    {
        const unsigned int vertsPerButton = 4;
        const unsigned int indsPerButton = 6;

        unsigned int inds[indsPerButton] = { 0, 1, 2, 2, 3, 0 };

        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        VSInput verts[vertsPerButton] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(65)                       , getUIAtlasCoordV(66) }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(65) + getUIAtlasSizeU(64), getUIAtlasCoordV(66) }
            },

            // 2
            {
                { 1.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(65) + getUIAtlasSizeU(64), getUIAtlasCoordV(66) - getUIAtlasSizeV(64) }
            },

            // 3
            {
                { 0.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(65)                       , getUIAtlasCoordV(66) - getUIAtlasSizeV(64) }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pCamIcolwertexBuf)))
        {
            HandleFailure();
        }
    }
    // Mouse lwrsors buffers

    {
        const unsigned int indsPerButton = 6;
        D3D11_BUFFER_DESC indexBufferDesc;
        D3D11_SUBRESOURCE_DATA indexData;

        unsigned int inds[indsPerButton] = { 0, 1, 2, 2, 3, 0 };

        indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerButton;
        indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        indexBufferDesc.CPUAccessFlags = 0;
        indexBufferDesc.MiscFlags = 0;
        indexBufferDesc.StructureByteStride = 0;

        indexData.pSysMem = inds;
        indexData.SysMemPitch = 0;
        indexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pMouseIndexBuf)))
        {
            HandleFailure();
        }
    }
    {
        const unsigned int vertsPerButton = 4;
        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float solidFillTCX = 0.0f, solidFillTCY = 0.0f;
        VSInput verts[vertsPerButton] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(260)                      , getUIAtlasCoordV(67) }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(260) + getUIAtlasSizeU(64), getUIAtlasCoordV(67) }
            },

            // 2
            {
                { 1.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(260) + getUIAtlasSizeU(64), getUIAtlasCoordV(67) - getUIAtlasSizeV(64) }
            },

            // 3
            {
                { 0.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(260)                      , getUIAtlasCoordV(67) - getUIAtlasSizeV(64) }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pMousePointerVertexBuf)))
        {
            HandleFailure();
        }
    }
    {
        const unsigned int vertsPerButton = 4;
        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const float solidFillTCX = 0.0f, solidFillTCY = 0.0f;
        VSInput verts[vertsPerButton] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(325)                  , getUIAtlasCoordV(67) }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(325) + getUIAtlasSizeU(64), getUIAtlasCoordV(67) }
            },

            // 2
            {
                { 1.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(325) + getUIAtlasSizeU(64), getUIAtlasCoordV(67) - getUIAtlasSizeV(64) }
            },

            // 3
            {
                { 0.0f, 1.0f, 0.0f, 1.0f },{ getUIAtlasCoordU(325)                      , getUIAtlasCoordV(67) - getUIAtlasSizeV(64) }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pMouseHandVertexBuf)))
        {
            HandleFailure();
        }
    }

    // Get various translated labels
    m_textGB = i18n::getLocalizedString(IDS_RESOLUTION_GB, m_langID);
    m_textFilterNone = i18n::getLocalizedString(IDS_FILTER_NONE, m_langID);
    m_textFilterLwstom = i18n::getLocalizedString(IDS_FILTER_LWSTOM, m_langID);
    m_textKindRegular = i18n::getLocalizedString(IDS_CAPTYPE_REGULAR, m_langID);
    m_textKindRegularHDR = L"EXR";
    m_textKindHighRes = i18n::getLocalizedString(IDS_CAPTYPE_HIGHRES, m_langID);
    m_textKind360 = i18n::getLocalizedString(IDS_CAPTYPE_360, m_langID);
    m_textKindStereo = i18n::getLocalizedString(IDS_CAPTYPE_STEREO, m_langID);
    m_textKind360Stereo = i18n::getLocalizedString(IDS_CAPTYPE_360STEREO, m_langID);
    m_textProgress = i18n::getLocalizedString(IDS_CAPTYPE_PROGRESS, m_langID);
    m_textAbort = i18n::getLocalizedString(IDS_ABORT, m_langID);
    m_textDone = i18n::getLocalizedString(IDS_DONE, m_langID);

    m_textFilter = i18n::getLocalizedString(IDS_FILTER, m_langID).c_str();
    m_textFilterType = i18n::getLocalizedString(IDS_FILTER_TYPE, m_langID);

#ifdef ENABLE_STYLETRANSFER
    m_textStyleNetLow = i18n::getLocalizedString(IDS_STYLE_NETLOW, m_langID);
    m_textStyleNetHigh = i18n::getLocalizedString(IDS_STYLE_NETHIGH, m_langID);

    m_textStyleProgress = i18n::getLocalizedString(IDS_PROGRESS, m_langID) + L":%3d%%";
    m_textStyleInstalling = i18n::getLocalizedString(IDS_STYLE_INSTALLING, m_langID);
#endif

    std::wstring intermNotifStorage;
    const size_t intermNotifStorageSize = 2048;
    intermNotifStorage.resize(intermNotifStorageSize);

    m_textNotifWelcome = i18n::getLocalizedString(IDS_NOTIF_WELCOME, m_langID);
    m_textNotifSessDeclined = i18n::getLocalizedString(IDS_NOTIF_SESSDECLINE, m_langID);
    m_textNotifDrvUpdate = i18n::getLocalizedString(IDS_NOTIF_DRVUPDATE, m_langID);

    m_lwGreenColor = ui::ColorF4(0x76 / 255.f, 0xb9 / 255.f, 0.000f, 1.000f);
    m_doneDoneColor = ui::ColorF4(0.196f, 0.212f, 0.231f, 1.000f);
    m_doneAbortColor = ui::ColorF4(0.623f, 0.031f, 0.059f, 1.000f);
    m_doneAbortColorBright = ui::ColorF4(1.0f, 0.05f, 0.094f, 1.000f);

    // Buttons
    struct ButtonLayout
    {
        float left, bottom;
        float sizeX, sizeY;
        float tcLeft, tcBottom;
        float tcSizeX, tcSizeY;
    };

    const float buttonHop = 0.025f;
    ButtonLayout defaultButton_lo = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    m_hotkeyModifierShift = false;
    m_hotkeyModifierCtrl = false;
    m_hotkeyModifierAlt = true;
    m_hotkeyModifierVKey = VK_F2;

    m_components.lblHotkey = new ui::ControlLabel();
    // [Shift] + [Ctrl] + [Alt] + F12
    m_components.lblHotkey->caption = allocateUIString(m_components.m_lblHotkeyCaptionMaxSize);

    // "S"
    m_components.btnHotkeySetup = new ui::ControlButtonClickable(&m_onButtonHotkeySetupClick);
    m_components.btnHotkeySetup->color = ui::ColorF4(0.196f, 0.212f, 0.231f, 1.000f);
    m_components.btnHotkeySetup->caption = L"E";
#endif

    // Temporary toggle test button
    m_components.btnToggleFilter = new ui::ControlButtonToggle(&m_onButtonToggleClick);
    m_components.btnToggleAdjustments = new ui::ControlButtonToggle(&m_onButtonToggleClick);
    m_components.btnToggleFX = new ui::ControlButtonToggle(&m_onButtonToggleClick);
#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer = new ui::ControlButtonToggle(&m_onButtonToggleClick);
#endif
    m_components.btnToggleGameSpecific = new ui::ControlButtonToggle(&m_onButtonToggleClick);
    m_components.btnToggleCamCapture = new ui::ControlButtonToggle(&m_onButtonToggleClick);

    // "Special FX" - flyout
    m_components.flySpecialFX = new ui::ControlFlyoutToggleShared(&m_onFlyoutClick, &m_onFlyoutSpecialFXChange);
    m_components.flySpecialFX->labelsStorage = &m_components.flySpecialFXLabels;

    // "Snap"
    m_components.btnSnap = new ui::ControlButtonSnap(&m_onButtonSnapClick);

    // "Done"
    m_components.btnDone = new ui::ControlButtonHide(&m_onButtonHideClick);

    // Sliders
    m_components.btnResetRoll = new ui::ControlButtonClickable(&m_onButtonResetRollClick);

    // Sliders share same vertex buffers

    // "Kind"
    m_components.sldKind = new ui::ControlSliderKind(&m_onSliderKindChange);

    // "HDR"
    m_components.chkHDR = new ui::ControlCheckbox();

    // "Grid of Thirds"
    m_components.chkGridOfThirds = new ui::ControlCheckbox();

    m_components.chkAllowModding = new ui::ControlCheckbox();
    ui::ControlCheckbox * chkAllowModding = static_cast<ui::ControlCheckbox *>(m_components.chkAllowModding);
    const std::wstring title(L"Filters in game");
    chkAllowModding->setTitle(title.c_str());
    chkAllowModding->isVisible = false;

    // "FOV"
    m_components.sldFOV = new ui::ControlSliderFOV;

    // "Roll"
    m_components.sldRoll = new ui::ControlSliderRoll;
    m_components.sldRoll->defaultValue = 0.5f;
    m_components.sldRoll->stickyValue = 0.5f;
    m_components.sldRoll->stickyRegion = 0.01f;
    m_components.sldRoll->step = 0.001f;

    // "HiRes Coeff"
    m_components.sldHiResMult = new ui::ControlSliderDiscr;

    // "360 quality"
    m_components.sldSphereFOV = new ui::ControlSlider360Quality(static_cast<ui::ControlSliderKind*>(m_components.sldKind), m_pAnselServer, m_textGB.c_str());

    // "Enhance (post-process) high-resolution captures"
    m_components.chkEnhanceHiRes = new ui::ControlCheckbox();

#ifdef ENABLE_STYLETRANSFER
    m_components.cntStyleTransfer = new ui::ControlContainer;

    if (!m_isStyleTransferAllowed)
    {
        m_components.btnToggleStyleTransfer->isVisible = false;
        m_components.cntStyleTransfer->isVisible = false;
    }
    else
    {
        m_components.btnToggleStyleTransfer->isVisible = true;
        m_components.cntStyleTransfer->isVisible = true;
    }

    m_components.chkEnableStyleTransfer = new ui::ControlCheckbox();
    ui::ControlCheckbox * chkEnableStyleTransfer = static_cast<ui::ControlCheckbox *>(m_components.chkEnableStyleTransfer);
    chkEnableStyleTransfer->setTitle(i18n::getLocalizedString(IDS_STYLE_ENABLE, m_langID).c_str());

    // "Styles" - flyout
    m_components.flyStyles = new ui::ControlFlyoutStylesToggle(&m_onFlyoutClick, &m_onFlyoutStylesChange);
    // "Network"
    m_components.flyStyleNetworks = new ui::ControlFlyoutStylesToggle(&m_onFlyoutClick, &m_onFlyoutStyleNetworksChange);
    
    // Download dialog
    m_components.dlgDownloadRestyle = new ui::ControlContainer;
    m_components.dlgDownloadRestyle->isVisible = false;

    m_components.lblDownloadRestyleText = new ui::ControlLabel();
    m_components.btnDownloadRestyleConfirm = new ui::ControlButtonClickable(&m_onButtonDownloadRestyleConfirmClick);
    m_components.btnDownloadRestyleCancel = new ui::ControlButtonClickable(&m_onButtonDownloadRestyleCancelClick);
    
    // Setting up progress controls (progress bar and progress indicator)
    m_components.pbRestyleProgress = new ui::ControlProgressBar();
    
    m_components.cntRestyleProgress = new ui::ControlContainer;
    m_components.cntRestyleProgress->m_isClipping = true;// false;
    m_components.cntRestyleProgress->m_isScrollable = false;// false;
    m_components.cntRestyleProgress->isInteractible = true;// false;
    m_components.cntRestyleProgress->m_needsContentAutoPlacement = false;
    m_components.cntRestyleProgress->sizeY = 0.0f;

    m_components.lblRestyleProgress = new ui::ControlLabel;
    m_components.lblRestyleProgress->fontAlignment = ui::ControlBase::FontAlignment::kLeft;
    m_components.lblRestyleProgress->caption = L"";

    m_components.lblRestyleProgressIndicator = new ui::ControlLabel;
    m_components.lblRestyleProgressIndicator->fontAlignment = ui::ControlBase::FontAlignment::kCenter;
    //m_components.lblRestyleProgressIndicator->caption = L"\u23F0 \u23F1 \u23F2 \u23F4 \u231A \u231B";
    m_components.lblRestyleProgressIndicator->caption = L"\u231B";
    m_components.lblRestyleProgressIndicator->fontColor = m_doneAbortColorBright;
    m_components.lblRestyleProgressIndicator->isVisible = false;
#endif

    // top slider, "kind" for darkroom
    m_components.sldKind->setTitle(i18n::getLocalizedString(IDS_CAPTURETYPE, m_langID).c_str());
    m_components.sldKind->isLeanStyle = false;

    ui::ControlSliderKind * sldKind = static_cast<ui::ControlSliderKind *>(m_components.sldKind);
    sldKind->setupLabels(&m_textKindRegular[0], &m_textKindHighRes[0], &m_textKind360[0], &m_textKindStereo[0], &m_textKind360Stereo[0]);

    ui::ControlCheckbox * chkHDR = static_cast<ui::ControlCheckbox *>(m_components.chkHDR);
    chkHDR->setTitle(i18n::getLocalizedString(IDS_HDR, m_langID).c_str());

    ui::ControlCheckbox * chkGridOfThirds = static_cast<ui::ControlCheckbox *>(m_components.chkGridOfThirds);
    chkGridOfThirds->setTitle(i18n::getLocalizedString(IDS_GRIDOFTHIRDS, m_langID).c_str());

    ui::ControlCheckbox * chkEnhanceHiRes = static_cast<ui::ControlCheckbox *>(m_components.chkEnhanceHiRes);
    chkEnhanceHiRes->setTitle(i18n::getLocalizedString(IDS_ENHANCEHIGHRES, m_langID).c_str());

    // "FOV" slider for camera
    m_components.sldFOV->setTitle(i18n::getLocalizedString(IDS_FOV, m_langID).c_str());
    m_components.sldFOV->percentage = 0.5f;

    // "Roll" slider for camera
    m_components.sldRoll->setTitle(i18n::getLocalizedString(IDS_ROLL, m_langID).c_str());
    m_components.sldRoll->percentage = 0.5f;

    // "HiRes Coeff"
    m_components.sldHiResMult->setTitle(i18n::getLocalizedString(IDS_RESOLUTION_HR, m_langID).c_str());
    m_components.sldHiResMultStringBufSize = 64;
    m_components.sldHiResMult->isEnabled = false;
    m_components.sldHiResMult->isVisible = true;
    m_components.sldHiResMult->isLeanStyle = false;

    // "360 quality"
    m_components.sldSphereFOV->setTitle(i18n::getLocalizedString(IDS_RESOLUTION_360, m_langID).c_str());
    m_components.sldSphereFOV->percentage = 0.0f;
    m_components.sldSphereFOV->isEnabled = false;
    m_components.sldSphereFOV->isVisible = false;
    m_components.sldSphereFOV->isLeanStyle = false;

    // Icons
    // Shouild be Y-flipped

    struct IconParameters
    {
        ButtonLayout layout;
        int pixelSizeX, pixelSizeY;
    };

    auto colwertPixelCoordsToParameters = [&](int xleft, int ybottom, int xright, int ytop) -> IconParameters
    {
        IconParameters icoParams;

        icoParams.layout.left = 0.0f;
        icoParams.layout.bottom = 0.0f;
        icoParams.layout.sizeX = 0.0f;
        icoParams.layout.sizeY = 0.0f;
        icoParams.layout.tcLeft = getUIAtlasCoordU(xleft);
        icoParams.layout.tcBottom = getUIAtlasCoordV(ybottom);
        icoParams.layout.tcSizeX = getUIAtlasSizeU(xright - xleft);
        icoParams.layout.tcSizeY = getUIAtlasSizeV(ytop - ybottom);

        icoParams.pixelSizeX = xright - xleft;
        icoParams.pixelSizeY = ybottom - ytop;

        return icoParams;
    };

    // Second version of LW ANSEL LOGO
    std::vector<IconParameters> lwAnselLogoParameters;

    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(0, 155, 424, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(426, 142, 787, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(789, 133, 1107, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(1109, 126, 1392, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(1394, 119, 1642, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(1644, 112, 1857, 67));
    lwAnselLogoParameters.push_back(colwertPixelCoordsToParameters(1859, 105, 2040, 67));

    m_components.icoLWLogo = new ui::ControlIcon;

    m_components.leftPane = new ui::ControlContainer;
    m_components.flyoutPane = new ui::ControlFlyoutContainer;
    m_components.cntControls = new ui::ControlContainer;
    m_components.cntFilter = new ui::ControlContainer;
    m_components.cntAdjustments = new ui::ControlContainer;
    m_components.cntFX = new ui::ControlContainer;

    m_components.cntCameraCapture = new ui::ControlContainer;
    m_components.cntGameSpecific = new ui::ControlContainer;

#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        m_components.btnAddFilter = new ui::ControlButtonClickable(&m_onButtonAddFilterClick);
        m_components.btnAddFilter->m_anchorY = m_components.cntFX;
    }
#endif

    m_components.flyoutPane->isVisible = false;

    m_components.cntFilter->m_needsContentAutoPlacement = true;
    m_components.cntAdjustments->m_needsContentAutoPlacement = true;
    m_components.cntFX->m_needsContentAutoPlacement = true;
    m_components.cntGameSpecific->m_needsContentAutoPlacement = true;

    m_components.cntFilter->m_renderSideVLine = true;
    m_components.cntAdjustments->m_renderSideVLine = true;
    m_components.cntFX->m_renderSideVLine = true;
#ifdef ENABLE_STYLETRANSFER
    m_components.cntStyleTransfer->m_renderSideVLine = true;
#endif
    m_components.cntCameraCapture->m_renderSideVLine = true;
    m_components.cntGameSpecific->m_renderSideVLine = true;

    // Hide Filter, Adjustments and SpecialFX containers by default
    m_components.cntFilter->isVisible = false;
    m_components.cntAdjustments->isVisible = false;
    m_components.cntFX->isVisible = false;

    // Hide Game Settings container and toggle button
    m_components.btnToggleGameSpecific->isVisible = false;
    m_components.cntGameSpecific->isVisible = false;

    m_components.btnToggleFilter->m_containerToggle = m_components.cntFilter;
    m_components.btnToggleAdjustments->m_containerToggle = m_components.cntAdjustments;
    m_components.btnToggleFX->m_containerToggle = m_components.cntFX;
#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer->m_containerToggle = m_components.cntStyleTransfer;
#endif
    m_components.btnToggleGameSpecific->m_containerToggle = m_components.cntGameSpecific;
    m_components.btnToggleCamCapture->m_containerToggle = m_components.cntCameraCapture;

    m_components.cntFilter->addControl(m_components.flySpecialFX);

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    m_components.cntControls->addControl(m_components.lblHotkey);
    m_components.cntControls->addControl(m_components.btnHotkeySetup);
#endif
    m_components.cntControls->addControl(m_components.btnToggleFilter);
    m_components.cntControls->addControl(m_components.cntFilter);
    m_components.cntControls->addControl(m_components.btnToggleAdjustments);
    m_components.cntControls->addControl(m_components.cntAdjustments);
    m_components.cntControls->addControl(m_components.btnToggleFX);
    m_components.cntControls->addControl(m_components.cntFX);
#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        m_components.cntControls->addControl(m_components.btnAddFilter);
    }
#endif
#ifdef ENABLE_STYLETRANSFER
    m_components.cntControls->addControl(m_components.btnToggleStyleTransfer);
    m_components.cntControls->addControl(m_components.cntStyleTransfer);
#endif
    m_components.cntControls->addControl(m_components.btnToggleGameSpecific);
    m_components.cntControls->addControl(m_components.cntGameSpecific);
    m_components.cntControls->addControl(m_components.btnToggleCamCapture);
    m_components.cntControls->addControl(m_components.cntCameraCapture);

#ifdef ENABLE_STYLETRANSFER
    m_components.cntControls->addControl(m_components.lblRestyleProgressIndicator);
    m_components.cntStyleTransfer->addControl(m_components.cntRestyleProgress);
    m_components.cntRestyleProgress->addControl(m_components.lblRestyleProgress);
    m_components.cntRestyleProgress->addControl(m_components.pbRestyleProgress);
#endif
    m_components.leftPane->addControl(m_components.cntControls);
    m_components.leftPane->addControl(m_components.chkAllowModding);
    m_components.leftPane->addControl(m_components.btnSnap);
    m_components.leftPane->addControl(m_components.btnDone);
    m_components.leftPane->addControl(m_components.icoLWLogo);
    m_components.cntCameraCapture->addControl(m_components.sldFOV);
    m_components.cntCameraCapture->addControl(m_components.sldRoll);
    m_components.cntCameraCapture->addControl(m_components.btnResetRoll);
    m_components.cntCameraCapture->addControl(m_components.sldKind);
    m_components.cntCameraCapture->addControl(m_components.chkGridOfThirds);
    m_components.cntCameraCapture->addControl(m_components.chkHDR);
    m_components.cntCameraCapture->addControl(m_components.sldHiResMult);
    m_components.cntCameraCapture->addControl(m_components.sldSphereFOV);
    m_components.cntCameraCapture->addControl(m_components.chkEnhanceHiRes);

#ifdef ENABLE_STYLETRANSFER
    m_components.cntStyleTransfer->addControl(m_components.chkEnableStyleTransfer);
    m_components.cntStyleTransfer->addControl(m_components.flyStyles);
    m_components.cntStyleTransfer->addControl(m_components.flyStyleNetworks);
#endif
    mainContainer.addControl(m_components.leftPane);
    mainContainer.addControl(m_components.flyoutPane);
#ifdef ENABLE_STYLETRANSFER
    mainContainer.addControl(m_components.dlgDownloadRestyle);

    m_components.dlgDownloadRestyle->addControl(m_components.lblDownloadRestyleText);
    m_components.dlgDownloadRestyle->addControl(m_components.btnDownloadRestyleConfirm);
    m_components.dlgDownloadRestyle->addControl(m_components.btnDownloadRestyleCancel);

    m_components.lblDownloadRestyleText->tabStop = TABSTOP_INIT * 3;    // 300
    m_components.btnDownloadRestyleConfirm->tabStop = TABSTOP_INIT * 3 - TABSTOP_STRIDE;    // 300
    m_components.btnDownloadRestyleCancel->tabStop = TABSTOP_INIT * 3 - 2 * TABSTOP_STRIDE; // 300
#endif

    m_components.leftPane->tabStop = TABSTOP_INIT * 3;  // 300
    m_components.flyoutPane->tabStop = TABSTOP_INIT * 3 - TABSTOP_STRIDE;   // 300
#ifdef ENABLE_STYLETRANSFER
    m_components.dlgDownloadRestyle->tabStop = TABSTOP_INIT * 3 - 2*TABSTOP_STRIDE; // 300
#endif

    // Order of selections
    m_components.icoLWLogo->tabStop = 0;
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    m_components.lblHotkey->tabStop = 0;
    m_components.btnHotkeySetup->tabStop = 0;
#endif
    m_components.cntControls->tabStop = TABSTOP_INIT * 3;   // 300

    m_components.flySpecialFX->tabStop = TABSTOP_INIT * 100;                            // Biggest tabstop, since this one should be at the top always

    m_components.btnToggleFilter->tabStop = TABSTOP_INIT * 10 - TABSTOP_STRIDE;         // 290
    m_components.cntFilter->tabStop = TABSTOP_INIT * 10 - 2*TABSTOP_STRIDE;             // 280
    m_components.btnToggleAdjustments->tabStop = TABSTOP_INIT * 10 - 3*TABSTOP_STRIDE;  // 270
    m_components.cntAdjustments->tabStop = TABSTOP_INIT * 10 - 4*TABSTOP_STRIDE;        // 260
    m_components.btnToggleFX->tabStop = TABSTOP_INIT * 10 - 5*TABSTOP_STRIDE;           // 250
    m_components.cntFX->tabStop = TABSTOP_INIT * 10 - 6*TABSTOP_STRIDE;                 // 240
#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        m_components.btnAddFilter->tabStop = TABSTOP_INIT * 10 - 70*TABSTOP_STRIDE;     // 240
    }
#endif

#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer->tabStop = TABSTOP_INIT * 10 - 70*TABSTOP_STRIDE;   // 250
    m_components.lblRestyleProgressIndicator->tabStop = TABSTOP_INIT * 10 - 70*TABSTOP_STRIDE;
    m_components.cntStyleTransfer->tabStop = TABSTOP_INIT * 10 - 71*TABSTOP_STRIDE;     // 240
#endif
    m_components.btnToggleGameSpecific->tabStop = TABSTOP_INIT * 10 - 73*TABSTOP_STRIDE;// 230
    m_components.cntGameSpecific->tabStop = TABSTOP_INIT * 10 - 74*TABSTOP_STRIDE;      // 220
    m_components.btnToggleCamCapture->tabStop = TABSTOP_INIT * 10 - 75*TABSTOP_STRIDE;  // 230
    m_components.cntCameraCapture->tabStop = TABSTOP_INIT * 10 - 76*TABSTOP_STRIDE;     // 220

#ifdef ENABLE_STYLETRANSFER
    m_components.cntRestyleProgress->tabStop = TABSTOP_INIT * 10 - 75 * TABSTOP_STRIDE;
    m_components.lblRestyleProgress->tabStop = TABSTOP_INIT * 10 - 74 * TABSTOP_STRIDE;
    m_components.pbRestyleProgress->tabStop = TABSTOP_INIT * 10 - 75 * TABSTOP_STRIDE;
#endif

    int tabStop = TABSTOP_INIT;
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    m_components.btnHotkeySetup->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
#endif
    m_components.btnDone->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.btnSnap->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.chkAllowModding->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.chkEnhanceHiRes->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.sldSphereFOV->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.sldHiResMult->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.sldKind->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.chkHDR->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.chkGridOfThirds->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.btnResetRoll->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.sldRoll->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.sldFOV->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
#ifdef ENABLE_STYLETRANSFER
    m_components.flyStyleNetworks->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.flyStyles->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
    m_components.chkEnableStyleTransfer->tabStop = tabStop;
    tabStop += TABSTOP_STRIDE;
#endif
    applyTabStop();

    m_components.btnToggleCamCapture->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleCamCapture->caption = i18n::getLocalizedString(IDS_CAMERACAPTURE, m_langID).c_str();
    m_components.btnToggleCamCapture->isBold = true;

    m_components.btnToggleFilter->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleFilter->caption = i18n::getLocalizedString(IDS_FILTER, m_langID).c_str();
    m_components.btnToggleFilter->isBold = true;

    m_components.btnToggleAdjustments->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleAdjustments->caption = i18n::getLocalizedString(IDS_ADJUSTMENTS, m_langID).c_str();
    m_components.btnToggleAdjustments->isBold = true;

    m_components.btnToggleFX->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleFX->caption = i18n::getLocalizedString(IDS_SPECIALFX, m_langID).c_str();
    m_components.btnToggleFX->isBold = true;

#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleStyleTransfer->caption = i18n::getLocalizedString(IDS_STYLE_TRANSFER, m_langID).c_str();//L"FX";
    m_components.btnToggleStyleTransfer->isBold = true;
#endif
    m_components.btnToggleGameSpecific->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 1.000f);
    m_components.btnToggleGameSpecific->caption = i18n::getLocalizedString(IDS_GAMESETTINGS, m_langID).c_str();
    m_components.btnToggleGameSpecific->isBold = true;

    m_components.flySpecialFX->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    m_components.flySpecialFX->renderType = ui::ControlButton::RenderType::kFlyoutToggle;
    m_components.flySpecialFX->caption = m_textFilterType.c_str();
    m_components.flySpecialFX->isBold = false;

#ifdef ENABLE_STYLETRANSFER
    m_components.flyStyles->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    m_components.flyStyles->renderType = ui::ControlButton::RenderType::kFlyoutToggle;
    m_components.flyStyles->caption = i18n::getLocalizedString(IDS_STYLE, m_langID).c_str();//L"Style";
    m_components.flyStyles->isBold = false;

    m_components.flyStyleNetworks->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
    m_components.flyStyleNetworks->renderType = ui::ControlButton::RenderType::kFlyoutToggle;
    m_components.flyStyleNetworks->caption = i18n::getLocalizedString(IDS_STYLE_NETQ, m_langID).c_str();
    m_components.flyStyleNetworks->isBold = false;
#endif

    m_components.btnSnap->color = m_lwGreenColor;
    m_components.btnSnap->caption = i18n::getLocalizedString(IDS_SNAP, m_langID).c_str();
    m_components.btnSnap->isBold = true;

    m_components.btnDone->color = m_doneDoneColor;
    m_components.btnDone->caption = m_textDone.c_str();
    m_components.btnDone->isBold = false;

    m_components.btnResetRoll->color = m_doneDoneColor;
    m_components.btnResetRoll->caption = L"Reset roll";
    m_components.btnResetRoll->isBold = false;

#ifdef ENABLE_STYLETRANSFER
    m_components.lblDownloadRestyleText->caption = const_cast<wchar_t*>(i18n::getLocalizedString(IDS_STYLE_WOULDYOULIKEDOWNLOAD, getLangId()).c_str());

    m_components.btnDownloadRestyleConfirm->color = m_lwGreenColor;
    m_components.btnDownloadRestyleConfirm->caption = const_cast<wchar_t*>(i18n::getLocalizedString(IDS_STYLE_YES, getLangId()).c_str());
    m_components.btnDownloadRestyleConfirm->isBold = true;

    m_components.btnDownloadRestyleCancel->color = m_doneDoneColor;
    m_components.btnDownloadRestyleCancel->caption = const_cast<wchar_t*>(i18n::getLocalizedString(IDS_STYLE_NO, getLangId()).c_str());
    m_components.btnDownloadRestyleCancel->isBold = false;
#endif

#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        m_components.btnAddFilter->color = ui::ColorF4(0.0f, 0.0f, 0.0f, 0.0f);
        m_components.btnAddFilter->hlType = ui::ControlButton::HighlightType::kFont;
        m_components.btnAddFilter->hlColor = m_lwGreenColor;
        m_components.btnAddFilter->fontAlignment = ui::ControlBase::FontAlignment::kRight;
        //m_components.btnAddFilter->caption = L"+";
        m_components.btnAddFilter->caption = L"\u2795"; // Heavy plus sign
        m_components.btnAddFilter->isBold = true;
    }
#endif

#if (DISABLE_STATIC_FILTERS == 1)
    m_components.btnToggleFilter->isVisible = false;
    m_components.cntFilter->isVisible = false;
    m_components.btnToggleAdjustments->isVisible = false;
    m_components.cntAdjustments->isVisible = false;
    m_components.btnToggleFX->isVisible = false;
    m_components.cntFX->isVisible = false;
#endif

    int checkboxCnt = 0;
    int buttonCnt = 0;
    int iconCnt = 0;
    int sliderCnt = 0;


#ifdef _DEBUG
    {
        m_components.btnToggleFilter->m_DBGname = L"btnToggleFilter";
        m_components.btnToggleAdjustments->m_DBGname = L"btnToggleAdjustments";
        m_components.btnToggleFX->m_DBGname = L"btnToggleFX";
        m_components.btnToggleCamCapture->m_DBGname = L"btnToggleCamCapture";
        m_components.btnToggleGameSpecific->m_DBGname = L"btnToggleGameSpecific";

        m_components.flySpecialFX->m_DBGname = L"flySpecialFX";

        m_components.btnSnap->m_DBGname = L"btnSnap";
        m_components.btnDone->m_DBGname = L"btnDone";

        m_components.sldKind->m_DBGname = L"sldKind";
        m_components.chkGridOfThirds->m_DBGname = L"chkGridOfThirds";
        m_components.chkHDR->m_DBGname = L"chkHDR";

        m_components.sldFOV->m_DBGname = L"sldFOV";
        m_components.sldRoll->m_DBGname = L"sldRoll";

        m_components.btnResetRoll->m_DBGname = L"btnResetRoll";

        m_components.sldHiResMult->m_DBGname = L"sldHiResMult";
        m_components.sldSphereFOV->m_DBGname = L"sldSphereFOV";

        m_components.chkEnhanceHiRes->m_DBGname = L"chkEnhanceHiRes";

        m_components.icoLWLogo->m_DBGname = L"icoLWLogo";
        //m_components.icoCamera->m_DBGname = L"icoCamera";
        //m_components.icoLight->m_DBGname = L"icoLight";
        //m_components.icoFilters->m_DBGname = L"icoFilters";

        m_components.leftPane->m_DBGname = L"leftPane";
        m_components.flyoutPane->m_DBGname = L"flyoutPane";

        m_components.cntControls->m_DBGname = L"cntControls";
        m_components.cntFilter->m_DBGname = L"cntFilter";
        m_components.cntAdjustments->m_DBGname = L"cntAdjustments";
        m_components.cntFX->m_DBGname = L"cntFX";
        m_components.cntGameSpecific->m_DBGname = L"cntGameSpecific";
        m_components.cntCameraCapture->m_DBGname = L"cntCameraCapture";

        m_components.chkAllowModding->m_DBGname = L"chkAllowModding";
    }
#endif

    ui::ControlBase * lwrContainer;
    containerHelper.init(&mainContainer);
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->getType() == ui::ControlType::kButton)
        {
            ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);

            const unsigned int vertsPerButton = 4;
            const unsigned int indsPerButton = 6;

            unsigned int inds[indsPerButton] = { 0, 1, 2, 2, 3, 0 };

            lwrButton->state = UI_CONTROL_ORDINARY;

            ButtonLayout buttonLayout = defaultButton_lo;

            D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
            D3D11_SUBRESOURCE_DATA vertexData, indexData;

            indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
            indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerButton;
            indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
            indexBufferDesc.CPUAccessFlags = 0;
            indexBufferDesc.MiscFlags = 0;
            indexBufferDesc.StructureByteStride = 0;

            indexData.pSysMem = inds;
            indexData.SysMemPitch = 0;
            indexData.SysMemSlicePitch = 0;

            if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pIndexBuf)))
            {
                HandleFailure();
            }

            vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
            vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
            vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            vertexBufferDesc.CPUAccessFlags = 0;
            vertexBufferDesc.MiscFlags = 0;
            vertexBufferDesc.StructureByteStride = 0;

            //  const float subPixelAddX = -0.5f/w, subPixelAddY = -0.5f/h;
            const float subPixelAddX = 0.0f, subPixelAddY = 0.0f;
            VSInput verts[vertsPerButton] =
            {
                // 0
                {
                    { 0.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                },

                // 1
                {
                    { 1.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                },

                // 2
                {
                    { 1.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom - buttonLayout.tcSizeY + subPixelAddY }
                },

                // 3
                {
                    { 0.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom - buttonLayout.tcSizeY + subPixelAddY }
                }
            };

            // Ordinary
            vertexData.pSysMem = verts;
            vertexData.SysMemPitch = 0;
            vertexData.SysMemSlicePitch = 0;

            if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pVertexBuf)))
            {
                HandleFailure();
            }

            // We have similar index buffers, so we can reuse rectangle buffers here
            lwrButton->renderBufsShared = false;    // Own idx and vertex buffers
            lwrButton->renderBufsAuxShared = true;  // HL/DN buffers shared
            lwrButton->pIndexBuf = pIndexBuf;
            lwrButton->pVertexBuf = pVertexBuf;
            lwrButton->pVertexBufHl = pRectVertexBuf;
            lwrButton->pVertexBufDn = pRectVertexBuf;

            ++buttonCnt;
        }
        if (lwrContainer->getType() == ui::ControlType::kCheckbox)
        {
            ui::ControlCheckbox * lwrCheckbox = static_cast<ui::ControlCheckbox *>(lwrContainer);

            lwrCheckbox->isVisible = true;
            lwrCheckbox->state = UI_CONTROL_ORDINARY;

            ++checkboxCnt;
        }
        else if (lwrContainer->getType() == ui::ControlType::kIcon)
        {
            ui::ControlIcon * lwrIcon = static_cast<ui::ControlIcon *>(lwrContainer);

            const unsigned int vertsPerButton = 4;
            const unsigned int indsPerButton = 6;

            unsigned int inds[indsPerButton] = { 0, 1, 2, 2, 3, 0 };

            lwrIcon->color = ui::ColorF4(1.0f, 1.0f, 1.0f, 1.0f);
            lwrIcon->isVisible = true;

            D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
            D3D11_SUBRESOURCE_DATA vertexData, indexData;

            indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
            indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerButton;
            indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
            indexBufferDesc.CPUAccessFlags = 0;
            indexBufferDesc.MiscFlags = 0;
            indexBufferDesc.StructureByteStride = 0;

            indexData.pSysMem = inds;
            indexData.SysMemPitch = 0;
            indexData.SysMemSlicePitch = 0;

            if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &pIndexBuf)))
            {
                HandleFailure();
            }

            vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
            vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
            vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            vertexBufferDesc.CPUAccessFlags = 0;
            vertexBufferDesc.MiscFlags = 0;
            vertexBufferDesc.StructureByteStride = 0;

            if (iconCnt == 0)
            {
                // Logo icon, features manual mipmapping
                ui::ControlIcon::VertexBufDesc vbDesc;

                for (size_t loi = 0, loiend = lwAnselLogoParameters.size(); loi < loiend; ++loi)
                {
                    ButtonLayout & buttonLayout = lwAnselLogoParameters[loi].layout;

                    const float subPixelAddX = 0.0f, subPixelAddY = 0.0f;
                    VSInput verts[vertsPerButton] =
                    {
                        // 0
                        {
                            { 0.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                        },

                        // 1
                        {
                            { 1.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                        },

                        // 2
                        {
                            { 1.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom + buttonLayout.tcSizeY + subPixelAddY }
                        },

                        // 3
                        {
                            { 0.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom + buttonLayout.tcSizeY + subPixelAddY }
                        }
                    };

                    // Ordinary
                    vertexData.pSysMem = verts;
                    vertexData.SysMemPitch = 0;
                    vertexData.SysMemSlicePitch = 0;

                    if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pVertexBuf)))
                    {
                        HandleFailure();
                    }

                    vbDesc.pVertexBuf = pVertexBuf;
                    vbDesc.pixelSizeX = lwAnselLogoParameters[loi].pixelSizeX;
                    vbDesc.pixelSizeY = lwAnselLogoParameters[loi].pixelSizeY;
                    lwrIcon->vertexBufDescs.push_back(vbDesc);
                }
            }
            else
            {
                ButtonLayout buttonLayout = { };

                const float subPixelAddX = 0.0f, subPixelAddY = 0.0f;
                VSInput verts[vertsPerButton] =
                {
                    // 0
                    {
                        { 0.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                    },

                    // 1
                    {
                        { 1.0f, 0.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom + subPixelAddY }
                    },

                    // 2
                    {
                        { 1.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + buttonLayout.tcSizeX + subPixelAddX, buttonLayout.tcBottom + buttonLayout.tcSizeY + subPixelAddY }
                    },

                    // 3
                    {
                        { 0.0f, 1.0f, 0.0f, 1.0f },{ buttonLayout.tcLeft + subPixelAddX, buttonLayout.tcBottom + buttonLayout.tcSizeY + subPixelAddY }
                    }
                };

                // Ordinary
                vertexData.pSysMem = verts;
                vertexData.SysMemPitch = 0;
                vertexData.SysMemSlicePitch = 0;

                if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &pVertexBuf)))
                {
                    HandleFailure();
                }
                ui::ControlIcon::VertexBufDesc vbDesc;
                vbDesc.pVertexBuf = pVertexBuf;
                vbDesc.pixelSizeX = 0;
                lwrIcon->vertexBufDescs.push_back(vbDesc);
            }

            lwrIcon->pIndexBuf = pIndexBuf;

            ++iconCnt;
        }
    }
    containerHelper.stopSearch();

    containerHelper.init(&mainContainer);
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
            lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
            lwrContainer->getType() == ui::ControlType::kSliderInt)
        {
            ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

            if (lwrSliderBase->getType() == ui::ControlType::kSliderCont)
            {
                ui::ControlSliderCont * slider = static_cast<ui::ControlSliderCont *>(lwrSliderBase);
                slider->percentage = 0.0f;
            }
            else if (lwrContainer->getType() == ui::ControlType::kSliderDiscr)
            {
                ui::ControlSliderDiscr * slider = static_cast<ui::ControlSliderDiscr *>(lwrSliderBase);
                slider->setSelected(0);
            }
            else if (lwrContainer->getType() == ui::ControlType::kSliderInt)
            {
                ui::ControlSliderInt * slider = static_cast<ui::ControlSliderInt *>(lwrSliderBase);
                slider->setSelected(0);
            }
            else
            {
                assert(false);
            }

            lwrSliderBase->state = UI_CONTROL_ORDINARY;
            lwrSliderBase->isVisible = true;
            lwrSliderBase->pIndexBuf = pRectIndexBuf;
            lwrSliderBase->pVertexBuf = pRectVertexBuf;

            ++sliderCnt;
        }
    }
    containerHelper.stopSearch();

    ui::ControlContainer * selectedControl = &mainContainer;
    while (selectedControl)
    {
        size_t controlsNum = selectedControl->getControlsNum();
        if (controlsNum > 0)
        {
            selectedControl->m_selectedControlIdx = 0; 
            selectedControl->state = UI_CONTROL_HIGHLIGHT;
            selectedControl = selectedControl->getControl(selectedControl->m_selectedControlIdx);
        }
        else
        {
            selectedControl->m_selectedControlIdx = -1; 
            selectedControl->state = UI_CONTROL_HIGHLIGHT;
            break;
        }
    }

    recallwlateUILayout(1920.f / 1080.f);

    setCWControlsVisibility(false);

    pPixelShader = pPS;
    pVertexShader = pVS;

#if (UI_ENABLE_TEXT == 1)
    IFW1Factory::initDWrite();

    // Initialize FW1 Font library
    IFW1Factory * pFW1Factory;
    if (!SUCCEEDED(status = FW1CreateFactory(FW1_VERSION, &pFW1Factory)))
    {
        HandleFailure();
    }

    IFW1FontWrapper * pFontWrapperSergoeUI;
    if (!SUCCEEDED(status = pFW1Factory->CreateFontWrapper(d3dDevice, L"Segoe UI", false, &pFontWrapperSergoeUI, pfnD3DCreateBlob)))
    {
        HandleFailureWMessage("%d", status);
    }

    IFW1FontWrapper * pFontWrapperSergoeUIBold;
    if (!SUCCEEDED(status = pFW1Factory->CreateFontWrapper(d3dDevice, L"Segoe UI", true, &pFontWrapperSergoeUIBold, pfnD3DCreateBlob)))
    {
        HandleFailureWMessage("%d", status);
    }

    // Get the DirectWrite factory used by the font-wrapper, for Segoe UI Bold (the widest)
    IDWriteFactory *pDWriteFactorySegoeUI;
    status = pFontWrapperSergoeUIBold->GetDWriteFactory(&pDWriteFactorySegoeUI);
    if (!SUCCEEDED(status))
    {
        HandleFailure();
    }
    IDWriteFactory *pDWriteFactorySegoeUIBold;
    status = pFontWrapperSergoeUIBold->GetDWriteFactory(&pDWriteFactorySegoeUIBold);
    if (!SUCCEEDED(status))
    {
        HandleFailure();
    }

    
    const float defaultFontSize = 16.0f;
    // Create the default DirectWrite text format to base layouts on
    IDWriteTextFormat  *pTextFormatSegoeUI;
    status = pDWriteFactorySegoeUI->CreateTextFormat(
                L"Segoe UI",
                NULL,
                DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                defaultFontSize,
                L"",
                &pTextFormatSegoeUI
                );
    if (!SUCCEEDED(status))
    {
        HandleFailure();
    }
    IDWriteTextFormat  *pTextFormatSegoeUIBold;
    status = pDWriteFactorySegoeUIBold->CreateTextFormat(
                L"Segoe UI",
                NULL,
                DWRITE_FONT_WEIGHT_SEMI_BOLD,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                defaultFontSize,
                L"",
                &pTextFormatSegoeUIBold
                );
    if (!SUCCEEDED(status))
    {
        HandleFailure();
    }

    IFW1GlyphRenderStates *pRenderStatesSergoeUI;
    pFontWrapperSergoeUI->GetRenderStates(&pRenderStatesSergoeUI);

    IFW1GlyphRenderStates *pRenderStatesSergoeUIBold;
    pFontWrapperSergoeUIBold->GetRenderStates(&pRenderStatesSergoeUIBold);

    // Custom Pixel Shader for FW1: Outline
    ID3D11PixelShader * pPSOutline;

    const BYTE * psTextByteCode = standaloneui_text_ps40::g_main;
    size_t psTextByteCodeSize = sizeof(standaloneui_text_ps40::g_main)/sizeof(BYTE);
    if (!SUCCEEDED(status = d3dDevice->CreatePixelShader(psTextByteCode, psTextByteCodeSize, NULL, &pPSOutline)))
    {
        HandleFailure();
    }

    FW1.pFontWrapperSergoeUI = pFontWrapperSergoeUI;
    FW1.pRenderStatesSergoeUI = pRenderStatesSergoeUI;
    FW1.pFontWrapperSergoeUIBold = pFontWrapperSergoeUIBold;
    FW1.pRenderStatesSergoeUIBold = pRenderStatesSergoeUIBold;
    FW1.pPSOutline = pPSOutline;

    FW1.defaultFontSize = defaultFontSize;
    FW1.pDWriteFactorySegoeUI = pDWriteFactorySegoeUI;
    FW1.pTextFormatSegoeUI = pTextFormatSegoeUI;
    FW1.pDWriteFactorySegoeUIBold = pDWriteFactorySegoeUIBold;
    FW1.pTextFormatSegoeUIBold = pTextFormatSegoeUIBold;
    FW1.pFactory = pFW1Factory;
#endif

    return S_OK;
}

void AnselUI::getTextRect(const wchar_t * str, size_t strLen, bool isBold, float * designSpaceW, float * designSpaceH) const
{
    if (!str || !designSpaceW || !designSpaceH)
        return;

    IDWriteFactory * pLwrrentFontFactory = isBold ? FW1.pDWriteFactorySegoeUIBold : FW1.pDWriteFactorySegoeUI;
    IDWriteTextFormat * pLwrrentTextFormat = isBold ? FW1.pTextFormatSegoeUIBold : FW1.pTextFormatSegoeUI;

    IDWriteTextLayout * pTextLayout;
    HRESULT status = FW1.pDWriteFactorySegoeUI->CreateTextLayout(
        str,
        (unsigned int)strLen,
        pLwrrentTextFormat,
        0.0f,
        0.0f,
        &pTextLayout
        );

    if (status != S_OK)
    {
        *designSpaceW = 0.0f;
        *designSpaceH = 0.0f;
        return;
    }

    pTextLayout->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

    // Get the layout measurements
    DWRITE_OVERHANG_METRICS overhangMetrics;
    pTextLayout->GetOverhangMetrics(&overhangMetrics);

    *designSpaceW = m_storedSizes.uiMulX * (overhangMetrics.right - overhangMetrics.left) / 1920.0f * 2.f;
    *designSpaceH = m_storedSizes.uiMulY * (overhangMetrics.bottom - overhangMetrics.top) / 1080.0f * 2.f;

    SAFE_RELEASE(pTextLayout);
}

void AnselUI::setShowMouseWhileDefolwsed(bool showMouseWhileDefolwsed)
{
    m_showMouseWhileDefolwsed = showMouseWhileDefolwsed;
}
bool AnselUI::getShowMouseWhileDefolwsed() const
{
    return m_showMouseWhileDefolwsed;
}

void AnselUI::renderGameplayOverlay(
        double dt,
        ID3D11DeviceContext * d3dctx,
        AnselResource * pPresentResourceData,
        AnselEffectState* pPassthroughEffect
        )
{
    m_gameplayOverlayNotifications.diminishLifeTime(dt * 0.001);

    float viewPort_Width = float(pPresentResourceData->toServerRes.width);
    float viewPort_Height = float(pPresentResourceData->toServerRes.height);

    const float aspect = viewPort_Width / viewPort_Height;

    float UImulX = 1.0f, UImulY = 1.0f;
    recallwlateUILayoutScale(aspect, &UImulX, &UImulY);

    // This is a pixel size in the design space (1920x1080)
    // allows to map design space onto normalized space
    float onePixelXDesign = 1.0f / 1920.f * 2.f * UImulX;
    float onePixelYDesign = 1.0f / 1080.f * 2.f * UImulY;

    // This is size of the pixel in real buffer that we're dealing with
    float onePixelXReal = 1.0f / (float)viewPort_Width * 2.f;
    float onePixelYReal = 1.0f / (float)viewPort_Height * 2.f;

    setRenderState(d3dctx, pPresentResourceData, pPassthroughEffect);

    UINT vbStride = vertexStrideUI;
    UINT offset = 0;

    // Offsets in the VS shader are hardcoded to slot 0
    d3dctx->VSSetConstantBuffers(0, 1, &pZeroOffsetsBuffer);

    {
        const int notificationSizeX = 300;
        const int notificationSizeY = 70;
        const int notificationPosX = 0;
        // Steam notifications ~100px
        const int notificationPosY = 1080 - (int)(110 * onePixelYReal / onePixelYDesign) - notificationSizeY;

        const int notificationIconSizeX = 64;
        const int notificationIconSizeY = 64;
        const int notificationIconOffsetX = 64;
        const int notificationIconOffsetY = (notificationSizeY - notificationIconSizeY) / 2;

        const int notificationOffsetY = notificationSizeY + 20;


        const float notificationPosXF = notificationPosX * onePixelXDesign - 1.0f;
        const float notificationPosYF = notificationPosY * onePixelYDesign - 1.0f;
        const float notificationSizeXF = notificationSizeX * onePixelXDesign;
        const float notificationSizeYF = notificationSizeY * onePixelYDesign;

        const float notificationIconOffsetXF = notificationIconOffsetX * onePixelXDesign;
        const float notificationIconOffsetYF = notificationIconOffsetY * onePixelYDesign;
        const float notificationIconSizeXF = notificationIconSizeX * onePixelXDesign;
        const float notificationIconSizeYF = notificationIconSizeY * onePixelYDesign;

        const float notificationOffsetYF = notificationOffsetY * onePixelYDesign;

        int notificationOffsetYAclwm = 0;

        auto cosineInterp = [&](float val1, float val2, float alpha) -> float
            {
                float alpha_cos = (1 - cosf(alpha * (float)M_PI)) / 2.0f;
                return val1 * (1.0f - alpha_cos) + val2 * alpha_cos;
            };

        const size_t errNum = (int)m_gameplayOverlayNotifications.getErrorCount();
        int errorsDisplayed = 0;
        for (size_t errCnt = 0u; errCnt < errNum; ++errCnt)
        {
            const size_t lwrErrorEntry = (m_gameplayOverlayNotifications.getFirstErrorIndex() + errCnt) % errNum;

            if (m_gameplayOverlayNotifications.getErrorLifetime(lwrErrorEntry) < 0.0f)
                continue;

            const float notificationLifetime = m_gameplayOverlayNotifications.getErrorLifetime(lwrErrorEntry);
            const float notificationElapsedTime = m_gameplayOverlayNotifications.getErrorElapsedTime(lwrErrorEntry);

            // Slides in 0.5s
            int notificationOffsetX = 0;
            if (notificationElapsedTime < 0.5f)
            {
                notificationOffsetX = int(cosineInterp(-notificationSizeX, 0.0f, notificationElapsedTime / 0.5f));
            }
            if (notificationLifetime < 0.5f)
            {
                notificationOffsetX = int(cosineInterp(-notificationSizeX, 0.0f, notificationLifetime / 0.5f));
            }

            float notificationOffsetXF = notificationOffsetX * onePixelXDesign;

            int errorOpacity = clamp(static_cast<int>(255 * notificationLifetime), 0, 255);
            float errorOpacityF = errorOpacity / 255.0f;

            float notificationOffsetYAclwmF = notificationOffsetYAclwm * onePixelYDesign;

            // Background
            {
                UIShaderConstBuf controlData_Left =
                {
                    0.0f, 0.0f, 0.0f, 0.8f,// * errorOpacityF,  // Color
                    notificationPosXF + notificationOffsetXF, notificationPosYF + notificationOffsetYAclwmF, notificationSizeXF, notificationSizeYF
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }
            // Green bar
            {
                UIShaderConstBuf controlData_Left =
                {
                    0x76 / 255.f, 0xb9 / 255.f, 0.0f, 1.0f,// * errorOpacityF,  // Color
                    notificationPosXF + notificationOffsetXF + notificationSizeXF, notificationPosYF + notificationOffsetYAclwmF, 3 * onePixelXDesign, notificationSizeYF
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            // Debug strype at the v-middle
            if (0)
            {
                UIShaderConstBuf controlData_Left =
                {
                    1.0f, 1.0f, 0.0f, 1.0f,// * errorOpacityF,  // Color
                    notificationPosXF + notificationOffsetXF, notificationPosYF + notificationSizeYF*0.5f + notificationOffsetYAclwmF, notificationSizeXF, 1.0f * onePixelYReal
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            // Icon
            {
                UIShaderConstBuf controlData_Left =
                {
                    1.0f, 1.0f, 1.0f, 1.0f,// * errorOpacityF,  // Color
                    notificationPosXF + notificationOffsetXF + notificationSizeXF - notificationIconOffsetXF, notificationPosYF + notificationOffsetYAclwmF + notificationIconOffsetYF,
                    notificationIconSizeXF, notificationIconSizeYF
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pCamIcolwertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            FW1.pRenderStatesSergoeUI->SetStates(d3dctx, 0);
            d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);

            {
                //wchar_t notification[128];
                //swprintf_s(notification, 128, L"Ansel can be activated using\n%s", m_pAnselServer->m_toggleHotkeyComboText.c_str());

                const int fonstSizeBase = 16;
                const float fontSizeF = fonstSizeBase / 1080.f * viewPort_Height;

                const float designToRealX = viewPort_Width / 1920.f;
                const float designToRealY = viewPort_Height / 1080.f;
                float textPosX = 10 + notificationOffsetX * designToRealX;
                float textPosY = viewPort_Height - (notificationPosY + notificationOffsetYAclwm + notificationSizeY * 0.5f) * designToRealY;

                // For better perceptual centering
                float additionalOffsetYF = -fonstSizeBase * 0.1f;

                UINT32 fontColor = 0x00FFffFF;
                fontColor += (errorOpacity << 24);
                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    m_gameplayOverlayNotifications.getErrorString(lwrErrorEntry).c_str(),//notification,    // String
                    fontSizeF,      // Font size
                    textPosX,       // X offset
                    textPosY + additionalOffsetYF,      // Y offset
                    fontColor,      // Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_VCENTER | FW1_STATEPREPARED// Flags
                    );
            }

            //
            d3dctx->HSSetShader(0, NULL, 0);
            d3dctx->DSSetShader(0, NULL, 0);
            d3dctx->GSSetShader(0, NULL, 0);

            setRenderState(d3dctx, pPresentResourceData, pPassthroughEffect);

            notificationOffsetYAclwm -= notificationOffsetY;
        }
    }

    setRenderState(d3dctx, pPresentResourceData, pPassthroughEffect);
}

void AnselUI::setRenderState(
                ID3D11DeviceContext* d3dctx,
                AnselResource * pPresentResourceData,
                AnselEffectState* pPassthroughEffect
                )
{
    d3dctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    d3dctx->IASetInputLayout(pInputLayout);
    d3dctx->VSSetShader(pVertexShader, NULL, 0);
    d3dctx->RSSetState(pRasterizerState);
    d3dctx->PSSetShader(pPixelShader, NULL, 0);
    d3dctx->PSSetShaderResources(0, 1, &pUIAtlasSRV);
    d3dctx->PSSetSamplers(0, 1, &pPassthroughEffect->pSamplerState);            // Using passthrough effect sampler state
    d3dctx->OMSetRenderTargets(1, &pPresentResourceData->toClientRes.pRTV, NULL);
    d3dctx->OMSetDepthStencilState(pDepthStencilState, 0xFFFFFFFF);
    d3dctx->OMSetBlendState(pBlendState, NULL, 0xffffffff);

    D3D11_RECT newRect;
    newRect.left = 0;
    newRect.bottom = pPresentResourceData->toServerRes.height;
    newRect.right = pPresentResourceData->toServerRes.width;
    newRect.top = 0;
    d3dctx->RSSetScissorRects(1, &newRect);
}

void AnselUI::render(
                ID3D11DeviceContext* d3dctx,
                AnselResource * pPresentResourceData,
                AnselEffectState* pPassthroughEffect,
                bool haveFolws,
                bool isSurfaceHDR,
                const ErrorManager & errorManager,
                const UIProgressInfo & progressInfo,
                const UIDebugPrintInfo & debugInfo,
                bool anselSDKDetected,
                const ansel::Camera& cam
                )
{
    if (isVisible())
    {
#ifdef ENABLE_STYLETRANSFER
        if (m_restyleProgressBarNeedsResizing)
        {
            const float restylePBChangeVel = 0.004f;
            if (m_restyleProgressBarVis > m_restyleProgressBarVisGoal)
            {
                m_restyleProgressBarVis -= restylePBChangeVel * debugInfo.dt;
                if (m_restyleProgressBarVis < m_restyleProgressBarVisGoal)
                {
                    m_restyleProgressBarVis = m_restyleProgressBarVisGoal;
                    m_restyleProgressBarNeedsResizing = false;
                }
            }
            else
            {
                m_restyleProgressBarVis += restylePBChangeVel * debugInfo.dt;
                if (m_restyleProgressBarVis > m_restyleProgressBarVisGoal)
                {
                    m_restyleProgressBarVis = m_restyleProgressBarVisGoal;
                    m_restyleProgressBarNeedsResizing = false;
                }
            }
            m_components.cntRestyleProgress->sizeY = m_restyleProgressBarVis * m_components.stylePBContainerHeight;
            m_needToRecallwILayout = true;
        }
#endif
        if (m_flyoutRebuildRequest.isValid)
        {
            rebuildFlyout(&m_flyoutRebuildRequest);
        }

#if (DBG_STACKING_PROTO == 1)
        if (m_allowDynamicFilterStacking)
        {
            if (m_needToAddDynamicFilter)
            {
                addDynamicFilter();
            }
            if (m_dynamicFilterIdxToRemove >= 0)
            {
                removeDynamicFilter();
                if (m_components.m_dynamicFilterContainers.size() == 0)
                {
                    m_needToAddDynamicFilter = true;
                }
            }
        }
#endif

        if (m_needToApplyTabStop)
        {
            applyTabStop();
            containerHelper.rebuildControlsArray(&mainContainer);
            m_needToApplyTabStop = false;
        }

        if (m_needToRecallwILayout)
        {
            float aspect = m_width / (float)m_height;
            recallwlateUILayout(aspect);
            m_needToRecallwILayout = false;
        }

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
        {
            bool renderModifierShift = false;
            bool renderModifierCtrl = false;
            bool renderModifierAlt = false;
            UINT renderModifierVKey = 0;

            if (m_selectingHotkey)
            {
                input::KeyboardState & kbdSt = m_inputstate.getKeyboardState();
                m_selectingHotkeyShift = kbdSt.isKeyDown(VK_SHIFT);
                m_selectingHotkeyCtrl = kbdSt.isKeyDown(VK_CONTROL);
                m_selectingHotkeyAlt = kbdSt.isKeyDown(VK_MENU);

                renderModifierShift = m_selectingHotkeyShift;
                renderModifierCtrl = m_selectingHotkeyCtrl;
                renderModifierAlt = m_selectingHotkeyAlt;
                renderModifierVKey = 0;
            }
            else
            {
                renderModifierShift = m_hotkeyModifierShift;
                renderModifierCtrl = m_hotkeyModifierCtrl;
                renderModifierAlt = m_hotkeyModifierAlt;
                renderModifierVKey = m_hotkeyModifierVKey;
            }

            const size_t labelMaxSize = ComponentsList::m_lblHotkeyCaptionMaxSize;
            wchar_t label[labelMaxSize];
            wchar_t vkName[labelMaxSize];

            label[0] = 0;
            vkName[0] = 0;

            if (renderModifierVKey)
            {
                UINT scanCode = MapVirtualKeyW(renderModifierVKey, MAPVK_VK_TO_VSC);
                LONG lParamValue = (scanCode << 16);

                int result = GetKeyNameTextW(lParamValue, vkName, labelMaxSize);

                swprintf_s(label, labelMaxSize, L"%s%s", label, vkName);
            }

            swprintf_s(label, labelMaxSize, L"%s%s%s%s%s",
                    renderModifierShift ? L"[Shift] + " : L"",
                    renderModifierCtrl ? L"[Ctrl] + " : L"",
                    renderModifierAlt ? L"[Alt] + " : L"",
                    (renderModifierShift || renderModifierCtrl || renderModifierAlt) ? L"\n" : L"",
                    vkName
                    );

            swprintf_s(m_components.lblHotkey->caption, labelMaxSize, L"%s", label);
        }
#endif

        //const float margin = 20.f / 1920.f * 2.f;
        float leftOffset = m_components.leftPane->sizeX;

        float viewPort_Width = float(pPresentResourceData->toServerRes.width);
        float viewPort_Height = float(pPresentResourceData->toServerRes.height);

        // This is size of the pixel in real buffer that we're dealing with
        float onePixelXReal = 1.0f / (float)m_width * 2.f;
        float onePixelYReal = 1.0f / (float)m_height * 2.f;

        // This is a pixel size in the design space (1920x1080)
        // allows to map design space onto normalized space
        float onePixelXDesign = 1.0f / 1920.f * 2.f;
        float onePixelYDesign = 1.0f / 1080.f * 2.f;

        UINT vbStride = vertexStrideUI;
        UINT offset = 0;

        setRenderState(d3dctx, pPresentResourceData, pPassthroughEffect);

        // Offsets in the VS shader are hardcoded to slot 0
        d3dctx->VSSetConstantBuffers(0, 1, &pZeroOffsetsBuffer);

        auto snapCoordsToPixelGrid = [&](float inX, float inY, float * outX, float * outY)
        {
            int inXint = int((inX * 0.5f + 0.5f) * (float)m_width);
            int inYint = int((inY * 0.5f + 0.5f) * (float)m_height);
            *outX = (inXint / (float)m_width) * 2.0f - 1.0f;
            *outY = (inYint / (float)m_height) * 2.0f - 1.0f;
        };
        auto snapXToPixelGridPos = [&](float inX) -> float
        {
            int inXint = int((inX * 0.5f + 0.5f) * (float)m_width + 0.5f);
            return (inXint / (float)m_width) * 2.0f - 1.0f;
        };
        auto snapXToPixelGridSize = [&](float inX) -> float
        {
            int inXint = int((inX * 0.5f) * (float)m_width + 0.5f);
            return (inXint / (float)m_width) * 2.0f;
        };
        auto snapYToPixelGridPos = [&](float inY) -> float
        {
            int inYint = int((inY * 0.5f + 0.5f) * (float)m_height + 0.5f);
            return (inYint / (float)m_height) * 2.0f - 1.0f;
        };
        auto snapYToPixelGridSize = [&](float inY) -> float
        {
            int inYint = int((inY * 0.5f) * (float)m_height + 0.5f);
            return (inYint / (float)m_height) * 2.0f;
        };

        // Render darkening rectangle with controls
        for (size_t paneIdx = 0, paneNum = mainContainer.getControlsNum(); paneIdx < paneNum; ++paneIdx)
        {
            ui::ControlContainer * lwrContainer = mainContainer.getControl((int)paneIdx);
            if (lwrContainer->getType() != ui::ControlType::kContainer)
            {
                continue;
            }

            if (lwrContainer->isVisible == false)
            {
                continue;
            }

            UIShaderConstBuf controlData_Left =
            {
                0.0f, 0.0f, 0.0f, isSurfaceHDR ? 1.0f : 0.8f,   // Color
                lwrContainer->absPosX - lwrContainer->m_renderingMarginX, lwrContainer->absPosY - lwrContainer->m_renderingMarginY,
                lwrContainer->sizeX + 2*lwrContainer->m_renderingMarginX, lwrContainer->sizeY + 2*lwrContainer->m_renderingMarginY
            };

            D3D11_MAPPED_SUBRESOURCE subResource;

            d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
            memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
            d3dctx->Unmap(pVariableOffsetsBuffer, 0);
            d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

            d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
            d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
            d3dctx->DrawIndexed(6, 0, 0);
        }

        if (1)
        {
            const size_t errNum = (int)errorManager.getErrorCount();
            int errorsDisplayed = 0;
            for (size_t errCnt = 0u; errCnt < errNum; ++errCnt)
            {
                const size_t lwrErrorEntry = (errorManager.getFirstErrorIndex() + errCnt) % errNum;

                if (errorManager.getErrorLifetime(lwrErrorEntry) < 0.0f)
                    continue;

                int errorOpacity = clamp(static_cast<int>(255 * errorManager.getErrorLifetime(lwrErrorEntry)), 0, 255);

                {
                    // First error line is offset by few pixels, work that around

                    const float margin = 20.f / 1920.f * 2.f;
                    UIShaderConstBuf controlData =
                    {
                        0.0f, 0.0f, 0.0f, isSurfaceHDR ? 1.0f : 0.8f,   // Color
                        -1.0f + leftOffset, ((1.0f - (errorsDisplayed == 0 ? 0.0f : 5.0f + 16.0f * errorsDisplayed) / 1080.f) * 2.0f - 1.0f), 2.0f, (errorsDisplayed == 0 ? -21.0f : -16.0f) / 1080.f * 2.0f
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }

                ++errorsDisplayed;
            }
        }

        float progressBarContainerHeight = 40.f / 1080.f*2.f;
        float progressBarContainerWidth = 1000.f / 1920.f*2.f;
        float progressBarHeight = 36.f / 1080.f*2.f;
        float progressBarWidth = 996.f / 1920.f*2.f;

        // Compact mode
        if (progressInfo.removeBlackTint)
        {
            progressBarContainerHeight = 20.f / 1080.f*2.f;
            progressBarContainerWidth = 500.f / 1920.f*2.f;
            progressBarHeight = 16.f / 1080.f*2.f;
            progressBarWidth = 496.f / 1920.f*2.f;
        }

        if (m_progressInfo_inProgress)
        {
            // Render rectangle darkening the remaining part of the screen
            if (!progressInfo.removeBlackTint)
            {
                UIShaderConstBuf controlData_Left =
                {
                    0.0f, 0.0f, 0.0f, 0.8f, // Color
                    -1.0f + leftOffset, -1.0f, 2.0f, 2.0f
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            // Render progress bar background
            {
                UIShaderConstBuf controlData_Left =
                {
                    0.2f, 0.2f, 0.2f, 1.0f, // Color
                    -progressBarContainerWidth / 2.0f + leftOffset * 0.5f, -progressBarContainerHeight / 2.0f, progressBarContainerWidth, progressBarContainerHeight
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            // Render progress bar
            {
                UIShaderConstBuf controlData_Left =
                {
                    0x76 / 255.f, 0xb9 / 255.f, 0.000f, 1.0f,   // Color
                    -progressBarWidth / 2.0f + leftOffset * 0.5f, -progressBarHeight / 2.0f, progressBarWidth * m_progressInfo_shotIdx / float(m_progressInfo_shotsTotal), progressBarHeight
                };

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }

            m_components.btnDone->color = m_doneAbortColor;
            m_components.btnDone->caption = m_textAbort.c_str();
        }
        else
        {
            m_components.btnDone->color = m_doneDoneColor;
            m_components.btnDone->caption = m_textDone.c_str();
        }

        ui::ControlContainer * lwrContainer;

        recallwlateUIPositions();

        const float sliderTitleFontSize = 16 / 1080.f * 2.f;
        const float sliderSubFontSize = 14 / 1080.f * 2.f;

        scissorRectStack.resize(0);

        auto scissorPop = [&](ui::ControlContainer * containerToRemove)
        {
            // Invisible containers can be reported, so we want to skip them
            //  (containers that invisible contains, won't be reported)
            if (containerToRemove->m_isClipping && containerToRemove->isVisible)
            {
                scissorRectStack.pop_back();

                D3D11_RECT newRect;
                if (scissorRectStack.size() > 0)
                {
                    newRect = scissorRectStack.back();
                }
                else
                {
                    newRect.left = (LONG)0;
                    newRect.bottom = (LONG)viewPort_Height;
                    newRect.right = (LONG)viewPort_Width;
                    newRect.top = (LONG)0;
                }
                d3dctx->RSSetScissorRects(1, &newRect);
            }
        };
        auto clipRect = [](D3D11_RECT & newRect, const D3D11_RECT & oldRect)
        {
            if (newRect.left < oldRect.left)
                newRect.left = oldRect.left;
            if (newRect.right > oldRect.right)
                newRect.right = oldRect.right;
            
            if (newRect.left > newRect.right)
                newRect.left = newRect.right;

            // Top is less than bottom, since window coordinates go top->bottom
            if (newRect.bottom > oldRect.bottom)
                newRect.bottom = oldRect.bottom;
            if (newRect.top < oldRect.top)
                newRect.top = oldRect.top;

            // We cannot allow ilwerted rectangles, or scissor tesdt won't work
            if (newRect.bottom < newRect.top)
                newRect.bottom = newRect.top;
        };

        int DBGcontainerCnt = 0, DBGslidersCnt = 0;
        containerHelper.startSearchHierarchical(&mainContainer);
        while (lwrContainer = containerHelper.getNextControlHierarchical(scissorPop))
        {
            if (!lwrContainer->isVisible)
            {
                containerHelper.jumpToNextControlHierarchical();
                continue;
            }
            
            // TODO avoroshilov UI
            //  this can be done more effectively, increasing offset each push_back, decreasing each pop_back
            //  or precallwlating it once per frame

            float controlAbsPosX = lwrContainer->absPosX;
            float controlAbsPosY = lwrContainer->absPosY;

            {
                ui::ControlContainer * cointainerTraverse = lwrContainer;
                while (cointainerTraverse->m_parent)
                {
                    controlAbsPosY += cointainerTraverse->m_parent->m_scrollValueY;
                    cointainerTraverse = cointainerTraverse->m_parent;
                }
            }

            auto renderHighlightRect = [&](
                float posX, float posY, float sizeX, float sizeY, float r, float g, float b, float a, float width,
                bool top, bool right, bool bottom, bool left
                )
            {
                const float onePixelRealX = 1.0f / m_width * 2.f;
                const float onePixelRealY = 1.0f / m_height * 2.f;
                float widthPixelX = width / 1920.f * 2.f;
                float widthPixelY = width / 1080.f * 2.f;
                if (fabsf(widthPixelX) < fabsf(onePixelRealX))
                {
                    widthPixelX = (widthPixelX) > 0.0f ? onePixelRealX : -onePixelRealX;
                }
                if (fabsf(widthPixelY) < fabsf(onePixelRealY))
                {
                    widthPixelY = (widthPixelY) > 0.0f ? onePixelRealY : -onePixelRealY;
                }

                float containerColor[4] =
                    { r, g, b, 1.0f };

                // top edge
                if (top)
                {
                    UIShaderConstBuf controlData =
                    {
                        r, g, b, a,
                        posX - widthPixelX, posY + sizeY, sizeX + widthPixelX, widthPixelY
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
                // right edge
                if (right)
                {
                    UIShaderConstBuf controlData =
                    {
                        r, g, b, a,
                        posX + sizeX, posY - widthPixelY, widthPixelX, sizeY + 2*widthPixelY
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
                // bottom edge
                if (bottom)
                {
                    UIShaderConstBuf controlData =
                    {
                        r, g, b, a,
                        posX - widthPixelX, posY - widthPixelY, sizeX + 2*widthPixelX, widthPixelY
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
                // left edge
                if (left)
                {
                    UIShaderConstBuf controlData =
                    {
                        r, g, b, a,
                        posX - widthPixelX, posY - widthPixelY, widthPixelX, sizeY + widthPixelY
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
            };

            if (lwrContainer->getType() == ui::ControlType::kContainer)
            {
                if (lwrContainer->m_isScrollable && lwrContainer->m_scrollContentSize > lwrContainer->sizeY)
                {
                    const float scrollRegionWidth = lwrContainer->m_scrollRegionRenderWidth;

                    const float scrollThumbAbs = lwrContainer->getScrollThumbSize();
                    const float scrollOffset = lwrContainer->getScrollThumbOffset();

                    ui::ColorF4 scrollThumbColor(0.7f, 0.7f, 0.7f, 1.0f);
                    if (!lwrContainer->m_isMouseOver && !lwrContainer->m_isDragScrolling && !lwrContainer->m_isBarScrolling)
                    {
                        //scrollThumbColor.val[0] = scrollThumbColor.val[1] = scrollThumbColor.val[2] = 0.4f;
                        scrollThumbColor.val[3] = 0.4f;
                    }
                    if (lwrContainer->m_isMouseOverScrollbar || lwrContainer->m_isBarScrolling)
                    {
                        scrollThumbColor.val[0] = 0.463f;
                        scrollThumbColor.val[1] = 0.725f;
                        scrollThumbColor.val[2] = 0.000f;
                    }

                    UIShaderConstBuf controlData_Border =
                    {
                        scrollThumbColor.val[0], scrollThumbColor.val[1], scrollThumbColor.val[2], scrollThumbColor.val[3], // Color
                        controlAbsPosX + lwrContainer->sizeX - scrollRegionWidth, controlAbsPosY + (lwrContainer->sizeY - scrollOffset - scrollThumbAbs), scrollRegionWidth, scrollThumbAbs
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;
                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData_Border, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }

                // DBG rendering control positioning data
#if 0
                bool scrollable = false;
                if (lwrContainer->m_isScrollable && lwrContainer->m_scrollContentSize > lwrContainer->sizeY)
                {
                    scrollable = true;
                }

                auto hue2rgb = [&](float hue)
                {
                    ui::ColorF4 K = ui::ColorF4(1.f, 2.f / 3.f, 1.f / 3.f, 3.f);

                    float p[3], tmp;
                    p[0] = std::abs(std::modf(hue + K.val[0], &tmp) * 6.0f - K.val[3]);
                    p[1] = std::abs(std::modf(hue + K.val[1], &tmp) * 6.0f - K.val[3]);
                    p[2] = std::abs(std::modf(hue + K.val[2], &tmp) * 6.0f - K.val[3]);

                    p[0] = std::min(std::max(p[0] - K.val[0], 0.0f), 1.0f);
                    p[1] = std::min(std::max(p[1] - K.val[0], 0.0f), 1.0f);
                    p[2] = std::min(std::max(p[2] - K.val[0], 0.0f), 1.0f);

                    return ui::ColorF4(p[0], p[1], p[2], 1.0f);
                };

                ui::ColorF4 containerColorF4 = hue2rgb(DBGcontainerCnt * 0.1f);

                renderHighlightRect(
                    controlAbsPosX, controlAbsPosY, lwrContainer->sizeX, lwrContainer->sizeY,
                    containerColorF4.val[0], containerColorF4.val[1], containerColorF4.val[2], 1.0f,
                    scrollable ? 5.0f : 1.0f,
                    true, true, true, true
                    );

                ++DBGcontainerCnt;
#endif
            }

            if (lwrContainer->isBasicContainer() && lwrContainer->m_isClipping)
            {
                D3D11_RECT newRect;
                newRect.left = (LONG)((controlAbsPosX + 1.0f) * 0.5f * viewPort_Width);
                newRect.right = (LONG)((controlAbsPosX + lwrContainer->sizeX + 1.0f) * 0.5f * viewPort_Width);
                newRect.top = (LONG)((1.0f - (controlAbsPosY + lwrContainer->sizeY + 1.0f) * 0.5f) * viewPort_Height);
                newRect.bottom = (LONG)((1.0f - (controlAbsPosY + 1.0f) * 0.5f) * viewPort_Height);

                if (scissorRectStack.size() > 0)
                {
                    const D3D11_RECT & oldRect = scissorRectStack.back();

                    // Intersecting old rect with the new rect
                    clipRect(newRect, oldRect);
                }

                scissorRectStack.push_back(newRect);

                d3dctx->RSSetScissorRects(1, &newRect);
            }

            if (lwrContainer->getType() == ui::ControlType::kContainer)
            {
                if (lwrContainer->m_renderSideVLine)
                {
                    renderHighlightRect(
                        snapXToPixelGridPos(controlAbsPosX + lwrContainer->m_renderSideVLineMargin), snapYToPixelGridPos(controlAbsPosY), snapXToPixelGridSize(lwrContainer->sizeX), snapYToPixelGridSize(lwrContainer->sizeY),
                        1.0f, 1.0f, 1.0f, 0.3f,
                        1.0f,
                        false, false, false, true
                        );
                }
            }

            if (lwrContainer->getType() == ui::ControlType::kLabel)
            {
                // DBG rendering control positioning data
#if 0
                renderHighlightRect(
                    snapXToPixelGridPos(controlAbsPosX), snapYToPixelGridPos(controlAbsPosY), snapXToPixelGridSize(lwrContainer->sizeX), snapYToPixelGridSize(lwrContainer->sizeY),
                    1.0f, 0.0f, 0.0f, 1.0f,
                    1.0f,
                    true, true, true, true
                    );
#endif
            }

            if (lwrContainer->getType() == ui::ControlType::kColorPicker)
            {
                // DBG rendering control positioning data
#if 0
                bool scrollable = false;
                if (lwrContainer->m_isScrollable && lwrContainer->m_scrollContentSize > lwrContainer->sizeY)
                {
                    scrollable = true;
                }

                auto hue2rgb = [&](float hue)
                {
                    ui::ColorF4 K = ui::ColorF4(1.f, 2.f / 3.f, 1.f / 3.f, 3.f);

                    float p[3], tmp;
                    p[0] = std::abs(std::modf(hue + K.val[0], &tmp) * 6.0f - K.val[3]);
                    p[1] = std::abs(std::modf(hue + K.val[1], &tmp) * 6.0f - K.val[3]);
                    p[2] = std::abs(std::modf(hue + K.val[2], &tmp) * 6.0f - K.val[3]);

                    p[0] = std::min(std::max(p[0] - K.val[0], 0.0f), 1.0f);
                    p[1] = std::min(std::max(p[1] - K.val[0], 0.0f), 1.0f);
                    p[2] = std::min(std::max(p[2] - K.val[0], 0.0f), 1.0f);

                    return ui::ColorF4(p[0], p[1], p[2], 1.0f);
                };

                ui::ColorF4 containerColorF4 = hue2rgb(DBGcontainerCnt * 0.1f);

                renderHighlightRect(
                    controlAbsPosX, controlAbsPosY, lwrContainer->sizeX, lwrContainer->sizeY,
                    containerColorF4.val[0], containerColorF4.val[1], containerColorF4.val[2], 1.0f,
                    scrollable ? 5.0f : 1.0f,
                    true, true, true, true
                );

                ++DBGcontainerCnt;
#endif
            }

            if (lwrContainer->getType() == ui::ControlType::kButton)
            {
                ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);

                if (lwrButton->renderType == ui::ControlButton::RenderType::kFlyoutToggle)
                {
                    ui::ColorF4 glyphColor(1.0f, 1.0f, 1.0f, 1.0f);

                    if (lwrButton->state != UI_CONTROL_ORDINARY)
                    {
                        glyphColor = ui::ColorF4(0.463f, 0.725f, 0.000f, 1.0f);
                    }
                    if (!isControlEnabled(lwrButton))
                    {
                        glyphColor = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                    }

                    const float glyphThickness = 2.0f;
                    const float glyphThicknessY = glyphThickness * onePixelYDesign;
                    const float glyphMarginWidth = 0.0f;//lwrToggle->m_glyphMargin;
                    const float glyphMarginHeight = 0.0f;
                    const float glyphWidth = 12.0f * onePixelXDesign;
                    const float glyphHeight = 12.0f * onePixelYDesign;

                    UIShaderConstBuf glyphControlData =
                    {
                        glyphColor.val[0], glyphColor.val[1], glyphColor.val[2], glyphColor.val[3],
                        snapXToPixelGridPos(controlAbsPosX + lwrContainer->sizeX + glyphMarginWidth), 
                                            snapYToPixelGridPos(controlAbsPosY + 1.0f*lwrButton->sizeY - 1.0f*glyphHeight - glyphMarginHeight),
                                            snapXToPixelGridSize(glyphWidth),
                                            snapYToPixelGridSize(glyphHeight)
                    };

                    D3D11_MAPPED_SUBRESOURCE subResourceGlyph;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResourceGlyph);
                    memcpy(subResourceGlyph.pData, &glyphControlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pArrowIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pArrowRightVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(12, 0, 0);
                    
                    continue;
                }


                if (lwrButton->state != UI_CONTROL_ORDINARY)
                {
                    if (lwrButton->renderType != ui::ControlButton::RenderType::kToggle && lwrButton->hlType == ui::ControlButton::HighlightType::kRectangle)
                    {
                        renderHighlightRect(
                            controlAbsPosX, controlAbsPosY, lwrContainer->sizeX, lwrContainer->sizeY,
                            lwrButton->hlColor.val[0], lwrButton->hlColor.val[1], lwrButton->hlColor.val[2], lwrButton->hlColor.val[3],
                            1.0f,
                            true, true, true, true
                            );
                    }
                }

                if (lwrButton->renderType == ui::ControlButton::RenderType::kToggle)
                {
                    // Rendering +/- for the toggling buttons
                    {
                        ui::ColorF4 glyphColor(1.0f, 1.0f, 1.0f, 1.0f);
                        ui::ControlButtonToggle * lwrToggle = static_cast<ui::ControlButtonToggle *>(lwrButton);
                        float linkedContainerSizeY = 0.0f;

                        if (!lwrToggle->m_containerToggle)
                            continue;

                        if (lwrButton->state != UI_CONTROL_ORDINARY)
                        {
                            glyphColor = ui::ColorF4(0.463f, 0.725f, 0.000f, 1.0f);
                        }
                        if (!isControlEnabled(lwrButton))
                        {
                            glyphColor = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                        }

                        linkedContainerSizeY = lwrToggle->m_containerToggle->sizeY;
                        if (linkedContainerSizeY == 0.0f)
                        {
                            glyphColor.val[0] = glyphColor.val[1] = glyphColor.val[2] = 0.2f;
                        }

                        const float glyphThickness = 2.0f;
                        const float glyphThicknessY = glyphThickness * onePixelYDesign;
                        const float glyphMarginWidth = lwrToggle->m_glyphMargin;
                        const float glyphMarginHeight = 0.0f;
                        const float glyphWidth = 10.0f * onePixelXDesign;
                        const float glyphHeight = 10.0f * onePixelYDesign;

                        const int width = (int)m_width;
                        const int height = (int)m_height;

                        if (lwrToggle->m_containerToggle->isVisible)
                        {
                            UIShaderConstBuf glyphControlData =
                            {
                                glyphColor.val[0], glyphColor.val[1], glyphColor.val[2], glyphColor.val[3],
                                snapXToPixelGridPos(controlAbsPosX + glyphMarginWidth), snapYToPixelGridPos(controlAbsPosY + 0.5f*lwrButton->sizeY - 0.5f*glyphHeight - glyphMarginHeight), snapXToPixelGridSize(glyphWidth), snapYToPixelGridSize(glyphHeight)
                            };

                            D3D11_MAPPED_SUBRESOURCE subResourceGlyph;

                            d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResourceGlyph);
                            memcpy(subResourceGlyph.pData, &glyphControlData, sizeof(UIShaderConstBuf));
                            d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                            d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                            d3dctx->IASetIndexBuffer(pArrowIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                            d3dctx->IASetVertexBuffers(0, 1, &pArrowUpVertexBuf, &vbStride, &offset);
                            d3dctx->DrawIndexed(12, 0, 0);
                        }
                        else
                        {
                            UIShaderConstBuf glyphControlData =
                            {
                                glyphColor.val[0], glyphColor.val[1], glyphColor.val[2], glyphColor.val[3],
                                snapXToPixelGridPos(controlAbsPosX + glyphMarginWidth), snapYToPixelGridPos(controlAbsPosY + 0.5f*lwrButton->sizeY - 0.5f*glyphHeight - glyphMarginHeight), snapXToPixelGridSize(glyphWidth), snapYToPixelGridSize(glyphHeight)
                            };

                            D3D11_MAPPED_SUBRESOURCE subResourceGlyph;

                            d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResourceGlyph);
                            memcpy(subResourceGlyph.pData, &glyphControlData, sizeof(UIShaderConstBuf));
                            d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                            d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                            d3dctx->IASetIndexBuffer(pArrowIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                            d3dctx->IASetVertexBuffers(0, 1, &pArrowDowlwertexBuf, &vbStride, &offset);
                            d3dctx->DrawIndexed(12, 0, 0);
                        }
                    }

                    continue;
                }

                UIShaderConstBuf controlData =
                {
                    lwrButton->color.val[0], lwrButton->color.val[1], lwrButton->color.val[2], lwrButton->color.val[3],
                    controlAbsPosX, controlAbsPosY, lwrButton->sizeX, lwrButton->sizeY
                };

                if (!isControlEnabled(lwrButton))
                {
                    controlData.cr = 0.2f;
                    controlData.cg = 0.2f;
                    controlData.cb = 0.2f;
                }

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->IASetIndexBuffer(lwrButton->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &lwrButton->pVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }
            else if (lwrContainer->getType() == ui::ControlType::kCheckbox)
            {
                ui::ControlCheckbox * lwrCheckbox = static_cast<ui::ControlCheckbox *>(lwrContainer);

#if 0
                // DBG rendering control positioning data
                // Renders outlining rectangle, for easier layout changes
                renderHighlightRect(
                    controlAbsPosX, controlAbsPosY, lwrCheckbox->sizeX, lwrCheckbox->sizeY,
                    0.0f, 1.0f, 0.0f, 1.0f,
                    1.0f,
                    true, true, true, true
                    );
#endif

                const float aspect = m_width / (float)m_height;
                const float checkSizeX = lwrCheckbox->checkSize / aspect;
                const float checkSizeY = lwrCheckbox->checkSize;
                const float checkOffsetY = 0.5f * (lwrCheckbox->sizeY - checkSizeY);

                ui::ColorF4 checkColor = lwrCheckbox->color;
                ui::ColorF4 checkBorderColor = lwrCheckbox->color;

                checkColor = ui::ColorF4(1.0f, 1.0f, 1.0f, 1.0f);
                checkBorderColor = ui::ColorF4(0.7f, 0.7f, 0.7f, 1.0f);

                if (lwrCheckbox->state != UI_CONTROL_ORDINARY)
                {
                    checkColor = ui::ColorF4(0.463f, 0.725f, 0.000f, 1.0f);
                    checkBorderColor = checkColor;
                }
                if (!isControlEnabled(lwrCheckbox))
                {
                    checkColor = ui::ColorF4(0.2f, 0.2f, 0.2f, 1.0f);
                    checkBorderColor = checkColor;
                }

                renderHighlightRect(
                    controlAbsPosX, controlAbsPosY + checkOffsetY, checkSizeX, checkSizeY,
                    checkBorderColor.val[0], checkBorderColor.val[1], checkBorderColor.val[2], checkBorderColor.val[3],
                    1.0f,
                    true, true, true, true
                    );

                if (lwrCheckbox->isChecked || lwrCheckbox->isPressed)
                {
                    UIShaderConstBuf controlData =
                    {
                        checkColor.val[0], checkColor.val[1], checkColor.val[2], lwrCheckbox->isPressed ? checkColor.val[3] * 0.5f : checkColor.val[3],
                        controlAbsPosX + 3*onePixelXDesign, controlAbsPosY + checkOffsetY + 3*onePixelYDesign, checkSizeX - 6*onePixelXDesign, checkSizeY - 6*onePixelYDesign
                    };

                    if (!isControlEnabled(lwrCheckbox))
                    {
                        controlData.cr = 0.2f;
                        controlData.cg = 0.2f;
                        controlData.cb = 0.2f;
                    }

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
            }
            else if (lwrContainer->getType() == ui::ControlType::kIcon)
            {
                ui::ControlIcon * lwrIcon = static_cast<ui::ControlIcon *>(lwrContainer);

                D3D11_MAPPED_SUBRESOURCE subResource;

                d3dctx->IASetIndexBuffer(lwrIcon->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);

                float iconSizeX, iconSizeY;
                if (lwrIcon->vertexBufDescs.size() == 1)
                {
                    iconSizeX = lwrIcon->vertexBufDescs[0].pixelSizeX * onePixelXReal;
                    iconSizeY = lwrIcon->vertexBufDescs[0].pixelSizeY * onePixelYReal;
                    d3dctx->IASetVertexBuffers(0, 1, &lwrIcon->vertexBufDescs[0].pVertexBuf, &vbStride, &offset);
                }
                else
                {
                    // "Mipmapping" that selects picture within some range, or closest candidate for downscaling
                    const int maxUpscaleDist = 0;
                    int bestMatchIndex = 0;
                    int pixelSizeX = (int)( (snapXToPixelGridSize(lwrIcon->sizeX) * 0.5f) * m_width );

                    int bestMatchDistance = -(int)m_width;
                    for (size_t vbi = 0, vbiend = lwrIcon->vertexBufDescs.size(); vbi < vbiend; ++vbi)
                    {
                        int pixelSizeDistance = pixelSizeX - lwrIcon->vertexBufDescs[vbi].pixelSizeX;
                        if (pixelSizeDistance >= 0 && (bestMatchDistance > pixelSizeDistance || bestMatchDistance < 0))
                        {
                            bestMatchDistance = pixelSizeDistance;
                            bestMatchIndex = (int)vbi;
                        }
                        // This check is needed in case the target size is smaller than all of the available sizes
                        else if (pixelSizeDistance < 0 && bestMatchDistance < 0 && bestMatchDistance < pixelSizeDistance)
                        {
                            bestMatchDistance = pixelSizeDistance;
                            bestMatchIndex = (int)vbi;
                        }
                    }

                    if (bestMatchDistance > 0)
                    {
                        iconSizeX = lwrIcon->vertexBufDescs[bestMatchIndex].pixelSizeX * onePixelXReal;
                        iconSizeY = lwrIcon->vertexBufDescs[bestMatchIndex].pixelSizeY * onePixelYReal;
                    }
                    else
                    {
                        // Target box is smaller than every buffer that we have, we need to scale it down to avoid clipping
                        iconSizeX = lwrIcon->sizeX;
                        iconSizeY = lwrIcon->sizeY;
                    }
                    d3dctx->IASetVertexBuffers(0, 1, &lwrIcon->vertexBufDescs[bestMatchIndex].pVertexBuf, &vbStride, &offset);
                }

                float centeringOffsetX = (lwrIcon->sizeX - iconSizeX) * 0.5f;

                UIShaderConstBuf controlData =
                {
                    lwrIcon->color.val[0], lwrIcon->color.val[1], lwrIcon->color.val[2], lwrIcon->color.val[3],
                    snapXToPixelGridPos(controlAbsPosX + centeringOffsetX), snapYToPixelGridPos(controlAbsPosY), snapXToPixelGridSize(iconSizeX), snapYToPixelGridSize(iconSizeY)
                };

                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                d3dctx->DrawIndexed(6, 0, 0);

#if 0
                // DBG rendering control positioning data
                // Renders outlining rectangle, for easier layout changes
                renderHighlightRect(
                    snapXToPixelGridPos(controlAbsPosX + centeringOffsetX), snapYToPixelGridPos(controlAbsPosY), snapXToPixelGridSize(iconSizeX), snapYToPixelGridSize(iconSizeY),
                    1.0f, 1.0f, 0.0f, 1.0f,
                    1.0f,
                    true, true, true, true
                );
#endif
            }
            else if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
                     lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
                     lwrContainer->getType() == ui::ControlType::kSliderInt)
            {
                ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

#if 0
                // DBG rendering control positioning data
                auto hue2rgb = [&](float hue)
                {
                    ui::ColorF4 K = ui::ColorF4(1.f, 2.f / 3.f, 1.f / 3.f, 3.f);

                    float p[3], tmp;
                    p[0] = std::abs(std::modf(hue + K.val[0], &tmp) * 6.0f - K.val[3]);
                    p[1] = std::abs(std::modf(hue + K.val[1], &tmp) * 6.0f - K.val[3]);
                    p[2] = std::abs(std::modf(hue + K.val[2], &tmp) * 6.0f - K.val[3]);

                    p[0] = std::min(std::max(p[0] - K.val[0], 0.0f), 1.0f);
                    p[1] = std::min(std::max(p[1] - K.val[0], 0.0f), 1.0f);
                    p[2] = std::min(std::max(p[2] - K.val[0], 0.0f), 1.0f);

                    return ui::ColorF4(p[0], p[1], p[2], 1.0f);
                };

                ui::ColorF4 containerColorF4 = hue2rgb(DBGslidersCnt * 0.1f);

                renderHighlightRect(
                    controlAbsPosX, controlAbsPosY, lwrContainer->sizeX, lwrContainer->sizeY,
                    containerColorF4.val[0], containerColorF4.val[1], containerColorF4.val[2], 1.0f,
                    1.0f,
                    true, true, true, true
                    );

                ++DBGslidersCnt;
#endif

                // Draw slider tracks
                {
                    UIShaderConstBuf controlData =
                    {
                        1.0f, 1.0f, 1.0f, 1.0f, // Color
                        controlAbsPosX, controlAbsPosY + lwrSliderBase->trackShiftY, lwrSliderBase->sizeX, lwrSliderBase->trackSizeY    // Offsets and scaling
                    };

                    if (lwrSliderBase->state != UI_CONTROL_ORDINARY)
                    {
                        controlData.cr = 0.463f;    // 0x76
                        controlData.cg = 0.725f;    // 0xb9
                        controlData.cb = 0.000f;    // 0x00
                    }

                    if (!isControlEnabled(lwrSliderBase))
                    {
                        controlData.cr = 0.2f;
                        controlData.cg = 0.2f;
                        controlData.cb = 0.2f;
                    }

                    D3D11_MAPPED_SUBRESOURCE subResource;
                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(lwrSliderBase->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &lwrSliderBase->pVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);

                    if (lwrSliderBase->getType() == ui::ControlType::kSliderDiscr)
                    {
                        // Render discrete slider notches
                        float notchPixelSizeX = 1.f;
                        float notchPixelSizeY = 3.f;
                        float notchOpacity = 1.0f;

                        const float notchMinSizeX = notchPixelSizeX / (float)viewPort_Width * 2.0f;
                        const float notchMinSizeY = notchPixelSizeY / (float)viewPort_Height * 2.0f; 

                        float notchSizeX = notchPixelSizeX / 1920.0f * 2.0f;
                        float notchSizeY = notchPixelSizeY / 1080.0f * 2.0f;

                        if (notchSizeX < notchMinSizeX)
                            notchSizeX = notchMinSizeX;
                        if (notchSizeY < notchMinSizeY)
                            notchSizeY = notchMinSizeY;

                        ui::ControlSliderDiscr * lwrDiscrSlider = static_cast<ui::ControlSliderDiscr *>(lwrSliderBase);
                        const size_t lblCntEnd = lwrDiscrSlider->getTotalNumTicks();
                        // If there are too many positions, no sense in visualizing them
                        if ((lblCntEnd <= 16) && (lblCntEnd > 1))
                        {
                            for (size_t lblCnt = 0; lblCnt < lblCntEnd; ++lblCnt)
                            {
                                float notchPosX = controlAbsPosX + (lblCnt / (float)(lblCntEnd - 1)) * lwrDiscrSlider->sizeX;
                                UIShaderConstBuf controlDataNotch =
                                {
                                    controlData.cr, controlData.cg, controlData.cb, notchOpacity,  // Color
                                    notchPosX, controlAbsPosY + lwrDiscrSlider->trackShiftY - notchSizeY, (lblCnt == lblCntEnd - 1) ? -notchSizeX : notchSizeX, lwrDiscrSlider->trackSizeY + notchSizeY  // Offsets and scaling
                                };

                                D3D11_MAPPED_SUBRESOURCE subResource;
                                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                                memcpy(subResource.pData, &controlDataNotch, sizeof(UIShaderConstBuf));
                                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                                d3dctx->IASetIndexBuffer(lwrSliderBase->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                                d3dctx->IASetVertexBuffers(0, 1, &lwrSliderBase->pVertexBuf, &vbStride, &offset);
                                d3dctx->DrawIndexed(6, 0, 0);
                            }
                        }
                    }
                    else if (lwrSliderBase->getType() == ui::ControlType::kSliderCont)
                    {
                        // Render continuous sliders default value
                        ui::ControlSliderCont * lwrContSlider = static_cast<ui::ControlSliderCont *>(lwrSliderBase);
                        if (lwrContSlider->defaultValue != -1.0f)
                        {
                            float notchSizeX = 5.0f / 1920.0f * 2.0f;
                            float notchSizeY = 5.0f / 1080.0f * 2.0f;
                            float notchPosX = controlAbsPosX + lwrContSlider->defaultValue * lwrContSlider->sizeX - notchSizeX * 0.5f;
                            UIShaderConstBuf controlDataNotch =
                            {
                                controlData.cr, controlData.cg, controlData.cb, 1.0f,  // Color
                                notchPosX, controlAbsPosY + lwrContSlider->trackShiftY - notchSizeY - 0.5f * lwrContSlider->thumbSizeY, notchSizeX, lwrContSlider->trackSizeY + notchSizeY  // Offsets and scaling
                            };

                            D3D11_MAPPED_SUBRESOURCE subResource;
                            d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                            memcpy(subResource.pData, &controlDataNotch, sizeof(UIShaderConstBuf));
                            d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                            d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                            d3dctx->IASetIndexBuffer(pTriUpIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                            d3dctx->IASetVertexBuffers(0, 1, &pTriUpVertexBuf, &vbStride, &offset);
                            d3dctx->DrawIndexed(3, 0, 0);
                        }
                    }
                }

                // Draw slider thumbs
                {
                    D3D11_MAPPED_SUBRESOURCE subResource;

                    float thumbCenterX, thumbCenterY;

                    lwrSliderBase->getThumbPosition(thumbCenterX, thumbCenterY);
                    thumbCenterX += controlAbsPosX;
                    thumbCenterY += controlAbsPosY;

                    // Thumb border
                    UIShaderConstBuf controlData_Border =
                    {
                        0.0f, 0.0f, 0.0f, 1.0f,  // Color

                        thumbCenterX - 0.5f * lwrSliderBase->thumbSizeX - lwrSliderBase->thumbBorderX, thumbCenterY - 0.5f * lwrSliderBase->thumbSizeY - lwrSliderBase->thumbBorderY,
                        lwrSliderBase->thumbSizeX + 2.0f * lwrSliderBase->thumbBorderX, lwrSliderBase->thumbSizeY + 2.0f * lwrSliderBase->thumbBorderY
                    };

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData_Border, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(lwrSliderBase->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &lwrSliderBase->pVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);

                    // Thumb itself
                    UIShaderConstBuf controlData =
                    {
                        1.0f, 1.0f, 1.0f, 1.0f,  // Color
                        thumbCenterX - 0.5f * lwrSliderBase->thumbSizeX, thumbCenterY - 0.5f * lwrSliderBase->thumbSizeY, lwrSliderBase->thumbSizeX, lwrSliderBase->thumbSizeY  // Offsets and scaling
                    };

                    if (lwrSliderBase->state != UI_CONTROL_ORDINARY)
                    {
                        controlData.cr = 0.463f;  // 0x76
                        controlData.cg = 0.725f;  // 0xb9
                        controlData.cb = 0.000f;  // 0x00
                    }

                    if (!isControlEnabled(lwrSliderBase))
                    {
                        controlData.cr = 0.2f;
                        controlData.cg = 0.2f;
                        controlData.cb = 0.2f;
                    }

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(lwrSliderBase->pIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &lwrSliderBase->pVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
            }
            else if (lwrContainer->getType() == ui::ControlType::kProgressBar)
            {
                float progressBarContainerHeight = lwrContainer->sizeY;
                float progressBarContainerWidth = lwrContainer->sizeX;
                // progressBar*Margin are later multiplied by uiMul* - keep that in mind if you will put them into onePixel*Design/onePixel*Real
                float progressBarHeightMargin = 2.f * onePixelYDesign;
                float progressBarWidthMargin = 2.f * onePixelXDesign;

                if (progressBarHeightMargin < onePixelYReal)
                    progressBarHeightMargin = onePixelYReal;
                if (progressBarWidthMargin < onePixelXReal)
                    progressBarWidthMargin = onePixelXReal;

                progressBarHeightMargin *= m_storedSizes.uiMulY;
                progressBarWidthMargin *= m_storedSizes.uiMulX;

                ui::ControlProgressBar * lwrProgressBar = static_cast<ui::ControlProgressBar *>(lwrContainer);

                float progress = lwrProgressBar->progress;

                // Render progress bar background
                {
                    UIShaderConstBuf controlData =
                    {
                        0.2f, 0.2f, 0.2f, 1.0f, // Color
                        controlAbsPosX, controlAbsPosY, lwrProgressBar->sizeX, lwrProgressBar->sizeY    // Offsets and scaling
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }

                // Render progress bar
                {
                    UIShaderConstBuf controlData =
                    {
                        0x76 / 255.f, 0xb9 / 255.f, 0.000f, 1.0f,   // Color
                        controlAbsPosX + progressBarWidthMargin, controlAbsPosY + progressBarHeightMargin,
                        (lwrProgressBar->sizeX - 2.f*progressBarWidthMargin) * progress, lwrProgressBar->sizeY - 2.f*progressBarHeightMargin
                    };

                    D3D11_MAPPED_SUBRESOURCE subResource;

                    d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                    memcpy(subResource.pData, &controlData, sizeof(UIShaderConstBuf));
                    d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                    d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

                    d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                    d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                    d3dctx->DrawIndexed(6, 0, 0);
                }
            }
        }
        containerHelper.stopSearchHierarchical();

#if (UI_ENABLE_TEXT == 1)
        const size_t ui_textBufSize = 256;
        wchar_t ui_text[ui_textBufSize];
//      swprintf_s(ui_text, ui_textBufSize, L"%d / %d", AnselSDK.DLLfound, m_bRunShaderMod ? 1 : 0);

#if 1
        FW1.pRenderStatesSergoeUIBold->SetStates(d3dctx, 0);
        d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);
        UINT fontFlags = FW1_CENTER | FW1_VCENTER | FW1_STATEPREPARED;
        UINT fontColor = 0xff442200;
#else
        // DBG : original state
        FW1.pRenderStates->SetStates(d3dctx, 0);
        UINT fontFlags = FW1_CENTER | FW1_VCENTER | FW1_STATEPREPARED;
        UINT fontColor = 0xffFFFFFF;
#endif

        auto colwertColorFloatToUint = [](const ui::ColorF4 & color) -> UINT32
        {
            unsigned char fcr = (unsigned char)(color.val[0] * 255);
            unsigned char fcg = (unsigned char)(color.val[1] * 255);
            unsigned char fcb = (unsigned char)(color.val[2] * 255);
            unsigned char fca = (unsigned char)(color.val[3] * 255);

            return (fca << 24) + (fcb << 16) + (fcg << 8) + fcr;
        };

        containerHelper.startSearchHierarchical(&mainContainer);
        while (lwrContainer = containerHelper.getNextControlHierarchical(scissorPop))
        {
            if (!lwrContainer->isVisible)
            {
                containerHelper.jumpToNextControlHierarchical();
                continue;
            }

            // TODO avoroshilov UI
            //  this can be done more effectively, increasing offset each push_back, decreasing each pop_back

            float controlAbsPosX = lwrContainer->absPosX;
            float controlAbsPosY = lwrContainer->absPosY;

            {
                ui::ControlContainer * cointainerTraverse = lwrContainer;
                while (cointainerTraverse->m_parent)
                {
                    controlAbsPosY += cointainerTraverse->m_parent->m_scrollValueY;
                    cointainerTraverse = cointainerTraverse->m_parent;
                }
            }

            if (lwrContainer->isBasicContainer() && lwrContainer->m_isClipping)
            {
                D3D11_RECT newRect;
                newRect.left = (LONG)((controlAbsPosX + 1.0f) * 0.5f * viewPort_Width);
                newRect.right = (LONG)((controlAbsPosX + lwrContainer->sizeX + 1.0f) * 0.5f * viewPort_Width);
                newRect.top = (LONG)((1.0f - (controlAbsPosY + lwrContainer->sizeY + 1.0f) * 0.5f) * viewPort_Height);
                newRect.bottom = (LONG)((1.0f - (controlAbsPosY + 1.0f) * 0.5f) * viewPort_Height);

                if (scissorRectStack.size() > 0)
                {
                    const D3D11_RECT & oldRect = scissorRectStack.back();

                    // Intersecting old rect with the new rect
                    clipRect(newRect, oldRect);
                }

                scissorRectStack.push_back(newRect);

                d3dctx->RSSetScissorRects(1, &newRect);
            }

            if (lwrContainer->getType() == ui::ControlType::kLabel)
            {
                ui::ControlLabel * lwrLabel = static_cast<ui::ControlLabel *>(lwrContainer);

                if (!lwrLabel->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                float btnCaptionCenterX = controlAbsPosX + 0.5f * lwrLabel->sizeX;
                float btnCaptionCenterY = controlAbsPosY + 0.5f * lwrLabel->sizeY;

                UINT32 fontFlagsLoc = fontFlags;
                if (lwrLabel->fontAlignment == ui::ControlBase::FontAlignment::kRight)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_RIGHT;
                    btnCaptionCenterX = controlAbsPosX + lwrLabel->sizeX;
                }
                else if (lwrLabel->fontAlignment == ui::ControlBase::FontAlignment::kLeft)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_LEFT;
                    btnCaptionCenterX = controlAbsPosX;
                }

                ui::ColorF4 fontColorF4 = lwrLabel->fontColor;
                if (!isControlEnabled(lwrLabel))
                {
                    fontColorF4 = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                }
                UINT32 fontColorLoc = colwertColorFloatToUint(fontColorF4);

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUIBold->DrawString(
                    d3dctx,
                    lwrLabel->caption,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    fontFlagsLoc// Flags
                    );
            }

            if (lwrContainer->getType() == ui::ControlType::kButton)
            {
                ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);

                if (lwrButton->renderType == ui::ControlButton::RenderType::kFlyoutToggle)
                {
                    ui::ControlFlyoutToggleShared * lwrFlyoutToggle = static_cast<ui::ControlFlyoutToggleShared *>(lwrButton);

                    const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                    float btnCaptionCenterX = controlAbsPosX;
                    float btnCaptionCenterY = controlAbsPosY + 1.0f * lwrButton->sizeY;

                    ui::ColorF4 btnFontColor = lwrButton->fontColor;
                    if (lwrButton->state != UI_CONTROL_ORDINARY)
                    {
                        // 118 185 0
                        btnFontColor = ui::ColorF4(0.463f, 0.725f, 0.000f, 1.0f);
                    }
                    if (!isControlEnabled(lwrFlyoutToggle))
                    {
                        btnFontColor = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                    }

                    UINT32 fontColorLoc = colwertColorFloatToUint(btnFontColor);
                    UINT32 fontFlagsLoc = FW1_LEFT | FW1_TOP | FW1_STATEPREPARED;

                    // Normalized -> absolute
                    btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                    btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                    FW1.pFontWrapperSergoeUIBold->DrawString(
                        d3dctx,
                        lwrFlyoutToggle->caption,// String
                        btnCaptionFontSize,// Font size
                        btnCaptionCenterX,// X offset
                        viewPort_Height - btnCaptionCenterY,// Y offset
                        fontColorLoc,// Text color, 0xAaBbGgRr
                        fontFlagsLoc// Flags
                        );

                    continue;
                }

                if (!lwrButton->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                float btnCaptionCenterX = controlAbsPosX + 0.5f * lwrButton->sizeX;
                float btnCaptionCenterY = controlAbsPosY + 0.5f * lwrButton->sizeY;

                UINT32 fontFlagsLoc = fontFlags;
                if (lwrButton->fontAlignment == ui::ControlBase::FontAlignment::kRight)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_RIGHT;
                    btnCaptionCenterX = controlAbsPosX + lwrButton->sizeX;
                }
                else if (lwrButton->fontAlignment == ui::ControlBase::FontAlignment::kLeft)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_LEFT;
                    btnCaptionCenterX = controlAbsPosX;
                }

                ui::ColorF4 btnFontColor = lwrButton->fontColor;
                if (!isControlEnabled(lwrButton))
                {
                    btnFontColor.val[3] = 0.2f;
                }

                if (lwrButton->renderType == ui::ControlButton::RenderType::kToggle)
                {
                    ui::ControlButtonToggle * lwrButtonToggle = static_cast<ui::ControlButtonToggle *>(lwrButton);

                    // TODO avoroshilov: unify this with +/- rendering
                    const float glyphThickness = 2.0f;
                    const float glyphMarginWidth = lwrButtonToggle->m_glyphMargin;
                    const float glyphMarginHeight = 0.0f;
                    const float glyphWidth = 10.0f * onePixelXDesign;
                    const float glyphHeight = 10.0f * onePixelYDesign;

                    if (lwrButton->state != UI_CONTROL_ORDINARY)
                    {
                        // 118 185 0
                        btnFontColor = ui::ColorF4(0.463f, 0.725f, 0.000f, 1.0f);
                    }
                    if (!isControlEnabled(lwrButton))
                    {
                        btnFontColor = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                    }

                    fontFlagsLoc = FW1_LEFT | FW1_VCENTER | FW1_STATEPREPARED;
                    btnCaptionCenterX = controlAbsPosX + glyphWidth * 0.5f + glyphMarginWidth + 2 / 1920.f * 2.f;

                    float linkedContainerSizeY = 0.0f;
                    if (lwrButtonToggle->m_containerToggle)
                    {
                        linkedContainerSizeY = lwrButtonToggle->m_containerToggle->sizeY;
                    }
                    if (linkedContainerSizeY == 0.0f)
                    {
                        btnFontColor.val[3] = 0.2f;
                    }
                }
                UINT32 fontColorLoc = colwertColorFloatToUint(btnFontColor);
                if (lwrButton->state != UI_CONTROL_ORDINARY && lwrButton->hlType == ui::ControlButton::HighlightType::kFont)
                    fontColorLoc = colwertColorFloatToUint(lwrButton->hlColor);

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUIBold->DrawString(
                    d3dctx,
                    lwrButton->caption,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    fontFlagsLoc// Flags
                    );
            }
            if (lwrContainer->getType() == ui::ControlType::kCheckbox )
            {
                ui::ControlCheckbox * lwrCheckbox  = static_cast<ui::ControlCheckbox  *>(lwrContainer);

                if (!lwrCheckbox ->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                const float aspect = m_width / (float)m_height;
                float btnCaptionCenterX = controlAbsPosX + 4*onePixelXDesign + lwrCheckbox->checkSize / aspect;
                float btnCaptionCenterY = controlAbsPosY;

                UINT32 fontColorLoc = 0xffFFffFF;
                if (!isControlEnabled(lwrCheckbox))
                {
                    fontColorLoc = 0x33FFffFF;  // 0x33 = 20% * 0xff
                }

                UINT32 fontFlagsLoc = fontFlags;

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUIBold->DrawString(
                    d3dctx,
                    lwrCheckbox->title,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_BOTTOM | FW1_STATEPREPARED// Flags
                );
            }
            else if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
                     lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
                     lwrContainer->getType() == ui::ControlType::kSliderInt)
            {
                ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

                // Title bold version
#if (UI_SLIDER_BOLD_CAPTIONS != 0)
                const float sliderTitleFontSizePixel = sliderTitleFontSize * 0.5f * viewPort_Height;
                float sliderTitleLeftX = controlAbsPosX;
                float sliderTitleCenterY = controlAbsPosY + lwrSliderBase->trackShiftY + 0.5f * lwrSliderBase->thumbSizeY + lwrSliderBase->thumbBorderY - 5.f / 1080.f * 2.f;

                // Normalized -> absolute
                sliderTitleLeftX = (sliderTitleLeftX + 1.0f) / 2.0f * viewPort_Width;

                sliderTitleCenterY = (sliderTitleCenterY + 1.0f) / 2.0f * viewPort_Height;
                sliderTitleCenterY += sliderTitleFontSizePixel * 0.5f;

                UINT32 fontColor = 0xffFFffFF;
                if (!isControlEnabled(lwrSliderBase))
                {
                    fontColor = 0x33FFffFF; // 0x33 = 20% * 0xff
                }

                FW1.pFontWrapperSergoeUIBold->DrawString(
                    d3dctx,
                    lwrSliderBase->title,// String
                    sliderTitleFontSizePixel,// Font size
                    sliderTitleLeftX,// X offset
                    viewPort_Height - sliderTitleCenterY,// Y offset
                    fontColor,// Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_BOTTOM | FW1_STATEPREPARED// Flags
                );
#endif
            }
        }
        containerHelper.stopSearchHierarchical();

        // Render error messages
        if (1)
        {
#if (DBG_USE_OUTLINE == 1)
            FontOutlineShaderConstBuf outlineData =
            {
                0.0f, 0.0f, 0.0f, 1.0f,
                1.0f
            };

            //d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);
            D3D11_MAPPED_SUBRESOURCE subResource;
            d3dctx->Map(pFontOutlineBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
            memcpy(subResource.pData, &outlineData, sizeof(FontOutlineShaderConstBuf));
            d3dctx->Unmap(pFontOutlineBuffer, 0);
            d3dctx->PSSetConstantBuffers(0, 1, &pFontOutlineBuffer);
#endif

            const int errNum = (int)errorManager.getErrorCount();
            int errorsDisplayed = 0;
            for (int errCnt = 0; errCnt < errNum; ++errCnt)
            {
                int lwrErrorEntry = (errorManager.getFirstErrorIndex() + errCnt) % errNum;

                if (errorManager.getErrorLifetime(lwrErrorEntry) < 0.0f)
                    continue;

                int errorOpacity = clamp(static_cast<int>(255 * errorManager.getErrorLifetime(lwrErrorEntry)), 0, 255);

                UINT32 fontColor = 0x00ffFFff + (errorOpacity<<24);

                // Draw some strings (Y goes top to bottom)
                FW1.pFontWrapperSergoeUIBold->DrawString(
                    d3dctx,
                    errorManager.getErrorString(lwrErrorEntry).c_str(),
                    16.0f / 1080.f * viewPort_Height,// Font size
                    (0.5f * m_components.leftPane->sizeX) * viewPort_Width,// X offset
                    ((0.f + 16.0f * errorsDisplayed) / 1080.f) * viewPort_Height,// Y offset
                    fontColor,// Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
                    );

                ++errorsDisplayed;
            }

#if (DBG_USE_OUTLINE == 1)
            FontOutlineShaderConstBuf outlineDefaultData =
            {
                1.0f, 1.0f, 1.0f, 1.0f,
                -1.0f
            };

            d3dctx->Map(pFontOutlineBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
            memcpy(subResource.pData, &outlineDefaultData, sizeof(FontOutlineShaderConstBuf));
            d3dctx->Unmap(pFontOutlineBuffer, 0);
            d3dctx->PSSetConstantBuffers(0, 1, &pFontOutlineBuffer);
#endif
        }

        FW1.pRenderStatesSergoeUI->SetStates(d3dctx, 0);
        d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);

        // Render auxiliary text above the progress bar
        if (m_progressInfo_inProgress)
        {
            const float fontSize = 16 / 1080.f * viewPort_Height;
            float captionRightmostX = progressBarContainerWidth / 2.0f + leftOffset * 0.5f;
            float captionRightmostY = progressBarContainerHeight / 2.0f;

            // Normalized -> absolute
            captionRightmostX = (captionRightmostX + 1.0f) / 2.0f * viewPort_Width;
            captionRightmostY = (captionRightmostY + 1.0f) / 2.0f * viewPort_Height;

            wchar_t progressBarCaption[128];
            const std::wstring localizedProgressString(m_textProgress);
            const auto formatString = localizedProgressString + L": %d / %d (%.0f%%)";
            swprintf_s(progressBarCaption, 128, formatString.c_str(), m_progressInfo_shotIdx, m_progressInfo_shotsTotal, m_progressInfo_shotIdx / (float)m_progressInfo_shotsTotal * 100.0f);

            UINT32 fontColor = 0xffFFffFF;
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                progressBarCaption,// String
                fontSize,// Font size
                captionRightmostX,// X offset
                viewPort_Height - captionRightmostY,// Y offset
                fontColor,// Text color, 0xAaBbGgRr
                FW1_RIGHT | FW1_BOTTOM | FW1_STATEPREPARED// Flags
            );
        }

        containerHelper.startSearchHierarchical(&mainContainer);
        while (lwrContainer = containerHelper.getNextControlHierarchical(scissorPop))
        {
            if (!lwrContainer->isVisible)
            {
                containerHelper.jumpToNextControlHierarchical();
                continue;
            }

            // TODO avoroshilov UI
            //  this can be done more effectively, increasing offset each push_back, decreasing each pop_back

            float controlAbsPosX = lwrContainer->absPosX;
            float controlAbsPosY = lwrContainer->absPosY;

            {
                ui::ControlContainer * cointainerTraverse = lwrContainer;
                while (cointainerTraverse->m_parent)
                {
                    controlAbsPosY += cointainerTraverse->m_parent->m_scrollValueY;
                    cointainerTraverse = cointainerTraverse->m_parent;
                }
            }

            if (lwrContainer->isBasicContainer() && lwrContainer->m_isClipping)
            {
                D3D11_RECT newRect;
                newRect.left = (LONG)((controlAbsPosX + 1.0f) * 0.5f * viewPort_Width);
                newRect.right = (LONG)((controlAbsPosX + lwrContainer->sizeX + 1.0f) * 0.5f * viewPort_Width);
                newRect.top = (LONG)((1.0f - (controlAbsPosY + lwrContainer->sizeY + 1.0f) * 0.5f) * viewPort_Height);
                newRect.bottom = (LONG)((1.0f - (controlAbsPosY + 1.0f) * 0.5f) * viewPort_Height);

                if (scissorRectStack.size() > 0)
                {
                    const D3D11_RECT & oldRect = scissorRectStack.back();

                    // Intersecting old rect with the new rect
                    clipRect(newRect, oldRect);
                }

                scissorRectStack.push_back(newRect);

                d3dctx->RSSetScissorRects(1, &newRect);
            }

            if (lwrContainer->getType() == ui::ControlType::kLabel)
            {
                ui::ControlLabel * lwrLabel = static_cast<ui::ControlLabel *>(lwrContainer);

                if (lwrLabel->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                float btnCaptionCenterX = controlAbsPosX + 0.5f * lwrLabel->sizeX;
                float btnCaptionCenterY = controlAbsPosY + 0.5f * lwrLabel->sizeY;

                UINT32 fontFlagsLoc = fontFlags;
                if (lwrLabel->fontAlignment == ui::ControlBase::FontAlignment::kRight)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_RIGHT;
                    btnCaptionCenterX = controlAbsPosX + lwrLabel->sizeX;
                }
                else if (lwrLabel->fontAlignment == ui::ControlBase::FontAlignment::kLeft)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_LEFT;
                    btnCaptionCenterX = controlAbsPosX;
                }

                ui::ColorF4 fontColorF4 = lwrLabel->fontColor;
                if (!isControlEnabled(lwrLabel))
                {
                    fontColorF4 = ui::ColorF4(1.0f, 1.0f, 1.0f, 0.2f);
                }
                UINT32 fontColorLoc = colwertColorFloatToUint(fontColorF4);

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    lwrLabel->caption,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    fontFlagsLoc// Flags
                );
            }
            if (lwrContainer->getType() == ui::ControlType::kButton)
            {
                ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);

                if (lwrButton->renderType == ui::ControlButton::RenderType::kFlyoutToggle)
                {
                    ui::ControlFlyoutToggleShared * lwrFlyoutToggle = static_cast<ui::ControlFlyoutToggleShared *>(lwrButton);

                    const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                    float btnCaptionCenterX = controlAbsPosX + 0.5f * lwrButton->sizeX;
                    float btnCaptionCenterY = controlAbsPosY + 2.f*onePixelYDesign;

                    ui::ColorF4 btnFontColor = lwrFlyoutToggle->fontColor;
                    if (!isControlEnabled(lwrButton))
                    {
                        btnFontColor.val[3] = 0.2f;
                    }

                    UINT32 fontColorLoc = colwertColorFloatToUint(btnFontColor);
                    UINT32 fontFlagsLoc = FW1_CENTER | FW1_BOTTOM | FW1_STATEPREPARED;

                    // Normalized -> absolute
                    btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                    btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                    FW1.pFontWrapperSergoeUI->DrawString(
                        d3dctx,
                        lwrFlyoutToggle->dynamicCaption,// String
                        btnCaptionFontSize,// Font size
                        btnCaptionCenterX,// X offset
                        viewPort_Height - btnCaptionCenterY,// Y offset
                        fontColorLoc,// Text color, 0xAaBbGgRr
                        fontFlagsLoc// Flags
                        );

                    continue;
                }

                if (lwrButton->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                float btnCaptionCenterX = controlAbsPosX + 0.5f * lwrButton->sizeX;
                float btnCaptionCenterY = controlAbsPosY + 0.5f * lwrButton->sizeY;

                UINT32 fontFlagsLoc = fontFlags;
                if (lwrButton->fontAlignment == ui::ControlBase::FontAlignment::kRight)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_RIGHT;
                    btnCaptionCenterX = controlAbsPosX + lwrButton->sizeX;
                }
                else if (lwrButton->fontAlignment == ui::ControlBase::FontAlignment::kLeft)
                {
                    fontFlagsLoc &= (~FW1_CENTER);
                    fontFlagsLoc |= FW1_LEFT;
                    btnCaptionCenterX = controlAbsPosX;
                }

                ui::ColorF4 btnFontColor = lwrButton->fontColor;
                if (!isControlEnabled(lwrButton))
                {
                    btnFontColor.val[3] = 0.2f;
                }

                if (lwrButton->renderType == ui::ControlButton::RenderType::kToggle)
                {
                    ui::ControlButtonToggle * lwrButtonToggle = static_cast<ui::ControlButtonToggle *>(lwrButton);

                    // TODO avoroshilov: unify this with +/- rendering
                    const float glyphThickness = 2.0f;
                    const float glyphMarginWidth = lwrButtonToggle->m_glyphMargin;
                    const float glyphMarginHeight = 0.0f;
                    const float glyphWidth = 10.0f * onePixelXDesign;
                    const float glyphHeight = 10.0f * onePixelYDesign;

                    fontFlagsLoc = FW1_LEFT | FW1_VCENTER | FW1_STATEPREPARED;
                    btnCaptionCenterX = controlAbsPosX + glyphWidth * 0.5f + glyphMarginWidth + 2 / 1920.f * 2.f;

                    float linkedContainerSizeY = 0.0f;
                    if (lwrButtonToggle->m_containerToggle)
                    {
                        linkedContainerSizeY = lwrButtonToggle->m_containerToggle->sizeY;
                    }
                    if (linkedContainerSizeY == 0.0f)
                    {
                        btnFontColor.val[3] = 0.2f;
                    }
                    if (!isControlEnabled(lwrButton))
                    {
                        btnFontColor.val[3] = 0.2f;
                    }
                }

                UINT32 fontColorLoc = colwertColorFloatToUint(btnFontColor);
                if (lwrButton->state != UI_CONTROL_ORDINARY && lwrButton->hlType == ui::ControlButton::HighlightType::kFont)
                    fontColorLoc = colwertColorFloatToUint(lwrButton->hlColor);

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    lwrButton->caption,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    fontFlagsLoc// Flags
                    );
            }
            if (lwrContainer->getType() == ui::ControlType::kCheckbox)
            {
                ui::ControlCheckbox * lwrCheckbox  = static_cast<ui::ControlCheckbox  *>(lwrContainer);

                if (lwrCheckbox->isBold)
                    continue;

                const float btnCaptionFontSize = 16 / 1080.f * viewPort_Height;
                const float aspect = m_width / (float)m_height;
                float btnCaptionCenterX = controlAbsPosX + 4*onePixelXDesign + lwrCheckbox->checkSize / aspect;
                float btnCaptionCenterY = controlAbsPosY;

                UINT32 fontFlagsLoc = fontFlags;
                UINT32 fontColorLoc = 0xffFFffFF;
                if (!isControlEnabled(lwrCheckbox))
                {
                    fontColorLoc = 0x33FFffFF;  // 0x33 = 20% * 0xff
                }

                // Normalized -> absolute
                btnCaptionCenterX = (btnCaptionCenterX + 1.0f) / 2.0f * viewPort_Width;
                btnCaptionCenterY = (btnCaptionCenterY + 1.0f) / 2.0f * viewPort_Height;

                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    lwrCheckbox->title,// String
                    btnCaptionFontSize,// Font size
                    btnCaptionCenterX,// X offset
                    viewPort_Height - btnCaptionCenterY,// Y offset
                    fontColorLoc,// Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_BOTTOM | FW1_STATEPREPARED// Flags
                    );
            }
            else if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
                     lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
                     lwrContainer->getType() == ui::ControlType::kSliderInt)
            {
                ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

                // Both render below, centered
                if (lwrSliderBase->getType() != ui::ControlType::kSliderCont &&
                    lwrSliderBase->getType() != ui::ControlType::kSliderDiscr &&
                    lwrSliderBase->getType() != ui::ControlType::kSliderInt)
                {
                    continue;
                }

                // Title non-bold version
#if (UI_SLIDER_BOLD_CAPTIONS == 0)
                {
                    const float sliderTitleFontSizePixel = sliderTitleFontSize * 0.5f * viewPort_Height;
                    float sliderTitleLeftX = controlAbsPosX;
                    float sliderTitleCenterY = controlAbsPosY + lwrSliderBase->trackShiftY + 0.5f * lwrSliderBase->thumbSizeY + lwrSliderBase->thumbBorderY - 5.f / 1080.f * 2.f;

                    // Normalized -> absolute
                    sliderTitleLeftX = (sliderTitleLeftX + 1.0f) / 2.0f * viewPort_Width;

                    sliderTitleCenterY = (sliderTitleCenterY + 1.0f) / 2.0f * viewPort_Height;
                    sliderTitleCenterY += sliderTitleFontSizePixel * 0.5f;

                    UINT32 fontColor = 0xffFFffFF;
                    if (!isControlEnabled(lwrSliderBase))
                    {
                        fontColor = 0x33FFffFF;  // 0x33 = 20% * 0xff
                    }

                    FW1.pFontWrapperSergoeUI->DrawString(
                        d3dctx,
                        lwrSliderBase->title,// String
                        sliderTitleFontSizePixel,// Font size
                        sliderTitleLeftX,// X offset
                        viewPort_Height - sliderTitleCenterY,// Y offset
                        fontColor,// Text color, 0xAaBbGgRr
                        FW1_LEFT | FW1_BOTTOM | FW1_STATEPREPARED// Flags
                        );
                }
#endif

                if (lwrSliderBase->isLeanStyle)
                {
                    const float sliderSubFontSizePixel = sliderSubFontSize * 0.5f * viewPort_Height;
                    float sliderTitleRightX = controlAbsPosX + lwrSliderBase->sizeX;
                    // -9px is because of the continuous sliders default notch
                    float sliderTitleCenterY = controlAbsPosY + lwrSliderBase->trackShiftY + 0.5f * lwrSliderBase->thumbSizeY + lwrSliderBase->thumbBorderY - 4.f / 1080.f * 2.f;

                    // Normalized -> absolute
                    sliderTitleRightX = (sliderTitleRightX + 1.0f) / 2.0f * viewPort_Width;
                    sliderTitleCenterY = (sliderTitleCenterY + 1.0f) / 2.0f * viewPort_Height;

                    sliderTitleCenterY += sliderSubFontSizePixel * 0.5f;

                    UINT32 fontColor = 0xb4FFffFF;
                    if (!isControlEnabled(lwrSliderBase))
                    {
                        fontColor = 0x24FFffFF;  // 0x24 = 20% * 0xb4
                    }

                    // This text should be vertically aligned by top edge, sicnce it could be multi-line
                    UINT sliderSubTextFontFlags = (fontFlags & (~FW1_VCENTER)) | FW1_TOP;

                    if (lwrSliderBase->isMinMaxLabeled == true)
                    {
                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            lwrSliderBase->minSubTitle,// String
                            sliderSubFontSizePixel,// Font size
                            (controlAbsPosX + 1.0f) / 2.0f * viewPort_Width,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            (sliderSubTextFontFlags & (~FW1_CENTER)) | FW1_LEFT// Flags
                            );

                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            lwrSliderBase->maxSubTitle,// String
                            sliderSubFontSizePixel,// Font size
                            (controlAbsPosX + lwrSliderBase->sizeX + 1.0f) / 2.0f * viewPort_Width,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            (sliderSubTextFontFlags & (~FW1_CENTER)) | FW1_RIGHT// Flags
                            );
                    }
                    else
                    {
                        wchar_t printed[256];

                        lwrSliderBase->getText(printed, 256);

                        UINT flags = (fontFlags & (~FW1_CENTER)) & (~FW1_VCENTER);
                        flags = flags | FW1_RIGHT | FW1_BOTTOM;
                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            printed,// String
                            sliderSubFontSizePixel,// Font size
                            sliderTitleRightX,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            flags// Flags
                            );
                    }
                }
                else
                {
                    const float sliderSubFontSizePixel = sliderSubFontSize * 0.5f * viewPort_Height;
                    float sliderTitleCenterX = controlAbsPosX + 0.5f * lwrSliderBase->sizeX;
                    // -9px is because of the continuous sliders default notch
                    float sliderTitleCenterY = controlAbsPosY + lwrSliderBase->trackShiftY - 0.5f * lwrSliderBase->thumbSizeY - 9.f / 1080.f*2.f;

                    // Normalized -> absolute
                    sliderTitleCenterX = (sliderTitleCenterX + 1.0f) / 2.0f * viewPort_Width;
                    sliderTitleCenterY = (sliderTitleCenterY + 1.0f) / 2.0f * viewPort_Height;

                    sliderTitleCenterY += sliderSubFontSizePixel * 0.5f;

                    UINT32 fontColor = 0xb4FFffFF;
                    if (!isControlEnabled(lwrSliderBase))
                    {
                        fontColor = 0x24FFffFF;  // 0x24 = 20% * 0xb4
                    }

                    // This text should be vertically aligned by top edge, sicnce it could be multi-line
                    UINT sliderSubTextFontFlags = (fontFlags & (~FW1_VCENTER)) | FW1_TOP;

                    if (lwrSliderBase->isMinMaxLabeled == true)
                    {
                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            lwrSliderBase->minSubTitle,// String
                            sliderSubFontSizePixel,// Font size
                            (controlAbsPosX + 1.0f) / 2.0f * viewPort_Width,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            (sliderSubTextFontFlags & (~FW1_CENTER)) | FW1_LEFT// Flags
                            );

                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            lwrSliderBase->maxSubTitle,// String
                            sliderSubFontSizePixel,// Font size
                            (controlAbsPosX + lwrSliderBase->sizeX + 1.0f) / 2.0f * viewPort_Width,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            (sliderSubTextFontFlags & (~FW1_CENTER)) | FW1_RIGHT// Flags
                            );
                    }
                    else
                    {
                        wchar_t printed[256];

                        lwrSliderBase->getText(printed, 256);

                        FW1.pFontWrapperSergoeUI->DrawString(
                            d3dctx,
                            printed,// String
                            sliderSubFontSizePixel,// Font size
                            sliderTitleCenterX,// X offset
                            viewPort_Height - sliderTitleCenterY,// Y offset
                            fontColor,// Text color, 0xAaBbGgRr
                            sliderSubTextFontFlags// Flags
                            );
                    }
                }
            }
        }
        containerHelper.stopSearchHierarchical();

#if 0
        FW1.pRenderStatesSergoeUIBold->SetStates(d3dctx, 0);
        d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);
        // Draw some strings (Y goes top to bottom)
        FW1.pFontWrapperSergoeUIBold->DrawString(
            d3dctx,
            L"SHARE (BETA)",// String
            32.0f,// Font size
            viewPort_Width*0.5f,// X offset
            (40.0f / 1080.0f) * viewPort_Height,// Y offset
            0xFFffFFff,// Text color, 0xAaBbGgRr
            fontFlags// Flags
        );
#endif

        FW1.pRenderStatesSergoeUI->SetStates(d3dctx, 0);
        d3dctx->PSSetShader(FW1.pPSOutline, NULL, 0);

        // Render debug perf text (FPS/st)
        if (debugInfo.renderDebugInfo)
        {
            double dtSeconds = debugInfo.dt * 0.001;
            swprintf_s(ui_text, ui_textBufSize, L"FPS: %.1f (%.2fms)", 1.f / dtSeconds, debugInfo.dt);

            // Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                ui_text,// String
                16.0f,// Font size
                (leftOffset / 2.0f) * viewPort_Width,// X offset
                (0.f / 1080.f) * viewPort_Height,// Y offset
                0xFFffFFff,// Text color, 0xAaBbGgRr
                FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
                );

            // Render debug latencies text
            swprintf_s(ui_text, ui_textBufSize, L"Latencies: capture %d, settle %d", debugInfo.shotCaptureLatency, debugInfo.shotSettleLatency);

            // Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                ui_text,// String
                16.0f,// Font size
                (leftOffset / 2.0f) * viewPort_Width,// X offset
                (16.f / 1080.f) * viewPort_Height,// Y offset
                0xFFffFFff,// Text color, 0xAaBbGgRr
                FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
                );

            // Render debug latencies text
            swprintf_s(ui_text, ui_textBufSize, L"network: %d", debugInfo.networkBytesTransferred);

            // Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                ui_text,// String
                16.0f,// Font size
                (leftOffset / 2.0f) * viewPort_Width,// X offset
                (32.0f / 1080.f) * viewPort_Height,// Y offset
                0xFFffFFff,// Text color, 0xAaBbGgRr
                FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
                );

            const float rowSizeY = 16.0f;
            const float lastHardcodedRowPosY = 32.0f;

            for (size_t cntRow = 0, cntRowEnd = debugInfo.additionalLines.size(); cntRow < cntRowEnd; ++cntRow)
            {
                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    debugInfo.additionalLines[cntRow].c_str(),// String
                    16.0f,// Font size
                    (leftOffset / 2.0f) * viewPort_Width,// X offset
                    ((lastHardcodedRowPosY + rowSizeY*(cntRow+1)) / 1080.f) * viewPort_Height,// Y offset
                    0xFFffFFff,// Text color, 0xAaBbGgRr
                    FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
                    );
            }
        }
        // Render debug gamepad text
        if (DEBUG_GAMEPAD)
        {
            swprintf_s(ui_text, ui_textBufSize, L"Gamepad: stick1 %d/%d, stick2 %d/%d, Z axis: %d Z axis float %f Dpad %d X %d Y %d A %d B %d LCap %d RCap %d LShld %d Rshld %d",
                       debugInfo.gamepadDebugInfo.lx, debugInfo.gamepadDebugInfo.ly, debugInfo.gamepadDebugInfo.rx, debugInfo.gamepadDebugInfo.ry,
                       debugInfo.gamepadDebugInfo.z, debugInfo.gamepadDebugInfo.fz, debugInfo.gamepadDebugInfo.dpad,
                       1ul * debugInfo.gamepadDebugInfo.x, 1ul * debugInfo.gamepadDebugInfo.y, 1ul * debugInfo.gamepadDebugInfo.a, 1ul * debugInfo.gamepadDebugInfo.b,
                       1ul * debugInfo.gamepadDebugInfo.lcap, 1ul * debugInfo.gamepadDebugInfo.rcap, 1ul * debugInfo.gamepadDebugInfo.lshoulder, 1ul * debugInfo.gamepadDebugInfo.rshoulder);

            //// Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                ui_text,// String
                16.0f,// Font size
                leftOffset * viewPort_Width,// X offset
                (32.f / 1080.f) * viewPort_Height,// Y offset
                0xFFffFFff,// Text color, 0xAaBbGgRr
                FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
            );
        }
        // Render debug bitness
        if (0)
        {
            // Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
#if _WIN32 && _WIN64
                L"64-bit",// String
#else
                L"32-bit",// String
#endif
                16.0f,// Font size
                (leftOffset / 2.0f) * viewPort_Width,// X offset
                (48.f / 1080.f) * viewPort_Height,// Y offset
                0xFFffFFff,// Text color, 0xAaBbGgRr
                FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
            );
        }

        if (0)
        {
            // Draw some strings (Y goes top to bottom)
            FW1.pFontWrapperSergoeUI->DrawString(
                d3dctx,
                ui_text,// String
                96.0f,// Font size
                viewPort_Width*0.5f,// X offset
                viewPort_Height*0.25f,// Y offset
                fontColor,// Text color, 0xAaBbGgRr
                fontFlags// Flags
            );

            if (anselSDKDetected)
            {
                swprintf_s(ui_text, ui_textBufSize, L"%f, %f, %f", cam.position.x, cam.position.y, cam.position.z);

                FW1.pFontWrapperSergoeUI->DrawString(
                    d3dctx,
                    ui_text,// String
                    50.0f,// Font size
                    viewPort_Width*0.5f,// X offset
                    viewPort_Height*0.5f,// Y offset
                    fontColor,// Text color, 0xAaBbGgRr
                    fontFlags// Flags
                );
            }
        }

        // TODO: maybe use FW1_RESTORESTATE

        //d3dctx->VSSetShader(0, NULL, 0);
        d3dctx->HSSetShader(0, NULL, 0);
        d3dctx->DSSetShader(0, NULL, 0);
        d3dctx->GSSetShader(0, NULL, 0);
        //d3dctx->PSSetShader(0, NULL, 0);
#endif

        // TODO: move this into separete setUIRenderingState
        d3dctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        d3dctx->IASetInputLayout(pInputLayout);
        d3dctx->VSSetShader(pVertexShader, NULL, 0);
        d3dctx->RSSetState(pRasterizerState);
        d3dctx->PSSetShader(pPixelShader, NULL, 0);
        d3dctx->PSSetShaderResources(0, 1, &pUIAtlasSRV);
        d3dctx->PSSetSamplers(0, 1, &pPassthroughEffect->pSamplerState);      // Using passthrough effect sampler state
        d3dctx->OMSetRenderTargets(1, &pPresentResourceData->toClientRes.pRTV, NULL);
        d3dctx->OMSetDepthStencilState(pDepthStencilState, 0xFFFFFFFF);
        d3dctx->OMSetBlendState(pBlendState, NULL, 0xffffffff);

        // Draw mouse cursor
        if (!isCameraDragActive)
        {
            // These sizes should not be relative to the design size (1920x1080), but rather relative to the real effect size (no scaling)
            float frameWidth = 1920, frameHeight = 1080;
            if (m_width > 0)
                frameWidth = (float)m_width;
            if (m_height > 0)
                frameHeight = (float)m_height;

            UIShaderConstBuf mouseLwrsorData =
            {
                1.0f, 1.0f, 1.0f, 1.0f,  // Color

                mouseCoordsAbsX - 32.0f / frameWidth*2.0f, mouseCoordsAbsY - 32.0f / frameHeight*2.0f,
                64.0f / frameWidth*2.0f, 64.0f / frameHeight*2.0f
            };

            if (!haveFolws && !m_showMouseWhileDefolwsed)
                mouseLwrsorData.ca = 0.0f;

            // DBG
#if 0
            {
                D3D11_MAPPED_SUBRESOURCE subResource;
                d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
                memcpy(subResource.pData, &mouseLwrsorData, sizeof(UIShaderConstBuf));
                d3dctx->Unmap(pVariableOffsetsBuffer, 0);
                d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);
                d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
                d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
                d3dctx->DrawIndexed(6, 0, 0);
            }
#endif

            // If mouse cursor is DONT_CARE - we set arrow
            ID3D11Buffer * pMouseLwrsorVertexBuf = pMousePointerVertexBuf;
            if (mouseLwrsor == UI_MOUSELWRSOR_ARROW)
            {
                pMouseLwrsorVertexBuf = pMousePointerVertexBuf;
            }
            else if (mouseLwrsor == UI_MOUSELWRSOR_HAND)
            {
                pMouseLwrsorVertexBuf = pMouseHandVertexBuf;
            }

            D3D11_MAPPED_SUBRESOURCE subResource;
            d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
            memcpy(subResource.pData, &mouseLwrsorData, sizeof(UIShaderConstBuf));
            d3dctx->Unmap(pVariableOffsetsBuffer, 0);
            d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

            d3dctx->IASetIndexBuffer(pMouseIndexBuf, DXGI_FORMAT_R32_UINT, 0);
            d3dctx->IASetVertexBuffers(0, 1, &pMouseLwrsorVertexBuf, &vbStride, &offset);
            d3dctx->DrawIndexed(6, 0, 0);
        }
    }

    if (m_fadeValue)
    {
        UINT vbStride = vertexStrideUI;
        UINT offset = 0;

        setRenderState(d3dctx, pPresentResourceData, pPassthroughEffect);

        float alpha = m_fadeValue;

        UIShaderConstBuf controlData_Left =
        {
            0.0f, 0.0f, 0.0f, alpha,  // Color
            -1.0f, -1.0f, 2.0f, 2.0f
        };

        D3D11_MAPPED_SUBRESOURCE subResource;

        d3dctx->Map(pVariableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
        memcpy(subResource.pData, &controlData_Left, sizeof(UIShaderConstBuf));
        d3dctx->Unmap(pVariableOffsetsBuffer, 0);
        d3dctx->VSSetConstantBuffers(0, 1, &pVariableOffsetsBuffer);

        d3dctx->IASetIndexBuffer(pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
        d3dctx->IASetVertexBuffers(0, 1, &pRectVertexBuf, &vbStride, &offset);
        d3dctx->DrawIndexed(6, 0, 0);
    }
}

void AnselUI::setModdingStatus(bool isModdingAllowed)
{
    m_isModdingAllowed = isModdingAllowed;
    m_components.chkAllowModding->isVisible = m_isStandaloneModdingEnabled && isModdingAllowed;
    if (!isModdingAllowed)
    {
        m_components.chkAllowModding->isChecked = false;
    }
    m_needToRecallwILayout = true;
}

void AnselUI::release()
{
    // Cleanup game-specific control
    {
        ui::ControlContainer * selectedControl = containerHelper.getSelectedControl(&mainContainer);
        bool needReselectControl = false;
        removeDynamicElementsInContainer(m_components.cntGameSpecific, &selectedControl, &needReselectControl);
        if (needReselectControl)
        {
            // Reselecting the same control to update selection indices
            containerHelper.setSelectedControl(selectedControl, selectedControl);
        }
    }

    // Cleanup dynamic effects controls
    while (!m_components.m_dynamicFilterContainers.empty())
    {
        m_dynamicFilterIdxToRemove = 0;
        removeDynamicFilter();
    }

    // Cleanup dynamic flyouts
    FlyoutRebuildRequest flyoutRebuildRequest;
#ifdef ENABLE_STYLETRANSFER
    {
        ui::ControlFlyoutStylesToggle * flyStyleNetworks = static_cast<ui::ControlFlyoutStylesToggle * >(m_components.flyStyleNetworks);
        flyoutRebuildRequest.isValid = true;
        flyoutRebuildRequest.dstFlyoutContainer = m_components.flyoutPane;
        flyoutRebuildRequest.srcFlyoutToggle = flyStyleNetworks;
        clearFlyout(&flyoutRebuildRequest);
    }
#endif

    {
        ui::ControlFlyoutToggleShared * srcFlyoutToggle = (ui::ControlFlyoutToggleShared *)(m_components.flySpecialFX);
        flyoutRebuildRequest.isValid = true;
        flyoutRebuildRequest.dstFlyoutContainer = m_components.flyoutPane;
        flyoutRebuildRequest.srcFlyoutToggle = srcFlyoutToggle;
        clearFlyout(&flyoutRebuildRequest);
    }

    m_inputstate.deinit();

    SAFE_RELEASE(pPixelShader);
    SAFE_RELEASE(pVertexShader);
    SAFE_RELEASE(pInputLayout);
    SAFE_RELEASE(pUIAtlasTexture);
    SAFE_RELEASE(pUIAtlasSRV);

    SAFE_RELEASE(pDepthStencilState);
    SAFE_RELEASE(pBlendState);
    SAFE_RELEASE(pRasterizerState);

    SAFE_RELEASE(pTriUpIndexBuf);
    SAFE_RELEASE(pTriUpVertexBuf);
    
    SAFE_RELEASE(pZeroOffsetsBuffer);
    SAFE_RELEASE(pVariableOffsetsBuffer);
#if (DBG_USE_OUTLINE == 1)
    SAFE_RELEASE(pFontOutlineBuffer);
#endif

    SAFE_RELEASE(pArrowDowlwertexBuf);
    SAFE_RELEASE(pArrowUpVertexBuf);
    SAFE_RELEASE(pArrowRightVertexBuf);
    SAFE_RELEASE(pArrowLeftVertexBuf);
    SAFE_RELEASE(pArrowIndexBuf);

    SAFE_RELEASE(pRectVertexBuf);
    SAFE_RELEASE(pRectIndexBuf);

    SAFE_RELEASE(pCamIcolwertexBuf);

    SAFE_RELEASE(pMouseIndexBuf);
    SAFE_RELEASE(pMousePointerVertexBuf);
    SAFE_RELEASE(pMouseHandVertexBuf);

    vertexStrideUI = 0;

    // TODO: add resourceManager here in order to avoid manually maintaining which buffer is which
    containerHelper.init(&mainContainer);
    ui::ControlContainer * lwrContainer;
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->getType() == ui::ControlType::kButton)
        {
            ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);
            if (!lwrButton->renderBufsShared)
            {
                SAFE_RELEASE(lwrButton->pIndexBuf);
                SAFE_RELEASE(lwrButton->pVertexBuf);
            }
            if (!lwrButton->renderBufsAuxShared)
            {
                SAFE_RELEASE(lwrButton->pVertexBufHl);
                SAFE_RELEASE(lwrButton->pVertexBufDn);
            }
        }
        else if (lwrContainer->getType() == ui::ControlType::kIcon)
        {
            ui::ControlIcon * lwrIcon = static_cast<ui::ControlIcon *>(lwrContainer);
            SAFE_RELEASE(lwrIcon->pIndexBuf);
            for (size_t vbi = 0, vbiend = lwrIcon->vertexBufDescs.size(); vbi < vbiend; ++vbi)
            {
                SAFE_RELEASE(lwrIcon->vertexBufDescs[vbi].pVertexBuf);
            }
        }
        else if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
                 lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
                 lwrContainer->getType() == ui::ControlType::kSliderInt)
        {
            ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

            // TODO avoroshilov UI
            if (lwrSliderBase->blockID == STATIC_BLOCKID)
            {
            }
            else
            {
                // TODO avoroshilov UI
                //  deleting as if they are dynamic sliders
                //  but dynamic elements could be of any type potentially in the future

                if (lwrContainer->getType() == ui::ControlType::kSliderCont)
                    static_cast<ui::ControlSliderCont *>(lwrSliderBase)->~ControlSliderCont();
                else if (lwrContainer->getType() == ui::ControlType::kSliderDiscr)
                    static_cast<ui::ControlSliderDiscr *>(lwrSliderBase)->~ControlSliderDiscr();
                else if (lwrContainer->getType() == ui::ControlType::kSliderInt)
                    static_cast<ui::ControlSliderInt *>(lwrSliderBase)->~ControlSliderInt();
            }
        }
    }
    containerHelper.stopSearch();

    // TODO avoroshilov UI
    //  change the traversal here - we need to de-initialize containers as well

    releaseDynamicControlPools();

    if (m_components.sldHiResMult->labels.size())
        delete[] m_components.sldHiResMult->labels[0];

    setIsVisible(false);
    setIsEnabled(false);

    SAFE_DELETE(m_components.btnToggleFilter);
    SAFE_DELETE(m_components.btnToggleAdjustments);
    SAFE_DELETE(m_components.btnToggleFX);
#ifdef ENABLE_STYLETRANSFER
    SAFE_DELETE(m_components.btnToggleStyleTransfer);
#endif
    SAFE_DELETE(m_components.btnToggleGameSpecific);
    SAFE_DELETE(m_components.btnToggleCamCapture);

    SAFE_DELETE(m_components.btnSnap);
    SAFE_DELETE(m_components.btnDone);

#ifdef ENABLE_STYLETRANSFER
    SAFE_DELETE(m_components.pbRestyleProgress);
    SAFE_DELETE(m_components.lblRestyleProgress);
    SAFE_DELETE(m_components.lblRestyleProgressIndicator);

    SAFE_DELETE(m_components.lblDownloadRestyleText);
    SAFE_DELETE(m_components.btnDownloadRestyleConfirm);
    SAFE_DELETE(m_components.btnDownloadRestyleCancel);
#endif

#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        SAFE_DELETE(m_components.btnAddFilter);
    }
#endif

    SAFE_DELETE(m_components.sldKind);
    SAFE_DELETE(m_components.chkGridOfThirds);
    SAFE_DELETE(m_components.chkHDR);

    SAFE_DELETE(m_components.chkAllowModding);

    SAFE_DELETE(m_components.flySpecialFX);

    SAFE_DELETE(m_components.sldFOV);
    SAFE_DELETE(m_components.sldRoll);
    SAFE_DELETE(m_components.btnResetRoll);

    SAFE_DELETE(m_components.sldHiResMult);
    SAFE_DELETE(m_components.sldSphereFOV);

    SAFE_DELETE(m_components.chkEnhanceHiRes);

#ifdef ENABLE_STYLETRANSFER
    SAFE_DELETE(m_components.chkEnableStyleTransfer);
    SAFE_DELETE(m_components.flyStyles);
    SAFE_DELETE(m_components.flyStyleNetworks);
#endif
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    SAFE_DELETE(m_components.lblHotkey);
    SAFE_DELETE(m_components.btnHotkeySetup);
#endif

    SAFE_DELETE(m_components.icoLWLogo);
    SAFE_DELETE(m_components.icoCamera);
    SAFE_DELETE(m_components.icoLight);
    SAFE_DELETE(m_components.icoFilters);

    SAFE_DELETE(m_components.leftPane);
    SAFE_DELETE(m_components.flyoutPane);
#ifdef ENABLE_STYLETRANSFER
    SAFE_DELETE(m_components.dlgDownloadRestyle);
    SAFE_DELETE(m_components.cntRestyleProgress);
#endif
    
    SAFE_DELETE(m_components.cntControls);
    SAFE_DELETE(m_components.cntFilter);
    SAFE_DELETE(m_components.cntAdjustments);
    SAFE_DELETE(m_components.cntFX);
#ifdef ENABLE_STYLETRANSFER
    SAFE_DELETE(m_components.cntStyleTransfer);
#endif
    SAFE_DELETE(m_components.cntGameSpecific);
    SAFE_DELETE(m_components.cntCameraCapture);

#if (UI_ENABLE_TEXT == 1)
    // Release FW1
    SAFE_RELEASE(FW1.pRenderStatesSergoeUI);
    SAFE_RELEASE(FW1.pFontWrapperSergoeUI);
    SAFE_RELEASE(FW1.pRenderStatesSergoeUIBold);
    SAFE_RELEASE(FW1.pFontWrapperSergoeUIBold);

    SAFE_RELEASE(FW1.pTextFormatSegoeUI);
    SAFE_RELEASE(FW1.pTextFormatSegoeUIBold);
    SAFE_RELEASE(FW1.pDWriteFactorySegoeUI);
    SAFE_RELEASE(FW1.pDWriteFactorySegoeUIBold);
    SAFE_RELEASE(FW1.pPSOutline);
    SAFE_RELEASE(FW1.pFactory);

    IFW1Factory::deinitDWrite();
#endif
    mouseCoordsAbsX = 0.0f;
    mouseCoordsAbsY = 0.0f;

    m_langID = (LANGID)-1;
}

void AnselUI::recallwlateUIPositions()
{
    ui::ControlContainer * lwrContainer;

    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->m_isScrollable)
        {
            lwrContainer->m_scrollContentSize = 0.0f;
        }

        if (lwrContainer->m_parent != nullptr)
        {
            lwrContainer->absPosX = FLT_MAX;
            lwrContainer->absPosY = FLT_MAX;
        }
    }
    containerHelper.stopSearch();

    int numUninitializedControls = 0;
    do
    {
        numUninitializedControls = 0;

        containerHelper.startSearch();
        while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
        {
            if (lwrContainer->absPosX != FLT_MAX && lwrContainer->absPosY != FLT_MAX)
                continue;

            // X-anchoring
            if (lwrContainer->m_anchorX != nullptr)
            {
                if (lwrContainer->m_anchorX->absPosX != FLT_MAX)
                {
                    lwrContainer->absPosX = lwrContainer->m_anchorY->absPosX + lwrContainer->posX;
                }
            }
            else if (lwrContainer->m_parent != nullptr)
            {
                if (lwrContainer->m_parent->absPosX != FLT_MAX)
                {
                    // Anchoring should be from the top of the parent element
                    lwrContainer->absPosX = lwrContainer->m_parent->absPosX + lwrContainer->posX;
                }
            }

            // Y-anchoring
            // We want coordinates to raise downwards
            //  while GAPI want them to raise upwards
            // TODO: check Y-shift for invisible elements that has nullptr Y-anchor
            // TODO: probably do not shift for posY if element is invisible
            if (lwrContainer->m_anchorY != nullptr)
            {
                if (lwrContainer->m_anchorY->absPosY != FLT_MAX)
                {
                    // Anchoring should be from the bottom of the anchor element
                    lwrContainer->absPosY = lwrContainer->m_anchorY->absPosY - (lwrContainer->posY + lwrContainer->sizeY);
                    if (!lwrContainer->m_anchorY->isVisible)
                    {
                        // Invisible elements do not have size
                        lwrContainer->absPosY += lwrContainer->m_anchorY->sizeY;
                        // They shoudln't have offsets
                        lwrContainer->absPosY += lwrContainer->m_anchorY->posY;
                    }
                }
            }
            else if (lwrContainer->m_parent != nullptr)
            {
                if (lwrContainer->m_parent->absPosY != FLT_MAX)
                {
                    // Anchoring should be from the top of the parent element
                    lwrContainer->absPosY = lwrContainer->m_parent->absPosY + lwrContainer->m_parent->sizeY - (lwrContainer->posY + lwrContainer->sizeY);
                }
            }

            ++numUninitializedControls;
        }
        containerHelper.stopSearch();
    }
    while (numUninitializedControls > 0);

    // Callwlate scrollable size
    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->m_parent && lwrContainer->m_parent->m_isScrollable)
        {
            float elementMax = lwrContainer->absPosY;
            if (!lwrContainer->isVisible)
            {
                elementMax += lwrContainer->sizeY;

                if (0)
                {
                    elementMax = 0.0f;
                }
            }
            lwrContainer->m_parent->m_scrollContentSize = std::max(lwrContainer->m_parent->m_scrollContentSize, lwrContainer->m_parent->m_scrollMarginBottom + (lwrContainer->m_parent->absPosY + lwrContainer->m_parent->sizeY) - elementMax);
        }
    }
    containerHelper.stopSearch();

    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        lwrContainer->m_absPosYScroll = lwrContainer->absPosY;
        if (lwrContainer->m_isScrollable)
        {
            lwrContainer->clampScrollValue();
        }
    }
    containerHelper.stopSearch();

#if 0
    // Pre callwlate global offsets
    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        ui::ControlContainer * childControl = lwrContainer;
        while (childControl->m_parent)
        {
            lwrContainer->m_absPosYScroll += childControl->m_parent->m_scrollValueY;
            childControl = childControl->m_parent;
        }
    }
    containerHelper.stopSearch();
#endif

    // Check if selected controls is actually visible
    //  adjust scrollValue if it is not
    ui::ControlContainer * selectedContainer = containerHelper.getSelectedControl(&mainContainer);
    float selectedControlPos = selectedContainer->absPosY;
    lwrContainer = selectedContainer;
    while (lwrContainer && containerHelper.wasControlJustSelected)
    {
        ui::ControlContainer * parentControl = lwrContainer->m_parent;
        if (parentControl == nullptr)
            break;

        if (parentControl->absPosY + parentControl->sizeY - parentControl->m_scrollValueY < selectedControlPos + selectedContainer->sizeY + m_globalScrollMargin)
        {
            parentControl->m_scrollValueY = (parentControl->absPosY + parentControl->sizeY) - (selectedControlPos + selectedContainer->sizeY + m_globalScrollMargin);
        }
        else if (parentControl->absPosY - parentControl->m_scrollValueY > selectedControlPos - m_globalScrollMargin)
        {
            parentControl->m_scrollValueY = parentControl->absPosY - (selectedControlPos - m_globalScrollMargin);
        }
        parentControl->clampScrollValue();

        selectedControlPos += parentControl->m_scrollValueY;
        lwrContainer = parentControl;
    }
    // We don't want to do this each frame, as it will limit scrolling
    //  selected element won't allow to scroll past it
    containerHelper.wasControlJustSelected = false;
}

void AnselUI::recallwlateUILayoutScale(float aspect, float * scaleX, float * scaleY) const
{
    bool portrait = aspect < 1.0f;
    const float aspect_default = 1920.f / 1080.f;
    float UImulX = portrait ? (aspect_default / aspect) : (aspect_default / aspect);
    float UImulY = 1.0f;

    if (scaleX)
        *scaleX = UImulX;

    if (scaleY)
        *scaleY = UImulY;
}

HRESULT AnselUI::recallwlateUILayout(float aspect)
{
    float UImulX = 1.0f, UImulY = 1.0f;
    recallwlateUILayoutScale(aspect, &UImulX, &UImulY);
    
    m_storedSizes.uiMarginX = UImulX * 20.f / 1920.f * 2.f;
    m_storedSizes.uiMulX = UImulX;
    m_storedSizes.uiMulY = UImulY;

    const float selectorStartX = UImulX * 7.f / 1920.f * 2.f;
    float selectorSizeX = UImulX * 186.f / 1920.f * 2.f;
    float selectorSizeY = UImulY * 30.f / 1080.f*2.f;
    float selectorSpacingY = UImulY * 1.f / 1080.f * 2.f;
    float selectorMarginY = UImulY * 2.f / 1080.f * 2.f;

    // Slider size is 160x3 in 1080@16:9
    // Ref screen -> normalized screen (-1 .. 1): * 2.0 - 1.0
    float sliderSizeX = UImulX * 160.f / 1920.f * 2.f, sliderSizeY = UImulY * 3.f / 1080.f * 2.f;
    m_storedSizes.defaultSliderSizeX = sliderSizeX;
    m_storedSizes.defaultCheckboxSizeX = sliderSizeX;
    m_storedSizes.defaultSelectorSizeX = selectorSizeX;

    const float sliderThumbSizeX = UImulX * 12.f / 1920.f * 2.f, sliderThumbSizeY = UImulY * 17.f / 1080.f * 2.f;
    const float thumbBorderX = UImulX * 1.f / 1920.f * 2.f;
    const float thumbBorderY = UImulX * 1.f / 1080.f * 2.f;

    const float btnSizeX = sliderSizeX;
    const float btnSizeY = UImulY * 40.f / 1080.f*2.f;
    const float flySizeY = UImulY * 40.f / 1080.f*2.f;
    const float lblSizeY = UImulY * 20.f / 1080.f*2.f;
    const float pbSizeY = UImulY * 10.f / 1080.f*2.f;

    const float margin = m_storedSizes.uiMarginX;//20.f / 1920.f * 2.f;
    float leftOffset = 2 * margin + sliderSizeX;

    ///////////////////////////////////////////////////
    // Callwlating controls layout

    // Spacing 80px
    const float sliderStartX = UImulX * 20.f / 1920.f * 2.f;// - 1.f;
    // Will be shifted later
    float sliderStartY = UImulY * (1.0f - 45.f / 1080.f) * 2.f;// - 1.f;
    const float sliderShiftY = UImulY * 60.f / 1080.f * 2.f;//-80.f / 1080.f * 2.f;
    const float sliderContShiftY = UImulY * 45.f / 1080.f * 2.f;//-80.f / 1080.f * 2.f;
    const float sliderTrackShiftY = UImulY * 27.f / 1080.f * 2.f;//-80.f / 1080.f * 2.f;
    const float sliderNotchSizeY = 7.f / 1080.f * 2.f;
    const float sliderContTrackShiftY = sliderThumbSizeY * 0.5f + sliderNotchSizeY;//-80.f / 1080.f * 2.f;
    const float sliderTrackSizeY = UImulY * 2.f / 1080.f * 2.f;

    const float checkboxShiftY = UImulY * 20.f / 1080.f * 2.f;

    // Anchor layout values
#if 0
    float iconLogoSizeX = UImulX * 180.f / 1920.f*2.f;
    float iconLogoSizeY = UImulY * 37.f / 1080.f*2.f;
#else
    float iconLogoSizeX = UImulX * 181.f / 1920.f*2.f;
    float iconLogoSizeY = UImulY * 38.f / 1080.f*2.f;
#endif
    //const float iconSizeX = UImulX * 64.f/1920.f*2.f;
    //const float iconSizeY = UImulY * 64.f/1080.f*2.f;

    const float iconLogoX = sliderStartX + 0.5f * sliderSizeX - 0.5f * iconLogoSizeX;

    const float iconSpacingY = UImulY * -20.f / 1080.f * 2.f;
    const float spacingY = UImulY * 15.f / 1080.f * 2.f;//10.f / 1080.f * 2.f;

    const float iconTopY = spacingY;//1027.f / 1080.f*2.f;// - 1.f;
    const float iconBottomY = UImulY * 8.f / 1080.f*2.f;// - 1.f;

    const float panelsDistanceMul = 1.5f;

    const float toggleButtonSizeX = UImulX * 9.f / 1920.f*2.f;

    // This is a pixel size in the design space (1920x1080)
    // allows to map design space onto normalized space
    float onePixelXDesign = 1.0f / 1920.f * 2.f * UImulX;
    float onePixelYDesign = 1.0f / 1080.f * 2.f * UImulY;

    // This is size of the pixel in real buffer that we're dealing with
    float onePixelXReal = onePixelXDesign;
    float onePixelYReal = onePixelYDesign;
    if (m_width != 0.0f && m_height != 0.0f)
    {
        onePixelXReal = 1.0f / (float)m_width * 2.f;
        onePixelYReal = 1.0f / (float)m_height * 2.f;
    }

    float scrollRegionRenderWidth = UImulX * 5.0f * onePixelXDesign;
    float scrollRegionWidth = UImulX * 5.0f * onePixelXDesign;
    if (scrollRegionWidth < 2.f * onePixelXReal)
        scrollRegionWidth = 2.f * onePixelXReal;


    m_globalScrollMargin = UImulY * 2.f / 1080.f * 2.f;

    mainContainer.posX = -1.0f;
    mainContainer.posY = -1.0f;
    mainContainer.absPosX = -1.0f;
    mainContainer.absPosY = -1.0f;
    mainContainer.sizeX = 2.0f;
    mainContainer.sizeY = 2.0f;

    m_components.leftPane->posX = 0.0f;
    m_components.leftPane->posY = 0.0f;
    m_components.leftPane->sizeX = leftOffset;
    m_components.leftPane->sizeY = 2.0f;

    const float flyoutPaneRenderingMarginX = UImulX * 0.f / 1920.f * 2.f;
    const float flyoutPaneRenderingMarginY = UImulY * 5.f / 1080.f * 2.f;
    const float flyoutPaneRenderingOffsetX = UImulX * 1.f / 1920.f * 2.f;

    m_components.flyoutPane->m_isScrollable = true;
    m_components.flyoutPane->m_anchorY = nullptr;
    m_components.flyoutPane->posX = leftOffset + flyoutPaneRenderingOffsetX + flyoutPaneRenderingMarginX;
//  m_components.flyoutPane->posY = 0.0f;
    m_components.flyoutPane->sizeX = leftOffset;
    m_components.flyoutPane->sizeY = 2*m_components.flyoutPane->m_scrollMarginBottom;
    m_components.flyoutPane->sizeYMax = 0.5f;
    m_components.flyoutPane->m_renderingMarginX = flyoutPaneRenderingMarginX;
    m_components.flyoutPane->m_renderingMarginY = flyoutPaneRenderingMarginY;

#ifdef ENABLE_STYLETRANSFER
    const float dlgRestyleSizeX = UImulX * 600.f / 1920.f * 2.f;
    const float dlgRestyleSizeY = UImulY * 275.f / 1080.f * 2.f;
    m_components.dlgDownloadRestyle->m_isScrollable = true;
    m_components.dlgDownloadRestyle->m_anchorY = nullptr;
    m_components.dlgDownloadRestyle->posX = 1.0f - dlgRestyleSizeX * 0.5f;
    m_components.dlgDownloadRestyle->posY = 1.0f - dlgRestyleSizeY * 0.5f;
    m_components.dlgDownloadRestyle->sizeX = dlgRestyleSizeX;
    m_components.dlgDownloadRestyle->sizeY = dlgRestyleSizeY;
#endif

    m_components.cntControls->m_isScrollable = true;
    m_components.cntControls->m_anchorY = m_components.icoLWLogo;
    m_components.cntControls->posX = 0.0f;
    m_components.cntControls->posY = spacingY;
    m_components.cntControls->sizeX = m_components.leftPane->sizeX;
    m_components.cntControls->sizeY = 2.0f - (iconLogoSizeY + iconTopY) - (2*btnSizeY + 3*spacingY);//13 * (sliderShiftY + spacingY);//6 * (sliderShiftY + spacingY) + spacingY + btnSizeY * 0.5f;

    // This one should go first
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    const float hotkeySizeY = UImulY * 40.f / 1080.f*2.f;

    m_components.btnHotkeySetup->m_anchorY = nullptr;
    float aspectDesign = 1920.0f / 1080.0f;
    const float onePixelXDesign = 1.0f / 1920.0f * 2.0f;
    const float onePixelYDesign = 1.0f / 1080.0f * 2.0f;
    // We need to take into account both button highlighting rectangle,
    // as well as the main container's scrollbar too, hence -5px width
    m_components.btnHotkeySetup->sizeX = btnSizeY / aspectDesign - 5*onePixelXDesign;
    m_components.btnHotkeySetup->sizeY = hotkeySizeY - 2*onePixelYDesign;
    // Scrollbar + highlighting rect will give total -4px x
    m_components.btnHotkeySetup->posX = m_components.leftPane->sizeX - m_components.btnHotkeySetup->sizeX - 3*onePixelYDesign;
    m_components.btnHotkeySetup->posY = onePixelYDesign;

    m_components.lblHotkey->m_anchorY = nullptr;
    m_components.lblHotkey->sizeX = m_components.btnHotkeySetup->posX;
    m_components.lblHotkey->sizeY = hotkeySizeY;//UImulY * 20.f / 1080.f*2.f;
    m_components.lblHotkey->posX = 0.0f;
    m_components.lblHotkey->posY = 0.0f;

    m_components.btnToggleFilter->m_anchorY = m_components.lblHotkey;
#else
    m_components.btnToggleFilter->m_anchorY = nullptr;
#endif
    m_components.btnToggleFilter->posY = spacingY;

    m_components.cntFilter->m_anchorY = m_components.btnToggleFilter;
    m_components.cntFilter->posX = 0.0f;
    m_components.cntFilter->posY = 0.0f;
    m_components.cntFilter->sizeX = m_components.cntControls->sizeX;
    m_components.cntFilter->sizeY = 0.0f;//5 * (sliderShiftY + spacingY) + 10.f / 1080.f * 2.f;

    m_components.btnToggleAdjustments->m_anchorY = m_components.cntFilter;

    m_components.cntAdjustments->m_anchorY = m_components.btnToggleAdjustments;
    m_components.cntAdjustments->posX = 0.0f;
    m_components.cntAdjustments->posY = 0.0f;
    m_components.cntAdjustments->sizeX = m_components.cntControls->sizeX;
    m_components.cntAdjustments->sizeY = 0.0f;//2 * (sliderShiftY + spacingY);

    m_components.btnToggleFX->m_anchorY = m_components.cntAdjustments;

    m_components.cntFX->m_anchorY = m_components.btnToggleFX;
    m_components.cntFX->posX = 0.0f;
    m_components.cntFX->posY = 0.0f;
    m_components.cntFX->sizeX = m_components.cntControls->sizeX;
    m_components.cntFX->sizeY = 0.0f;//2 * (sliderShiftY + spacingY);

#if (DBG_STACKING_PROTO == 1)
    if (m_allowDynamicFilterStacking)
    {
        m_components.btnAddFilter->posX = UImulX * 15.f / 1920.f*2.f;
        m_components.btnAddFilter->posY = 0.5f * spacingY;
        m_components.btnAddFilter->sizeX = UImulX * 170.f / 1920.f * 2.f;
        m_components.btnAddFilter->sizeY = btnSizeY * 0.5f;

#ifdef ENABLE_STYLETRANSFER
        m_components.btnToggleStyleTransfer->m_anchorY = m_components.btnAddFilter;
#else
        m_components.btnToggleGameSpecific->m_anchorY = m_components.btnAddFilter;
#endif
        for (size_t dci = 0, dciEnd = m_components.m_dynamicFilterContainers.size(); dci < dciEnd; ++dci)
        {
            ui::ControlDynamicFilterContainer * lwrDynamicContainer = m_components.m_dynamicFilterContainers[dci];

            lwrDynamicContainer->m_renderSideVLineMargin = toggleButtonSizeX;
            lwrDynamicContainer->m_scrollRegionRenderWidth = scrollRegionRenderWidth;
            lwrDynamicContainer->m_scrollRegionWidth = scrollRegionWidth;

            lwrDynamicContainer->posX = 0.0f;
            lwrDynamicContainer->posY = 0.0f;

            lwrDynamicContainer->sizeX = m_components.cntControls->sizeX;
            lwrDynamicContainer->sizeY = 0.0f;

            // dynFilterButtonsShiftX == toggleButtonSizeX
            const float dynFilterButtonsToggleLwtX = UImulX * 60.f / 1920.f*2.f;
            const float dynFilterButtonsShiftX = UImulX * 140.f / 1920.f*2.f;
            const float dynFilterButtonsRemoveShiftX = UImulX * 7.f / 1920.f*2.f;
            const float dynFilterButtonsSizeX = UImulX * 15.f / 1920.f*2.f;
            const float dynFilterButtonsSizeY = btnSizeY * 0.5f;

            // Up button
            lwrDynamicContainer->m_btnUp->posX = dynFilterButtonsShiftX;
            lwrDynamicContainer->m_btnUp->posY = -dynFilterButtonsSizeY;
            lwrDynamicContainer->m_btnUp->sizeX = dynFilterButtonsSizeX;
            lwrDynamicContainer->m_btnUp->sizeY = btnSizeY * 0.5f;

            // Down button
            lwrDynamicContainer->m_btnDown->posX = dynFilterButtonsShiftX + lwrDynamicContainer->m_btnUp->sizeX;
            lwrDynamicContainer->m_btnDown->posY = -dynFilterButtonsSizeY;
            lwrDynamicContainer->m_btnDown->sizeX = dynFilterButtonsSizeX;
            lwrDynamicContainer->m_btnDown->sizeY = btnSizeY * 0.5f;

            // Remove button
            lwrDynamicContainer->m_btnRemove->posX = dynFilterButtonsShiftX + lwrDynamicContainer->m_btnUp->sizeX + lwrDynamicContainer->m_btnDown->sizeX + dynFilterButtonsRemoveShiftX;
            lwrDynamicContainer->m_btnRemove->posY = -dynFilterButtonsSizeY;
            lwrDynamicContainer->m_btnRemove->sizeX = dynFilterButtonsSizeX;
            lwrDynamicContainer->m_btnRemove->sizeY = btnSizeY * 0.5f;

            // Setup button visibility toggle
            lwrDynamicContainer->m_toggleButton->m_glyphMargin = toggleButtonSizeX;
            lwrDynamicContainer->m_toggleButton->posX = 0.0f;
            lwrDynamicContainer->m_toggleButton->posY = panelsDistanceMul * spacingY;
            lwrDynamicContainer->m_toggleButton->sizeX = lwrDynamicContainer->m_toggleButton->m_parent->sizeX - 2.f * lwrDynamicContainer->m_toggleButton->posX - dynFilterButtonsToggleLwtX;
            m_storedSizes.defaultDynamicToggleSizeX = lwrDynamicContainer->m_toggleButton->sizeX;
            lwrDynamicContainer->m_toggleButton->sizeY = btnSizeY * 0.5f;

            // Setup flyout filter toggle
            lwrDynamicContainer->m_filterToggle->posX = UImulX * 20.f / 1920.f*2.f;
            lwrDynamicContainer->m_filterToggle->posY = UImulY * 5.f / 1080.f*2.f;
            lwrDynamicContainer->m_filterToggle->sizeX = btnSizeX;
            lwrDynamicContainer->m_filterToggle->sizeY = btnSizeY;
        }
    }
    else
    {
#ifdef ENABLE_STYLETRANSFER
        m_components.btnToggleStyleTransfer->m_anchorY = m_components.cntFX;
#else
        m_components.btnToggleGameSpecific->m_anchorY = m_components.cntFX;
#endif
    }
#else
#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer->m_anchorY = m_components.cntFX;
#else
    m_components.btnToggleGameSpecific->m_anchorY = m_components.cntFX;
#endif
#endif

#ifdef ENABLE_STYLETRANSFER
    m_components.cntStyleTransfer->m_anchorY = m_components.btnToggleStyleTransfer;
    m_components.cntStyleTransfer->posX = 0.0f;
    m_components.cntStyleTransfer->posY = 0.0f;
    m_components.cntStyleTransfer->sizeX = m_components.cntControls->sizeX;
    m_components.cntStyleTransfer->sizeY = 0.0f;//2 * (sliderShiftY + spacingY);

    m_components.btnToggleGameSpecific->m_anchorY = m_components.cntStyleTransfer;
#endif
    m_components.cntGameSpecific->m_anchorY = m_components.btnToggleGameSpecific;
    m_components.cntGameSpecific->posX = 0.0f;
    m_components.cntGameSpecific->posY = 0.0f;
    m_components.cntGameSpecific->sizeX = m_components.cntControls->sizeX;
    m_components.cntGameSpecific->sizeY = 0.0f;//2 * (sliderShiftY + spacingY);

    m_components.btnToggleCamCapture->m_anchorY = m_components.cntGameSpecific;

    m_components.cntCameraCapture->m_anchorY = m_components.btnToggleCamCapture;
    m_components.cntCameraCapture->posX = 0.0f;
    m_components.cntCameraCapture->posY = 0.0f;
    m_components.cntCameraCapture->sizeX = m_components.cntControls->sizeX;
    m_components.cntCameraCapture->sizeY = 0.0f;//2 * (sliderShiftY + spacingY) + 2 * (sliderContShiftY + spacingY) + spacingY * 2 + checkboxShiftY + 0.5f * spacingY;  // additional spacing since last slider can wrap text

    // Camera & Capture - posY
    m_components.sldFOV->posY = spacingY;
    m_components.sldRoll->posY = spacingY;
    m_components.sldKind->posY = spacingY;
    m_components.chkGridOfThirds->posY = spacingY;
    m_components.chkHDR->posY = spacingY;
    m_components.sldHiResMult->posY = spacingY;
    m_components.sldSphereFOV->posY = spacingY;
    m_components.chkEnhanceHiRes->posY = spacingY;

#ifdef ENABLE_STYLETRANSFER
    // Style transfer - posY
    m_components.chkEnableStyleTransfer->posY = spacingY;
#endif

    // Camera & Capture - anchorY
    m_components.sldFOV->m_anchorY = nullptr;
    m_components.sldRoll->m_anchorY = m_components.sldFOV;

    m_components.btnResetRoll->m_anchorY = m_components.sldRoll;
    m_components.btnResetRoll->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.btnResetRoll->posY = UImulY * 5.f / 1080.f*2.f;
    m_components.btnResetRoll->sizeX = btnSizeX;
    m_components.btnResetRoll->sizeY = UImulY * 30.f / 1080.f*2.f;

    m_components.chkGridOfThirds->m_anchorY = m_components.btnResetRoll;
    m_components.chkHDR->m_anchorY = m_components.chkGridOfThirds;
    m_components.sldKind->m_anchorY = m_components.chkHDR;
    m_components.sldHiResMult->m_anchorY = m_components.sldKind;
    m_components.sldSphereFOV->m_anchorY = m_components.sldKind;
    m_components.chkEnhanceHiRes->m_anchorY = m_components.sldHiResMult;

#ifdef ENABLE_STYLETRANSFER
    // Style transfer - anchorY
    m_components.chkEnableStyleTransfer->m_anchorY = nullptr;

    m_components.flyStyles->m_anchorY = m_components.cntRestyleProgress;//m_components.chkEnableStyleTransfer;
    m_components.flyStyles->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.flyStyles->posY = UImulY * 5.f / 1080.f*2.f;
    m_components.flyStyles->sizeX = btnSizeX;
    m_components.flyStyles->sizeY = btnSizeY;

    m_components.flyStyleNetworks->m_anchorY = m_components.flyStyles;
    m_components.flyStyleNetworks->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.flyStyleNetworks->posY = UImulY * 5.f / 1080.f*2.f;
    m_components.flyStyleNetworks->sizeX = btnSizeX;
    m_components.flyStyleNetworks->sizeY = btnSizeY;
#endif
    m_components.flySpecialFX->m_anchorY = nullptr;
    m_components.flySpecialFX->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.flySpecialFX->posY = UImulY * 5.f / 1080.f*2.f;
    m_components.flySpecialFX->sizeX = btnSizeX;
    m_components.flySpecialFX->sizeY = flySizeY;

#ifdef ENABLE_STYLETRANSFER
    m_components.cntRestyleProgress->m_anchorY = m_components.chkEnableStyleTransfer;
    m_components.cntRestyleProgress->posX = 0.0f;
    m_components.cntRestyleProgress->posY = 0.0f;
    m_components.cntRestyleProgress->sizeX = m_components.cntControls->sizeX;
    //m_components.cntPB->sizeY = 0.0f + btnSizeY;

    m_components.lblRestyleProgress->m_anchorY = nullptr;
    m_components.lblRestyleProgress->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.lblRestyleProgress->posY = 0.0f;
    m_components.lblRestyleProgress->sizeX = btnSizeX;
    m_components.lblRestyleProgress->sizeY = lblSizeY;

    m_components.pbRestyleProgress->m_anchorY = m_components.lblRestyleProgress;
    m_components.pbRestyleProgress->posX = UImulX * 20.f / 1920.f*2.f;
    m_components.pbRestyleProgress->posY = UImulY * 5.f / 1080.f*2.f;
    m_components.pbRestyleProgress->sizeX = btnSizeX;
    m_components.pbRestyleProgress->sizeY = pbSizeY;

    m_components.lblRestyleProgressIndicator->m_anchorY = m_components.btnToggleStyleTransfer;
    m_components.lblRestyleProgressIndicator->posX = UImulX * 175.f / 1920.f*2.f;
    m_components.lblRestyleProgressIndicator->posY = -lblSizeY;
    m_components.lblRestyleProgressIndicator->sizeX = UImulX * 20.f / 1920.f*2.f;
    m_components.lblRestyleProgressIndicator->sizeY = lblSizeY;

    // dlgDownloadRestyle
    m_components.lblDownloadRestyleText->m_anchorY = nullptr;
    m_components.lblDownloadRestyleText->posX = UImulX * 10.f / 1920.f*2.f;
    m_components.lblDownloadRestyleText->posY = UImulY * 10.f / 1080.f*2.f;
    m_components.lblDownloadRestyleText->sizeX = UImulX * 580.f / 1920.f*2.f;
    m_components.lblDownloadRestyleText->sizeY = UImulY * 200.f / 1080.f*2.f;

    m_components.btnDownloadRestyleConfirm->m_anchorY = nullptr;
    m_components.btnDownloadRestyleConfirm->posX = UImulX *  50.f / 1920.f*2.f;
    m_components.btnDownloadRestyleConfirm->posY = UImulY * 200.f / 1080.f*2.f;
    m_components.btnDownloadRestyleConfirm->sizeX = btnSizeX;
    m_components.btnDownloadRestyleConfirm->sizeY = btnSizeY;

    m_components.btnDownloadRestyleCancel->m_anchorY = nullptr;
    m_components.btnDownloadRestyleCancel->posX = UImulX * 550.f / 1920.f*2.f - btnSizeX;
    m_components.btnDownloadRestyleCancel->posY = UImulY * 200.f / 1080.f*2.f;
    m_components.btnDownloadRestyleCancel->sizeX = btnSizeX;
    m_components.btnDownloadRestyleCancel->sizeY = btnSizeY;
#endif
    if (m_isModdingAllowed)
    {
        m_components.chkAllowModding->m_anchorY = m_components.cntControls;
        m_components.chkAllowModding->posY = 0;

        m_components.btnSnap->m_anchorY = m_components.chkAllowModding;

        // "Snap"
        m_components.btnSnap->posX = UImulX * 20.f / 1920.f*2.f;
        m_components.btnSnap->posY = UImulY * 5.f / 1080.f*2.f;

        // "Done"
        m_components.btnDone->posX = UImulX * 20.f / 1920.f*2.f;
        m_components.btnDone->posY = UImulY * 10.f / 1080.f*2.f;

        const float btnSizeYnew = UImulY * 35.f / 1080.f*2.f;
        m_components.btnSnap->sizeX = btnSizeX;
        m_components.btnSnap->sizeY = btnSizeYnew;

        m_components.btnDone->sizeX = btnSizeX;
        m_components.btnDone->sizeY = btnSizeYnew;
    }
    else
    {
        m_components.btnSnap->m_anchorY = m_components.cntControls;

        // "Snap"
        m_components.btnSnap->posX = UImulX * 20.f / 1920.f*2.f;
        m_components.btnSnap->posY = UImulY * 0.0f;//76.f / 1080.f*2.f + sliderShiftY * 0.5f;

        // "Done"
        m_components.btnDone->posX = UImulX * 20.f / 1920.f*2.f;
        m_components.btnDone->posY = UImulY * 20.f / 1080.f*2.f;

        m_components.btnSnap->sizeX = btnSizeX;
        m_components.btnSnap->sizeY = btnSizeY;

        m_components.btnDone->sizeX = btnSizeX;
        m_components.btnDone->sizeY = btnSizeY;
    }

    m_components.btnDone->m_anchorY = m_components.btnSnap;

    // Buttons
    m_components.cntFilter->m_renderSideVLineMargin = toggleButtonSizeX;
    m_components.cntAdjustments->m_renderSideVLineMargin = toggleButtonSizeX;
    m_components.cntFX->m_renderSideVLineMargin = toggleButtonSizeX;
#ifdef ENABLE_STYLETRANSFER
    m_components.cntStyleTransfer->m_renderSideVLineMargin = toggleButtonSizeX;
#endif
    m_components.cntGameSpecific->m_renderSideVLineMargin = toggleButtonSizeX;
    m_components.cntCameraCapture->m_renderSideVLineMargin = toggleButtonSizeX;

    // "Filters"
    m_components.btnToggleFilter->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleFilter->posX = 0.0f;//toggleButtonSizeX;
    m_components.btnToggleFilter->posY = spacingY;
    m_components.btnToggleFilter->sizeX = m_components.btnToggleFilter->m_parent->sizeX - 2.f * m_components.btnToggleFilter->posX;
    m_components.btnToggleFilter->sizeY = btnSizeY * 0.5f;

    // "Adjustments"
    m_components.btnToggleAdjustments->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleAdjustments->posX = 0.0f;//toggleButtonSizeX;
    m_components.btnToggleAdjustments->posY = panelsDistanceMul * spacingY;
    m_components.btnToggleAdjustments->sizeX = m_components.btnToggleAdjustments->m_parent->sizeX - 2.f * m_components.btnToggleAdjustments->posX;
    m_components.btnToggleAdjustments->sizeY = btnSizeY * 0.5f;

    // "FX"
    m_components.btnToggleFX->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleFX->posX = 0.0f;//toggleButtonSizeX;
    m_components.btnToggleFX->posY = panelsDistanceMul * spacingY;
    m_components.btnToggleFX->sizeX = m_components.btnToggleFX->m_parent->sizeX - 2.f * m_components.btnToggleFX->posX;
    m_components.btnToggleFX->sizeY = btnSizeY * 0.5f;

    // "Style transfer"
#ifdef ENABLE_STYLETRANSFER
    m_components.btnToggleStyleTransfer->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleStyleTransfer->posX = 0.0f;//toggleButtonSizeX;
    m_components.btnToggleStyleTransfer->posY = panelsDistanceMul * spacingY;
    m_components.btnToggleStyleTransfer->sizeX = m_components.btnToggleStyleTransfer->m_parent->sizeX - 2.f * m_components.btnToggleStyleTransfer->posX;
    m_components.btnToggleStyleTransfer->sizeY = btnSizeY * 0.5f;
#endif
    // "Game specific"
    m_components.btnToggleGameSpecific->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleGameSpecific->posX = 0.0f;//toggleButtonSizeX;
    if (isGameSettingsPanelDisplayed)
    {
        m_components.btnToggleGameSpecific->posY = panelsDistanceMul * spacingY;
    }
    else
    {
        m_components.btnToggleGameSpecific->posY = 0.0f;
    }
    m_components.btnToggleGameSpecific->sizeX = m_components.btnToggleFX->m_parent->sizeX - 2.f * m_components.btnToggleFX->posX;
    m_components.btnToggleGameSpecific->sizeY = btnSizeY * 0.5f;

    // "Camera & Capture"
    m_components.btnToggleCamCapture->m_glyphMargin = toggleButtonSizeX;
    m_components.btnToggleCamCapture->posX = 0.0f;//toggleButtonSizeX;
    m_components.btnToggleCamCapture->posY = panelsDistanceMul * spacingY;
    m_components.btnToggleCamCapture->sizeX = m_components.btnToggleCamCapture->m_parent->sizeX - 2.f * m_components.btnToggleCamCapture->posX;
    m_components.btnToggleCamCapture->sizeY = btnSizeY * 0.5f;

    m_components.icoLWLogo->posX = iconLogoX;
    m_components.icoLWLogo->posY = iconTopY;
    m_components.icoLWLogo->sizeX = iconLogoSizeX;
    m_components.icoLWLogo->sizeY = iconLogoSizeY;

    m_components.icoLWLogo->maxPixelWidth = 180;
    m_components.icoLWLogo->maxPixelHeight = 38;

    m_components.icoLWLogo->storeBaseParameters();

    auto setContainerScrollRegion = [](ui::ControlContainer * container, float srWidth, float srRenderWidth)
    {
        container->m_scrollRegionWidth = srWidth;
        container->m_scrollRegionRenderWidth = srRenderWidth;
    };

    // Containers scrolling parameters
    setContainerScrollRegion(m_components.flyoutPane,    scrollRegionWidth, scrollRegionRenderWidth);
    m_components.flyoutPane->m_scrollMarginBottom = selectorMarginY;

    setContainerScrollRegion(m_components.cntControls,    scrollRegionWidth, scrollRegionRenderWidth);
    setContainerScrollRegion(m_components.cntFilter,    scrollRegionWidth, scrollRegionRenderWidth);
    setContainerScrollRegion(m_components.cntAdjustments,  scrollRegionWidth, scrollRegionRenderWidth);
    setContainerScrollRegion(m_components.cntFX,      scrollRegionWidth, scrollRegionRenderWidth);
#ifdef ENABLE_STYLETRANSFER
    setContainerScrollRegion(m_components.cntStyleTransfer,  scrollRegionWidth, scrollRegionRenderWidth);
#endif
    setContainerScrollRegion(m_components.cntGameSpecific,  scrollRegionWidth, scrollRegionRenderWidth);
    setContainerScrollRegion(m_components.cntCameraCapture,  scrollRegionWidth, scrollRegionRenderWidth);

    ui::ControlContainer * lwrContainer;

    // Additional pass to auto-setup sliders sizes
    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        lwrContainer->m_autoPlacementElement = nullptr;

        if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
            lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
            lwrContainer->getType() == ui::ControlType::kSliderInt)
        {
            ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

            if (lwrSliderBase->isLeanStyle)
            {
                lwrSliderBase->sizeX = sliderSizeX;
                lwrSliderBase->sizeY = sliderContShiftY;
                lwrSliderBase->trackShiftY = sliderContTrackShiftY;
            }
            else
            {
                lwrSliderBase->sizeX = sliderSizeX;
                lwrSliderBase->sizeY = sliderShiftY;
                lwrSliderBase->trackShiftY = sliderTrackShiftY;
            }
        }
        else if (lwrContainer->getType() == ui::ControlType::kButton)
        {
            ui::ControlButton * lwrButton = static_cast<ui::ControlButton *>(lwrContainer);
            if (!lwrButton->needsAutosize)
                continue;

            if (lwrButton->renderType == ui::ControlButton::RenderType::kSelector)
            {
                lwrButton->posX = selectorStartX;
                lwrButton->posY = 0.0f; // Will be recallwlated in the dynamic container recalc loop
                lwrButton->sizeX = selectorSizeX;
                lwrButton->sizeY = selectorSizeY;
            }
            else
            {
                lwrButton->posX = sliderStartX;
                lwrButton->posY = spacingY;
                lwrButton->sizeX = btnSizeX;
                lwrButton->sizeY = btnSizeY;
            }
        }
    }
    containerHelper.stopSearch();

    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->getType() == ui::ControlType::kColorPicker)
        {
            ui::ControlColorPickerEffectTweak * lwrColorPicker = static_cast<ui::ControlColorPickerEffectTweak *>(lwrContainer);

            lwrColorPicker->setSizes(sliderSizeX, sliderShiftY);
        }
    }
    containerHelper.stopSearch();

    const float resolutionSlidersSecondLineSize = UImulY * 20.f / 1080.f * 2.f;
    m_components.sldHiResMult->sizeY = sliderShiftY + resolutionSlidersSecondLineSize;
    m_components.sldSphereFOV->sizeY = sliderShiftY + resolutionSlidersSecondLineSize;
    m_components.sldHiResMult->trackShiftY += resolutionSlidersSecondLineSize;
    m_components.sldSphereFOV->trackShiftY += resolutionSlidersSecondLineSize;

    // TODO avoroshilov stacking:
    //  fix this
    ui::ControlContainer * prevFilterControlFilter = nullptr;
    ui::ControlContainer * prevFilterControlAdjustments = nullptr;
    ui::ControlContainer * prevFilterControlFX = nullptr;
    ui::ControlContainer * prevFilterControlGameSpecific = nullptr;

    ui::ControlContainer * prevFlyoutSelector = nullptr;

    containerHelper.startSearch();
    while (lwrContainer = containerHelper.getNextControl((int)ui::ControlType::kALL, false))
    {
        if (lwrContainer->getType() == ui::ControlType::kSliderCont ||
            lwrContainer->getType() == ui::ControlType::kSliderDiscr ||
            lwrContainer->getType() == ui::ControlType::kSliderInt)
        {
            ui::ControlSliderBase * lwrSliderBase = static_cast<ui::ControlSliderBase *>(lwrContainer);

            lwrSliderBase->posX = sliderStartX;

            lwrSliderBase->thumbSizeX = sliderThumbSizeX;
            lwrSliderBase->thumbSizeY = sliderThumbSizeY;
            lwrSliderBase->thumbBorderX = thumbBorderX;
            lwrSliderBase->thumbBorderY = thumbBorderY;
            lwrSliderBase->trackSizeY = sliderTrackSizeY;

            // We don't want the internal control sliders to have additional offset
            if (lwrContainer->m_parent && (lwrContainer->m_parent->getType() != ui::ControlType::kContainer))
            {
                lwrSliderBase->posX = 0.0f;
            }
        }
        else if (lwrContainer->getType() == ui::ControlType::kCheckbox)
        {
            ui::ControlCheckbox * lwrCheckbox = static_cast<ui::ControlCheckbox *>(lwrContainer);

            lwrCheckbox->posX = sliderStartX;

            lwrCheckbox->sizeX = sliderSizeX;
            lwrCheckbox->sizeY = checkboxShiftY;

            lwrCheckbox->checkSize = checkboxShiftY * 0.7f;
        }
        else if (lwrContainer->getType() == ui::ControlType::kColorPicker)
        {
            ui::ControlColorPickerEffectTweak * lwrColorPicker = static_cast<ui::ControlColorPickerEffectTweak *>(lwrContainer);

            lwrColorPicker->posX = sliderStartX;
            //lwrColorPicker->posY = 0.0f;
        }

        bool contentsAutoPlacement = false;
        if (lwrContainer->m_parent && lwrContainer->m_parent->m_needsContentAutoPlacement)
        {
            contentsAutoPlacement = true;
        }

        // Elements that belong to these three containers shoud be chained together, and the layout needs to be recallwlated
        if (contentsAutoPlacement)
        {
            lwrContainer->posY = spacingY;

            if (lwrContainer->m_parent)
            {
                if (lwrContainer->isVisible)
                {
                    lwrContainer->m_parent->sizeY += lwrContainer->sizeY + spacingY;
                }

                lwrContainer->m_anchorY = lwrContainer->m_parent->m_autoPlacementElement;
                lwrContainer->m_parent->m_autoPlacementElement = lwrContainer;
            }
        }
        else if (lwrContainer->m_parent == m_components.cntCameraCapture)
        {
            if (lwrContainer->isVisible)
            {
                float containerRelY = 0.0f;
                ui::ControlContainer * anchorY = lwrContainer;
                while (anchorY)
                {
                    if (anchorY->isVisible)
                        containerRelY += anchorY->posY + anchorY->sizeY;
                    anchorY = anchorY->m_anchorY;
                }
                lwrContainer->m_parent->sizeY = std::max(lwrContainer->m_parent->sizeY, containerRelY);
            }
        }
#ifdef ENABLE_STYLETRANSFER
        else if (lwrContainer->m_parent == m_components.cntStyleTransfer)
        {
            if (lwrContainer->isVisible)
            {
                float containerRelY = 0.0f;
                ui::ControlContainer * anchorY = lwrContainer;
                while (anchorY)
                {
                    containerRelY += anchorY->posY + anchorY->sizeY;
                    anchorY = anchorY->m_anchorY;
                }
                lwrContainer->m_parent->sizeY = std::max(lwrContainer->m_parent->sizeY, containerRelY);
            }
        }
#endif
        else if (lwrContainer->m_parent == m_components.flyoutPane)
        {
            if (prevFlyoutSelector)
                lwrContainer->posY = selectorSpacingY;
            else
                lwrContainer->posY = selectorMarginY;

            if (lwrContainer->isVisible)
            {
                lwrContainer->m_parent->sizeY += lwrContainer->sizeY + selectorSpacingY;

                if (lwrContainer->m_parent->sizeY > lwrContainer->m_parent->sizeYMax)
                    lwrContainer->m_parent->sizeY = lwrContainer->m_parent->sizeYMax;
            }

            lwrContainer->m_anchorY = prevFlyoutSelector;
            prevFlyoutSelector = lwrContainer;
        }
    }
    containerHelper.stopSearch();

    // Special code for the sliding progress bar container
#ifdef ENABLE_STYLETRANSFER
    m_components.stylePBContainerHeight = 0.0f;
    for (size_t ci = 0, ciend = m_components.cntRestyleProgress->getControlsNum(); ci < ciend; ++ci)
    {
        ui::ControlContainer * childControl = m_components.cntRestyleProgress->getControl((int)ci);
        if (childControl->isVisible)
        {
            m_components.stylePBContainerHeight += childControl->sizeY + childControl->posY;
        }
    }
#endif
    recallwlateUIPositions();

    return S_OK;
}

void AnselUI::getTelemetryData(UISpecificTelemetryData &ret) const
{
    if (m_components.sldRoll)// should be like getRollDegrees(), but with safe checks
    {
        ret.roll = (float)(m_components.sldRoll->percentage - 0.5f) * 360.0f;
    }

    if (m_components.sldFOV)// should be like getFOVDegrees(), but with safe checks
    {
        ret.fov = (float) static_cast<ui::ControlSliderFOV *>(m_components.sldFOV)->getFOV();
    }

    ret.kindOfShot = m_shotToTake;
    ret.isShotHDR = m_shotHDREnabled;

    if (m_components.sldHiResMult)
        ret.highresMult = m_components.sldHiResMult->getSelected() + 2;

    if (m_components.sldSphereFOV)
    {
        ret.resolution360 = static_cast<ui::ControlSlider360Quality *>(m_components.sldSphereFOV)->getResolution();
    }
    
    ret.usedGamepadForUIDuringTheSession = m_usedGamepadForUIDuringTheSession;
}

float AnselUI::getMessageDurationByType(MessageType msgType)
{
    if (msgType == MessageType::kShotWithUISaved)
        return 10.0f;
    else if (msgType == MessageType::kUnableToSaveShotWithUI)
        return 10.0f;

    return 5.0f;
}
void AnselUI::displayMessageInternal(float duration, const std::wstring & msgText)
{
    // Message should be split according to max length of a single string and oclwrence of linebreaks
    std::vector<std::wstring> msgTextParts;

    const size_t msgMaxSize = 512;
    const size_t msgLength = msgText.length();
    size_t msgPointer = 0;
    while (msgPointer < msgLength)
    {
        const auto start = msgText.cbegin() + msgPointer;
        size_t len = std::min(msgLength - msgPointer, msgMaxSize);
        
        size_t linebreakPos = msgText.find(L'\n', msgPointer);
        if (linebreakPos != std::wstring::npos && linebreakPos - msgPointer < len)
        {
            len = linebreakPos - msgPointer;
            msgPointer += len+1;  // we would want to skip '\n'
        }
        else
        {
            msgPointer += len;
        }

        if (len > 0)
            msgTextParts.push_back(std::wstring(start, start + len));
    }

    // Error reporter is a stack (new messages on top), so split messages need to be reversed
    for (int msgIdx = (int)msgTextParts.size()-1; msgIdx >= 0; --msgIdx)
        m_errorManager->addError(duration, msgTextParts[msgIdx]);
}
void AnselUI::displayMessage(MessageType msgType)
{
    displayMessageInternal(getMessageDurationByType(msgType), buildDisplayMessage(msgType));
}
void AnselUI::displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                        , bool removeLastLine
#endif
)
{
    displayMessageInternal(getMessageDurationByType(msgType), buildDisplayMessage(msgType, parameters));
}

void AnselUI::reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, ...) const
{
    va_list args;
    va_start(args, format);


    if(m_pAnselServer)
        m_pAnselServer->reportFatalError(filename, lineNumber, code, format, args);

    va_end(args);
}

void AnselUI::reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, ...) const
{
    va_list args;
    va_start(args, format);

    if (m_pAnselServer)
        m_pAnselServer->reportNonFatalError(filename, lineNumber, code, format, args);

    va_end(args);
}



