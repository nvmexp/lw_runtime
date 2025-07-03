#pragma once

#include "Config.h"
#if (UI_ENABLE_TEXT == 1)
#include "ui/fw1/FW1FontWrapper.h"
#endif
#include "OverlayDetection.h"
#include "UIBase.h"
#include "darkroom/Director.h"
#include "CommonStructs.h"
#include "AnselInput.h"
#include "ui/classes.h"
#include "ui/elements.h"
#include "ir/SpecializedPool.h"
#include "AnselSDKState.h"

#include <d3d11.h>
#include <vector>

class AnselServer;

#define DISABLE_STATIC_FILTERS      1

#define DBG_USE_OUTLINE             0
#define DBG_ENABLE_HOTKEY_SETUP     0

#define DBG_STACKING_PROTO          1

#if (DISABLE_STATIC_FILTERS == 1)
#  define STATIC_BLOCKID            0
#  define TEMP_DYNAMIC_BLOCKID      1
#  define ADJUSTMENTS_BLOCKID       -1
#  define FX_BLOCKID                -1
#  define GAMESPEC_BLOCKID          2
#  define DYNAMIC_FILTERS_BLOCKID   3
#else
#  define STATIC_BLOCKID            0
#  define TEMP_DYNAMIC_BLOCKID      1
#  define ADJUSTMENTS_BLOCKID       2
#  define FX_BLOCKID                3
#  define GAMESPEC_BLOCKID          4
#  define DYNAMIC_FILTERS_BLOCKID   5
#endif

#define TABSTOP_INIT  1000
#define TABSTOP_STRIDE  10

struct UIDebugPrintInfo
{
    float dt;
    int shotCaptureLatency;
    int shotSettleLatency;
    bool renderDebugInfo;
    uint32_t networkBytesTransferred = 0u;

    std::vector<std::wstring> additionalLines;

    struct GamepadDebugInfo
    {
        short lx, ly, rx, ry, z;
        input::EDPadDirection::Enum dpad;
        float fz;
        bool a, b, x, y, lcap, rcap, lshoulder, rshoulder;
    } gamepadDebugInfo;
};

struct UIProgressInfo
{
    bool removeBlackTint;
};

class AnselUI : public input::InputEventsConsumerInterface, public AnselUIBase, public OverlayDetector
{
protected:

    bool m_isEnabled = false;  // This shows if Ansel is active
    bool m_isVisible = false;  // This determines whether UI is visible or not
    bool m_enableAnselSDKUI = false; // True if AnselSDK was detected

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    bool m_selectingHotkey = false;
    bool m_selectingHotkeyShift = false;
    bool m_selectingHotkeyCtrl = false;
    bool m_selectingHotkeyAlt = false;

    bool m_hotkeyModifierShift = false;
    bool m_hotkeyModifierCtrl = false;
    bool m_hotkeyModifierAlt = false;
    USHORT m_hotkeyModifierVKey = 0;
#endif

    bool                                    m_areMouseButtonsSwapped = false;
    input::InputHandlerForStandalone        m_inputstate;
    bool                                    m_usedGamepadForUIDuringTheSession = false; //only use for stats!

    bool                                    m_allowDynamicFilterStacking = false;

public:
    AnselUI() :
        m_onButtonToggleClick(this, &AnselUI::onButtonToggleClick),
        m_onFlyoutClick(this, &AnselUI::onFlyoutClick),
        m_onFlyoutHidePane(this, &AnselUI::onFlyoutHidePane),
        m_onFlyoutSelectorClick(this, &AnselUI::onFlyoutSelectorClick),
        m_onButtonSnapClick(this, &AnselUI::onButtonSnapClick),
        m_onButtonHideClick(this, &AnselUI::onButtonHideClick),
        m_onButtonResetRollClick(this, &AnselUI::onButtonResetRollClick),
#if (DBG_STACKING_PROTO == 1)
        m_onButtonAddFilterClick(this, &AnselUI::onButtonAddFilterClick),
        m_onButtonRemoveFilterClick(this, &AnselUI::onButtonRemoveFilterClick),
        m_onButtonMoveFilterUpClick(this, &AnselUI::onButtonMoveFilterUpClick),
        m_onButtonMoveFilterDownClick(this, &AnselUI::onButtonMoveFilterDownClick),
        m_onFlyoutSpecialFXChangeDynamic(this, &AnselUI::onFlyoutSpecialFXChangeDynamic),
#endif
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
        m_onButtonHotkeySetupClick(this, &AnselUI::onButtonHotkeySetupClick),
#endif
        m_onFlyoutSpecialFXChange(this, &AnselUI::onFlyoutSpecialFXChange),

#ifdef ENABLE_STYLETRANSFER
        m_onFlyoutStylesChange(this, &AnselUI::onFlyoutStylesChange),
        m_onFlyoutStyleNetworksChange(this, &AnselUI::onFlyoutStyleNetworksChange),

        m_onButtonDownloadRestyleConfirmClick(this, &AnselUI::onButtonDownloadRestyleConfirmClick),
        m_onButtonDownloadRestyleCancelClick(this, &AnselUI::onButtonDownloadRestyleCancelClick),
#endif

        m_onSliderKindChange(this, &AnselUI::onSliderKindChange),
        m_onSliderEffectTweakChange(this, &AnselUI::onSliderEffectTweakChange),
        m_onSliderListEffectTweakChange(this, &AnselUI::onSliderListEffectTweakChange),
        m_onSliderIntEffectTweakChange(this, &AnselUI::onSliderIntEffectTweakChange),
        m_onCheckboxEffectTweakChange(this, &AnselUI::onCheckboxEffectTweakChange),
        m_onColorPickerEffectTweakChange(this, &AnselUI::onColorPickerEffectTweakChange)
    {
        m_inputstate.addEventConsumer(this);
    }

    virtual ~AnselUI()
    {
        release();
        m_inputstate.removeEventConsumer(this);
    }

    bool m_makeScreenshot = false;
    bool m_makeScreenshotWithUI = false;

    void setDynamicFilterStackingState(bool allowDynamicFilterStacking)
    {
        m_allowDynamicFilterStacking = allowDynamicFilterStacking;
    }

    class ContainerTraverseHelper
    {
    public:

        bool m_searchInProgress = false;
        bool m_searchHierarchicalInProgress = false;
        void startSearch(size_t selectedControl = 0)
        {
            assert(m_searchInProgress == false);
            m_searchInProgress = true;
            isInitialized = false;
            numControlsSearched = 0;
            baseSelectedControl = selectedControl;
        }
        void stopSearch()
        {
            m_searchInProgress = false;
        }

        void rebuildControlsArray(ui::ControlContainer * baseContainer, size_t selectedControl = 0)
        {
            // TODO avoroshilov UI
            //  perform this once after control insertion/deletion

            linearControls.resize(0);

            ui::ControlContainer * lwrContainer = baseContainer;
            offsetsStack.resize(0);
            offsetsStack.push_back(0);

            while (true)
            {
                if (lwrContainer == nullptr)
                    break;

                int offsetIdx = (int)offsetsStack.size() - 1;
                int offset = offsetsStack.back();

                ui::ControlType DBGcontrolType = lwrContainer->getType();

                const bool emptyContainer = (offset == 0) && (lwrContainer->getControlsNum() == 0) && lwrContainer->isBasicContainer();
                if (emptyContainer)
                {
                    linearControls.push_back(lwrContainer);
                }

                if (offset < int(lwrContainer->getControlsNum()))
                {
                    if (offset == 0)
                    {
                        linearControls.push_back(lwrContainer);
                    }

                    ++offsetsStack.back();
                    if (lwrContainer->isChildBasicContainer(offset))
                    {
                        offsetsStack.push_back(0);
                        lwrContainer = lwrContainer->getControl(offset);
                    }
                    else
                    {
                        linearControls.push_back(lwrContainer->getControl(offset));
                    }
                    continue;
                }

                offsetsStack.pop_back();
                lwrContainer = lwrContainer->m_parent;
            }
        }

        void init(ui::ControlContainer * baseContainer, size_t selectedControl = 0)
        {
            rebuildControlsArray(baseContainer, selectedControl);
            startSearch(selectedControl);
        }

        void startSearchHierarchical(ui::ControlContainer * baseContainer, size_t selectedControl = 0)
        {
            assert(m_searchHierarchicalInProgress == false);
            m_searchHierarchicalInProgress = true;
            baseSelectedControl = selectedControl;
            m_lwrContainerHierarchical = baseContainer;
            offsetsStack.resize(0);
            offsetsStack.push_back(-1);
        }
        void stopSearchHierarchical()
        {
            m_searchHierarchicalInProgress = false;
        }

        ui::ControlContainer * jumpToNextControlHierarchical()
        {
            ui::ControlContainer * lwrContainer = m_lwrContainerHierarchical;
            int offset = offsetsStack.back();

            if (offset == 0)
            {
                // We're looking at parent container, to skip it hierarchically, we can just set offsets
                offsetsStack.back() = (int)lwrContainer->getControlsNum();
            }

            // In case we're on simple control, we don't need to skip anything
            //  since counter was already incremented
            if (offset < int(lwrContainer->getControlsNum()))
            {
                return lwrContainer->getControl(offset);
            }

            return lwrContainer;
        }
        void reportElementDeletedHierarchical()
        {
            --offsetsStack.back();
        }
        void skipChildrenHierarchical()
        {
            offsetsStack.pop_back();
            m_lwrContainerHierarchical = m_lwrContainerHierarchical->m_parent;
        }
        ui::ControlContainer * getNextControlHierarchical()
        {
            while (true)
            {
                if (m_lwrContainerHierarchical == nullptr)
                    return nullptr;

                int offsetIdx = (int)offsetsStack.size() - 1;
                int offset = offsetsStack.back();

                if (offset == -1)
                {
                    // We need to report container as well
                    ++offsetsStack.back();
                    return m_lwrContainerHierarchical;
                }

                if (offset < int(m_lwrContainerHierarchical->getControlsNum()))
                {
                    ++offsetsStack.back();
                    if (m_lwrContainerHierarchical->isChildBasicContainer(offset))
                    {
                        offsetsStack.push_back(-1);
                        m_lwrContainerHierarchical = m_lwrContainerHierarchical->getControl(offset);
                        continue;
                    }
                    else
                    {
                        return m_lwrContainerHierarchical->getControl(offset);
                    }
                }

                offsetsStack.pop_back();
                m_lwrContainerHierarchical = m_lwrContainerHierarchical->m_parent;
            }
            return nullptr;
        }
        template <typename Fn>
        ui::ControlContainer * getNextControlHierarchical(Fn & lambda)
        {
            while (true)
            {
                if (m_lwrContainerHierarchical == nullptr)
                    return nullptr;

                int offsetIdx = (int)offsetsStack.size() - 1;
                int offset = offsetsStack.back();

                if (offset == -1)
                {
                    // We need to report container as well
                    ++offsetsStack.back();
                    return m_lwrContainerHierarchical;
                }

                if (offset < int(m_lwrContainerHierarchical->getControlsNum()))
                {
                    ++offsetsStack.back();
                    if (m_lwrContainerHierarchical->isChildBasicContainer(offset))
                    {
                        offsetsStack.push_back(-1);
                        m_lwrContainerHierarchical = m_lwrContainerHierarchical->getControl(offset);
                        continue;
                    }
                    else
                    {
                        return m_lwrContainerHierarchical->getControl(offset);
                    }
                }

                lambda(m_lwrContainerHierarchical);
                offsetsStack.pop_back();
                m_lwrContainerHierarchical = m_lwrContainerHierarchical->m_parent;
            }
            return nullptr;
        }

        ui::ControlContainer * getNextControl(int requestedTypes = (int)ui::ControlType::kALL, bool shouldBeActive = true)
        {
            // TODO avoroshilov UI
            //  add check for 'baseContainer', this will allow to limit search for a sub-tree
            //  also, add sign to 'numControlsSearched' to allow interleaved search in both directions
            size_t numControlsTotal = linearControls.size();

            if (!isInitialized)
            {
                // We want first invocation of getNextControl to return proper control
                //  either the one passed as 'selectedControl' or the next one that fits requirements
                if (baseSelectedControl > 0)
                {
                    lastSelectedControl = baseSelectedControl - 1;
                }
                else
                {
                    lastSelectedControl = numControlsTotal - 1;
                }
                isInitialized = true;
            }

            while (numControlsSearched < numControlsTotal)
            {
                if (lastSelectedControl < numControlsTotal - 1)
                {
                    ++lastSelectedControl;
                }
                else
                {
                    lastSelectedControl = 0;
                }
                ++numControlsSearched;

                ui::ControlContainer * lwrControl = linearControls[lastSelectedControl];
                if (!shouldBeActive || lwrControl->isInteractive())
                {
                    if ((requestedTypes & (int)lwrControl->getType()) != 0)
                    {
                        return lwrControl;
                    }
                }
            }

            return nullptr;
        }

        ui::ControlContainer * getPrevControl(int requestedTypes = (int)ui::ControlType::kALL, bool shouldBeActive = true)
        {
            size_t numControlsTotal = linearControls.size();

            if (!isInitialized)
            {
                // We want first invocation of getNextControl to return proper control
                //  either the one passed as 'selectedControl' or the next one that fits requirements
                if (baseSelectedControl < numControlsTotal - 1)
                {
                    lastSelectedControl = baseSelectedControl + 1;
                }
                else
                {
                    lastSelectedControl = 0;
                }
                isInitialized = true;
            }

            while (numControlsSearched < numControlsTotal)
            {
                if (lastSelectedControl > 0)
                {
                    --lastSelectedControl;
                }
                else
                {
                    lastSelectedControl = numControlsTotal - 1;
                }
                ++numControlsSearched;

                ui::ControlContainer * lwrControl = linearControls[lastSelectedControl];
                if (!shouldBeActive || lwrControl->isInteractive())
                {
                    if ((requestedTypes & (int)lwrControl->getType()) != 0)
                    {
                        return lwrControl;
                    }
                }
            }

            return nullptr;
        }

        size_t findLinearIndex(ui::ControlContainer * control)
        {
            for (size_t i = 0, iend = linearControls.size(); i < iend; ++i)
            {
                if (linearControls[i] == control)
                    return i;
            }
            return linearControls.size();
        }

        bool checkIfPresentInSelectionChain(ui::ControlContainer * baseContainer, ui::ControlContainer * control)
        {
            ui::ControlContainer * selectedControl = getSelectedControl(baseContainer);
            while (selectedControl)
            {
                if (selectedControl == control)
                    return true;
                selectedControl = selectedControl->m_parent;
            }
            return false;
        }

        ui::ControlContainer * getSelectedControl(ui::ControlContainer * baseContainer)
        {
            // TODO avoroshilov UI
            //  this can be sped up using selected control storage proxy

            ui::ControlContainer * lwrContainer = baseContainer;
            while (lwrContainer->m_selectedControlIdx != -1)
            {
                lwrContainer = static_cast<ui::ControlContainer *>(lwrContainer->getSelectedControl());
            }

            return lwrContainer;
        }

        bool wasControlJustSelected = true;
        ui::ControlContainer * setSelectedControl(ui::ControlContainer * newSelectedControl, ui::ControlContainer * oldSelectedControl)
        {
            // newSelectedControl can legitimately be equal to oldSelectedControl
            //  the reason is that sometimes we need to update selection indices
            //  we just don't want to do anything fancy in that case (selection wasn't really updated)
            if (newSelectedControl != oldSelectedControl)
            {
                wasControlJustSelected = true;
            }

            // Deselect old chain
            ui::ControlContainer * selectedControl = oldSelectedControl;
            while (selectedControl)
            {
                selectedControl->m_selectedControlIdx = -1;
                selectedControl->state = UI_CONTROL_ORDINARY;
                selectedControl = selectedControl->m_parent;
            }

            // Select deepest element
            selectedControl = newSelectedControl;
            selectedControl->m_selectedControlIdx = -1;
            selectedControl->state = UI_CONTROL_HIGHLIGHT;

            // Select the whole new chain
            while (selectedControl)
            {
                ui::ControlContainer * selectedParentControl = selectedControl->m_parent;
                // If parent exists, find index of the selected component
                if (selectedParentControl != nullptr)
                {
                    size_t selCnt = 0;
                    for (size_t selCntEnd = selectedParentControl->getControlsNum(); selCnt < selCntEnd; ++selCnt)
                    {
                        if (selectedParentControl->getControl((int)selCnt) == selectedControl)
                            break;
                    }
                    selectedParentControl->m_selectedControlIdx = (int)selCnt;
                }
                selectedControl->state = UI_CONTROL_HIGHLIGHT;
                selectedControl = selectedParentControl;
            }

            return selectedControl;
        }

        std::vector<int> offsetsStack;

    protected:

        std::vector<ui::ControlContainer *> linearControls;
        size_t numControlsSearched;
        size_t lastSelectedControl, baseSelectedControl;
        bool isInitialized = false;

        // Hierarchical container traversal
        ui::ControlContainer * m_lwrContainerHierarchical = nullptr;
    };
    // Traversing helpers
    std::vector<ui::ControlContainer *> containersStack;
    std::vector<bool> isMouseOverStack;
    std::vector<D3D11_RECT> scissorRectStack;
    ContainerTraverseHelper containerHelper;

    void addFloatingContainer(ui::ControlContainer * container, ui::ControlContainer * containerToSelect = nullptr)
    {
        bool alreadyPresent = false;
        for (int i = 0, iend = (int)floatingContainers.size(); i < iend; ++i)
        {
            if (floatingContainers[i].first == container)
            {
                alreadyPresent = true;
                break;
            }
        }

        if (!alreadyPresent)
        {
            floatingContainers.push_back(std::make_pair(container, containerToSelect));
        }
    }
    void removeFloatingContainerFast(ui::ControlContainer * container)
    {
        for (int i = 0, iend = (int)floatingContainers.size(); i < iend; ++i)
        {
            if (floatingContainers[i].first == container)
            {
                if (i != (int)floatingContainers.size() - 1)
                {
                    std::swap(floatingContainers[i], floatingContainers.back());
                }
                floatingContainers.pop_back();

                return;
            }
        }
    }
    void hideFloatingContainer(ui::ControlContainer * floatingContainer)
    {
        removeFloatingContainerFast(floatingContainer);
        floatingContainer->isVisible = false;
    }
    // 1st - container, 2nd - related element to focus
    std::vector<std::pair<ui::ControlContainer *, ui::ControlContainer *>> floatingContainers;

    float m_globalScrollMargin = 0.0f;
    ui::ControlContainer mainContainer;

    // "Done" button parameters
    ui::ColorF4 m_lwGreenColor;
    ui::ColorF4 m_doneDoneColor;
    ui::ColorF4 m_doneAbortColor, m_doneAbortColorBright;

    std::wstring m_textGB;
    std::wstring m_textFilterNone;
    std::wstring m_textFilterLwstom;
    std::wstring m_textKindRegular;
    std::wstring m_textKindRegularHDR;
    std::wstring m_textKindHighRes;
    std::wstring m_textKind360;
    std::wstring m_textKindStereo;
    std::wstring m_textKind360Stereo;
    std::wstring m_textAbort;
    std::wstring m_textDone;
    std::wstring m_textProgress;

    std::wstring m_textFilter;
    std::wstring m_textFilterType;

#ifdef ENABLE_STYLETRANSFER
    std::wstring m_textStyleNetLow;
    std::wstring m_textStyleNetHigh;
    std::wstring m_textStyleProgress;
    std::wstring m_textStyleInstalling;
#endif

    // Notifications
    std::wstring m_textNotifWelcome;
    std::wstring m_textNotifSessDeclined;
    std::wstring m_textNotifDrvUpdate;

    class ComponentsList
    {
    public:
#ifdef ENABLE_STYLETRANSFER
        ui::ControlButtonToggle * btnToggleStyleTransfer = nullptr;
        ui::ControlFlyoutStylesToggle * flyStyles = nullptr;
        ui::ControlFlyoutStylesToggle * flyStyleNetworks = nullptr;
        ui::ControlContainer * cntStyleTransfer = nullptr;
        // Style transfer progress bar
        float stylePBContainerHeight = 0.0f;
        static const size_t stylePBLabelSize = 64;
        wchar_t stylePBLabel[stylePBLabelSize];
        ui::ControlContainer * cntRestyleProgress = nullptr;
        ui::ControlLabel * lblRestyleProgress = nullptr;
        ui::ControlProgressBar * pbRestyleProgress = nullptr;

        // Style transfer progress indicator
        ui::ControlLabel * lblRestyleProgressIndicator = nullptr;

        ui::ControlCheckbox * chkEnableStyleTransfer = nullptr;
        ui::ControlContainer * dlgDownloadRestyle = nullptr;
        ui::ControlLabel * lblDownloadRestyleText = nullptr;
        ui::ControlButtonClickable * btnDownloadRestyleConfirm = nullptr;
        ui::ControlButtonClickable * btnDownloadRestyleCancel = nullptr;
#endif

        ui::ControlButtonToggle * btnToggleFilter = nullptr;
        ui::ControlButtonToggle * btnToggleAdjustments = nullptr;
        ui::ControlButtonToggle * btnToggleFX = nullptr;
        ui::ControlButtonToggle * btnToggleCamCapture = nullptr;
        ui::ControlButtonToggle * btnToggleGameSpecific = nullptr;

        ui::ControlFlyoutToggleShared::LabelsStorage flySpecialFXLabels;
        ui::ControlFlyoutToggleShared * flySpecialFX = nullptr;

        ui::ControlButton * btnSnap = nullptr;
        ui::ControlButton * btnDone = nullptr;

        ui::ControlSliderDiscr * sldKind = nullptr;
        ui::ControlCheckbox * chkGridOfThirds = nullptr;
        ui::ControlCheckbox * chkHDR = nullptr;
        ui::ControlSliderCont * sldFOV = nullptr;
        ui::ControlSliderCont * sldRoll = nullptr;
        ui::ControlButton * btnResetRoll = nullptr;

        // DarkRoom capture quality sliders
        int sldHiResMultStringBufSize;
        ui::ControlSliderDiscr * sldHiResMult = nullptr;
        ui::ControlSliderCont * sldSphereFOV = nullptr;

        ui::ControlCheckbox * chkEnhanceHiRes = nullptr;

        ui::ControlIcon * icoLWLogo = nullptr;
        ui::ControlIcon * icoCamera = nullptr;
        ui::ControlIcon * icoLight = nullptr;
        ui::ControlIcon * icoFilters = nullptr;

        ui::ControlContainer * leftPane = nullptr;
        ui::ControlFlyoutContainer * flyoutPane = nullptr;

        ui::ControlContainer * cntControls = nullptr;
        ui::ControlContainer * cntFilter = nullptr;
        ui::ControlContainer * cntAdjustments = nullptr;
        ui::ControlContainer * cntFX = nullptr;
        ui::ControlContainer * cntGameSpecific = nullptr;
        ui::ControlContainer * cntCameraCapture = nullptr;

        ui::ControlCheckbox * chkAllowModding = nullptr;


#if (DBG_STACKING_PROTO == 1)
        ui::ControlButton * btnAddFilter = nullptr;
        std::vector<ui::ControlDynamicFilterContainer *> m_dynamicFilterContainers;
#endif

#if (DBG_ENABLE_HOTKEY_SETUP == 1)
        static const unsigned int m_lblHotkeyCaptionMaxSize = 64;
        ui::ControlButton * btnHotkeySetup = nullptr;
        ui::ControlLabel * lblHotkey = nullptr;
#endif
    };

    class FlyoutRebuildRequest
    {
    public:
        bool isValid = false;
        ui::ControlFlyoutToggleShared * srcFlyoutToggle = nullptr;
        ui::ControlContainer * dstFlyoutContainer = nullptr;
    } m_flyoutRebuildRequest;

    ComponentsList m_components;
    bool m_fovChangeAllowed = true;

    bool m_isModdingAllowed = false;


    bool areCameraInteractionsDisabled = false;
    bool areControlsInteractionsDisabled = false;
    ui::ControlContainer * excludeContainer = nullptr;
    bool isControlEnabled(ui::ControlContainer * control);


    // Dynamic controls memory management
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DynamicControlRawData
    {
        char rawData[768];
    };

    shadermod::ir::Pool<DynamicControlRawData> m_dynamicControlsPool;

    void releaseDynamicControlPools()
    {
        m_dynamicControlsPool.destroy();
    }

    template<typename T>
    T * getNewDynamicControlMem()
    {
        static_assert(sizeof(T) < sizeof(DynamicControlRawData), "Not enough data in the DynamicControlRawData");
        DynamicControlRawData * dynamicControlMem = m_dynamicControlsPool.getElement();
        return reinterpret_cast<T *>(dynamicControlMem);
    }

    template<typename T>
    void deleteOldDynamicControlMem(T * dynamicControl)
    {
        assert(false);
        // This is invalid thing to do, as it doesn't call proper destructor
        //m_dynamicControlsPool.deleteElement(reinterpret_cast<DynamicControlRawData *>(dynamicControl));
    }

    void releaseDynamicControl(ui::ControlContainer * dynamicControl)
    {
        for (uint32_t extIdx = 0, extIdxEnd = dynamicControl->getNumExternalChildren(); extIdx < extIdxEnd; ++extIdx)
        {
            ui::ControlContainer * extChild = dynamicControl->getExternalChild(extIdx);
            extChild->~ControlContainer();
            m_dynamicControlsPool.putElement(reinterpret_cast<DynamicControlRawData *>(extChild));
        }
        dynamicControl->~ControlContainer();
        m_dynamicControlsPool.putElement(reinterpret_cast<DynamicControlRawData *>(dynamicControl));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool m_useHybridController = false;
    void setUseHybridController(bool useHybridController) { m_useHybridController = useHybridController; }
    bool getUseHybridController() const { return m_useHybridController; }

    bool m_isCameraWorksSessionActive;
    void onSessionStarted(bool isCameraWorksSessionActive) override;
    void onSessionStopped() override;

    bool m_isStandaloneModdingEnabled = false;
    void setStandaloneModding(bool isStandaloneModdingEnabled) { m_isStandaloneModdingEnabled = isStandaloneModdingEnabled; }
    bool getStandaloneModding() const { return m_isStandaloneModdingEnabled; }

    void repopulateEffectsList(const std::vector<std::wstring> & filterIds, const std::vector<std::wstring> & filterNames) override;
#ifdef ENABLE_STYLETRANSFER
    void repopulateStylesList(const std::vector<std::wstring> & styleIds, const std::vector<std::wstring> & styleNames, const std::vector<std::wstring> & stylePaths) override;
    void repopulateStyleNetworksList(const std::vector<std::wstring> & styleIds, const std::vector<std::wstring> & styleNames) override;
#endif

    bool isStackFiltersListRequested() override { return false; }
    void stackFiltersListDone(const std::vector<std::wstring> & filterIds) override { }

    std::vector<std::wstring> filterIDs;

    size_t getLwrrentFilterNum() const override;

    void clampStringNumChars(std::wstring * str, const size_t maxStrLength) const;
    void clampStringFloatSize(std::wstring * str, float maxSize) const;

    const std::wstring getLwrrentFilter(size_t effectStackIdx) const override;
    bool getLwrrentFilterInfoQuery(size_t effectStackIdx) override { /* We don't need this at the moment */ return false; }
    bool getLwrrentFilterResetValues(size_t effectStackIdx) override { /* We don't need this at the moment */ return false; }
    void lwrrentFilterResetValuesDone(size_t effectStackIdx) override { /* We don't need this at the moment */ }
    int getLwrrentFilterOldStackIdx(size_t effectStackIdx) const override;
    void updateLwrrentFilterStackIndices() override;

    ErrorManager * m_errorManager = nullptr;
    float getMessageDurationByType(MessageType msgType);
    void displayMessageInternal(float duration, const std::wstring & msgText);
    void displayMessage(MessageType msgType) override;
    void displayMessage(MessageType msgType, const std::vector<std::wstring> & parameters
#if ANSEL_SIDE_PRESETS
                        , bool removeLastLine = false
#endif
        ) override;

    void reportFatalError(FatalErrorCode code, const std::string& filename, uint32_t line, const std::string& data) override {}
    void reportNonFatalError(uint32_t code, const std::string& filename, uint32_t line, const std::string& data) override {}
#ifdef ENABLE_STYLETRANSFER
    bool m_isStyleTransferAllowed = false;
    void mainSwitchStyleTransfer(bool status) override;

    const std::wstring & getLwrrentStyle() const override;
    const std::wstring & getLwrrentStylePath() const override;

    const std::wstring & getLwrrentStyleNetwork() const override;
    void setLwrrentStyleNetwork(const std::wstring &) override;

    void setStyleTransferStatus(bool isStyleTransferEnabled) override;

    void showRestyleDownloadConfirmation() override;
    AnselUIBase::RestyleDownloadStatus m_restyleDownloadConfirmationStatus = AnselUIBase::RestyleDownloadStatus::kNone;
    RestyleDownloadStatus getRestyleDownloadConfirmationStatus() override;
    void clearRestyleDownloadConfirmationStatus() override;

    bool m_restyleProgressBarNeedsResizing = false;
    float m_restyleProgressBarVis = 0.0f;
    float m_restyleProgressBarVisGoal = 0.0f;
    void toggleRestyleProgressBar(bool status) override;
    void setRestyleProgressBarValue(float value) override;
    void formatRestyleProgressLabel();
    RestyleProgressState m_restyleProgressState = RestyleProgressState::kDownloadProgress;
    void setRestyleProgressState(RestyleProgressState progressState) override;
    void toggleRestyleProgressIndicator(bool status) override;
#endif

    std::vector<EffectChange> m_controlChangeQueue;
    void getEffectChangesDone() override { m_controlChangeQueue.resize(0); }
    std::vector<AnselUIBase::EffectChange>& getEffectChanges() override { return m_controlChangeQueue; }

    void removeDynamicElement(ui::ControlContainer * element, ui::ControlContainer ** selectedControl, bool * needToReselectControl);
    void removeDynamicElementsInContainer(ui::ControlContainer * neededContainer, ui::ControlContainer ** selectedControl, bool * needToReselectControl);

    void updateContainerControls(size_t effectStackIdx, ui::ControlContainer * neededContainer, int neededBlockID, EffectPropertiesDescription * effectDesc);
    void updateEffectControls(size_t effectStackIdx, EffectPropertiesDescription * effectDesc) override;
    void updateEffectControlsInfo(size_t effectStackIdx, EffectPropertiesDescription * effectDesc) override {}

    bool isGameSettingsPanelDisplayed = false;
    bool isUpdateGameSpecificControlsRequired() override { return false; }
    void updateGameSpecificControls(EffectPropertiesDescription * effectDesc) override;
    void updateGameSpecificControlsInfo(EffectPropertiesDescription * effectDesc) override {}

    void setFOVLimitsDegrees(double lo, double hi) override;
    void setRollLimitsDegrees(double lo, double hi) override;
    void set360WidthLimits(uint64_t lo, uint64_t hi) override;
    void setFovControlEnabled(bool enabled) override;

    bool isGridOfThirdsEnabled() const override;

    bool isShotEXR() const override { return static_cast<ui::ControlCheckbox *>(m_components.chkHDR)->isChecked; }
    bool isShotJXR() const override { return false; }
    bool isShotPreviewRequired() const override { return false; }
    double getFOVDegrees() const override;
    double getRollDegrees() const override;
    bool processFOVChange() override;
    bool processRollChange() override;
    bool getCameraDragActive() override;
    bool isCameraInteractive() override;

    bool isModdingAllowed() override;
    bool queryIsModdingEnabled() override;
    bool isModdingEnabled() override;

    std::wstring m_defaultEffectPath;
    void setDefaultEffectPath(const wchar_t * defaultEffectPath) override
    {
        m_defaultEffectPath = defaultEffectPath;
    }

    void updateSettings(const std::map<std::string, std::wstring>& settings) override {}

    bool isHighresEnhance() const override
    {
        return m_components.chkEnhanceHiRes->isChecked;
    }

    bool isHighQualityEnabled() const override { return m_isHighQualityEnabled; }
    void setHighQualityEnabled(bool setting) override { m_isHighQualityEnabled = setting; }
    bool m_isHighQualityEnabled = false;

#ifdef ENABLE_STYLETRANSFER
    bool isStyleTransferEnabled() const override
    {
        return m_components.chkEnableStyleTransfer->isChecked;
    }
#endif

    void setFOVDegrees(double fov) override;
    void setRollDegrees(double roll) override;
    
    void setHighresEnhance(bool enhance) override
    {
        m_components.chkEnhanceHiRes->isChecked = enhance;
    }

    virtual input::InputHandler& getInputHandler() override
    {
        return m_inputstate;
    }
    virtual const input::InputHandler& getInputHandler() const override
    {
        return m_inputstate;
    }

    // Set to true if you want primary and secondary buttons meaning be swapped
    void setAreMouseButtonsSwapped(bool areMouseButtonsSwapped)
    {
        m_areMouseButtonsSwapped = areMouseButtonsSwapped;
    }
    //this is the callback triggered for each input event
    void onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
        const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
        const input::FolwsChecker& folwsChecker, const input::MouseTrapper& mouseTrapper) override;
    //this is to process input state once per frame (lwrrently for timer-based reactions to buttons held permanently)
    void processInputState(const AnselSDKState & sdkState, float dt);

    virtual void getTelemetryData(UISpecificTelemetryData &) const override;
    void reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, ...) const;
    void reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, ...) const;

    enum { 
        kStateDeactive = 0,
        kStateTransitionToSession,
        kStateSessionActive,
        kStateTransitionFromSession,
        kStateTransitionToDeactive,
        kStateCount
    };
    uint32_t m_stateLwrrent = kStateDeactive;
    uint32_t m_stateRequested = kStateDeactive;
    float m_stateTime = 0.0f;
    float m_stateDuration = 0.0f;

#define FADE_OUT_DURATION_IN_SECONDS 0.4f
#define FADE_IN_DURATION_IN_SECONDS 0.5f

    void advanceState();
    void updateState(double dt);
    void forceImmediateStateChange();

    void update(double dt) override;

    bool m_isSDKDetected = false;
    void camWorksToggle();
    void checkToggleCamWorks();

    bool m_isAnselPrestartRequested = false;
    bool m_isAnselStartRequested = false;
    bool m_isAnselStopRequested = false;
    bool m_isAnselPoststopRequested = false;

    bool isAnselPrestartRequested() override;
    bool isAnselStartRequested() override;
    bool isAnselSDKSessionRequested() override;
    bool isAnselStopRequested() override;
    bool isAnselPoststopRequested() override;
    bool isAnselFeatureSetRequested() override;

    void anselPrestartDone(AnselUIBase::Status status, bool isSDKDetected, bool requireSDK) override;
    void anselStartDone(AnselUIBase::Status status) override;
    void anselStopDone(AnselUIBase::Status status) override;
    void anselPoststopDone(AnselUIBase::Status status) override;
    void anselFeatureSetRequestDone() override {}

    void forceEnableUI() override;
    void forceDisableUI() override;

    bool m_needToResetRoll = false;
    bool isResetRollNeeded() const override;
    void resetRollDone() override;
    void setResetRollStatus(bool isAvailable) override;

    bool m_isSDKCaptureAbortRequested = false;
    bool isSDKCaptureAbortRequested() override;
    void sdkCaptureAbortDone(int status) override;

    bool m_needToggleCamWorks = false;

    bool m_hotkeyModifierDown = false;
    bool m_altKeyDown = false;
    bool m_ctrlKeyDown = false;
    bool m_toggleKeyDown = false;
    bool m_f12KeyDown = false;

    bool m_isAnselActive = true;  // This is about deactivating Ansel completely (so that it can't get called in current process until it restarts)
    void processAnselStatus(const AnselSDKState & sdkState, bool * needToSkipRestFrame, int * anselSessionStateChangeRequest, bool * forceNotSkipRestFrame);

    void dormantUpdate(const AnselSDKState & sdkState, bool * needToAbortCapture, bool * needToSkipRestFrame);

    int m_progressInfo_shotsTotal = 0;
    int m_progressInfo_shotIdx = 0;
    bool m_progressInfo_inProgress = false;

    void setFilterControlsEnabled(bool isEnabled);
    void onCaptureStarted(int numShotsTotal) override;
    void onCaptureTaken(int shotIdx) override;
    void onCaptureStopped(AnselUIBase::MessageType status) override;

    ui::ControlContainer * clearFlyout(FlyoutRebuildRequest * request);
    void rebuildFlyout(FlyoutRebuildRequest * request);

    void hideFloatingContainers();

#if (DBG_STACKING_PROTO == 1)
    struct DynamicFilterParameters
    {
    } m_dynamicFilterParams;
    int m_dynamicFilterIdxToRemove = -1;
    bool m_needToAddDynamicFilter = true;
    void addDynamicFilter();
    void removeDynamicFilter();
    void dynamicFilterExchangeTabStops(ui::ControlDynamicFilterContainer * first, ui::ControlDynamicFilterContainer * second);
    void checkUpDownButtons();
#endif

    void onButtonToggleClick(void * object);
    void onFlyoutClick(void * object);
    void onFlyoutHidePane(void * object);
    void onFlyoutSelectorClick(void * object);
    void onButtonSnapClick(void * object);
    void onButtonHideClick(void * object);
    void onButtonResetRollClick(void * object);
#if (DBG_STACKING_PROTO == 1)
    void onButtonAddFilterClick(void * object);
    void onButtonRemoveFilterClick(void * object);
    void onButtonMoveFilterUpClick(void * object);
    void onButtonMoveFilterDownClick(void * object);
    void onFlyoutSpecialFXChangeDynamic(void * object);
#endif
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    void onButtonHotkeySetupClick(void * object);
#endif
    void onFlyoutSpecialFXChange(void * object);
#ifdef ENABLE_STYLETRANSFER
    void onFlyoutStylesChange(void * object);
    void onFlyoutStyleNetworksChange(void * object);
    void onButtonDownloadRestyleConfirmClick(void * object);
    void onButtonDownloadRestyleCancelClick(void * object);
#endif
    
    void onSliderKindChange(void * object);

    void onSliderEffectTweakChange(void * object);
    void onSliderListEffectTweakChange(void * object);
    void onSliderIntEffectTweakChange(void * object);
    void onCheckboxEffectTweakChange(void * object);
    void onColorPickerEffectTweakChange(void * object);

    UIControlReactorFunctor<AnselUI> m_onButtonToggleClick;
    
    UIControlReactorFunctor<AnselUI> m_onFlyoutClick;
    UIControlReactorFunctor<AnselUI> m_onFlyoutHidePane;
    UIControlReactorFunctor<AnselUI> m_onFlyoutSelectorClick;
    UIControlReactorFunctor<AnselUI> m_onFlyoutSpecialFXChange;

    UIControlReactorFunctor<AnselUI> m_onButtonSnapClick;
    UIControlReactorFunctor<AnselUI> m_onButtonHideClick;
    UIControlReactorFunctor<AnselUI> m_onButtonResetRollClick;
#if (DBG_STACKING_PROTO == 1)
    UIControlReactorFunctor<AnselUI> m_onButtonAddFilterClick;
    UIControlReactorFunctor<AnselUI> m_onButtonRemoveFilterClick;
    UIControlReactorFunctor<AnselUI> m_onButtonMoveFilterUpClick;
    UIControlReactorFunctor<AnselUI> m_onButtonMoveFilterDownClick;
    UIControlReactorFunctor<AnselUI> m_onFlyoutSpecialFXChangeDynamic;
#endif
#if (DBG_ENABLE_HOTKEY_SETUP == 1)
    UIControlReactorFunctor<AnselUI> m_onButtonHotkeySetupClick;
#endif
#ifdef ENABLE_STYLETRANSFER
    UIControlReactorFunctor<AnselUI> m_onFlyoutStylesChange;
    UIControlReactorFunctor<AnselUI> m_onFlyoutStyleNetworksChange;
    UIControlReactorFunctor<AnselUI> m_onButtonDownloadRestyleConfirmClick;
    UIControlReactorFunctor<AnselUI> m_onButtonDownloadRestyleCancelClick;
#endif

    UIControlReactorFunctor<AnselUI> m_onSliderKindChange;

    UIControlReactorFunctor<AnselUI> m_onSliderEffectTweakChange;
    UIControlReactorFunctor<AnselUI> m_onSliderListEffectTweakChange;
    UIControlReactorFunctor<AnselUI> m_onSliderIntEffectTweakChange;
    UIControlReactorFunctor<AnselUI> m_onCheckboxEffectTweakChange;
    UIControlReactorFunctor<AnselUI> m_onColorPickerEffectTweakChange;

    LCID m_forcedLocale = 0;
    void setForcedLocale(LCID forcedLocale) { m_forcedLocale = forcedLocale; }
    LCID getForcedLocale() const { return m_forcedLocale; }

    ShotType m_shotToTake = ShotType::kNone;

    bool m_shotHDREnabled = false;
    bool m_shotTypeEnabled[(unsigned int)ShotType::kNumEntries];

    void setShotTypePermissions(bool shotHDREnabled, const bool * shotTypeEnabled, int arraySize) override;

    int mouseLwrsor = UI_MOUSELWRSOR_ARROW;

    bool isCameraDragActive = false;
#ifdef ENABLE_STYLETRANSFER
    bool isUIInteractActive = false;
#endif

    // TODO: mark it as a copy or something
    const float mouseSensititvityUI = 1.0f;
    const float mouseWheelSensitivityUI = 1.0f;
    double gamepadFilterTime = -1.0;
    float mouseCoordsAbsX = 0.0f, mouseCoordsAbsY = 0.0f;
    unsigned int resolutionBaseWidth = 0, resolutionBaseHeight = 0;

    float m_fadeValue = 0.0f;
    void setFadeValue(float fade) { m_fadeValue = fade; }

    // IPC API
    void setScreenSize(uint32_t, uint32_t) override;
    void setShotType(ShotType type);
    bool isVisible() const;
    void setIsVisible(bool isVisible);
    void setIsEnabled(bool isEnabled);
    bool isEnabled() const;
    // Whether UI receives messages and processes them
    void setAnselSDKDetected(bool) override;
    bool isActive() const;
    bool doBlockMessageTransfer() const;
    bool isControlInteractiveHierarchical(ui::ControlContainer * controlToCheck);
    void selectNextElement();
    void selectPrevElement();
    void checkSelectedControl();
    void setCWControlsVisibility(bool visibility);
#ifdef ENABLE_STYLETRANSFER
    bool isUIInteractionActive() override;
#endif

    bool m_isEffectListRequested = false;
    bool isEffectListRequested() override
    {
        return m_isEffectListRequested;
    }

#ifdef ENABLE_STYLETRANSFER
    bool m_isStylesListRequested = false;
    bool isStylesListRequested() override
    {
        return m_isStylesListRequested;
    }

    bool m_isStyleNetworksListRequested = false;
    bool isStyleNetworksListRequested() override
    {
        return m_isStyleNetworksListRequested;
    }
#endif


    bool m_fadeEnabled = true;
    void setFadeState(bool fadeState) override
    {
        m_fadeEnabled = fadeState;
    }

    // These are values that are retrieved from GFE. LwCamera does not have a way to get these on its own.
    virtual std::wstring getAppCMSID() const override { return L""; }
    virtual std::wstring getAppShortName() const override { return L""; }

    ShotDesc shotCaptureRequested() override
    {
        ShotDesc shotDesc;
        shotDesc.shotType = m_shotToTake;
        shotDesc.highResMult = m_components.sldHiResMult->getSelected() + 2;
        shotDesc.resolution360 = static_cast<ui::ControlSlider360Quality *>(m_components.sldSphereFOV)->getResolution();
#if 0
        shotDesc.thumbnailRequired = true;
#endif
        return shotDesc;
    }
    void shotCaptureDone(AnselUIBase::Status status) override
    {
        m_shotToTake = ShotType::kNone;
    }
    void onCaptureProcessingDone(int status, const std::wstring & absPath) override
    {
    }

    bool m_isHighResolutionRecalcRequested = false;
    bool isHighResolutionRecalcRequested() override
    {
        return m_isHighResolutionRecalcRequested;
    }
    void highResolutionRecalcDone(const std::vector<HighResolutionEntry> & highResEntries) override;

    void recallwlateUIPositions();
    
    void recallwlateUILayoutScale(float aspect, float * scaleX, float * scaleY) const;
    
    struct UIStoredSizes
    {
        float uiMulX = 1.0f, uiMulY = 1.0f;
        float uiMarginX = 20.f / 1920.f * 2.f;
        float defaultSliderSizeX = 160.f / 1920.f * 2.f;
        float defaultCheckboxSizeX = 160.f / 1920.f * 2.f;
        float defaultSelectorSizeX = 160.f / 1920.f * 2.f;
        float defaultDynamicToggleSizeX = 100.f / 1920.f * 2.f;
    } m_storedSizes;
    HRESULT recallwlateUILayout(float aspect);
    void applyTabStop();

    LANGID getLangId() const override
    {
        return m_langID;
    };

    HRESULT init(HMODULE hResourceModule, ID3D11Device* d3dDevice, AnselServer* pAnselServer, const std::wstring& installationFolderPath);
    void release();

    bool m_showMouseWhileDefolwsed = false;
    void setShowMouseWhileDefolwsed(bool showMouseWhileDefolwsed);
    bool getShowMouseWhileDefolwsed() const;

    void getTextRect(const wchar_t * str, size_t strLen, bool isBold, float * designSpaceW, float * designSpaceH) const;

    void setRenderState(
            ID3D11DeviceContext* d3dctx,
            AnselResource * pPresentResourceData,
            AnselEffectState* pPassthroughEffect
            );
    void render(
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
            );

    void setModdingStatus(bool isModdingAllowed) override;

    ErrorManager m_gameplayOverlayNotifications;
    bool isGameplayOverlayRequired()
    {
        return !m_gameplayOverlayNotifications.isEmpty();
    }
    void addGameplayOverlayNotification(AnselUIBase::NotificationType notificationType, ErrorManager::ErrorEntry notification, bool allowSame)
    {
        const size_t errNum = (int)m_gameplayOverlayNotifications.getErrorCount();
        int errorsDisplayed = 0;

        std::wstring formattedMessage;
        switch (notificationType)
        {
        case AnselUIBase::NotificationType::kWelcome:
            {
                formattedMessage = m_textNotifWelcome + notification.message;
                break;
            }
        case AnselUIBase::NotificationType::kSessDeclined:
            {
                formattedMessage = m_textNotifSessDeclined;
                break;
            }
        case AnselUIBase::NotificationType::kDrvUpdate:
            {
                formattedMessage = m_textNotifDrvUpdate;
                break;
            }
        default:
            {
                formattedMessage = notification.message;
                break;
            }
        };

        bool isSameFound = false;

        if (!allowSame)
        {
            const size_t newStringSize = formattedMessage.size();

            for (size_t errCnt = 0u; errCnt < errNum; ++errCnt)
            {
                const size_t lwrErrorEntry = (m_gameplayOverlayNotifications.getFirstErrorIndex() + errCnt) % errNum;

                if (m_gameplayOverlayNotifications.getErrorLifetime(lwrErrorEntry) < 0.0f)
                    continue;

                size_t lwrErrorEntrySize = m_gameplayOverlayNotifications.getErrorString(lwrErrorEntry).size();
                bool lwrErrorStringEqual = (m_gameplayOverlayNotifications.getErrorString(lwrErrorEntry) == formattedMessage);

                if (lwrErrorEntrySize == newStringSize && lwrErrorStringEqual)
                {
                    isSameFound = true;
                    break;
                }
            }
        }

        if (!isSameFound)
        {
            m_gameplayOverlayNotifications.addError(notification.lifeTime, formattedMessage);
        }
    }
    void renderGameplayOverlay(
            double dt,
            ID3D11DeviceContext * d3dctx,
            AnselResource * pPresentResourceData,
            AnselEffectState* pPassthroughEffect
            );

    void emergencyAbort() override
    {
        m_inputstate.deinit();
    }

    bool isRequestingControl() override
    {
        return m_needToggleCamWorks || m_isAnselPrestartRequested;
    }
    void rejectControlRequest() override
    {
        m_needToggleCamWorks = false;
        m_stateRequested = kStateDeactive;
        m_isAnselPrestartRequested = false;
    }

#if (UI_ENABLE_TEXT == 1)
    struct FW1State
    {
        IFW1FontWrapper *       pFontWrapperSergoeUI = nullptr;
        IFW1GlyphRenderStates * pRenderStatesSergoeUI = nullptr;

        IFW1FontWrapper *       pFontWrapperSergoeUIBold = nullptr;
        IFW1GlyphRenderStates * pRenderStatesSergoeUIBold = nullptr;

        IDWriteFactory *        pDWriteFactorySegoeUI = nullptr;
        IDWriteFactory *        pDWriteFactorySegoeUIBold = nullptr;
        IDWriteTextFormat *     pTextFormatSegoeUI = nullptr;
        IDWriteTextFormat *     pTextFormatSegoeUIBold = nullptr;
        ID3D11PixelShader *     pPSOutline = nullptr;

        IFW1Factory *           pFactory = nullptr;
        float                   defaultFontSize = 0.0f;
    } FW1;
#endif

    ID3D11VertexShader * pVertexShader = nullptr;
    ID3D11PixelShader * pPixelShader = nullptr;
    ID3D11Texture2D * pUIAtlasTexture = nullptr;
    ID3D11ShaderResourceView * pUIAtlasSRV = nullptr;

    ID3D11InputLayout * pInputLayout = nullptr;

    ID3D11DepthStencilState * pDepthStencilState = nullptr;
    ID3D11BlendState * pBlendState = nullptr;
    ID3D11RasterizerState * pRasterizerState = nullptr;

    ID3D11Buffer * pZeroOffsetsBuffer = nullptr;
    ID3D11Buffer * pVariableOffsetsBuffer = nullptr;

#if (DBG_USE_OUTLINE == 1)
    ID3D11Buffer * pFontOutlineBuffer = nullptr;
#endif

    ID3D11Buffer * pArrowDowlwertexBuf = nullptr;
    ID3D11Buffer * pArrowUpVertexBuf = nullptr;
    ID3D11Buffer * pArrowRightVertexBuf = nullptr;
    ID3D11Buffer * pArrowLeftVertexBuf = nullptr;
    ID3D11Buffer * pArrowIndexBuf = nullptr;

    ID3D11Buffer * pCamIcolwertexBuf = nullptr;

    ID3D11Buffer * pRectVertexBuf = nullptr;
    ID3D11Buffer * pRectIndexBuf = nullptr;

    ID3D11Buffer * pTriUpVertexBuf = nullptr;
    ID3D11Buffer * pTriUpIndexBuf = nullptr;

    ID3D11Buffer * pMouseIndexBuf = nullptr;
    ID3D11Buffer * pMousePointerVertexBuf = nullptr;
    ID3D11Buffer * pMouseHandVertexBuf = nullptr;

    unsigned int vertexStrideUI;

    input::InputState* m_inputCapture = nullptr;

    uint32_t m_width = 0, m_height = 0;
    bool m_needToRecallwILayout = false;
    bool m_needToApplyTabStop = false;

    AnselServer* m_pAnselServer = nullptr;
};