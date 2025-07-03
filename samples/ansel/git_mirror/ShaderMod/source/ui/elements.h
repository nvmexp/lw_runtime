#pragma once

#include "AnselServer.h"
#include "classes.h"
#include "darkroom/Director.h"

class UIControlReactorFunctorInterface
{
public:
    virtual void operator()(void * object) = 0;
    virtual ~UIControlReactorFunctorInterface() {}
};

template<typename TClass>
class UIControlReactorFunctor: public UIControlReactorFunctorInterface
{
public:
    typedef void (TClass::*TMethod)(void *);
    UIControlReactorFunctor(TClass* classPtr, TMethod methodPtr):
        m_classPtr(classPtr), m_methodPtr(methodPtr)
    {
    }

    void operator()(void * object) override
    {
        (m_classPtr->*m_methodPtr)(object);
    }

protected:
    TClass*     m_classPtr;
    TMethod     m_methodPtr;
};

namespace ui
{
    class ControlButtonToggle : public ControlButton
    {
    public:

        ControlContainer * m_containerToggle = nullptr;
        float m_glyphMargin = 0.0f;

        ControlButtonToggle(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
            renderType = RenderType::kToggle;
        }

        virtual int onClick()
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        };

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlButtonClickable : public ControlButton
    {
    public:

        ControlButtonClickable(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        virtual int onClick()
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        };

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlFlyoutToggleShared : public ControlButton
    {
    public:

        class LabelsStorage
        {
        public:
            std::vector<std::wstring> labels;
            std::vector<std::wstring> ids;
        };

        std::wstring lastSelectedLabel;
        LabelsStorage * labelsStorage = nullptr;

        std::wstring emptyString;
        static const size_t dynamicCaptionSize = 64;
        wchar_t dynamicCaption[dynamicCaptionSize];

        int selectedId = 0;

        ControlFlyoutToggleShared(UIControlReactorFunctorInterface* pClickReactor, UIControlReactorFunctorInterface* pChangeReactor):
            m_pClickReactor(pClickReactor),
            m_pChangeReactor(pChangeReactor)
        {
        }

        virtual int onClick()
        {
            if (m_pClickReactor)
                (*m_pClickReactor)(this);

            return 1;
        };

        virtual int onChange()
        {
            if (m_pChangeReactor)
                (*m_pChangeReactor)(this);

            return 1;
        };

        bool needsChangeOnCWToggle()
        {
            return (labelsStorage->labels.size() > (size_t)selectedId) ? ((lastSelectedLabel == labelsStorage->labels[selectedId]) != 0) : true;
        }

        void setSelected(int selected, const wchar_t * str)
        {
            if (selected >= int(labelsStorage->labels.size()))
            {
                return;
            }

            selectedId = selected;
            swprintf_s(dynamicCaption, dynamicCaptionSize, L"%s", str);
        }

        int getSelected() { return selectedId; }
        wchar_t * getLabel(int id) { return (id < int(labelsStorage->labels.size())) ? labelsStorage->labels[id].c_str() : L""; }
        wchar_t * getSelectedLabel() { return getLabel(selectedId); }
        wchar_t * getSelectedId() { return (selectedId < int(labelsStorage->ids.size())) ? labelsStorage->ids[selectedId].c_str() : L""; }
        const std::wstring & getSelectedIdStr()
        {
            return (selectedId < int(labelsStorage->ids.size())) ? labelsStorage->ids[selectedId] : emptyString;
        }

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:

        UIControlReactorFunctorInterface * m_pClickReactor = nullptr;
        UIControlReactorFunctorInterface * m_pChangeReactor = nullptr;
    };

    class ControlFlyoutToggleIndependent : public ControlFlyoutToggleShared
    {
    public:

        ControlFlyoutToggleShared::LabelsStorage m_dataStorage;

        ControlFlyoutToggleIndependent(UIControlReactorFunctorInterface* pClickReactor, UIControlReactorFunctorInterface* pChangeReactor):
            ControlFlyoutToggleShared(pClickReactor, pChangeReactor)
        {
            labelsStorage = &m_dataStorage;
        }
    };

    class ControlFlyoutSelector : public ControlButton
    {
    public:

        static const size_t dynamicCaptionSize = 64;
        wchar_t dynamicCaption[dynamicCaptionSize];

        int id;

        ui::ControlFlyoutToggleShared * srcToggle = nullptr;
        ui::ControlContainer * dstContainer = nullptr;

        ControlFlyoutSelector(UIControlReactorFunctorInterface* pReactor, UIControlReactorFunctorInterface* pHideReactor):
            m_pReactor(pReactor),
            m_pHideReactor(pHideReactor)
        {
            // Force caption pointer to point to the dynamicCaption buffer
            caption = dynamicCaption;
        }

        virtual int onKeyPress(DWORD vkey) override
        {
            if (!isEnabled)
                return 0;

            if (vkey == VK_LEFT)
            {
                return onDecrease();
            }

            return ControlButton::onKeyPress(vkey);
        }

        virtual int onClick() override
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        };

        virtual int onDecrease() override
        {
            if (m_pHideReactor)
            {
                (*m_pHideReactor)(this);
                return 1;
            }

            return ControlButton::onDecrease();
        };

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
        UIControlReactorFunctorInterface * m_pHideReactor = nullptr;
    };

    class ControlFlyoutContainer : public ControlContainer
    {
    public:

        ControlFlyoutToggleShared * srcToggle = nullptr;
    };

    class ControlDynamicFilterContainer : public ControlContainer
    {
    public:
        
        int stackIdx = -1;
        ControlButtonToggle * m_toggleButton = nullptr;      // Toggle that shows/hides container
        ControlButtonClickable * m_btnRemove = nullptr;
        ControlButtonClickable * m_btnUp = nullptr;
        ControlButtonClickable * m_btnDown = nullptr;
        ControlFlyoutToggleShared * m_filterToggle = nullptr;  // Filter type flyout toggle

        static const size_t maxFilterNameSize = 64;
        wchar_t filterName[maxFilterNameSize];

        ControlDynamicFilterContainer()
        {

        }
    };

    class ControlButtonSnap : public ControlButton
    {
    public:

        ControlButtonSnap(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        virtual int onClick()
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        };

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlButtonHide : public ControlButton
    {
    public:

        ControlButtonHide(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        virtual int onClick()
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        };

        virtual int getPreferredLwrsor() override { return UI_MOUSELWRSOR_HAND; }

    private:
        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlCheckboxEffectTweak : public ControlCheckbox
    {
    public:

        AnselUIBase::DataType dataType;

        // TODO: move this into a separate interface
        std::wstring filterId;
        int controlIdx = -1;
        int stackIdx = -1;

        shadermod::ir::TypelessVariable milwalue;
        shadermod::ir::TypelessVariable maxValue;

        ControlCheckboxEffectTweak(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        // Function to see if the checkbox will be checked with the lwrValue
        virtual bool checkLwrrentValue(const shadermod::ir::TypelessVariable & lwrValue)
        {
            switch (lwrValue.getType())
            {
            case shadermod::ir::UserConstDataType::kFloat:
                {
                    float floatValue;
                    lwrValue.get(&floatValue, 1);

                    float milwalF, maxValF;
                    milwalue.get(&milwalF, 1);
                    maxValue.get(&maxValF, 1);
                    float distValMin = fabsf(floatValue - milwalF);
                    float distValMax = fabsf(floatValue - maxValF);

                    return (distValMax < distValMin);
                }
            case shadermod::ir::UserConstDataType::kBool:
                {
                    bool boolValue;
                    lwrValue.get(&boolValue, 1);
                    return boolValue;
                }
            case shadermod::ir::UserConstDataType::kInt:
                {
                    int intValue;
                    lwrValue.get(&intValue, 1);

                    int milwalI, maxValI;
                    milwalue.get(&milwalI, 1);
                    maxValue.get(&maxValI, 1);
                    int distValMin = std::abs(intValue - milwalI);
                    int distValMax = std::abs(intValue - maxValI);

                    return (distValMax < distValMin);
                }
            case shadermod::ir::UserConstDataType::kUInt:
                {
                    unsigned int uintValue;
                    lwrValue.get(&uintValue, 1);

                    unsigned int milwalU, maxValU;
                    milwalue.get(&milwalU, 1);
                    maxValue.get(&maxValU, 1);
                    int distValMin = (int)(uintValue - milwalU);
                    int distValMax = (int)(uintValue - maxValU);

                    return (distValMax < distValMin);
            }
            default:
                {
                    // This shouldn't happen
                    return false;
                }
            }
        }

        virtual void setValue()
        {
            if (m_pReactor)
                (*m_pReactor)(this);
        }

        virtual int onClick()
        {
            ControlCheckbox::onClick();
            setValue();
            return 1;
        }

        virtual int onIncrease()
        {
            ControlCheckbox::onClick();
            setValue();
            return 1;
        }

        virtual int onDecrease()
        {
            ControlCheckbox::onClick();
            setValue();
            return 1;
        }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlSliderEffectTweak : public ControlSliderCont
    {
        const float defaultStep = 0.05f;

    public:

        AnselUIBase::DataType dataType;

        std::wstring filterId;
        int controlIdx = -1;
        int stackIdx = -1;

        ControlSliderEffectTweak(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        virtual void setValue(float val)
        {
            if (m_pReactor)
                (*m_pReactor)(this);
        }

        virtual int onIncrease() override
        {
            const float multiplier = fineTuning ? 0.1f : 1.0f;

            // TODO: differentiate between real slider step and keyboard/gamepad defaultStep
            percentage += multiplier * (step != 0.0f ? step : defaultStep);
            if (percentage >= 1.0f)
                percentage = 1.0f;

            setValue(percentage);

            onChange();

            return 1;
        }

        virtual int onDecrease() override
        {
            const float multiplier = fineTuning ? 0.1f : 1.0f;

            // TODO: differentiate between real slider step and keyboard/gamepad defaultStep
            percentage -= multiplier * (step != 0.0f ? step : defaultStep);
            if (percentage <= 0.0f)
                percentage = 0.0f;

            setValue(percentage);

            onChange();

            return 1;
        }

        void calcPercentageFromMouseX(float mx)
        {
            percentage = mx / sizeX;
            if (percentage >= 1.0f)
                percentage = 1.0f;
            if (percentage <= 0.0f)
                percentage = 0.0f;

            if (!fineTuning && (std::abs(percentage - stickyValue) < stickyRegion))
                percentage = stickyValue;

            setValue(percentage);
            onChange();
        }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlSliderIntEffectTweak : public ControlSliderInt
    {
    public:

        AnselUIBase::DataType dataType;

        std::wstring filterId;
        int controlIdx = -1;
        int stackIdx = -1;

        ControlSliderIntEffectTweak(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        virtual void setValue(int val)
        {
            if (m_pReactor)
                (*m_pReactor)(this);
        }

        virtual int onIncrease() override
        {
            ControlSliderInt::onIncrease();
            setValue(calcIntFromSelected());
            return 1;
        }

        virtual int onDecrease() override
        {
            ControlSliderInt::onDecrease();
            setValue(calcIntFromSelected());
            return 1;
        }

        virtual void calcSelectedFromMouseX(float mx)
        {
            ControlSliderInt::calcSelectedFromMouseX(mx);
            setValue(calcIntFromSelected());
            onChange();
        }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlSliderListEffectTweak : public ControlSliderDiscr
    {
    public:

        AnselUIBase::DataType dataType;

        wchar_t * labelsMem = nullptr;
        std::vector<shadermod::ir::TypelessVariable> data;

        std::wstring filterId;
        int controlIdx = -1;
        int stackIdx = -1;

        shadermod::ir::TypelessVariable emptyData;

        ControlSliderListEffectTweak(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor),
            emptyData(0)
        {
        }

        virtual const shadermod::ir::TypelessVariable & getData(int idx)
        {
            if (idx >= 0 && idx < (int)data.size())
            {
                return data[idx];
            }
            return emptyData;
        }

        virtual void allocLabels(size_t totalNumCharacters)
        {
            labelsMem = (wchar_t *)malloc(totalNumCharacters * sizeof(wchar_t));
        }
        virtual void deallocLabels()
        {
            free(labelsMem);
            labelsMem = nullptr;
        }

        virtual void setValue()
        {
            if (m_pReactor)
                (*m_pReactor)(this);
        }

        virtual int onChange() override
        {
            ControlSliderDiscr::onChange();
            setValue();
            return 1;
        }

        ~ControlSliderListEffectTweak()
        {
            deallocLabels();
        }

    private:

        UIControlReactorFunctorInterface * m_pReactor = nullptr;
    };

    class ControlColorPickerEffectTweak : public ControlContainer
    {
    protected:
    public:

        class SliderTweak : public ControlSliderCont
        {
        public:

            ControlColorPickerEffectTweak * m_parentColorPicker = nullptr;

            void setParent(ControlColorPickerEffectTweak * parentColorPicker)
            {
                m_parentColorPicker = parentColorPicker;
            }

            virtual void setValue(float val)
            {
                if (m_parentColorPicker && m_parentColorPicker->m_pReactor)
                    (*(m_parentColorPicker->m_pReactor))(m_parentColorPicker);
            }

            virtual int onIncrease() override
            {
                const float multiplier = fineTuning ? 0.1f : 1.0f;

                percentage += multiplier * step;
                if (percentage >= 1.0f)
                    percentage = 1.0f;

                setValue(percentage);

                onChange();

                return 1;
            }

            virtual int onDecrease() override
            {
                const float multiplier = fineTuning ? 0.1f : 1.0f;

                percentage -= multiplier * step;
                if (percentage <= 0.0f)
                    percentage = 0.0f;

                setValue(percentage);

                onChange();

                return 1;
            }

            void calcPercentageFromMouseX(float mx)
            {
                percentage = mx / sizeX;
                if (percentage >= 1.0f)
                    percentage = 1.0f;
                if (percentage <= 0.0f)
                    percentage = 0.0f;

                if (!fineTuning && (std::abs(percentage - stickyValue) < stickyRegion))
                    percentage = stickyValue;

                setValue(percentage);
                onChange();
            }
        };

        friend class SliderTweak;

        AnselUIBase::DataType dataType;

        std::wstring filterId;
        int controlIdx = -1;
        int stackIdx = -1;

        ControlColorPickerEffectTweak(
            UIControlReactorFunctorInterface * pReactor,
            unsigned int numChannels
            ):
                m_numChannels(numChannels)
        {
            m_isClipping = false;

            m_pReactor = pReactor;
            for (uint32_t i = 0; i < 4; ++i)
                m_sliders[i] = nullptr;
        }

        void initSliders()
        {
            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->setParent(this);
            }

            SliderTweak * prevElement = nullptr;
            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->m_anchorX = nullptr;
                m_sliders[i]->m_anchorY = prevElement;

                m_sliders[i]->state = UI_CONTROL_ORDINARY;
                m_sliders[i]->isVisible = true;
                m_sliders[i]->isLeanStyle = true;

                addControl(m_sliders[i]);

                prevElement = m_sliders[i];
            }
        }

        virtual ControlType getType() const override { return ControlType::kColorPicker; }
        virtual bool isBasicContainer() const override { return true; }

        void setOffsets(float x, float y)
        {
            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->posX = x;
                m_sliders[i]->posY = y;
            }
        }

        void setBuffers(ID3D11Buffer * idxBuf, ID3D11Buffer * vtxBuf)
        {
            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->pIndexBuf = idxBuf;
                m_sliders[i]->pVertexBuf = vtxBuf;
            }
        }

        void setBlockID(int newBlockID)
        {
            blockID = newBlockID;

            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->blockID = newBlockID;
            }
        }

        void setTitle(const  wchar_t* str, size_t len = (size_t ) -1)
        {
            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->setTitle(str, len);
            }
        }

        void setSizes(float elementSizeX, float elementSizeY)
        {
            sizeX = m_sliders[0]->sizeX;
            sizeY = m_numChannels*m_sliders[0]->sizeY;

            for (unsigned int i = 0; i < m_numChannels; ++i)
            {
                m_sliders[i]->posX = 0.0f;
                m_sliders[i]->posY = 0.0f;
            }
        }

        uint32_t getNumExternalChildren() const override { return m_numChannels; }
        ControlContainer * getExternalChild(uint32_t idx) override
        {
            // This could potentially be replaced with simple container controls search
            //  but for increased flexibility, the explicit function exists
            if (idx >= m_numChannels)
                return nullptr;
            else
                return m_sliders[idx];
        }

        unsigned int m_numChannels;
        SliderTweak * m_sliders[4];

    private:

        UIControlReactorFunctorInterface * m_pReactor;
    };

    class ControlSliderSpecialFX : public ControlSliderDiscr
    {
    public:

        std::vector<std::wstring> filterIds;
        wchar_t lastSelectedFX[UI_STRINGBUFSIZE_FXNAME];

        ControlSliderSpecialFX(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
            lastSelectedFX[0] = L'\0';
        }

        bool needsChangeOnCWToggle()
        {
            return (labels.size() > size_t(m_selected)) ? (wcscmp(lastSelectedFX, labels[m_selected]) != 0) : true;
        }

        int onChange() override
        {
            if (m_pReactor)
                (*m_pReactor)(this);
            return 1;
        }

    private:
        UIControlReactorFunctorInterface* m_pReactor = nullptr;
    };

    class ControlSliderStyles : public ControlSliderDiscr
    {
    public:
        std::vector<std::wstring> styleIds;
        std::vector<std::wstring> stylePaths;
        wchar_t lastSelectedStyle[UI_STRINGBUFSIZE_FXNAME];

        ControlSliderStyles(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
            lastSelectedStyle[0] = L'\0';
        }

        bool needsChangeOnCWToggle()
        {
            return (labels.size() > size_t(m_selected)) ? (wcscmp(lastSelectedStyle, labels[m_selected]) != 0) : true;
        }

        int onChange() override
        {
            if (m_pReactor)
                (*m_pReactor)(this);
            return 1;
        }

    private:

        UIControlReactorFunctorInterface* m_pReactor = nullptr;
    };

    class ControlFlyoutStylesToggle : public ControlFlyoutToggleIndependent
    {
    public:

        std::vector<std::wstring> paths;

        ControlFlyoutStylesToggle(UIControlReactorFunctorInterface* pClickReactor, UIControlReactorFunctorInterface* pChangeReactor):
            ControlFlyoutToggleIndependent(pClickReactor, pChangeReactor)
        {
        }
    };

    class ControlSliderFOV : public ControlSliderCont
    {
    public:

        float getFOV()
        {
            return percentage * (m_maxFOV - m_minFOV) + m_minFOV;
        }
        void setFOV(float fov)
        {
            percentage = (fov - m_minFOV) / (m_maxFOV - m_minFOV);

            if (percentage < 0.0f)
                percentage = 0.0f;
            if (percentage > 1.0f)
                percentage = 1.0f;
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            swprintf_s(textBuf, bufSize, L"%.0f\u00B0", getFOV());
            return;
        }

        void setFOVLimits(double lo, double hi)
        {
            m_minFOV = (float)lo;
            m_maxFOV = (float)hi;
        }

    protected:

        float m_minFOV = 10.0f;
        float m_maxFOV = 140.0f;
    };

    namespace
    {
        const float kPi = 3.14159265358979323846f;
    }

    class ControlSliderRoll : public ControlSliderCont
    {
    public:

        ControlSliderRoll()
        {
            //step = 0.05f;
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            double roll = m_minRoll + percentage * (m_maxRoll - m_minRoll);
            if (std::abs(roll) < 1e-3f)
                roll = 0.0f;

            swprintf_s(textBuf, bufSize, L"%.0f\u00B0", (float)roll);
        }

        void setRollDegrees(float roll)
        {
            percentage = (float)( (roll - m_minRoll) / (m_maxRoll - m_minRoll) );

            if (percentage < 0.0f)
                percentage = 0.0f;
            if (percentage > 1.0f)
                percentage = 1.0f;
        }

        void setRollRangeDegrees(double lo, double hi)
        {
            m_minRoll = lo;
            m_maxRoll = hi;
        }

    protected:

        double m_minRoll = -180.0;
        double m_maxRoll =  180.0;
    };

    class ControlSliderKind : public ControlSliderDiscr
    {
    public:

        ControlSliderKind(UIControlReactorFunctorInterface* pReactor):
            m_pReactor(pReactor)
        {
        }

        void setupLabels(wchar_t * textKindRegular, wchar_t * textKindHighRes, wchar_t * textKind360, wchar_t * textKindStereo, wchar_t * textKind360Stereo)
        {
            labels.resize(UI_DIRCETORSTATE_TOTAL);
            labels[UI_DIRCETORSTATE_NONE] = textKindRegular;
            labels[UI_DIRCETORSTATE_360] =    textKind360;
            labels[UI_DIRCETORSTATE_HIGHRES] =  textKindHighRes;
#if (ENABLE_STEREO_SHOTS == 1)
            labels[UI_DIRCETORSTATE_STEREO] =  textKindStereo;
            labels[UI_DIRCETORSTATE_STEREO_360] =  textKind360Stereo;
#endif
        }

        void setShotPermissions(bool * permissions)
        {
            m_shotTypeEnabled[UI_DIRCETORSTATE_NONE] =          permissions[(int)ShotType::kRegular];
            m_shotTypeEnabled[UI_DIRCETORSTATE_360] =           permissions[(int)ShotType::k360];
            m_shotTypeEnabled[UI_DIRCETORSTATE_HIGHRES] =       permissions[(int)ShotType::kHighRes];
#if (ENABLE_STEREO_SHOTS == 1)
            m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO] =        permissions[(int)ShotType::kStereo];
            m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO_360] =    permissions[(int)ShotType::k360Stereo];
#endif

            bool anyShotTypeDisabled = !(m_shotTypeEnabled[UI_DIRCETORSTATE_NONE]
                                        && m_shotTypeEnabled[UI_DIRCETORSTATE_360]
                                        && m_shotTypeEnabled[UI_DIRCETORSTATE_HIGHRES]
#if (ENABLE_STEREO_SHOTS == 1)
                                        && m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO]
                                        && m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO_360]
#endif
                                        );
            if (anyShotTypeDisabled)
            {
                tickEnabled.resize(labels.size());
                tickEnabled[UI_DIRCETORSTATE_NONE] =        m_shotTypeEnabled[UI_DIRCETORSTATE_NONE];
                tickEnabled[UI_DIRCETORSTATE_360] =         m_shotTypeEnabled[UI_DIRCETORSTATE_360];
                tickEnabled[UI_DIRCETORSTATE_HIGHRES] =     m_shotTypeEnabled[UI_DIRCETORSTATE_HIGHRES];
#if (ENABLE_STEREO_SHOTS == 1)
                tickEnabled[UI_DIRCETORSTATE_STEREO] =      m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO];
                tickEnabled[UI_DIRCETORSTATE_STEREO_360] =  m_shotTypeEnabled[UI_DIRCETORSTATE_STEREO_360];
#endif
            }
            else
            {
                tickEnabled.resize(0);
            }
        }

        bool isShotTypeEnabled(int selected)
        {
            if (selected >= 0 && selected < (int)labels.size())
                return m_shotTypeEnabled[selected];
            return false;
        }

        int onChange() override
        {
            if (m_pReactor)
                (*m_pReactor)(this);

            return 1;
        }
    private:
        UIControlReactorFunctorInterface* m_pReactor = nullptr;
        bool m_shotTypeEnabled[UI_DIRCETORSTATE_TOTAL];
    };

    class ControlSlider360Quality : public ControlSliderCont
    {
    public:

        ControlSlider360Quality(ControlSliderKind* sldKind, AnselServer* pAnselServer, const wchar_t* textGB) :
            m_pKind(sldKind), m_pAnselServer(pAnselServer), m_textGB(textGB)
        {
        }

        virtual int onChange() override
        {
            uint64_t resolution360 = getResolution();

            using darkroom::ShotDescription;
            ShotDescription desc;
            if (m_pKind->getSelected() == UI_DIRCETORSTATE_STEREO_360)
                desc.type = ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA;
            else
                desc.type = ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA;
            desc.overlap = 1.5f;
            desc.bmpWidth = m_pAnselServer->getWidth();
            desc.bmpHeight = m_pAnselServer->getHeight();
            desc.horizontalFov = (float)darkroom::CameraDirector::estimateTileHorizontalFovSpherical(uint32_t(resolution360 * 2), uint32_t(desc.bmpWidth));

            darkroom::CaptureTaskEstimates estimates;
            estimates = darkroom::CameraDirector::estimateCaptureTask(desc);

            outputResolutionX = size_t(estimates.outputResolutionX) / 2;
            outputResolutionY = size_t(estimates.outputResolutionY) / 2;

            totalMemSizeGB = (estimates.inputDatasetSizeTotalInBytes + estimates.outputSizeInBytes) / 1024. / 1024. / 1024.;

            return ControlSliderCont::onChange();
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            const size_t strDimBufSize = 32;
            wchar_t strDimX[strDimBufSize];
            wchar_t strDimY[strDimBufSize];

            lwanselutils::buildSplitStringFromNumber(outputResolutionX, strDimX, strDimBufSize);
            lwanselutils::buildSplitStringFromNumber(outputResolutionY, strDimY, strDimBufSize);

            swprintf_s(textBuf, bufSize, L"%s \u00D7 %s\n%.1f%s", strDimX, strDimY, totalMemSizeGB, m_textGB);
        }

        uint64_t getResolution()
        {
            return m_sphericalResolutionLo + (uint64_t)((double)percentage * (m_sphericalResolutionHi - m_sphericalResolutionLo));
        }
        void setResolutionLimits(uint64_t lo, uint64_t hi)
        {
            m_sphericalResolutionLo = lo;
            m_sphericalResolutionHi = hi;
        }
        void getsetResolutionLimits(uint64_t & lo, uint64_t & hi)
        {
            lo = m_sphericalResolutionLo;
            hi = m_sphericalResolutionHi;
        }

    private:
        AnselServer* m_pAnselServer = nullptr;
        ControlSliderKind* m_pKind = nullptr;
        const wchar_t* m_textGB = nullptr;

        uint64_t outputResolutionX;
        uint64_t outputResolutionY;

        double totalMemSizeGB;

        uint64_t m_sphericalResolutionLo = 0;
        uint64_t m_sphericalResolutionHi = 0;
    };

}
