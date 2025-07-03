#pragma once

#include "ir/TypeEnums.h"
#include "ui/common.h"

//namespace shadermod
//{
namespace ui
{

    class ColorF4
    {
    public:

        float val[4];
        ColorF4()
        {
            val[0] = 1.0f;
            val[1] = 1.0f;
            val[2] = 1.0f;
            val[3] = 1.0f;
        }

        ColorF4(float cr, float cg, float cb, float ca)
        {
            val[0] = cr;
            val[1] = cg;
            val[2] = cb;
            val[3] = ca;
        }
    };

    enum class ControlType
    {
        kContainer          = (1<<0),
        kButton             = (1<<1),
        kSliderCont         = (1<<2),
        kSliderDiscr        = (1<<3),
        kIcon               = (1<<4),
        kCheckbox           = (1<<5),
        kLabel              = (1<<6),
        kColorPicker        = (1<<7),
        kSliderInt          = (1<<8),
        kProgressBar        = (1<<9),

        kALL      = kContainer | kButton | kSliderCont | kSliderDiscr | kIcon | kCheckbox | kLabel | kColorPicker | kSliderInt | kProgressBar
    };

    class ControlBase
    {
    public:

        float sizeX = FLT_MAX, sizeY = FLT_MAX;
        float posX = FLT_MAX, posY = FLT_MAX;    // relative
        float absPosX = FLT_MAX, absPosY = FLT_MAX;  // absolute

        float sizeXMax = FLT_MAX, sizeYMax = FLT_MAX;  // For auto-size containers

        int state;
        int tabStop = 0;

        int blockID = 0;    // For dynamic controls, for faster indexing
        bool isStatic = true;  // Whether control should survive dynamic deletions within parent container

        enum class FontAlignment
        {
            kLeft,
            kCenter,
            kRight,

            kNUM_ENTRIES
        };
        FontAlignment fontAlignment = FontAlignment::kCenter;

        ColorF4 color;
        ColorF4 fontColor = ColorF4(1.0f, 1.0f, 1.0f, 1.0f);
        ColorF4 hlColor = ColorF4(1.0f, 1.0f, 1.0f, 1.0f);

        bool isVisible = true;

        bool isEnabled = true;

        bool isInteractible = true;

        virtual bool isInteractive()  { return isInteractible && isVisible && isEnabled; }

        virtual ControlType getType() const = 0;

        virtual int onClick() { return 0; };
        virtual int onChange() { return 0; };
        virtual int onKeyPress(DWORD vkey) { return 0; };
        virtual int onIncrease() { return 0; };
        virtual int onDecrease() { return 0; };

        virtual int onMouseMove(float mx, float my, float dmz) { return 0; }
        virtual int onMouseDown(float mx, float my) { return 0; }
        virtual int onMouseUp(float mx, float my) { return 0; }

        virtual int getPreferredLwrsor() { return UI_MOUSELWRSOR_DONTCARE; }

        virtual ~ControlBase() {};

    protected:


    };

    class ControlContainer : public ControlBase
    {
    public:

#ifdef _DEBUG
        wchar_t * m_DBGname = nullptr;
#endif

        ControlContainer * m_parent = nullptr;
        ControlContainer * m_anchorX = nullptr;
        ControlContainer * m_anchorY = nullptr;

        bool m_renderSideVLine = false;
        float m_renderSideVLineMargin = 0.0f;

        float m_renderingMarginX = 0.0f;
        float m_renderingMarginY = 0.0f;

        bool m_isMouseOver = false;
        bool m_isMouseOverScrollbar = false;

        bool m_isDragScrolling = false;
        bool m_isBarScrolling = false;
        float m_scrollValueY = 0.0f;
        float m_absPosYScroll = 0.0f;

        float m_scrollRegionRenderWidth = 1.0f;
        float m_scrollRegionWidth = 1.0f;

        bool m_isClipping = true;
        bool m_isScrollable = false;
        float m_scrollContentSize = 0.0f;
        float m_scrollMarginBottom = 0.0f;

        float m_prevMouseX = 0.0f;
        float m_prevMouseY = 0.0f;

        int m_selectedControlIdx = -1;

        int m_dynamicControlType = -1;

        bool m_needsContentAutoPlacement = false;
        ControlContainer * m_autoPlacementElement = nullptr;

        virtual ControlType getType() const override { return ControlType::kContainer; };

        float getScrollThumbPercentage()
        {
            return sizeY / m_scrollContentSize;
        }
        float getScrollThumbSize()
        {
            return sizeY * getScrollThumbPercentage();
        }
        float getScrollThumbOffset()
        {
            return m_scrollValueY / (m_scrollContentSize - sizeY) * (sizeY - getScrollThumbSize());
        }

        ControlContainer * addControl(ControlContainer * control)
        {
            m_controls.push_back(control);
            control->m_parent = this;
            return control;
        }

        ControlContainer * removeControlFast(ControlContainer * control)
        {
            for (int uiCnt = 0; uiCnt < (int)m_controls.size(); ++uiCnt)
            {
                if (m_controls[uiCnt] != control)
                    continue;

                if (uiCnt != m_controls.size() - 1)
                {
                    std::swap(m_controls[uiCnt], m_controls.back());
                }
                m_controls.pop_back();

                return control;
            }
            return nullptr;
        }

        size_t getControlsNum() const
        {
            return m_controls.size();
        }

        ControlContainer * getSelectedControl()
        {
            assert(m_selectedControlIdx >= 0 && m_selectedControlIdx < int(getControlsNum()));
            return m_controls[m_selectedControlIdx];
        }

        ControlContainer * getControl(int idx)
        {
            assert(idx >= 0 && idx < int(getControlsNum()));
            return m_controls[idx];
        }

        std::vector<ControlContainer *> & getControlsRaw()
        {
            return m_controls;
        }

        virtual bool isBasicContainer() const
        {
            return getType() == ControlType::kContainer;
        }
        bool isChildBasicContainer(int idx) const
        {
            assert(idx >= 0 && idx < int(getControlsNum()));
            return m_controls[idx]->isBasicContainer();
        }

        float getScrollMaxValue()
        {
            return m_scrollContentSize - sizeY;
        }
        void clampScrollValue()
        {
            if (m_scrollValueY < 0.0f)
                m_scrollValueY = 0.0f;

            float scrollMaxValue = getScrollMaxValue();
            if (scrollMaxValue > 0.0f)
            {
                if (m_scrollValueY > scrollMaxValue)
                    m_scrollValueY = scrollMaxValue;
            }
            else
            {
                m_scrollValueY = 0.0f;
            }
        }

        bool isOverVertScrollbarRegion(float mx)
        {
            return (mx > sizeX - m_scrollRegionWidth);
        }
        bool isOverVertScrollbar(float mx, float my)
        {
            const float vertScrollBarThumbOffset = sizeY - getScrollThumbOffset() - getScrollThumbSize();
            return isOverVertScrollbarRegion(mx) && ((my > vertScrollBarThumbOffset) && (my < vertScrollBarThumbOffset + getScrollThumbSize()));
        }

        int isMouseOver(float mx, float my)
        {
            m_isMouseOver = false;
            m_isMouseOverScrollbar = false;
            if (mx > 0 && mx < sizeX &&
                my > 0 && my < sizeY)
            {
                m_isMouseOver = true;

                if (isOverVertScrollbarRegion(mx))
                    m_isMouseOverScrollbar = true;

                return 1;
            }
            else
            {
                return 0;
            }
        }

        virtual int onMouseMove(float mx, float my, float dmz) override
        {
            isMouseOver(mx, my);

            bool isWheelScrolling = m_isMouseOver && (fabs(dmz) > FLT_EPSILON);

            if (m_isDragScrolling || m_isBarScrolling || isWheelScrolling)
            {
                if (m_isDragScrolling)
                {
                    const float multiplier = 1.0f;
                    //m_scrollValueX += multiplier * (mx - m_prevMouseX);
                    m_scrollValueY += multiplier * (my - m_prevMouseY);
                }
                else if (m_isBarScrolling)
                {
                    // Since content size in pixels could be far greater than scrollable space in pixels,
                    //  we need to adjust the multiplier to match scrolling speed
                    const float multiplier = getScrollMaxValue() / (sizeY - getScrollThumbSize());
                    //m_scrollValueX -= multiplier * (mx - m_prevMouseX);
                    m_scrollValueY -= multiplier * (my - m_prevMouseY);
                }
                else if (isWheelScrolling)
                {
                    const float wheelScrollingSensitivity = 0.05f * (1.0f / 120.0f); //120 is how much dz is for a single click of the wheel. See WHEEL_DELTA
                    m_scrollValueY -= wheelScrollingSensitivity * dmz;
                }

                clampScrollValue();

                m_prevMouseX = mx;
                m_prevMouseY = my;

                return 0;
            }

            return 0;
        }
        virtual int onMouseDown(float mx, float my) override
        {
            if (!isEnabled)
                return 0;

            if (isMouseOver(mx, my) != 0)
            {
                if (m_isScrollable)
                {
                    if (isOverVertScrollbarRegion(mx))
                    {
                        if (!isOverVertScrollbar(mx, my))
                        {
                            // If mouse didn't hit the scroll bar thumb,
                            //  we need to jump the thumb here first

                            // We callwlate thumbOffset as if thumb center is in the point of clicking
                            const float thumbSize = getScrollThumbSize();
                            const float thumbOffset = (sizeY - my) - 0.5f * thumbSize;
                            m_scrollValueY = thumbOffset * (m_scrollContentSize - sizeY) / (sizeY - thumbSize);

                            clampScrollValue();
                        }
                        m_isBarScrolling = true;
                    }
                    else
                    {
                        m_isDragScrolling = true;
                    }

                    m_prevMouseX = mx;
                    m_prevMouseY = my;

                    return 1;
                }
            }
            return 0;
        }
        virtual int onMouseUp(float mx, float my) override
        {
            if (m_isDragScrolling || m_isBarScrolling)
            {
                m_isDragScrolling = false;
                m_isBarScrolling = false;
                return 1;
            }
            return 0;
        }

        virtual uint32_t getNumExternalChildren() const { return 0; }
        virtual ControlContainer * getExternalChild(uint32_t idx) { return nullptr; }

    protected:

        std::vector<ControlContainer *> m_controls;
    };

    class ControlLabel : public ControlContainer
    {
    public:

        bool isBold = false;
        wchar_t * caption;

        virtual ControlType getType() const override { return ControlType::kLabel; };

        ControlLabel()
        {
            isInteractible = false;
        }

    protected:

    };

    class ControlProgressBar : public ControlContainer
    {
    public:

        float progress = 0.0f;

        virtual ControlType getType() const override { return ControlType::kProgressBar; };

        ControlProgressBar()
        {
            isInteractible = false;
        }

    protected:

    };

    class ControlButton : public ControlContainer
    {
    public:

        bool renderBufsShared = false;  // Idx/vtx buffers, owned by default
        bool renderBufsAuxShared = true;// Vtx HL/DN buffers, they are shared by default
        ID3D11Buffer * pIndexBuf = nullptr;
        ID3D11Buffer * pVertexBuf = nullptr;  // Ordinary
        ID3D11Buffer * pVertexBufHl = nullptr;  // Highlighted
        ID3D11Buffer * pVertexBufDn = nullptr;  // Pressed

        enum class RenderType
        {
            kBasic,
            kToggle,
            kFlyoutToggle,
            kSelector,
            kNUM_ENTRIES
        };
        RenderType renderType = RenderType::kBasic;

        enum class HighlightType
        {
            kRectangle,
            kFont,
            kNUM_ENTRIES
        };
        HighlightType hlType = HighlightType::kRectangle;

        bool isPressed = false;
        bool isBold = false;
        bool needsAutosize = false;

        const wchar_t * caption;

        virtual ControlType getType() const override { return ControlType::kButton; };

        virtual int onKeyPress(DWORD vkey) override
        {
            if (!isEnabled)
                return 0;

            if (vkey == VK_RIGHT || vkey == VK_SPACE)
            {
                return onClick();
            }

            return 0;
        }

        int isMouseOnButton(float mx, float my)
        {
            if (mx > 0 && mx < sizeX &&
                my > 0 && my < sizeY)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        virtual int onMouseMove(float mx, float my, float) override
        {
            return isMouseOnButton(mx, my);
        }
        virtual int onMouseDown(float mx, float my) override
        {
            if (!isEnabled)
                return 0;

            if (isMouseOnButton(mx, my) != 0)
            {
                isPressed = true;
                return 1;
            }
            return 0;
        }
        virtual int onMouseUp(float mx, float my) override
        {
            if (isPressed && (isMouseOnButton(mx, my) != 0))
            {
                isPressed = false;
                onClick();
                return 1;
            }
            return 0;
        }

    protected:

    };

    class ControlCheckbox : public ControlContainer
    {
    public:

        enum class RenderType
        {
            kBasic,
            kNUM_ENTRIES
        };

        RenderType renderType = RenderType::kBasic;
        bool isBold = false;

        float checkSize = 0.0f;

        bool isPressed = false;
        bool isChecked = false;

        static const unsigned int maxTitleLength = 64;
        wchar_t title[maxTitleLength + 1];

        void setTitle(const  wchar_t* str, size_t len = (size_t ) -1)
        {
            if (len == (size_t) -1)
                len = wcslen(str);

            if (len > maxTitleLength)
                len = maxTitleLength;

            memcpy(title, str, sizeof(wchar_t) * len);
            title[len] = L'\0';
        }


        virtual ControlType getType() const override { return ControlType::kCheckbox; };

        virtual int getPreferredLwrsor() { return UI_MOUSELWRSOR_HAND; }

        virtual int onClick()
        {
            isChecked = !isChecked;
            return 1;
        };

        virtual int onIncrease()
        {
            isChecked = !isChecked;
            return 1;
        };

        virtual int onDecrease()
        {
            isChecked = !isChecked;
            return 1;
        };

        virtual int onKeyPress(DWORD vkey) override
        {
            if (!isEnabled)
                return 0;

            if (vkey == VK_RIGHT || vkey == VK_SPACE)
            {
                return onClick();
            }

            return 0;
        }

        int isMouseOnCheckbox(float mx, float my)
        {
            if (mx > 0 && mx < sizeX &&
                my > 0 && my < sizeY)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        virtual int onMouseMove(float mx, float my, float) override
        {
            return isMouseOnCheckbox(mx, my);
        }
        virtual int onMouseDown(float mx, float my) override
        {
            if (!isEnabled)
                return 0;

            if (isMouseOnCheckbox(mx, my) != 0)
            {
                isPressed = true;
                return 1;
            }
            return 0;
        }
        virtual int onMouseUp(float mx, float my) override
        {
            bool wasPressed = isPressed;
            isPressed = false;
            if (wasPressed && (isMouseOnCheckbox(mx, my) != 0))
            {
                onClick();
                return 1;
            }
            return 0;
        }

    protected:

    };

    class ControlSliderBase : public ControlContainer
    {
    public:

        ControlSliderBase()
        {
            m_isClipping = false;
        }

        ID3D11Buffer * pVertexBuf;    // Ordinary
        ID3D11Buffer * pIndexBuf;

        float thumbSizeX, thumbSizeY;
        float thumbBorderX, thumbBorderY;
        float trackShiftY, trackSizeY;

        //feodorb TODO: the UI shouldn't work with the mem owned by the IR or wnything else external to UI, and this is especially obvious with the
        //localization code. So I changed the pointer to an array.
        //This is somehwta ugly, as it is doesn't go in line with the allocateUiString mechanism, but unfortunately the latter one doesn't have
        //a way to reallocate, so it doesn't suit to the dynamic sliders at all, which can be recrearted multiple times as the user switches effects.
        //On the otehr hand, the allocateUiString always allocates a fixed-length string, so lwrrently there's no need for anything more complex than the
        //array. Either way, this requires some cleanup.
        static const unsigned int maxTitleLength = 64;
        wchar_t title[maxTitleLength + 1];

        bool isMinMaxLabeled = false;
        bool isLeanStyle = true;
        wchar_t minSubTitle[16];
        wchar_t maxSubTitle[16];

        bool isDragged = false;

        virtual void getThumbPosition(float & thumbPosX, float & thumbPosY) const { }

        virtual void getText(wchar_t * textBuf, size_t bufSize) { return; }

        void setMinMaxText(wchar_t * minTextBuf, wchar_t * maxTextBuf)
        {
            swprintf_s(minSubTitle, 16, L"%s", minTextBuf);
            swprintf_s(maxSubTitle, 16, L"%s", maxTextBuf);
        }

        void setTitle(const  wchar_t* str, size_t len = (size_t ) -1)
        {
            if (len == (size_t) -1)
                len = wcslen(str);

            if (len > maxTitleLength)
                len = maxTitleLength;

            memcpy(title, str, sizeof(wchar_t) * len);
            title[len] = L'\0';
        }

        virtual int onKeyPress(DWORD vkey) override
        {
            if (vkey == VK_RIGHT)
            {
                onIncrease();
                return 1;
            }
            else if (vkey == VK_LEFT)
            {
                onDecrease();
                return 1;
            }

            return 0;
        }

        virtual int onMouseMove(float mx, float my, float) override
        {
            float thumbHalfSizeX = 0.5f * thumbSizeX;
            float thumbHalfSizeY = 0.5f * thumbSizeY;
            if (mx > -thumbHalfSizeX && mx < sizeX + thumbHalfSizeX &&
                my > trackShiftY - thumbHalfSizeY && my < trackShiftY + trackSizeY + thumbHalfSizeY)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        virtual int onMouseDown(float mx, float my) override
        {
            float thumbHalfSizeX = 0.5f * thumbSizeX;
            float thumbHalfSizeY = 0.5f * thumbSizeY;

            float thumbCenterX;
            float thumbCenterY;
            getThumbPosition(thumbCenterX, thumbCenterY);

            if (mx > thumbCenterX - thumbHalfSizeX && mx < thumbCenterX + thumbHalfSizeX &&
                my > thumbCenterY - thumbHalfSizeY && my < thumbCenterY + thumbHalfSizeY)
            {
                isDragged = true;
                return 1;
            }
            else
            {
                return 0;
            }
        }
        virtual int onMouseUp(float mx, float my) override
        {
            if (isDragged)
            {
                isDragged = false;
                return 1;
            }
            return 0;
        }

    protected:

    };

    static float colwertPercentageToValue(float percentage, float baseValue, float valueMin, float valueMax)
    {
        float colwPercentage = (percentage < baseValue) ? (0.5f * percentage / baseValue) : (0.5f * (percentage - baseValue) / (1.0f - baseValue) + 0.5f);
        return colwPercentage * (valueMax - valueMin) + valueMin;
    }
    static float colwertValueToPercentage(float value, float baseValue, float valueMin, float valueMax)
    {
        float colwValue = (value - valueMin) / (valueMax - valueMin);
        return (colwValue > 0.5f) ? ((colwValue - 0.5f) / 0.5f * (1.0f - baseValue) + baseValue) : colwValue / 0.5f * baseValue;
    }

    class ControlSliderCont : public ControlSliderBase
    {
    public:

        bool fineTuning = false;
        float percentage = 0.0f;
        float step = 0.1f;
        
        float baseValue = 0.0f;      // Sliders have 3 possible ranges: -100%..0%, -100%..100% (continuous, variable) and 0..100%

        float defaultValue = -1.0f;

        float stickyValue = -1.0f;
        float stickyRegion = 0.0f;

        float milwalue = 0.0f, maxValue = 1.0f;    // range for the shader constant
        float uiMilwalue = 0.0f, uiMaxValue = 1.0f;  // range for the UI display

        std::wstring uiMeasurementUnit;        // what to add post-number on the UI display , e.g. 50% - '%' is a unit

        virtual ControlType getType() const override { return ControlType::kSliderCont; };

        virtual void getThumbPosition(float & thumbPosX, float & thumbPosY) const override
        {
            thumbPosX = percentage * sizeX;
            thumbPosY = trackShiftY + 0.5f * trackSizeY;
        }

        virtual int onIncrease() override
        {
            const float multiplier = fineTuning ? 0.1f : 1.0f;
            percentage += multiplier * step;

            if (percentage >= 1.0f)
                percentage = 1.0f;

            onChange();

            return 1;
        }

        virtual int onDecrease() override
        {
            const float multiplier = fineTuning ? 0.1f : 1.0f;
            percentage -= multiplier * step;

            if (percentage <= 0.0f)
                percentage = 0.0f;

            onChange();

            return 1;
        }

        virtual void calcPercentageFromMouseX(float mx)
        {
            percentage = mx / sizeX;
            if (percentage >= 1.0f)
                percentage = 1.0f;
            if (percentage <= 0.0f)
                percentage = 0.0f;

            if (!fineTuning && (std::abs(percentage - stickyValue) < stickyRegion))
                percentage = stickyValue;

            onChange();
        }

        virtual int onMouseMove(float mx, float my, float dmz) override
        {
            if (isDragged)
            {
                const float multiplier = fineTuning ? 0.1f : 1.0f;
                aclwmMouseX += multiplier * (mx - prevMouseX);
                aclwmMouseY += multiplier * (my - prevMouseY);

                calcPercentageFromMouseX(aclwmMouseX);

                prevMouseX = mx;
                prevMouseY = my;

                return 1;
            }

            return ControlSliderBase::onMouseMove(mx, my, dmz);
        }
        virtual int onMouseDown(float mx, float my) override
        {
            int baseMouseDown = ControlSliderBase::onMouseDown(mx, my);
            if (baseMouseDown)
            {
                aclwmMouseX = mx;
                aclwmMouseY = my;
                prevMouseX = mx;
                prevMouseY = my;

                return baseMouseDown;
            }

            float thumbHalfSizeX = 0.5f * thumbSizeX;
            float thumbHalfSizeY = 0.5f * thumbSizeY;
            if (mx > 0 && mx < sizeX &&
                my > trackShiftY - thumbHalfSizeY && my < trackShiftY + trackSizeY + thumbHalfSizeY)
            {
                aclwmMouseX = mx;
                aclwmMouseY = my;
                prevMouseX = mx;
                prevMouseY = my;

                calcPercentageFromMouseX(mx);
                isDragged = true;
                return 1;
            }
            else
            {
                return 0;
            }
        }
        virtual int onMouseUp(float mx, float my) override
        {
            aclwmMouseX = 0.0f;
            aclwmMouseY = 0.0f;
            return ControlSliderBase::onMouseUp(mx, my);
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            // Selecting number of digits after decimal
            float absRange = fabsf(uiMaxValue - uiMilwalue);
            int precision = getPrecisionFromRange(absRange);

            wchar_t * decimalFormat;
            if (precision == 0)
            {
                decimalFormat = L"%.0f%s";
            }
            else if (precision == 1)
            {
                decimalFormat = L"%.1f%s";
            }
            else
            {
                decimalFormat = L"%.2f%s";
            }
            swprintf_s(textBuf, bufSize, decimalFormat, colwertPercentageToValue(percentage, baseValue, uiMilwalue, uiMaxValue), uiMeasurementUnit.c_str());

            return;
        }

        virtual int onChange() override
        {
            isChanged = true;
            return 0;
        }

        // Use this method to process changes from last time this method was called.
        // Will return true if the state of the control has changed since - otherwise false
        // and nothing needs to happen. This method resets the internal change tracking.
        bool processChange()
        {
            bool ret = isChanged;
            isChanged = false;
            return ret;
        }

    protected:

        float aclwmMouseX = 0.0f, aclwmMouseY = 0.0f;
        float prevMouseX = 0.0f, prevMouseY = 0.0f;
        bool isChanged = false;

    };

    class ControlSliderInt : public ControlSliderBase
    {
    public:

        ControlSliderInt():
            m_selected(0),
            m_prevSelected(0)
        {
        }

        int m_milwal = 0;
        int m_maxVal = 100;
        int m_step = 1;
        int defaultValue = 0;

        std::wstring uiMeasurementUnit;        // what to add post-number on the UI display , e.g. 50% - '%' is a unit

        virtual ControlType getType() const override { return ControlType::kSliderInt; };

        virtual int getPrevSelected() const { return m_prevSelected; }
        virtual int getSelected() const { return m_selected; }
        virtual void setSelectedRaw(int selected)
        {
            m_prevSelected = m_selected;
            m_selected = selected;
        }
        virtual bool setSelected(int selected)
        {
            bool newSelectedValid = true;
            int totalNumTicks = (int)getTotalNumTicks();
            if (selected < 0 || selected >= totalNumTicks)
                newSelectedValid = false;

            if (newSelectedValid)
            {
                setSelectedRaw(selected);
                onChange();
            }

            return newSelectedValid;
        }

        virtual void getThumbPosition(float & thumbPosX, float & thumbPosY) const override
        {
            size_t totalNumTicks = getTotalNumTicks();
            if (totalNumTicks > 1)
            {
                thumbPosX = m_selected / (float)(totalNumTicks - 1) * sizeX;
            }
            else
            {
                thumbPosX = 0.0f;
            }
            thumbPosY = trackShiftY + 0.5f * trackSizeY;
        }

        virtual int onIncrease() override
        {
            m_prevSelected = m_selected;

            int totalNumTicks = (int)getTotalNumTicks();
            int maxVal = getMaxValue();

            ++m_selected;
            if (m_selected > maxVal)
                m_selected = maxVal;

            onChange();

            return 1;
        }
        virtual int onDecrease() override
        {
            m_prevSelected = m_selected;

            int totalNumTicks = (int)getTotalNumTicks();
            int milwal = getMilwalue();

            --m_selected;
            if (m_selected < milwal)
                m_selected = milwal;

            onChange();

            return 1;
        }

        virtual int getMilwalue() const
        {
            return m_milwal < m_maxVal ? m_milwal : m_maxVal;
        }
        virtual int getMaxValue() const
        {
            return m_maxVal > m_milwal ? m_maxVal : m_milwal;
        }
        virtual size_t getTotalNumTicks() const
        {
            return (size_t)((getMaxValue() - getMilwalue()) / m_step) + 1;
        }

        virtual void calcSelectedFromMouseX(float mx)
        {
            size_t totalNumTicks = getTotalNumTicks();
            if (totalNumTicks > 1)
            {
                int newSelected = (int)( (mx / sizeX) * (int)(totalNumTicks - 1) + 0.5f );
                if (newSelected >= (int)totalNumTicks)
                    newSelected = (int)totalNumTicks - 1;
                if (newSelected < 0)
                    newSelected = 0;

                if (newSelected != m_selected)
                {
                    m_prevSelected = m_selected;
                    m_selected = newSelected;
                    onChange();
                }
            }
            else
            {
                m_prevSelected = m_selected;
                m_selected = 0;
            }
        }

        virtual int onMouseMove(float mx, float my, float dmz) override
        {
            if (isDragged)
            {
                calcSelectedFromMouseX(mx);
                return 1;
            }

            return ControlSliderBase::onMouseMove(mx, my, dmz);
        }
        virtual int onMouseDown(float mx, float my) override
        {
            int baseMouseDown = ControlSliderBase::onMouseDown(mx, my);
            if (baseMouseDown)
                return baseMouseDown;

            float thumbHalfSizeY = 0.5f * thumbSizeY;
            if (mx > 0 && mx < sizeX &&
                my > trackShiftY - thumbHalfSizeY && my < trackShiftY + trackSizeY + thumbHalfSizeY)
            {
                calcSelectedFromMouseX(mx);
                isDragged = true;
                return 1;
            }
            else
            {
                return 0;
            }
        }

        virtual int calcIntFromSelected() const
        {
            size_t totalNumTicks = getTotalNumTicks();
            int milwal = getMilwalue();
            if (totalNumTicks > 0)
            {
                return m_selected * m_step + milwal;
            }
            else
            {
                return milwal;
            }
        }

        virtual void clampSelected()
        {
            int totalNumTicks = (int)getTotalNumTicks();
        }

        virtual void calcSelectedFromInt(int intVal)
        {
            int totalNumTicks = (int)getTotalNumTicks();
            int milwal = getMilwalue();
            if (totalNumTicks > 0)
            {
                m_selected = (intVal - milwal) / m_step;

                if (m_selected < 0)
                    m_selected = 0;
                else if (m_selected >= totalNumTicks)
                {
                    m_selected = totalNumTicks - 1;
                }
            }
            else
            {
                m_selected = 0;
            }
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            swprintf_s(textBuf, bufSize, L"%d%s", calcIntFromSelected(), uiMeasurementUnit.c_str());
            return;
        }

    protected:

        int m_prevSelected;
        int m_selected;

    };

    class ControlSliderDiscr : public ControlSliderBase
    {
    public:

        ControlSliderDiscr():
            m_selected(0),
            m_prevSelected(0)
        {
        }

        std::vector<wchar_t*> labels;
        std::vector<bool> tickEnabled;

        virtual int getPrevSelected() const { return m_prevSelected; }
        virtual int getSelected() const { return m_selected; }
        virtual void setSelectedRaw(int selected)
        {
            m_prevSelected = m_selected;
            m_selected = selected;
        }
        virtual bool setSelected(int selected)
        {
            bool newSelectedValid = true;
            if (selected < 0 || selected >= (int)labels.size())
                newSelectedValid = false;
            else if (tickEnabled.size() > 0 && !tickEnabled[selected])
                newSelectedValid = false;

            if (newSelectedValid)
            {
                setSelectedRaw(selected);
                onChange();
            }

            return newSelectedValid;
        }
        virtual ControlType getType() const override { return ControlType::kSliderDiscr; };

        virtual void getThumbPosition(float & thumbPosX, float & thumbPosY) const override
        {
            size_t totalNumTicks = getTotalNumTicks();
            int selectedUnmapped;
            if (totalNumTicks != labels.size())
            {
                selectedUnmapped = 0;
                for (int i = 0; i < m_selected; ++i)
                {
                    if (tickEnabled[i])
                    {
                        ++selectedUnmapped;
                    }
                }
            }
            else
            {
                selectedUnmapped = m_selected;
            }

            if (totalNumTicks > 1)
            {
                thumbPosX = selectedUnmapped / (float)(totalNumTicks - 1) * sizeX;
            }
            else
            {
                thumbPosX = 0.0f;
            }
            thumbPosY = trackShiftY + 0.5f * trackSizeY;
        }

        virtual int onIncrease() override
        {
            m_prevSelected = m_selected;

            int totalNumTicks = (int)getTotalNumTicks();
            if (totalNumTicks == (int)labels.size())
            {
                ++m_selected;
                if (m_selected >= (int)labels.size())
                    m_selected = (int)labels.size() - 1;
            }
            else
            {
                int numRepetitions = 0, totalNumLabels = (int)labels.size();
                do
                {
                    ++m_selected;

                    if (m_selected >= (int)labels.size())
                    {
                        // In case we're hitting the limit, we roll back to latest enabled
                        // we need this to be robust in case we started from the disabled position
                        for (m_selected = UI_DIRCETORSTATE_TOTAL - 1; m_selected > 0; --m_selected)
                        {
                            if (tickEnabled[m_selected])
                                break;
                        }
                        break;
                    }
                } while (!tickEnabled[m_selected] && numRepetitions <= totalNumLabels);
            }

            onChange();

            return 1;
        }
        virtual int onDecrease() override
        {
            m_prevSelected = m_selected;

            int totalNumTicks = (int)getTotalNumTicks();
            if (totalNumTicks == (int)labels.size())
            {
                --m_selected;
                if (m_selected < 0)
                    m_selected = 0;
            }
            else
            {
                int numRepetitions = 0, totalNumLabels = (int)labels.size();
                do
                {
                    --m_selected;

                    if (m_selected < 0)
                    {
                        // In case we're hitting the limit, we roll back to latest enabled
                        // we need this to be robust in case we started from the disabled position
                        int m_selectedEnd = (int)labels.size();
                        for (m_selected = 0; m_selected < m_selectedEnd; ++m_selected)
                        {
                            if (tickEnabled[m_selected])
                                break;
                        }
                        break;
                    }
                } while (!tickEnabled[m_selected] && numRepetitions <= totalNumLabels);
            }

            onChange();

            return 1;
        }

        size_t getTotalNumTicks() const
        {
            size_t totalNumTicks = 0;
            if (tickEnabled.size() > 0)
            {
                for (size_t i = 0, iend = tickEnabled.size(); i < iend; ++i)
                {
                    if (tickEnabled[i])
                        ++totalNumTicks;
                }
            }
            else
            {
                totalNumTicks = labels.size();
            }
            return totalNumTicks;
        }

        void calcSelectedFromMouseX(float mx)
        {
            size_t totalNumTicks = getTotalNumTicks();

            if (totalNumTicks > 1)
            {
                int newSelected = (int)( (mx / sizeX) * (int)(totalNumTicks - 1) + 0.5f );
                if (newSelected >= (int)totalNumTicks)
                    newSelected = (int)totalNumTicks - 1;
                if (newSelected < 0)
                    newSelected = 0;

                // Remap to the total amount of labels
                if (totalNumTicks != labels.size())
                {
                    int numSelectedUnmapped = newSelected, numSelectedUnmappedCounter = 0;
                    int numLabels = (int)tickEnabled.size();
                    newSelected = 0;
                    for (newSelected = 0; newSelected < numLabels - 1; ++newSelected)
                    {
                        if (tickEnabled[newSelected])
                        {
                            if (numSelectedUnmappedCounter == numSelectedUnmapped)
                                break;
                            ++numSelectedUnmappedCounter;
                        }
                    }
                }

                if (newSelected != m_selected)
                {
                    m_prevSelected = m_selected;
                    m_selected = newSelected;
                    onChange();
                }
            }
            else
            {
                m_prevSelected = m_selected;
                m_selected = 0;
            }
        }

        virtual int onMouseMove(float mx, float my, float dmz) override
        {
            if (isDragged)
            {
                calcSelectedFromMouseX(mx);
                return 1;
            }

            return ControlSliderBase::onMouseMove(mx, my, dmz);
        }
        virtual int onMouseDown(float mx, float my) override
        {
            int baseMouseDown = ControlSliderBase::onMouseDown(mx, my);
            if (baseMouseDown)
                return baseMouseDown;

            float thumbHalfSizeY = 0.5f * thumbSizeY;
            if (mx > 0 && mx < sizeX &&
                my > trackShiftY - thumbHalfSizeY && my < trackShiftY + trackSizeY + thumbHalfSizeY)
            {
                calcSelectedFromMouseX(mx);
                isDragged = true;
                return 1;
            }
            else
            {
                return 0;
            }
        }

        virtual void getText(wchar_t * textBuf, size_t bufSize) override
        {
            swprintf_s(textBuf, bufSize, L"%s", labels.size() ? labels[m_selected] : L"");
            return;
        }

    protected:

        int m_prevSelected;
        int m_selected;

    };

    class ControlIcon : public ControlContainer
    {
    public:

        ControlIcon()
        {
            isInteractible = false;
        }

        int maxPixelWidth = -1;
        int maxPixelHeight = -1;

        void resize(int screenWidth, int screenHeight)
        {
            if (maxPixelWidth != -1 || maxPixelHeight != -1)
            {
                float maxNormalizedWidth = maxPixelWidth / (float)screenWidth * 2.f;
                float maxNormalizedHeight = maxPixelHeight / (float)screenHeight * 2.f;

                posX = m_basePosX;
                posY = m_basePosY;

                sizeX = m_baseSizeX;
                sizeY = m_baseSizeY;

                float scaleWidth = m_baseSizeX / maxNormalizedWidth;
                float scaleHeight = m_baseSizeY / maxNormalizedHeight;

                if (scaleWidth > scaleHeight)
                {
                    if (maxPixelWidth > 0 && m_baseSizeX > maxNormalizedWidth)
                    {
                        sizeX = maxNormalizedWidth;
                        sizeY = m_baseSizeY / scaleWidth;
                    }
                }
                else
                {
                    if (maxPixelHeight > 0 && m_baseSizeY > maxNormalizedHeight)
                    {
                        sizeX = m_baseSizeX / scaleHeight;
                        sizeY = maxNormalizedHeight;
                    }
                }

                posX += 0.5f * (m_baseSizeX - sizeX);
                posY += 0.5f * (m_baseSizeY - sizeY);
            }
        }

        struct VertexBufDesc
        {
            ID3D11Buffer * pVertexBuf = nullptr;
            int pixelSizeX = 0;
            int pixelSizeY = 0;
        };
        std::vector<VertexBufDesc> vertexBufDescs;//ID3D11Buffer * pVertexBuf = nullptr;
        ID3D11Buffer * pIndexBuf = nullptr;

        virtual ControlType getType() const override { return ControlType::kIcon; };

        void storeBaseParameters()
        {
            m_baseSizeX = sizeX;
            m_baseSizeY = sizeY;
            m_basePosX = posX;
            m_basePosY = posY;
        }

    protected:
        
        float m_baseSizeX, m_baseSizeY;
        float m_basePosX, m_basePosY;

    };

}
//}