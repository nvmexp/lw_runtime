#define ANSEL_SDK_EXPORTS
#include <array>
#include <Windows.h>
#include <ansel/Configuration.h>
#include <ansel/Hints.h>
#include <LwCameraVersion.h>

namespace
{
    struct BufferInfo
    {
        bool bufferBindHintActive = false;
        bool bufferFinishedHintActive = false;
        uint64_t markBufferBindThreadingId = ansel::kThreadingBehaviourNoMatching;
        uint64_t markBufferFinishedThreadingId = ansel::kThreadingBehaviourNoMatching;
        ansel::HintType hintType = ansel::kHintTypePreBind;
    };

    std::array<BufferInfo, ansel::kBufferTypeCount> s_bufferInfo;

    inline bool checkBufferType(const ansel::BufferType bufferType)
    {
        return uint32_t(bufferType) < s_bufferInfo.size();
    }
}

namespace ansel
{
    typedef void (__cdecl * BufferFinishedCallback)(void * userData, ansel::BufferType bufferType, uint64_t threadId);
    ansel::BufferFinishedCallback s_bufferFinishedCallback = nullptr;

    extern void * s_userData;

    void markBufferBind(BufferType bufferType, HintType hintType, uint64_t threadId)
    { 
        if (checkBufferType(bufferType))
        {
            s_bufferInfo[bufferType].bufferBindHintActive = true;
            s_bufferInfo[bufferType].markBufferBindThreadingId = threadId == kThreadingBehaviourMatchAutomatic ? GetLwrrentThreadId() : threadId;
            s_bufferInfo[bufferType].hintType = hintType;
        }
    }
    
    void markBufferFinished(BufferType bufferType, uint64_t threadId)
    { 
        if (checkBufferType(bufferType))
        {
            s_bufferInfo[bufferType].bufferFinishedHintActive = true;
            s_bufferInfo[bufferType].markBufferFinishedThreadingId = threadId == kThreadingBehaviourMatchAutomatic ? GetLwrrentThreadId() : threadId;

            if (s_bufferFinishedCallback)
            {
                s_bufferFinishedCallback(s_userData, bufferType, threadId);
            }
        }
    }

    // These functions gets called by the driver.
    ANSEL_SDK_INTERNAL_API bool getBufferBindHintActive(BufferType bufferType, uint64_t& threadingMode, HintType& hintType)
    {
        if (!checkBufferType(bufferType))
            return false;

        internal::Version lwCameraVersion;
        internal::getLwCameraVersion(lwCameraVersion);
        // 4.0.164 (375GA3) changed the signature of this function from nonary returning bool, to binary returning bool.
        // It is safe to treat this function as nonary if this function behaves as such in case LwCamera version is sufficiently old
        // This means this function shouldn't touch its arguments (read or write), because older LwCamera are not passing anything to this function
        if (lwCameraVersion.major >= 4 || (lwCameraVersion.major == 4 && lwCameraVersion.build >= 164))
        {
            threadingMode = s_bufferInfo[bufferType].markBufferBindThreadingId;
            hintType = s_bufferInfo[bufferType].hintType;
        }
        return s_bufferInfo[bufferType].bufferBindHintActive;
    }

    ANSEL_SDK_INTERNAL_API bool getBufferFinishedHintActive(BufferType bufferType, uint64_t& threadingMode)
    { 
        if (!checkBufferType(bufferType))
            return false;

        internal::Version lwCameraVersion;
        internal::getLwCameraVersion(lwCameraVersion);
        // 4.0.164 (375GA3) changed the signature of this function from nonary returning bool, to binary returning bool.
        // It is safe to treat this function as nonary if this function behaves as such in case LwCamera version is sufficiently old
        // This means this function shouldn't touch its arguments (read or write), because older LwCamera are not passing anything to this function
        if (lwCameraVersion.major >= 4 || (lwCameraVersion.major == 4 && lwCameraVersion.build >= 164))
        {
            threadingMode = s_bufferInfo[bufferType].markBufferFinishedThreadingId;
        }
        return s_bufferInfo[bufferType].bufferFinishedHintActive;
    }

    ANSEL_SDK_INTERNAL_API void clearBufferBindHint(BufferType bufferType)
    { 
        if (checkBufferType(bufferType))
        {
            s_bufferInfo[bufferType].bufferBindHintActive = false;
            s_bufferInfo[bufferType].hintType = ansel::kHintTypePreBind;
            s_bufferInfo[bufferType].markBufferBindThreadingId = ansel::kThreadingBehaviourNoMatching;
        }
    }

    ANSEL_SDK_INTERNAL_API void clearBufferFinishedHint(BufferType bufferType)
    { 
        if (checkBufferType(bufferType))
        {
            s_bufferInfo[bufferType].bufferFinishedHintActive = false;
            s_bufferInfo[bufferType].markBufferFinishedThreadingId = ansel::kThreadingBehaviourNoMatching;
        }
    }

    ANSEL_SDK_INTERNAL_API void setBufferFinishedCallback(BufferFinishedCallback bufferFinishedCallback)
    {
        s_bufferFinishedCallback = bufferFinishedCallback;
    }
}
