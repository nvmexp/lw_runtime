#include "stdafx.h"
#include "CppUnitTest.h"
#include "AssertExt.h"
#include "Lock.h"
#include <array>
#include <Windows.h>
#include <tchar.h>
#include <ansel/Hints.h>
#include <ansel/Configuration.h>
#include <lw/Vec3.inl>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace ansel;
using namespace lw;

namespace UnitTest
{
	TEST_CLASS(SetConfiguration)
	{
	public:
		TEST_CLASS_INITIALIZE(Setup)
		{
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
		}

		TEST_METHOD(setConfigurationCorrectConfiguration)
		{
			UnitTestLock lock;
			// Configure Ansel
			ansel::Configuration config;
			config.rotationalSpeedInDegreesPerSecond = 220.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 50.0f;
			config.right.x = 1.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			config.forward.x = 0.0f;
			config.forward.y = 0.0f;
			config.forward.z = 1.0f;
			config.up.x = 0.0f;
			config.up.y = 1.0f;
			config.up.z = 0.0f;
			config.captureLatency = 0;
			config.captureSettleLatency = 0;
			config.fovType = ansel::kVerticalFov;
			config.startSessionCallback = [](SessionConfiguration& settings, void* userPointer) -> StartSessionStatus { return kAllowed; };
			config.stopSessionCallback = [](void* userPointer) {};
			config.startCaptureCallback = nullptr;
			config.stopCaptureCallback = nullptr;
			config.isCameraFovSupported = true;
			config.isCameraOffcenteredProjectionSupported = true;
			config.isCameraRotationSupported = true;
			config.isCameraTranslationSupported = true;
			config.userPointer = nullptr;
			config.gameWindowHandle = reinterpret_cast<void*>(0x12345678);
			const auto retcode = ansel::setConfiguration(config);
			AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
		}

		TEST_METHOD(setConfigurationHints)
		{
			UnitTestLock lock;
			// Configure Ansel
			ansel::Configuration config;
			config.rotationalSpeedInDegreesPerSecond = 220.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 50.0f;
			config.right.x = 1.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			config.forward.x = 0.0f;
			config.forward.y = 0.0f;
			config.forward.z = 1.0f;
			config.up.x = 0.0f;
			config.up.y = 1.0f;
			config.up.z = 0.0f;
			config.captureLatency = 0;
			config.captureSettleLatency = 0;
			config.fovType = ansel::kVerticalFov;
			config.startSessionCallback = [](SessionConfiguration& settings, void* userPointer) -> StartSessionStatus { return kAllowed; };
			config.stopSessionCallback = [](void* userPointer) {};
			config.startCaptureCallback = nullptr;
			config.stopCaptureCallback = nullptr;
			config.isCameraFovSupported = true;
			config.isCameraOffcenteredProjectionSupported = true;
			config.isCameraRotationSupported = true;
			config.isCameraTranslationSupported = true;
			config.userPointer = nullptr;
			config.gameWindowHandle = reinterpret_cast<void*>(0x12345678);
			const auto retcode = ansel::setConfiguration(config);
			AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");

			HMODULE hAnselSDK = NULL;
			// defined in AnselSDK too
			typedef void(__cdecl *PFNCWGETCONFIGURATIONFUNC) (ansel::Configuration& cfg);
			typedef bool(__cdecl *PFNBUFFERFINISHEDHINTFUNC)(ansel::BufferType, uint64_t&);
			typedef void(__cdecl *PFNCLEARHINTFUNC)(ansel::BufferType);
			typedef bool(__cdecl *PFNBUFFERBINDHINTFUNC)(ansel::BufferType, uint64_t&, ansel::HintType& hintType);
			typedef bool(__cdecl *PFNBUFFERFINISHEDHINTFUNC)(ansel::BufferType, uint64_t&);

			PFNCLEARHINTFUNC clearHdrBufferBindHint;
			PFNCLEARHINTFUNC clearHdrBufferFinishedHint;
			PFNBUFFERBINDHINTFUNC getHdrBufferBindHintActive;
			PFNBUFFERFINISHEDHINTFUNC getHdrBufferFinishedHintActive;

			const std::array<std::wstring, 10> anselSDKnames = {
				_T("AnselSDK64.dll"),
				_T("AnselSDK64d.dll"),
				_T("AnselSDK32.dll"),
				_T("AnselSDK32d.dll"),
				// fall back to older version (for temporary backwards compatibility)
				_T("LwCameraSDK64.dll"),
				_T("LwCameraSDK64d.dll"),
				// fall back to older version (for temporary backwards compatibility)
				_T("LwCameraSDK32.dll"),
				_T("LwCameraSDK32d.dll"),
				_T("LwCameraSDK.dll"),
				_T("LwCameraSDKd.dll"),
			};

			for (auto& moduleName : anselSDKnames)
				if (hAnselSDK = GetModuleHandle(moduleName.c_str()))
					break;

			AssertExt::AreEqual(hAnselSDK != NULL, true, "Expected: hAnselSDK to be not NULL");

			// HDR hints (introduced in Ansel SDK 0.13)
			if (!(clearHdrBufferBindHint = (PFNCLEARHINTFUNC)GetProcAddress(hAnselSDK, "clearBufferBindHint")))
				AssertExt::AreEqual(false, true, "Expected: clearBufferBindHint present in Ansel SDK");
			if (!(clearHdrBufferFinishedHint = (PFNCLEARHINTFUNC)GetProcAddress(hAnselSDK, "clearBufferFinishedHint")))
				AssertExt::AreEqual(false, true, "Expected: clearBufferFinishedHint present in Ansel SDK");
			if (!(getHdrBufferBindHintActive = (PFNBUFFERBINDHINTFUNC)GetProcAddress(hAnselSDK, "getBufferBindHintActive")))
				AssertExt::AreEqual(false, true, "Expected: getBufferBindHintActive present in Ansel SDK");
			if (!(getHdrBufferFinishedHintActive = (PFNBUFFERFINISHEDHINTFUNC)GetProcAddress(hAnselSDK, "getBufferFinishedHintActive")))
				AssertExt::AreEqual(false, true, "Expected: getBufferFinishedHintActive present in Ansel SDK");

			// Call this right before setting HDR render target active
			ansel::markBufferBind();
			// Call this right after the last draw call into the HDR render target
			ansel::markBufferFinished();

			uint64_t threadingMode = 0;
			ansel::HintType hintType = ansel::kHintTypePreBind;
			{
				const auto retcode = getHdrBufferBindHintActive(ansel::kBufferTypeHDR, threadingMode, hintType);
				AssertExt::AreEqual(retcode, true, "Expected: getHdrBufferBindHintActive to return true");
			}
			{
				const auto retcode = getHdrBufferFinishedHintActive(ansel::kBufferTypeHDR, threadingMode);
				AssertExt::AreEqual(retcode, true, "Expected: getHdrBufferFinishedHintActive to return true");
			}

			clearHdrBufferBindHint(ansel::kBufferTypeHDR);
			{
				const auto retcode = getHdrBufferBindHintActive(ansel::kBufferTypeHDR, threadingMode, hintType);
				AssertExt::AreEqual(retcode, false, "Expected: getHdrBufferBindHintActive to return false");
			}

			clearHdrBufferFinishedHint(ansel::kBufferTypeHDR);
			{
				const auto retcode = getHdrBufferFinishedHintActive(ansel::kBufferTypeHDR, threadingMode);
				AssertExt::AreEqual(retcode, false, "Expected: getHdrBufferFinishedHintActive to return false");
			}
			{
				const auto retcode = getHdrBufferBindHintActive(ansel::kBufferTypeHDR, threadingMode, hintType);
				AssertExt::AreEqual(retcode, false, "Expected: getHdrBufferBindHintActive to return false");
			}
		}


		TEST_METHOD(setConfigurationGetConfiguration)
		{
			// Configure Ansel
			ansel::Configuration config;
			config.rotationalSpeedInDegreesPerSecond = 220.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 50.0f;
			config.right.x = 1.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			config.forward.x = 0.0f;
			config.forward.y = 0.0f;
			config.forward.z = 1.0f;
			config.up.x = 0.0f;
			config.up.y = 1.0f;
			config.up.z = 0.0f;
			config.captureLatency = 0;
			config.captureSettleLatency = 0;
			config.fovType = ansel::kVerticalFov;
			config.startSessionCallback = [](SessionConfiguration& settings, void* userPointer) -> StartSessionStatus { return kAllowed; };
			config.stopSessionCallback = [](void* userPointer) {};
			config.startCaptureCallback = nullptr;
			config.stopCaptureCallback = nullptr;
			config.isCameraFovSupported = true;
			config.isCameraOffcenteredProjectionSupported = true;
			config.isCameraRotationSupported = true;
			config.isCameraTranslationSupported = true;
			config.userPointer = nullptr;
			config.gameWindowHandle = reinterpret_cast<void*>(0x12345678);
			const auto retcode = ansel::setConfiguration(config);
			AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");

			HMODULE hAnselSDK = NULL;
			// defined in AnselSDK too
			typedef void(__cdecl *PFNCWGETCONFIGURATIONFUNC) (ansel::Configuration& cfg);
			typedef uint32_t(__cdecl *PFNCWGETCONFIGURATIONSIZE) ();
			typedef uint32_t(__cdecl *PFNCWGETSESSIONCONFIGURATIONSIZE) ();
			typedef void(__cdecl *PFLWOIDHINTFUNC)();
			typedef bool(__cdecl *PFNBOOLHINTFUNC)();

			PFNCWGETCONFIGURATIONSIZE getConfigurationSize;
			PFNCWGETSESSIONCONFIGURATIONSIZE getSessionConfigurationSize;
			PFNCWGETCONFIGURATIONFUNC getConfigurationPfn;

			const std::array<std::wstring, 10> anselSDKnames = {
				_T("AnselSDK64.dll"),
				_T("AnselSDK64d.dll"),
				_T("AnselSDK32.dll"),
				_T("AnselSDK32d.dll"),
				// fall back to older version (for temporary backwards compatibility)
				_T("LwCameraSDK64.dll"),
				_T("LwCameraSDK64d.dll"),
				// fall back to older version (for temporary backwards compatibility)
				_T("LwCameraSDK32.dll"),
				_T("LwCameraSDK32d.dll"),
				_T("LwCameraSDK.dll"),
				_T("LwCameraSDKd.dll"),
			};

			for (auto& moduleName : anselSDKnames)
				if (hAnselSDK = GetModuleHandle(moduleName.c_str()))
					break;

			AssertExt::AreEqual(hAnselSDK != NULL, true, "Expected: hAnselSDK to be not NULL");

			if (!(getConfigurationPfn = (PFNCWGETCONFIGURATIONFUNC)GetProcAddress(hAnselSDK, "getConfiguration")))
				AssertExt::AreEqual(false, true, "Expected: getConfiguration present in Ansel SDK");
			if (!(getConfigurationSize = (PFNCWGETCONFIGURATIONSIZE)GetProcAddress(hAnselSDK, "getConfigurationSize")))
				AssertExt::AreEqual(false, true, "Expected: getConfigurationSize present in Ansel SDK");
			if (!(getSessionConfigurationSize = (PFNCWGETSESSIONCONFIGURATIONSIZE)GetProcAddress(hAnselSDK, "getSessionConfigurationSize")))
				AssertExt::AreEqual(false, true, "Expected: getSessionConfigurationSize present in Ansel SDK");

			// HDR hints (introduced in Ansel SDK 0.13)
			if (!GetProcAddress(hAnselSDK, "clearBufferBindHint"))
				AssertExt::AreEqual(false, true, "Expected: clearBufferBindHint present in Ansel SDK");
			if (!GetProcAddress(hAnselSDK, "clearBufferFinishedHint"))
				AssertExt::AreEqual(false, true, "Expected: clearBufferFinishedHint present in Ansel SDK");
			if (!GetProcAddress(hAnselSDK, "getBufferBindHintActive"))
				AssertExt::AreEqual(false, true, "Expected: getBufferBindHintActive present in Ansel SDK");
			if (!GetProcAddress(hAnselSDK, "getBufferFinishedHintActive"))
				AssertExt::AreEqual(false, true, "Expected: getBufferFinishedHintActive present in Ansel SDK");

			ansel::Configuration cfg;
			getConfigurationPfn(cfg);
			{
				const auto retcode = memcmp(&cfg, &config, sizeof(Configuration));
				AssertExt::AreEqual(retcode, 0, "Expected: Two configurations to be the same");
			}
		}

		TEST_METHOD(setConfigurationIncorrectConfiguration)
		{
			// Configure Ansel using incorrect configuration - callbacks are nullptr
			ansel::Configuration config;
			config.rotationalSpeedInDegreesPerSecond = 220.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 50.0f;
			config.right.x = 1.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			config.forward.x = 0.0f;
			config.forward.y = 0.0f;
			config.forward.z = 1.0f;
			config.up.x = 0.0f;
			config.up.y = 1.0f;
			config.up.z = 0.0f;
			config.captureLatency = 0;
			config.captureSettleLatency = 0;
			config.fovType = ansel::kVerticalFov;
			config.startSessionCallback = nullptr;
			config.stopSessionCallback = nullptr;
			config.startCaptureCallback = nullptr;
			config.stopCaptureCallback = nullptr;
			config.isCameraFovSupported = true;
			config.isCameraOffcenteredProjectionSupported = true;
			config.isCameraRotationSupported = true;
			config.isCameraTranslationSupported = true;
			config.userPointer = nullptr;
			config.gameWindowHandle = reinterpret_cast<void*>(0x12345678);
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncorrectConfiguration, "Expected: setConfiguration to return kSetConfigurationIncorrectConfiguration");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}

			// fix callbacks, test if it helps
			config.startSessionCallback = [](SessionConfiguration& settings, void* userPointer) -> StartSessionStatus { return kAllowed; };
			config.stopSessionCallback = [](void* userPointer) {};
			{
				const auto retcode = ansel::setConfiguration(config);
				AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
			}
			// break basis vectors, test if it breaks
			config.right.x = 0.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncorrectConfiguration, "Expected: setConfiguration to return kSetConfigurationIncorrectConfiguration");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}
			// fix basis vectors, test if it helps
			config.right.x = 1.0f;
			config.right.y = 0.0f;
			config.right.z = 0.0f;
			{
				const auto retcode = ansel::setConfiguration(config);
				AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
			}
			// break fovType, test if it breaks
			config.fovType = static_cast<FovType>(0xFFFFFFFF);
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncorrectConfiguration, "Expected: setConfiguration to return kSetConfigurationIncorrectConfiguration");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}
			// fix fovType, test if it helps
			config.fovType = ansel::kVerticalFov;
			{
				const auto retcode = ansel::setConfiguration(config);
				AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
			}
			// break speed multipliers, test if it break
			config.rotationalSpeedInDegreesPerSecond = 0.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 0.0f;
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncorrectConfiguration, "Expected: setConfiguration to return kSetConfigurationIncorrectConfiguration");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}
			// fix fovType, test if it helps
			config.rotationalSpeedInDegreesPerSecond = 220.0f;
			config.translationalSpeedInWorldUnitsPerSecond = 50.0f;
			{
				const auto retcode = ansel::setConfiguration(config);
				AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
			}
			// break gameWindowHandle, test if it breaks
			config.gameWindowHandle = nullptr;
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncorrectConfiguration, "Expected: setConfiguration to return kSetConfigurationIncorrectConfiguration");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}
			// fix gameWindowHandle, test if it helps
			config.gameWindowHandle = reinterpret_cast<void*>(0x12345678);
			{
				const auto retcode = ansel::setConfiguration(config);
				AssertExt::AreEqual(retcode, kSetConfigurationSuccess, "Expected: setConfiguration to return kSetConfigurationSuccess");
			}
			// break version, test if it breaks
			config.sdkVersion = 0;
			{
				const auto retcode = ansel::setConfiguration(config);
				const auto isAvailable = ansel::isAnselAvailable();
				AssertExt::AreEqual(retcode, kSetConfigurationIncompatibleVersion, "Expected: setConfiguration to return kSetConfigurationIncompatibleVersion");
				AssertExt::AreEqual(isAvailable, false, "Expected: isAnselAvailable to return false");
			}
		}
	};
}