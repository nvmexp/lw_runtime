#include "stdafx.h"
#include "CppUnitTest.h"
#include "AssertExt.h"
#include "Lock.h"
#include <Windows.h>
#include <tchar.h>
#include <array>
#include <ansel/Camera.h>
#include <ansel/Configuration.h>
#include <ansel/UserControls.h>
#include <lw/Vec3.inl>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace ansel;
using namespace lw;

namespace
{
	const ansel::Camera goldenCamera{ lw::Vec3{ 1.0f, 2.0f, 3.0f }, lw::Quat{ 0.0f, 0.0f, 0.0f, 1.0f }, 90.0f, 0.0f, 0.0f };
}

namespace UnitTest
{
	TEST_CLASS(UpdateCamera)
	{
	public:
		TEST_CLASS_INITIALIZE(Setup)
		{
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
		}

		TEST_METHOD(updateCamera)
		{
			UnitTestLock lock;
			HMODULE hAnselSDK = NULL;
			// defined in AnselSDK too
			typedef void(*SessionFunc)(void* userData);
			typedef void(__cdecl *PFNCWGETCONFIGURATIONFUNC) (ansel::Configuration& cfg);
			typedef uint32_t(__cdecl *PFNCWGETCONFIGURATIONSIZE) ();
			typedef uint32_t(__cdecl *PFNCWGETSESSIONCONFIGURATIONSIZE) ();
			typedef void(__cdecl *PFNCWSETSESSIONFUNCTIONS) (SessionFunc start, SessionFunc stop, void* userData);
			typedef void(__cdecl *PFNCWSETUPDATECAMERAFUNC) (decltype(ansel::updateCamera)* updateCameraFunc);
			typedef void(__cdecl *PFLWOIDHINTFUNC)();
			typedef bool(__cdecl *PFNBOOLHINTFUNC)();
			typedef UserControlStatus(__cdecl *PFNADDUSERCONTROL)(const UserControlDesc&);
			typedef UserControlStatus(__cdecl *PFNSETUSERCONTROLLABELLOCALIZATION)(uint32_t, const char*, const char*);
			typedef UserControlStatus(__cdecl *PFNREMOVEUSERCONTROL)(uint32_t);
			typedef UserControlStatus(__cdecl *PFNGETUSERCONTROLVALUE)(uint32_t, void*);

			PFNCWGETCONFIGURATIONSIZE getConfigurationSize;
			PFNCWGETSESSIONCONFIGURATIONSIZE getSessionConfigurationSize;
			PFNCWSETSESSIONFUNCTIONS setSessionFunctions;
			PFNCWSETUPDATECAMERAFUNC setUpdateCameraFunc;
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

			if (!(setUpdateCameraFunc = (PFNCWSETUPDATECAMERAFUNC)GetProcAddress(hAnselSDK, "setUpdateCameraFunc")))
				AssertExt::AreEqual(false, true, "Expected: setUpdateCameraFunc present in Ansel SDK");
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

			// SDK can start/stop session (introduced in Ansel SDK 0.15) - this will be nullptr for older games
			if (!(setSessionFunctions = (PFNCWSETSESSIONFUNCTIONS)GetProcAddress(hAnselSDK, "setSessionFunctions")))
				AssertExt::AreEqual(false, true, "Expected: setSessionFunctions present in Ansel SDK");

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
			setUpdateCameraFunc([](ansel::Camera& cam) {
				cam = goldenCamera;
			});

			ansel::Camera cam{ lw::Vec3{ 0.0f, 0.0f, 0.0f }, lw::Quat{ 0.0f, 0.0f, 0.0f, 1.0f }, 110.0f, 0.0f, 0.0f };
			ansel::updateCamera(cam);
			AssertExt::AreEqual(cam, goldenCamera, "Expected: camera object equal to golden value");
		}
	};
}