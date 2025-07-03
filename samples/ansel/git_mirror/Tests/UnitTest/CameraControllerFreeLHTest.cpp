#include "stdafx.h"
#include "CppUnitTest.h"
#include "AssertExt.h"

#include <anselutils/CameraControllerFreeGlobalRotation.h>
#include <lw/Vec3.inl>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace ansel;
using namespace anselutils;
using namespace lw;

extern void rotateBasis(const Quat& q, const Vec3 basis[3], Vec3 out[3]);

namespace {
	Vec3 basis[3] = {
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f }
	};
}

namespace UnitTest
{
	const Vec3 kAxisX = { 1.0f, 0.0f, 0.0f };
	const Vec3 kAxisY = { 0.0f, 1.0f, 0.0f };
	const Vec3 kAxisZ = { 0.0f, 0.0f, 1.0f };


	TEST_CLASS(CameraControllerFreeGlobalRotationLHTest)
	{
	public:
		static Camera m_cam;
		static CameraControllerFree* m_freeCam;

		TEST_CLASS_INITIALIZE(Setup)
		{
			Logger::WriteMessage("Setting up LH class\n");
			m_cam.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
			m_cam.position = { 0.0f, 0.0f, 0.0f };
			m_freeCam = new CameraControllerFreeGlobalRotation(basis[0], basis[1], basis[2]);
			m_freeCam->setTranslationalSpeed(1.0f);
			m_freeCam->setFixedTimeStep(1.0f);
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
			Logger::WriteMessage("Cleaning up LH class\n");
			delete m_freeCam;
		}
		
		TEST_METHOD(moveForward)
		{
			m_freeCam->reset();
			m_freeCam->moveCameraForward(1.0f);
			Camera cam = m_cam;
			m_freeCam->update(cam);
			Vec3 expectedValue = { 0.0f, 0.0f, 1.0f };
			AssertExt::AreEqual(expectedValue, cam.position);
		}

		TEST_METHOD(adjustCameraPitch)
		{
			m_freeCam->reset();
			m_freeCam->setRotationalSpeed(180.0f);
			m_freeCam->adjustCameraPitch(0.5f);
			Camera cam = m_cam;
			m_freeCam->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(kAxisY, newCam[2]);
			AssertExt::AreEqual(-1.0f*kAxisZ, newCam[1]);
			AssertExt::AreEqual(basis[0], newCam[0]);
		}

		TEST_METHOD(adjustCameraYaw)
		{
			m_freeCam->reset();
			m_freeCam->setRotationalSpeed(360.0f);
			m_freeCam->adjustCameraYaw(0.25f);
			Camera cam = m_cam;
			m_freeCam->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(newCam[2], kAxisX);
			AssertExt::AreEqual(newCam[1], basis[1]);
			AssertExt::AreEqual(newCam[0], -kAxisZ);
		}

		TEST_METHOD(adjustCameraRoll)
		{
			m_freeCam->reset();
			m_freeCam->setRotationalSpeed(360.0f);
			m_freeCam->adjustCameraRoll(0.25f);
			Camera cam = m_cam;
			m_freeCam->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(basis[2], newCam[2]);
			AssertExt::AreEqual(-kAxisY, newCam[0]);
			AssertExt::AreEqual(kAxisX, newCam[1]);
		}

	};

	CameraControllerFree* CameraControllerFreeGlobalRotationLHTest::m_freeCam = 0;
	Camera CameraControllerFreeGlobalRotationLHTest::m_cam;

}