#include "stdafx.h"
#include "CppUnitTest.h"
#include "AssertExt.h"

#include <anselutils/CameraControllerRelative.h>
#include <lw/Vec3.inl>
#define _USE_MATH_DEFINES
#include <math.h>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace ansel;
using namespace anselutils;
using namespace lw;

extern void rotateBasis(const Quat& q, const Vec3 basis[3], Vec3 out[3]);
extern void makeRotationMatrixFromEulerAngles(float rot[3][3], float pitch, float yaw, float roll);
extern void makeQuaternionFromRotationMatrix(Quat& q, const float r[3][3]);
extern void matSetAndTranspose(float dest[3][3], float source[3][3]);
extern void matMult(float res[3][3], const float a[3][3], const float b[3][3]);

namespace
{
	Vec3 basis[3] =
	{
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f },
		{ 1.0f, 0.0f, 0.0f }
	};

}

namespace UnitTest
{
	const Vec3 kAxisX = { 1.0f, 0.0f, 0.0f };
	const Vec3 kAxisY = { 0.0f, 1.0f, 0.0f };
	const Vec3 kAxisZ = { 0.0f, 0.0f, 1.0f };

	TEST_CLASS(CameraControllerRelativeUE4Test)
	{
	public:
		static Camera m_cam;
		static CameraControllerRelative* m_cc;

		TEST_CLASS_INITIALIZE(Setup)
		{
			Logger::WriteMessage("Setting up UE4 class\n");
			m_cam.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
			m_cam.position = { 0 };
			m_cc = new CameraControllerRelative(basis[0], basis[1], basis[2]);
			m_cc->setCameraBase(m_cam);
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
			Logger::WriteMessage("Cleaning up UE4 class\n");
			delete m_cc;
		}


		TEST_METHOD(setCameraPitch)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5*float(M_PI), 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(kAxisZ, newCam[2]);
			AssertExt::AreEqual(-kAxisX, newCam[1]);
			AssertExt::AreEqual(basis[0], newCam[0]);
		}

		TEST_METHOD(setCameraYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisX, newCam[0]);
			AssertExt::AreEqual(basis[1], newCam[1]);
			AssertExt::AreEqual(kAxisY, newCam[2]);
		}

		TEST_METHOD(setCameraRoll)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(basis[2], newCam[2]);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(kAxisY, newCam[1]);
		}

		TEST_METHOD(setCameraPitchAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5f*float(M_PI), 0.5f*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisX, newCam[0]);
			AssertExt::AreEqual(-kAxisY, newCam[1]);
			AssertExt::AreEqual(kAxisZ, newCam[2]);
		}

		TEST_METHOD(setCameraRollAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(-kAxisX, newCam[1]);
			AssertExt::AreEqual(kAxisY, newCam[2]);
		}

	};

	CameraControllerRelative* CameraControllerRelativeUE4Test::m_cc = 0;
	Camera CameraControllerRelativeUE4Test::m_cam;

	///////////////////

	TEST_CLASS(CameraControllerRelativeWithRotationUE4Test)
	{
	public:
		static Camera m_cam;
		static CameraControllerRelative* m_cc;

		TEST_CLASS_INITIALIZE(Setup)
		{
			Logger::WriteMessage("Setting up UE4 class\n");
			float rot[3][3];
			makeRotationMatrixFromEulerAngles(rot, 0.5f*float(M_PI), 0.0f, 0.0f);
			float base[3][3] =
			{
				{ basis[0].x, basis[1].x, basis[2].x },
				{ basis[0].y, basis[1].y, basis[2].y },
				{ basis[0].z, basis[1].z, basis[2].z }
			};

			float base_rot[3][3];
			matMult(base_rot, base, rot);

			float baseilw[3][3];
			matSetAndTranspose(baseilw, base);

			float base_rot_baseilw[3][3];
			matMult(base_rot_baseilw, base_rot, baseilw);

			makeQuaternionFromRotationMatrix(m_cam.rotation, base_rot_baseilw);
			m_cam.position = { 0 };
			m_cc = new CameraControllerRelative(basis[0], basis[1], basis[2]);
			m_cc->setCameraBase(m_cam);
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
			Logger::WriteMessage("Cleaning up UE4 class\n");
			delete m_cc;
		}


		TEST_METHOD(setCameraPitch)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5*float(M_PI), 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual( kAxisY, newCam[0]);
			AssertExt::AreEqual(-kAxisZ, newCam[1]);
			AssertExt::AreEqual(-kAxisX, newCam[2]);
		}

		TEST_METHOD(setCameraYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(-kAxisX, newCam[1]);
			AssertExt::AreEqual( kAxisY, newCam[2]);
		}

		TEST_METHOD(setCameraRoll)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(kAxisX, newCam[0]);
			AssertExt::AreEqual(kAxisY, newCam[1]);
			AssertExt::AreEqual(kAxisZ, newCam[2]);
		}

		TEST_METHOD(setCameraPitchAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5f*float(M_PI), 0.5f*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(-kAxisY, newCam[1]);
			AssertExt::AreEqual(-kAxisX, newCam[2]);
		}

		TEST_METHOD(setCameraRollAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual( kAxisX, newCam[0]);
			AssertExt::AreEqual(-kAxisZ, newCam[1]);
			AssertExt::AreEqual( kAxisY, newCam[2]);
		}

	};

	CameraControllerRelative* CameraControllerRelativeWithRotationUE4Test::m_cc = 0;
	Camera CameraControllerRelativeWithRotationUE4Test::m_cam;

	////////////////
	TEST_CLASS(CameraControllerRelativeNoBaseUE4Test)
	{
	public:
		static Camera m_cam;
		static CameraControllerRelative* m_cc;

		TEST_CLASS_INITIALIZE(Setup)
		{
			Logger::WriteMessage("Setting up UE4 class\n");
			m_cam.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
			m_cam.position = { 0 };
			m_cc = new CameraControllerRelative(basis[0], basis[1], basis[2]);
		}

		TEST_CLASS_CLEANUP(CleanUp)
		{
			Logger::WriteMessage("Cleaning up UE4 class\n");
			delete m_cc;
		}


		TEST_METHOD(setCameraPitch)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5*float(M_PI), 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(kAxisZ, newCam[2]);
			AssertExt::AreEqual(-kAxisX, newCam[1]);
			AssertExt::AreEqual(basis[0], newCam[0]);
		}

		TEST_METHOD(setCameraYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisX, newCam[0]);
			AssertExt::AreEqual(basis[1], newCam[1]);
			AssertExt::AreEqual(kAxisY, newCam[2]);
		}

		TEST_METHOD(setCameraRoll)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.0f);
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(basis[2], newCam[2]);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(kAxisY, newCam[1]);
		}

		TEST_METHOD(setCameraPitchAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.0f, 0.5f*float(M_PI), 0.5f*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisX, newCam[0]);
			AssertExt::AreEqual(-kAxisY, newCam[1]);
			AssertExt::AreEqual(kAxisZ, newCam[2]);
		}

		TEST_METHOD(setCameraRollAndYaw)
		{
			m_cc->setCameraRotationRelativeToBase(0.5f*float(M_PI), 0.0f, 0.5*float(M_PI));
			Camera cam = m_cam;
			m_cc->update(cam);
			Vec3 newCam[3];
			rotateBasis(cam.rotation, basis, newCam);
			AssertExt::AreEqual(-kAxisZ, newCam[0]);
			AssertExt::AreEqual(-kAxisX, newCam[1]);
			AssertExt::AreEqual(kAxisY, newCam[2]);
		}

	};

	CameraControllerRelative* CameraControllerRelativeNoBaseUE4Test::m_cc = 0;
	Camera CameraControllerRelativeNoBaseUE4Test::m_cam;

}