#include <catch.hpp>
#include <anselutils/CameraControllerFreeLocalFrame.h>

void CheckActualVsExpected(const lw::Vec3& actual, const lw::Vec3& expected)
{
	CHECK(actual.x == Approx(expected.x));
	CHECK(actual.y == Approx(expected.y));
	CHECK(actual.z == Approx(expected.z));
}

TEST_CASE("CameraControllerFreeLocalFrame.moves correctly in left handed coordinate system", 
	"[CameraControllerFreeLocalFrame][left handed]")
{
	using namespace lw;
	using namespace anselutils;
	using namespace ansel;

	const Vec3 kAxisX = { 1.0f, 0.0f, 0.0f };
	const Vec3 kAxisY = { 0.0f, 1.0f, 0.0f };
	const Vec3 kAxisZ = { 0.0f, 0.0f, 1.0f };

	const Vec3 basis[3] = {
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f }
	};

	Camera originalCam;
	originalCam.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
	originalCam.position = { 0.0f, 0.0f, 0.0f };
	CameraControllerFreeLocalFrame freeCam(basis[0], basis[1], basis[2]);
	freeCam.setTranslationalSpeed(1.0f);
	freeCam.setFixedTimeStep(1.0f);

	SECTION("moving forward")
	{
		freeCam.moveCameraForward(1.0f);
		Camera cam = originalCam;
		freeCam.update(cam);
		CHECK(cam.position.x == Approx(originalCam.position.x));
		CHECK(cam.position.y == Approx(originalCam.position.y));
		CHECK(cam.position.z == Approx(1.0f));
	}
}