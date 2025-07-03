#pragma once

#include "stdafx.h"
#include "CppUnitTest.h"
#include <ansel/Camera.h>
#include <lw/Vec3.h>

inline bool equal(float a, float b)
{
	return fabsf(a - b) < 1e-6f ? true : false;
}

template<typename T>
inline bool equal(const T& a, const T& b, const char* msg = nullptr)
{
	if (a == b)
		return true;

	if (msg)
		Microsoft::VisualStudio::CppUnitTestFramework::Logger::WriteMessage(msg);

	return false;
}

template<>
inline bool equal<lw::Vec3>(const lw::Vec3& a, const lw::Vec3& b, const char* msg)
{
	if (equal(a.x, b.x) && equal(a.y, b.y) && equal(a.z, b.z))
		return true;

	char buffer[512];
	sprintf_s(buffer, "Expected (%f, %f, %f) but got (%f, %f, %f)\n", a.x, a.y, a.z, b.x, b.y, b.z);
	Microsoft::VisualStudio::CppUnitTestFramework::Logger::WriteMessage(buffer);
	return false;
}

template<>
inline bool equal<ansel::Camera>(const ansel::Camera& b, const ansel::Camera& a, const char* msg)
{
	if ((a.fov == b.fov) &&
		(a.projectionOffsetX == b.projectionOffsetX) &&
		(a.projectionOffsetY == b.projectionOffsetY) &&
		(a.position.x == b.position.x) &&
		(a.position.y == b.position.y) &&
		(a.position.z == b.position.z) &&
		(a.rotation.x == b.rotation.x) &&
		(a.rotation.y == b.rotation.y) &&
		(a.rotation.z == b.rotation.z) &&
		(a.rotation.w == b.rotation.w))
		return true;

	char buffer[512];
	sprintf_s(buffer, "Expected { (%f, %f, %f), (%f, %f, %f, %f), fov = %f, projectionOffsetX = %f, projectionOffsetY = %f } but got \
								{ (%f, %f, %f), (%f, %f, %f, %f), fov = %f, projectionOffsetX = %f, projectionOffsetY = %f }\n", 
		a.position.x, a.position.y, a.position.z, a.rotation.x, a.rotation.y, a.rotation.z, a.rotation.w, a.fov, a.projectionOffsetX, a.projectionOffsetY,
		b.position.x, b.position.y, b.position.z, b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w, b.fov, b.projectionOffsetX, b.projectionOffsetY);
	Microsoft::VisualStudio::CppUnitTestFramework::Logger::WriteMessage(buffer);
	return false;
}

namespace AssertExt
{
	template<typename T>
	inline void AreEqual(const T& a, const T& b, const char* msg = nullptr)
	{
		Microsoft::VisualStudio::CppUnitTestFramework::Assert::IsTrue(equal(a, b, msg));
	}
}