#pragma once
#include <Windows.h>

class UnitTestLock
{
private:
	HANDLE m_lock = nullptr;
public:
	UnitTestLock();
	~UnitTestLock();
};