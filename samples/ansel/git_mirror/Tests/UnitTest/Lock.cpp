#include "stdafx.h"
#include "Lock.h"

UnitTestLock::UnitTestLock() 
{ 
	m_lock = CreateMutex(NULL, FALSE, L"UnitTestLock"); 
	WaitForSingleObject(m_lock, INFINITE);
}

UnitTestLock::~UnitTestLock() 
{	
	ReleaseMutex(m_lock);
	CloseHandle(m_lock); 
}
