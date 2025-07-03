/*
 *  Base/common class for all enumerated devices in a system
 */

#ifndef _LWVS_LWVS_DEVICE_H
#define _LWVS_LWVS_DEVICE_H

#include <string>

using namespace std;

class Device
{
/***************************PUBLIC***********************************/
public:
    Device() {};
    ~Device() {};

    void setDeviceName (string name) { this->name = name; }
    string getDeviceName () { return name; }
    
/***************************PRIVATE**********************************/
private:

/***************************PROTECTED********************************/
protected:
    string name;
};

#endif //_LWVS_LWVS_DEVICE_H
