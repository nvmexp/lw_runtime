#ifndef _LWVS_LWVS_EUD_H
#define _LWVS_LWVS_EUD_H

#include <vector>
#include "common.h"
#include "Test.h"
#include "Gpu.h"

using namespace std;

class Eud : public Test
{
public:
    Eud();
    ~Eud();

    string getTestName()
        { return testName; }

    void setPathToEud(string startingPath)
        { pathToEud = startingPath; }

    void setEudExeName(string exeName)
        { eudExeName = exeName; }

    void beginTest(vector<Gpu *> gpus);
private:
    bool eudPresent(string startingPath);
    void exelwteEud(int gpuNumber);

    string pathToEud;
    string eudExeName;
};

#endif //_LWVS_LWVS_EUD_H

