#include "Eud.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

Eud::Eud()
{
    testName = "EUD";
    setEudExeName("mods");

    if (!eudPresent("./tests"))
    {
        throw std::runtime_error("Unable to locate EUD test binary");
    }
}

Eud::~Eud()
{
}

void Eud::beginTest(vector<Gpu *> gpus)
{
    for (vector<Gpu *>::iterator it = gpus.begin(); it != gpus.end(); ++it)
    {
        exelwteEud((*it)->getDeviceIndex());
    }
}

void Eud::exelwteEud(int gpuNumber)
{
#ifdef _UNIX
    FILE * filep;
    char retBuffer[128];
    std::ostringstream out;

    cout << "Attempting to run EUD test for device#" << gpuNumber << endl;
    out << "cd " << "./tests && sudo ./mods" << " device=" << gpuNumber << endl;
    cout << out.str() << endl;
    filep = popen(out.str().c_str(),"r");
    while (NULL != fgets(retBuffer, sizeof(retBuffer), filep))
    {
        cout << retBuffer << endl;
    }
#endif
}

bool Eud::eudPresent(const string startingPath)
{
    string pathToSearch;
    string fullPath;

    if (startingPath.empty())
    {
        pathToSearch = "./";
    }
    char lastChar = *startingPath.rbegin();
    if (lastChar != '/')
        pathToSearch = startingPath + string("/");

    fullPath = pathToSearch + eudExeName;

    ifstream fileStream(fullPath.c_str());
    if (fileStream.good())
    {
        fileStream.close();
        setPathToEud(fullPath);
        return true;
    }

    fileStream.close();
    return false;
}
 
