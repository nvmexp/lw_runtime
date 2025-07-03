#ifndef _LWVS_LWVS_WRAPPER_H_
#define _LWVS_LWVS_WRAPPER_H_
#include "Plugin.h"
#include "TestParameters.h"
#include "StatCollection.h"
#include <stdio.h>

#define LWVS_EUD_SYMBOLS_PATH "/usr/share/lwpu/diagnostic/liblwvs-diagnostic.so.1"

class EUDWrapper : public Plugin
{
public:
    EUDWrapper();
    ~EUDWrapper();

    void go(TestParameters *testParameters)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
    void go(TestParameters *testParameters, unsigned int)
    {
        throw std::runtime_error("Not implemented in this test.");
    }

    void go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);

private:
    int createTempFile(FILE **file, char *path);
    int deflate(FILE *src, FILE *dest);
    int untar(char *tarPath, char *tarDir);
    void cleanDirectory(char *tarDir);
    int checkForGraphicsProcesses(const std::vector<unsigned int> &gpuList);
    TestParameters *defaultTp;
    int launch(std::string args, std::string eudPath);
    int WriteLog(std::string, int, StatCollection*);
    int openEudSymbols(std::string symbolsPath);

    StatCollection * m_StatCollection;
    char *m_EudDir;
};

extern "C" {
    unsigned int *eud_bytes;
    unsigned int eud_size;
}

extern const unsigned int EUD_SIZE;

#endif // _LWVS_LWVS_WRAPPER_H_
