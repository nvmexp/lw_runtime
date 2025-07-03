/* 
 * File:   TestVersioning.h
 */

#ifndef TESTVERSIONING_H
#define	TESTVERSIONING_H

#include "TestLwcmModule.h"
#include "dcgm_structs_internal.h"
#include "dcgm_structs.h"

class TestVersioning : public TestLwcmModule {
public:
    TestVersioning();
    virtual ~TestVersioning();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();  
    
private:
    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    /*****************************************************************************
     * This method is used to test basic recognition of API versions
     *****************************************************************************/
    int TestBasicAPIVersions();
    
#if 0
    int AddInt32FieldToCommand(command_t *pcmd, int fieldType, int32_t val);
    int AddInt64FieldToCommand(command_t *pcmd, int fieldType, int64_t val);
    int AddDoubleFieldToCommand(command_t *pcmd, int fieldType, double val);
    int AddStrFieldToCommand(command_t *pcmd, int fieldType, char *pChar);
    int TestFieldEncodeDecode(int fieldType, int valType);

    int UpdateCommandToEncode(command_t *pcmd, unsigned int fieldType, unsigned int valType);
    int VerifyDecodedCommand(command_t *pcmd, int fieldType, int valType);
#endif    
};

#endif	/* TESTVERSIONING_H */

