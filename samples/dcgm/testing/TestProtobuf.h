/* 
 * File:   TestProtobuf.h
 */

#ifndef TESTPROTOBUF_H
#define	TESTPROTOBUF_H

#include "TestLwcmModule.h"
#include "LwcmProtobuf.h"

class TestProtobuf : public TestLwcmModule {
public:
    TestProtobuf();
    virtual ~TestProtobuf();
    
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
     * This method is used to test encoding/decoding of basic types int, double
     *****************************************************************************/
    int TestExchangeBasicTypes();
    
    /*****************************************************************************
     * This method is used to test encoding/decoding of structs
     *****************************************************************************/
    int TestExchangeStructs();
    
    /*****************************************************************************
     * This method is used to test encoding/decoding of batch of commands
     *****************************************************************************/
    int TestExchangeBatchCommands();

    /*************************************************************************/    

    int TestFieldEncodeDecode(int fieldType, int valType);
    int UpdateFieldValueToTx(lwcm::FieldValue *pFieldValue, unsigned int fieldType, unsigned int valType);
    int VerifyDecodedCommand(lwcm::Command *pcmd, int fieldType, int valType);
    
};

#endif	/* TESTPROTOBUF_H */

