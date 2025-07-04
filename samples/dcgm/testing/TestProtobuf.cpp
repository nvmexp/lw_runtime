/* 
 * File:   TestProtobuf.cpp
 */
#include "TestProtobuf.h"
#include <iostream>

using namespace std;

TestProtobuf::TestProtobuf() {
}

TestProtobuf::~TestProtobuf() {
    google::protobuf::ShutdownProtobufLibrary();
}

int TestProtobuf::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

int TestProtobuf::Run()
{

    int st;
    int Nfailed = 0;

    st = TestExchangeBasicTypes();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestProtobuf::TestExchangeBasicTypes FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    
    printf("TestProtobuf::TestExchangeBasicTypes PASSED\n");
    
    st = TestExchangeBatchCommands();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestProtobuf::TestExchangeBatchCommands FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    
    printf("TestProtobuf::TestExchangeBatchCommands PASSED\n");    
    

    if(Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    return 0;    
}

int TestProtobuf::Cleanup()
{
    return 0;
}

std::string TestProtobuf::GetTag()
{
    return std::string("encoder_decoder");
}

#define TEST_PROTOBUF_INT_VALUE 321
#define TEST_PROTOBUF_DBL_VALUE 123.56
#define TEST_PROTOBUF_STR_VALUE "Test String For Protobuf"
#define TEST_PROTOBUF_DEV_INDEX 8
#define TEST_PROTOBUF_STATUS    4
#define TEST_PROTOBUF_TS        1432244113

/*****************************************************************************
 * Test helper to create the command to be encoded
 *****************************************************************************/
int TestProtobuf::UpdateFieldValueToTx(lwcm::FieldValue *pFieldValue, unsigned int fieldType, unsigned int valType)
{
    pFieldValue->set_version(dcgmFieldValue_version1);
    pFieldValue->set_fieldid(fieldType);
    pFieldValue->set_fieldtype(valType);
    pFieldValue->set_ts(TEST_PROTOBUF_TS);
    pFieldValue->set_status(TEST_PROTOBUF_STATUS);
    
    lwcm::Value *value = pFieldValue->mutable_val();
    
    switch (valType)
    {
        case DCGM_FT_INT64:
            value->set_i64(TEST_PROTOBUF_INT_VALUE);
            break;
            
        case DCGM_FT_DOUBLE:
            value->set_dbl(TEST_PROTOBUF_DBL_VALUE);
            break;
            
        case DCGM_FT_STRING:
            value->set_str(TEST_PROTOBUF_STR_VALUE);
            break;

        default:
            return 1;
    }
            
    return 0;
}

/*****************************************************************************
 * Test helper to verify the correctness of decoded message
 *****************************************************************************/
int TestProtobuf::VerifyDecodedCommand(lwcm::Command *pcmd, int fieldId, int fieldType)
{
    if (pcmd->cmdtype() != lwcm::GET_FIELD_LATEST_VALUE) {
        return 10;
    }
    
    if (pcmd->id() != TEST_PROTOBUF_DEV_INDEX) {
        return 20;
    }
    
    if (pcmd->status() != TEST_PROTOBUF_STATUS) {
        return 30;
    }
    
    if (pcmd->arg(0).fieldvalue().fieldid() != fieldId) {
        return 40;
    }
    
    if (pcmd->arg(0).fieldvalue().ts() != TEST_PROTOBUF_TS) {
        return 50;
    }    
    
    
    switch(fieldType)
    {
        case DCGM_FT_INT64:
            if (pcmd->arg(0).fieldvalue().val().i64() != TEST_PROTOBUF_INT_VALUE) {
                return 52;
            }                        

            break;            
            
        case DCGM_FT_DOUBLE:
            if (pcmd->arg(0).fieldvalue().val().dbl() != TEST_PROTOBUF_DBL_VALUE) {
                return 53;
            }
            
            break;
            
        case DCGM_FT_STRING:
            if (0 != strcmp(pcmd->arg(0).fieldvalue().val().str().c_str(), TEST_PROTOBUF_STR_VALUE))
            {
                return 54;
            }
            
            break;

        default:
            return 55;
    }
    
    if (pcmd->arg(0).fieldvalue().version() != dcgmFieldValue_version1) {
        return 60;
    }
    return 0; 
}

/*****************************************************************************
 * Test helper to encode and decode messages based on the input types
 *****************************************************************************/
int TestProtobuf::TestFieldEncodeDecode(int fieldId, int fieldType)
{
    LwcmProtobuf encodePrb;             /* Protobuf message to be encoded for sending over network */
    LwcmProtobuf decodePrb;             /* Protobuf message recvd back from the DCGM HostEngine */
    char *bufToSend;                    /* The byte array buffer to be sent */
    unsigned int length;                /* Length of message to be sent */
    lwcm::Command *pCmd;                /* Pointer to proto command */
    vector<lwcm::Command *> cmdsRecvd;  /* Vector of proto commands recvd from remote end */    
    int ret;
    lwcm::FieldValue *pLwcmFieldValueToTx;
    
    pLwcmFieldValueToTx = new lwcm::FieldValue;
    
    UpdateFieldValueToTx(pLwcmFieldValueToTx, fieldId, fieldType);
    
    /* Add Command to be sent over the network */
    pCmd = encodePrb.AddCommand(lwcm::GET_FIELD_LATEST_VALUE, lwcm::OPERATION_GROUP_ENTITIES, 
                            TEST_PROTOBUF_DEV_INDEX, TEST_PROTOBUF_STATUS);
    if (NULL == pCmd) {
        return 100;
    }
    
    pCmd->add_arg()->set_allocated_fieldvalue(pLwcmFieldValueToTx);

    /* Get the protobuf encoded message */
    encodePrb.GetEncodedMessage(&bufToSend, &length);    
    
    // Display the length of encoded message
    // cout << "Val Type:" <<  fieldType << " Length of encoded message:" << length << endl;
    
    // Decode the message
    // Initialize Decoder object
    if (0 != decodePrb.ParseRecvdMessage(bufToSend, length, &cmdsRecvd)) {
        return 300;
    }    
    
    ret = VerifyDecodedCommand(cmdsRecvd[0], fieldId, fieldType);
    if (0 != ret) {
        return ret;
    }
    
    return 0;
}

/*****************************************************************************
 * This method is used to test encoding/decoding of basic types int, double
 *****************************************************************************/
int TestProtobuf::TestExchangeBasicTypes() {
    int ret;
    
    ret = TestFieldEncodeDecode(DCGM_FI_DEV_POWER_USAGE, DCGM_FT_INT64);
    if (0 != ret) {
        return ret;
    }
    
    ret = TestFieldEncodeDecode(DCGM_FI_DEV_POWER_USAGE, DCGM_FT_DOUBLE);
    if (0 != ret) {
        return ret;
    }
    
    ret = TestFieldEncodeDecode(DCGM_FI_DEV_POWER_USAGE, DCGM_FT_STRING);
    if (0 != ret) {
        return ret;
    }
    
    return 0;
}

/*****************************************************************************
 * This method is used to test encoding/decoding of structs
 *****************************************************************************/
int TestProtobuf::TestExchangeStructs() {
    /* todo: Will be implemented later */
    return 0;
}

/*****************************************************************************
 * This method is used to test encoding/decoding of batch of commands
 *****************************************************************************/
int TestProtobuf::TestExchangeBatchCommands() {
    LwcmProtobuf encodePrb;             /* Protobuf message to be encoded for sending over network */
    LwcmProtobuf decodePrb;             /* Protobuf message recvd back from the DCGM HostEngine */
    char *bufToSend;                    /* The byte array buffer to be sent */
    unsigned int length;                /* Length of message to be sent */
    lwcm::Command *pCmd;                /* Pointer to proto command */
    vector<lwcm::Command *> cmdsRecvd;  /* Vector of proto commands recvd from remote end */    
    int ret;
    int countToBatch = 10;
    int index;


    for (index = 0; index < countToBatch; index++) {
        lwcm::FieldValue *pLwcmFieldValueToTx;
        pLwcmFieldValueToTx = new lwcm::FieldValue;
        UpdateFieldValueToTx(pLwcmFieldValueToTx, DCGM_FI_DEV_SERIAL, DCGM_FT_STRING);
        
        pCmd = encodePrb.AddCommand(lwcm::GET_FIELD_LATEST_VALUE, lwcm::OPERATION_GROUP_ENTITIES,
                TEST_PROTOBUF_DEV_INDEX, TEST_PROTOBUF_STATUS);
        if (NULL == pCmd) {
            delete pLwcmFieldValueToTx;
            return 100;
        }
        
        pCmd->add_arg()->set_allocated_fieldvalue(pLwcmFieldValueToTx);
    }
    
    
    /* Get the protobuf encoded message */
    encodePrb.GetEncodedMessage(&bufToSend, &length);        
    
    
    // Decode the message
    if (0 != decodePrb.ParseRecvdMessage(bufToSend, length, &cmdsRecvd)) {
        return 300;
    }
    
    for (index = 0; index < countToBatch; index++) {
        ret = VerifyDecodedCommand(cmdsRecvd[index], DCGM_FI_DEV_SERIAL, DCGM_FT_STRING);
        if (0 != ret) {
            return ret;
        }        
    }
    
    return 0;
}

