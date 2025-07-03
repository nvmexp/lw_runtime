

#include <vector>
#include "logging.h"
#include "dcgm_agent_internal.h" /* DCGM_EMBEDDED_HANDLE */
#include "LwcmProtobuf.h"
#include "dcgm_module_structs.h"
#include "LwcmRequest.h"

/*****************************************************************************/
/* Stubs */
dcgmReturn_t processAtHostEngine(dcgmHandle_t pDcgmHandle, LwcmProtobuf *encodePrb,
                                 LwcmProtobuf *decodePrb, vector<lwcm::Command *> *vecCmdsRef,
                                 LwcmRequest *request=0, unsigned int timeout=60000);

/*****************************************************************************/
dcgmReturn_t dcgmModuleSendBlockingFixedRequest(dcgmHandle_t pLwcmHandle,
                                                dcgm_module_command_header_t *moduleCommand,
                                                LwcmRequest *request, unsigned int timeout)
{
    LwcmProtobuf encodePrb;              /* Protobuf message for encoding */
    LwcmProtobuf decodePrb;              /* Protobuf message for decoding */
    lwcm::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    std::vector<lwcm::Command *> vecCmdsRef; /* Vector of proto commands. Used as output parameter */
    dcgmReturn_t ret;
    lwcm::CmdArg *cmdArg = 0;

    if(!moduleCommand)
        return DCGM_ST_BADPARAM;

    unsigned int sizeBefore = moduleCommand->length;
    
    if(moduleCommand->length < sizeof(*moduleCommand))
    {
        PRINT_ERROR("%u", "Bad module param length %u", moduleCommand->length);
        if(request)
            delete request;
        return DCGM_ST_BADPARAM;
    }
    if(moduleCommand->moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Bad module ID %u", moduleCommand->moduleId);
        if(request)
            delete request;
        return DCGM_ST_BADPARAM;
    }

    /* Add Command to the protobuf encoder object */
    pCmdTemp = encodePrb.AddCommand(lwcm::MODULE_COMMAND, lwcm::OPERATION_SYSTEM, -1, 0);
    if (NULL == pCmdTemp)
    {
        PRINT_ERROR("", "Error from AddCommand");
        if(request)
            delete request;
        return DCGM_ST_GENERIC_ERROR;
    }

    cmdArg = pCmdTemp->add_arg();
    cmdArg->set_blob(moduleCommand, moduleCommand->length);

    ret = processAtHostEngine(pLwcmHandle, &encodePrb, &decodePrb, 
                              &vecCmdsRef, request, timeout);
    if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d", "processAtHostEngine returned %d", ret);
        return ret;
    }    

    if(!vecCmdsRef[0]->arg_size())
    {
        PRINT_ERROR("", "Arg size of 0 unexpected");
        return DCGM_ST_GENERIC_ERROR;
    }

    if(!vecCmdsRef[0]->arg(0).has_blob())
    {
        PRINT_ERROR("", "Response missing blob");
        return DCGM_ST_GENERIC_ERROR;
    }

    if(vecCmdsRef[0]->arg(0).blob().size() > sizeBefore)
    {
        PRINT_ERROR("%d %u", "Returned blob size %d > sizeBefore %u",
                    (int)vecCmdsRef[0]->arg(0).blob().size(), sizeBefore);
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(moduleCommand, (void *)vecCmdsRef[0]->arg(0).blob().c_str(),
           vecCmdsRef[0]->arg(0).blob().size());

    /* Check the status of the DCGM command */
    ret = (dcgmReturn_t)vecCmdsRef[0]->status();
    return ret;
}

/*****************************************************************************/
