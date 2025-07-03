/*
 * CommandOutputController.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: chris
 */

#include "CommandOutputController.h"
#include "dcgm_agent.h"
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

/***************************************************************/
CommandOutputController::CommandOutputController() {
    // TODO Auto-generated constructor stub

}

CommandOutputController::~CommandOutputController() {
    // TODO Auto-generated destructor stub
}

/***************************************************************/
void CommandOutputController::clearDisplayParameters(){
    displayParams.clear();
}

/***************************************************************/
void CommandOutputController::setDisplayStencil(char stencil[]){
    displayStencil = stencil;
}

/***************************************************************/
void CommandOutputController::display(){
    char *display_tmp = new char[strlen(displayStencil) + 2]();
    strncpy(display_tmp, displayStencil, strlen(displayStencil));
    display_tmp[strlen(displayStencil)] = '\0';

    // Fill in tags
    for (std::vector<dcgmDisplayParameter_t>::iterator it = displayParams.begin() ; it != displayParams.end(); ++it){
        ReplaceTag(display_tmp, it->tag.c_str() , (char *)"%s", it->val.c_str() );
    }

    //Print to screen
    std::cout << display_tmp;

    clearDisplayParameters();

    // Clean up
    delete [] display_tmp;
}

/*****************************************************************************/
int CommandOutputController::OnlySpacesBetween(char *start, char *end)
{
    while (start < end)
    {
        if (' ' != *start)
            return 0;
        start++;
    }
    return 1;
}

/*****************************************************************************/
void CommandOutputController::ReplaceTag(char *buff, const char *tag, char *fmt, ...)
{
    char *tagstart;
    char *tagend = buff;
    size_t taglen = strlen(tag);
    size_t len;
    size_t vallen;
    char val[255];
    va_list args;
    do
    {
        tagstart = strstr(tagend, tag);
        if (NULL == tagstart){
            std::cout << "Debug Error: Parser unable to find tag start. Tag: " << tag << "Is stencil set? \n";
            return;
        }
        tagend = strstr(tagstart, ">");
        if (NULL == tagend){
            std::cout << "Debug Error: Parser unable to find tag end. (2)\n";
            return;
        }
    } while (!OnlySpacesBetween(tagstart + taglen, tagend));
    len = tagend - tagstart + 1;

    va_start(args, fmt);
    vsnprintf(val, sizeof(val), fmt, args);
    val[sizeof(val) - 1] = '\0'; // make sure that val is null terminated
    va_end(args);
    vallen = strlen(val);

    // clear tag from output string
    memset(tagstart, ' ', len);

    // fill the val
    strncpy(tagstart, val, (vallen < len) ? vallen:len);

    // indicate if not whole val fitted into the space
    if (vallen > len)
        strncpy(tagend - 2, "...", 3);
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(unsigned int val)
{
    std::stringstream ss;

    if (DCGM_INT32_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT32_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT32_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT32_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT32_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(long long val)
{
    std::stringstream ss;

    if (DCGM_INT64_IS_BLANK(val)) {
        switch (val)
        {
            case DCGM_INT64_BLANK:
                ss <<  "Not Specified";
                break;

            case DCGM_INT64_NOT_FOUND:
                ss <<  "Not Found";
                break;

            case DCGM_INT64_NOT_SUPPORTED:
                ss <<  "Not Supported";
                break;

            case DCGM_INT64_NOT_PERMISSIONED:
                ss <<  "Insf. Permission";
                break;

            default:
                ss <<  "Unknown";
                break;
        }
    } else {
        ss << val;
    }

    return ss.str();
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(double val)
{
    std::stringstream ss;

    if (DCGM_FP64_IS_BLANK(val)) {
        if (val == DCGM_FP64_BLANK)
            ss << "Not Specified";
        else if (val == DCGM_FP64_NOT_FOUND)
            ss << "Not Found";
        else if (val == DCGM_FP64_NOT_SUPPORTED)
            ss << "Not Supported";
        else if (val == DCGM_FP64_NOT_PERMISSIONED)
            ss << "Insf. Permission";
        else
            ss << "Unknown";
    } else {
        ss << val;
    }

    return ss.str();
}

void CommandOutputController::RemoveTabsAndNewlines(std::string &str)
{
    std::replace(str.begin(), str.end(), '\t', ' ');
    std::replace(str.begin(), str.end(), '\n', ' ');
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(std::string val)
{
    std::string str;

    if (DCGM_STR_IS_BLANK(val.c_str()))
    {
        if (!val.compare(DCGM_STR_BLANK))
        {
            str = "Not Specified";
        }
        else if (!val.compare(DCGM_STR_NOT_FOUND))
        {
            str = "Not Found";
        }
        else if (!val.compare(DCGM_STR_NOT_SUPPORTED))
        {
            str = "Not Supported";
        }
        else if (!val.compare(DCGM_STR_NOT_PERMISSIONED))
        {
            str = "Insf. Permission";
        }
        else
        {
            str = "Unknown";
        }
    }
    else
    {
        str = val;
        RemoveTabsAndNewlines(str);
    }

    return str;
}

/*****************************************************************************/
std::string CommandOutputController::HelperDisplayValue(char *val)
{
    std::string str;

    if (DCGM_STR_IS_BLANK(val))
    {
        if (!strcmp(val,DCGM_STR_BLANK))
        {
            str = "Not Specified";
        }
        else if (!strcmp(val,DCGM_STR_NOT_FOUND))
        {
            str = "Not Found";
        }
        else if (!strcmp(val,DCGM_STR_NOT_SUPPORTED))
        {
            str = "Not Supported";
        }
        else if (!strcmp(val,DCGM_STR_NOT_PERMISSIONED))
        {
            str = "Insf. Permission";
        }
        else
        {
            str = "Unknown";
        }
    }
    else
    {
        str = val;
        RemoveTabsAndNewlines(str);
    }

    return str;
}

/***************************************************************************************
 **
 **    GPU Error     *******************************************************************/

char ERROR_HEADER[] =
        "+---------+------------------------------------------------------------------+\n"
        "| GPU ID  | Error Message                                                    |\n"
        "+=========+==================================================================+\n";

char ERROR_DISPLAY[] =
        "| <GPUID >| <ERROR_MESSAGE                                                  >|\n";

char ERROR_FOOTER[] =
        "+---------+------------------------------------------------------------------+\n";

#define GPU_ID_TAG "<GPUID"
#define ERROR_MESSAGE_TAG "<ERROR_MESSAGE"


/************************************************************************************/
GPUErrorOutputController::GPUErrorOutputController(){

}
GPUErrorOutputController::~GPUErrorOutputController(){

}

/************************************************************************************/
void GPUErrorOutputController::display(){
    std::stringstream ss;
    dcgmErrorInfo_t lwrrentError;
    dcgmReturn_t result;
    dcgm_field_meta_p errorID;
    bool hadData = false;
    bool isOverridden = false;

    DcgmFieldsInit();

    /* Look at status to get individual errors */
    result = dcgmStatusPopError(mErrHandle, &lwrrentError);

    if (result != DCGM_ST_NO_DATA){
        std::cout << ERROR_HEADER;
        hadData = true;
    }

    this->setDisplayStencil(ERROR_DISPLAY);

    while (result != DCGM_ST_NO_DATA) {

        // Fill in tags
        if (lwrrentError.gpuId > 512){
            this->addDisplayParameter(GPU_ID_TAG, "N/A");
        } else {
            this->addDisplayParameter(GPU_ID_TAG, lwrrentError.gpuId);
        }

        // Create error message
        errorID = DcgmFieldGetById(lwrrentError.fieldId);
        ss.str("");

        // Check if error message has been overridden.
        isOverridden = false;
        for (unsigned int i = 0; i < mStringOverriders.size(); i++){
            if ( (mStringOverriders[i].fieldId == lwrrentError.fieldId) && (mStringOverriders[i].errorCode == lwrrentError.status) ){
                ss << mStringOverriders[i].overrideString;
                isOverridden = true;
                break;
            }
        }

        if (!isOverridden){
            if(errorID)
                ss << errorID->tag << " - " << errorString((dcgmReturn_t) lwrrentError.status);
            else
                ss << "No Field ID - " << errorString((dcgmReturn_t) lwrrentError.status);
        }

        // Display Error
        this->addDisplayParameter(ERROR_MESSAGE_TAG, ss.str());
        CommandOutputController::display();

        // Get next error
        result = dcgmStatusPopError(mErrHandle, &lwrrentError);
    }

    if (hadData){
        std::cout << ERROR_FOOTER;
    }

}

/************************************************************************************/
void GPUErrorOutputController::addError(dcgmStatus_t errHandle){
    mErrHandle = errHandle;
}

/************************************************************************************/
void GPUErrorOutputController::addErrorStringOverride(short fieldId, dcgmReturn_t errorCode, std::string replacement){
    dcgmErrorStringOverride_t temp = {replacement, fieldId, errorCode};
    mStringOverriders.push_back(temp);
}



