/*
 * CommandOutputController.h
 *
 *  Created on: Oct 23, 2015
 *      Author: chris
 */

#ifndef COMMANDOUTPUTCONTROLLER_H_
#define COMMANDOUTPUTCONTROLLER_H_

#include <iostream>
#include <vector>
#include "logging.h"
#include "dcgm_structs.h"

typedef struct
{
    std::string         tag;     //!<  Tag to be parsed for and replaced with value
    std::string         val;     //!<  Value to replaced parsed tag
} dcgmDisplayParameter_t;

class CommandOutputController {
public:
    CommandOutputController();
    virtual ~CommandOutputController();

    /* Clear all display parameters in memory. This is automatically called from display(). */
    void clearDisplayParameters();

    /* Setter for display stencil. The stencil is the base that will be parsed for tags which
     * will be replaced with values from the display parameters */
    void setDisplayStencil(char stencil[]);

    /* This displays the current stencil, with all stored display paramter tags swapped for
     * their values  */
    virtual void display();

    /* Add display parameter to list. toReplace is the tag to be parsed for and it is
     * replaced with replacedWith. See helper functions for formatting. */
    template<typename T>
    void addDisplayParameter(std::string toReplace, T replaceWith)
    {
            dcgmDisplayParameter_t temp = {toReplace, HelperDisplayValue(replaceWith)};

            displayParams.push_back(temp);
    }

    /*****************************************************************************
     * Helper method to give correct output for 32 bit integers
     *****************************************************************************/
    std::string HelperDisplayValue(int val);

    /*****************************************************************************
     * Helper method to give correct output for 32 bit unsigned integers
     *****************************************************************************/
    std::string HelperDisplayValue(unsigned int val);

    /*****************************************************************************
     * Helper method to give correct output for 64 bit integers
     *****************************************************************************/
    std::string HelperDisplayValue(long long val);

    /*****************************************************************************
     * Helper method to give correct output for doubles
     *****************************************************************************/
    std::string HelperDisplayValue(double val);

    /*****************************************************************************
     * Helper method to give proper output to strings
     *****************************************************************************/
    std::string HelperDisplayValue(std::string val);

    /*****************************************************************************
     * Helper method to give proper output Enabled/Disabled Values
     *****************************************************************************/
    std::string HelperDisplayValue(char * val);

private:
    /*****************************************************************************
     * Replaces the tag with the information given
     *****************************************************************************/
    void ReplaceTag(char *buff, const char *tag, char *fmt, ...);

    /*****************************************************************************
     * Returns the number of conselwtive spaces from start to end
     *****************************************************************************/
    int OnlySpacesBetween(char *start, char *end);

    /*****************************************************************************
     * Removes the tabs and newlines from str
     *****************************************************************************/
    void RemoveTabsAndNewlines(std::string &str);

    char *displayStencil;
    std::vector <dcgmDisplayParameter_t> displayParams;

};

typedef struct
{
    std::string overrideString;
    short fieldId;
    dcgmReturn_t errorCode;
}dcgmErrorStringOverride_t;

// Class that is used to display GPU errors.
class GPUErrorOutputController : CommandOutputController{
public:
    GPUErrorOutputController();
    virtual ~GPUErrorOutputController();

    /* Display a preset error display containing all the error codes and error
     * strings given by the error previously added */
    void display();

    /* A handle used to get all of the errors and fill display parameters with
     *  their corresponding information */
    void addError(dcgmStatus_t errHandle);

    /* Adds an string to override an error message in the error output
     * before an error is displayed, the field id and error code are checked
     * and if it matches an override it will use the replacement string*/
    void addErrorStringOverride(short fieldId, dcgmReturn_t errorCode, std::string replacement);
private:
    dcgmStatus_t mErrHandle;
    std::vector <dcgmErrorStringOverride_t> mStringOverriders;
};

#endif /* COMMANDOUTPUTCONTROLLER_H_ */
