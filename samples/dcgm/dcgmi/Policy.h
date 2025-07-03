/*
 * Policy.h
 *
 *  Created on: Oct 5, 2015
 *      Author: chris
 */

#ifndef POLICY_H_
#define POLICY_H_

#include "Command.h"
#include "dcgm_client_internal.h"

class Policy {
public:
    Policy();
    virtual ~Policy();

    dcgmReturn_t DisplayLwrrentViolationPolicy(dcgmHandle_t mLwcmHandle, unsigned int groupId, bool verbose, bool json);
    dcgmReturn_t SetLwrrentViolationPolicy(dcgmHandle_t mLwcmHandle, unsigned int groupId, dcgmPolicy_t policy);
    dcgmReturn_t RegisterForPolicyUpdates(dcgmHandle_t mLwcmHandle, unsigned int groupId, unsigned int condition);
    dcgmReturn_t UnregisterPolicyUpdates(dcgmHandle_t mLwcmHandle, unsigned int groupId, unsigned int condition);
private:
    static int ListenForViolations(void *data);
    static std::string HelperFormatTimestamp(long long timestamp);
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Get Policy ilwoker class
 */
class GetPolicy : public Command
{
public:
    GetPolicy(std::string hostname, unsigned int groupId, bool verbose, bool json);
    virtual ~GetPolicy();

    int Execute();

private:
    Policy policyObj;
    unsigned int groupId;
    bool verbose;
};

/**
 * Set Policy ilwoker class
 */
class SetPolicy : public Command
{
public:
    SetPolicy(std::string hostname, dcgmPolicy_t setPolicy, unsigned int groupId);
    virtual ~SetPolicy();

    int Execute();

private:
    Policy policyObj;
    dcgmPolicy_t setPolicy;
    unsigned int groupId;
};

/**
 * Register Policy ilwoker class
 */
class RegPolicy : public Command
{
public:
    RegPolicy(std::string hostname, unsigned int groupId, unsigned int condition);
    virtual ~RegPolicy();

    int Execute();

private:
    Policy policyObj;
    unsigned int groupId;
    unsigned int condition;
};
#endif /* POLICY_H_ */
