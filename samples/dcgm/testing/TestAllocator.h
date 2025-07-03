/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   TestAllocator.h
 * Author: ankjain
 *
 * Created on July 31, 2017, 4:06 PM
 */
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "TestLwcmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"
#include "LwcmCacheManager.h"


#ifndef TESTALLOCATOR_H
#define TESTALLOCATOR_H

class TestAllocator : public TestLwcmModule
{
public:
    TestAllocator();
    virtual ~TestAllocator();
    
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
     int testAllocateVarSize_FreeInorder();
     int testAllocateVarSize_FreeReverse();
     int testAllocateFixedSize_FreeInorder();  
     int testAllocateFixedSize_FreeReverse(); 
     int testAllocateVarSize_FreeSerial();
     int testAllocateFixedSize_FreeSerial();
     int testInternalFragFixedSize();
};

#endif /* TESTALLOCATOR_H */

