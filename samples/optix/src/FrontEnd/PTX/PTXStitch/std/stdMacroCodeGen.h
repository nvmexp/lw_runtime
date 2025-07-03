/*
 *  Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 * 
 *  NOTICE TO USER: The source code, and related code and software
 *  ("Code"), is copyrighted under U.S. and international laws.  
 * 
 *  LWPU Corporation owns the copyright and any patents issued or 
 *  pending for the Code.  
 * 
 *  LWPU CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY 
 *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
 *  OR IMPLIED WARRANTY OF ANY KIND.  LWPU CORPORATION DISCLAIMS ALL
 *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND 
 *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE.  IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
 *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
 *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE 
 *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO LWPU
 *  CORPORATION.
 * 
 *  Module name              : stdMacroCodeGen.h
 *
 *  Last update              :
 *
 *  Description              :
 *     
 *         This module provides support for C/C++ code generation 
 *         (mis)using the preprocessor.
 */

#ifndef stdMacroCodeGen_INCLUDED
#define stdMacroCodeGen_INCLUDED

/*---------------------------------- Macros ----------------------------------*/


#define __stdLOG0()
#define __stdLOG1(t1,v1)                                                                                                         __stdLOG(v1);
#define __stdLOG2(t1,v1,t2,v2)                                                                                                   __stdLOG(v1);__stdLOG(v2);
#define __stdLOG3(t1,v1,t2,v2,t3,v3)                                                                                             __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);
#define __stdLOG4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                       __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);
#define __stdLOG5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                                 __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);
#define __stdLOG6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                           __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);
#define __stdLOG7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                                     __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);
#define __stdLOG8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                               __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);
#define __stdLOG9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                         __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);
#define __stdLOG10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                                __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);
#define __stdLOG11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                        __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);__stdLOG(v11);                
#define __stdLOG12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                                __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);__stdLOG(v11);__stdLOG(v12);               
#define __stdLOG13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                        __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);__stdLOG(v11);__stdLOG(v12);__stdLOG(v13);              
#define __stdLOG14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)                __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);__stdLOG(v11);__stdLOG(v12);__stdLOG(v13);__stdLOG(v14);
#define __stdLOG15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)        __stdLOG(v1);__stdLOG(v2);__stdLOG(v3);__stdLOG(v4);__stdLOG(v5);__stdLOG(v6);__stdLOG(v7);__stdLOG(v8);__stdLOG(v9);__stdLOG(v10);__stdLOG(v11);__stdLOG(v12);__stdLOG(v13);__stdLOG(v14);__stdLOG(v15);

#define __stdVARDECL0()                                                     
#define __stdVARDECL1(t1,v1)                                                                                                      t1 v1;
#define __stdVARDECL2(t1,v1,t2,v2)                                                                                                t1 v1;t2 v2;
#define __stdVARDECL3(t1,v1,t2,v2,t3,v3)                                                                                          t1 v1;t2 v2;t3 v3;
#define __stdVARDECL4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                    t1 v1;t2 v2;t3 v3;t4 v4;
#define __stdVARDECL5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                              t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;
#define __stdVARDECL6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                        t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;
#define __stdVARDECL7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                                  t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;
#define __stdVARDECL8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                            t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;
#define __stdVARDECL9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                      t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;
#define __stdVARDECL10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                             t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;
#define __stdVARDECL11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                     t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;t11 v11;
#define __stdVARDECL12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                             t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;t11 v11;t12 v12;
#define __stdVARDECL13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                     t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;t11 v11;t12 v12;t13 v13;
#define __stdVARDECL14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)             t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;t11 v11;t12 v12;t13 v13;t14 v14;
#define __stdVARDECL15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)     t1 v1;t2 v2;t3 v3;t4 v4;t5 v5;t6 v6;t7 v7;t8 v8;t9 v9;t10 v10;t11 v11;t12 v12;t13 v13;t14 v14;t15 v15;

#define __stdPARMDECL0()                                                    
#define __stdPARMDECL1(t1,v1)                                                                                                     t1 v1
#define __stdPARMDECL2(t1,v1,t2,v2)                                                                                               t1 v1,t2 v2
#define __stdPARMDECL3(t1,v1,t2,v2,t3,v3)                                                                                         t1 v1,t2 v2,t3 v3
#define __stdPARMDECL4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                   t1 v1,t2 v2,t3 v3,t4 v4
#define __stdPARMDECL5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                             t1 v1,t2 v2,t3 v3,t4 v4,t5 v5
#define __stdPARMDECL6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                       t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6
#define __stdPARMDECL7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                                 t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7
#define __stdPARMDECL8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                           t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8
#define __stdPARMDECL9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                     t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9
#define __stdPARMDECL10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                            t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10
#define __stdPARMDECL11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                    t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11
#define __stdPARMDECL12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                            t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12
#define __stdPARMDECL13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                    t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13
#define __stdPARMDECL14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)            t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13,t14 v14
#define __stdPARMDECL15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)    t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13,t14 v14,t15 v15

#define __stdCPARMDECL0()                                                    
#define __stdCPARMDECL1(t1,v1)                                                                                                   ,t1 v1
#define __stdCPARMDECL2(t1,v1,t2,v2)                                                                                             ,t1 v1,t2 v2
#define __stdCPARMDECL3(t1,v1,t2,v2,t3,v3)                                                                                       ,t1 v1,t2 v2,t3 v3
#define __stdCPARMDECL4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                 ,t1 v1,t2 v2,t3 v3,t4 v4
#define __stdCPARMDECL5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                           ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5
#define __stdCPARMDECL6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                     ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6
#define __stdCPARMDECL7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                               ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7
#define __stdCPARMDECL8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                         ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8
#define __stdCPARMDECL9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                   ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9
#define __stdCPARMDECL10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                          ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10
#define __stdCPARMDECL11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                  ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11
#define __stdCPARMDECL12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                          ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12
#define __stdCPARMDECL13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                  ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13
#define __stdCPARMDECL14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)          ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13,t14 v14
#define __stdCPARMDECL15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)  ,t1 v1,t2 v2,t3 v3,t4 v4,t5 v5,t6 v6,t7 v7,t8 v8,t9 v9,t10 v10,t11 v11,t12 v12,t13 v13,t14 v14,t15 v15

#define __stdINITINSTVAR0()                                                
#define __stdINITINSTVAR1(t1,v1)                                                                                                  : v1(v1)
#define __stdINITINSTVAR2(t1,v1,t2,v2)                                                                                            : v1(v1), v2(v2) 
#define __stdINITINSTVAR3(t1,v1,t2,v2,t3,v3)                                                                                      : v1(v1), v2(v2), v3(v3)
#define __stdINITINSTVAR4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                : v1(v1), v2(v2), v3(v3), v4(v4)
#define __stdINITINSTVAR5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                          : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5) 
#define __stdINITINSTVAR6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                    : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6) 
#define __stdINITINSTVAR7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                              : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7) 
#define __stdINITINSTVAR8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                        : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8) 
#define __stdINITINSTVAR9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                  : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9) 
#define __stdINITINSTVAR10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                         : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10) 
#define __stdINITINSTVAR11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                 : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11)
#define __stdINITINSTVAR12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                         : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12)
#define __stdINITINSTVAR13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                 : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12), v13(v13)
#define __stdINITINSTVAR14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)         : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12), v13(v13), v14(v14)
#define __stdINITINSTVAR15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15) : v1(v1), v2(v2), v3(v3), v4(v4), v5(v5), v6(v6), v7(v7), v8(v8), v9(v9), v10(v10), v11(v11), v12(v12), v13(v13), v14(v14), v15(v15)

#define __stdARGLIST0()
#define __stdARGLIST1(t1,v1)                                                                                                      v1
#define __stdARGLIST2(t1,v1,t2,v2)                                                                                                v1,v2
#define __stdARGLIST3(t1,v1,t2,v2,t3,v3)                                                                                          v1,v2,v3
#define __stdARGLIST4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                    v1,v2,v3,v4
#define __stdARGLIST5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                              v1,v2,v3,v4,v5
#define __stdARGLIST6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                        v1,v2,v3,v4,v5,v6
#define __stdARGLIST7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                                  v1,v2,v3,v4,v5,v6,v7
#define __stdARGLIST8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                            v1,v2,v3,v4,v5,v6,v7,v8
#define __stdARGLIST9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                      v1,v2,v3,v4,v5,v6,v7,v8,v9
#define __stdARGLIST10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                             v1,v2,v3,v4,v5,v6,v7,v8,v9,v10
#define __stdARGLIST11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                     v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11
#define __stdARGLIST12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                             v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12
#define __stdARGLIST13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                     v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13
#define __stdARGLIST14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)             v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14
#define __stdARGLIST15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)     v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15


#define __stdCARGLIST0()
#define __stdCARGLIST1(t1,v1)                                                                                                    ,v1
#define __stdCARGLIST2(t1,v1,t2,v2)                                                                                              ,v1,v2
#define __stdCARGLIST3(t1,v1,t2,v2,t3,v3)                                                                                        ,v1,v2,v3
#define __stdCARGLIST4(t1,v1,t2,v2,t3,v3,t4,v4)                                                                                  ,v1,v2,v3,v4
#define __stdCARGLIST5(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5)                                                                            ,v1,v2,v3,v4,v5
#define __stdCARGLIST6(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6)                                                                      ,v1,v2,v3,v4,v5,v6
#define __stdCARGLIST7(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7)                                                                ,v1,v2,v3,v4,v5,v6,v7
#define __stdCARGLIST8(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8)                                                          ,v1,v2,v3,v4,v5,v6,v7,v8
#define __stdCARGLIST9(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9)                                                    ,v1,v2,v3,v4,v5,v6,v7,v8,v9
#define __stdCARGLIST10(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10)                                           ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10
#define __stdCARGLIST11(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11)                                   ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11
#define __stdCARGLIST12(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12)                           ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12
#define __stdCARGLIST13(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13)                   ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13
#define __stdCARGLIST14(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14)           ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14
#define __stdCARGLIST15(t1,v1,t2,v2,t3,v3,t4,v4,t5,v5,t6,v6,t7,v7,t8,v8,t9,v9,t10,v10,t11,v11,t12,v12,t13,v13,t14,v14,t15,v15)   ,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15



#endif
