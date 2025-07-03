/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */


#include "mex.h" /* Always include this */
#include <math.h>
#include "matrix.h"

//input variables:
#define x_in prhs[0]  
#define n_in prhs[1]
#define g_in prhs[2]
#define r_in prhs[3]

//output variables:
#define crc_out plhs[0]


void crc_main(double* x, double* g, double* crc, int n, int r)
{
    //init augmented polynomial
    double *y;
    y = (double *)malloc(sizeof(double)*(n + r));
    
    //copy x
    for (int i = 0; i < n; ++i){
        y[i] = x[i];
    }
    
    //append zeros
    for (int i = 0; i < r; ++i)
    {
        y[n+i] = 0;
    }
    
    
    //main loop
    for (int i = 0; i < n; ++i)
    {
        if (y[i] == 1){
            for (int j = 1; j <= r; ++j)
            {
                y[i+j] = (y[i+j] + g[j]) * (2 - y[i+j] - g[j]);
            }
        }
    }
    
    //copy crc
    for (int i = 0; i < r; ++i)
    {
        crc[i] = y[n + i];
    }
}
            
    


void mexFunction(int nlhs, mxArray *plhs[], /* Outputs */
int nrhs, const mxArray *prhs[]) /* Inputs */
{
    double *x, *g, *crc;
    int n, r;
    
    x = mxGetPr(x_in);       //input bits
    g = mxGetPr(g_in);       //generator polynomial
    n = mxGetScalar(n_in);   //number of input puts
    r = mxGetScalar(r_in);   //degree of generator polynomial
    
    
    crc_out = mxCreateDoubleMatrix(r, 1, mxREAL);
    crc = mxGetPr(crc_out);
    

    
    crc_main(x,g,crc,n,r);
}
