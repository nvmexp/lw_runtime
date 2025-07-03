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

//input variables:
#define b_in prhs[0]
#define n_in prhs[1]
#define qam_in prhs[2]  
#define qam_mapping_real_in prhs[3]
#define qam_mapping_imag_in prhs[4]
#define z_in prhs[5]

//output variables:
#define x_real_out plhs[0]
#define x_imag_out plhs[1]

void modulate_main(int *b, double *qam_mapping_real, double *qam_mapping_imag,
        double *x_real, double *x_imag, int n, int qam, int *z)
{
    for (int i = 0; i <  n; ++i){
        int idx = 0;
        
        for (int j = 0; j < qam; ++j){
            idx = idx + b[qam*i + j]*z[j];
        }
        
        x_real[i] = qam_mapping_real[idx];
        x_imag[i] = qam_mapping_imag[idx];
    }
}


    
    
    
    






        
        

        void mexFunction(int nlhs, mxArray *plhs[], /* Outputs */
int nrhs, const mxArray *prhs[]) /* Inputs */
{
    double *qam_mapping_real, *qam_mapping_imag, *x_real, *x_imag;
    int *b, *z;
    int n, qam;
    
    n = mxGetScalar(n_in);
    qam = mxGetScalar(qam_in);
    
    b = mxGetPr(b_in);
    qam_mapping_real = mxGetPr(qam_mapping_real_in);
    qam_mapping_imag = mxGetPr(qam_mapping_imag_in);
    z = mxGetPr(z_in);
            
    x_real_out = mxCreateDoubleMatrix(n, 1, mxREAL);
    x_real = mxGetPr(x_real_out);
//     
    x_imag_out = mxCreateDoubleMatrix(n, 1, mxREAL);
    x_imag = mxGetPr(x_imag_out);
    
    modulate_main(b,qam_mapping_real,qam_mapping_imag,x_real,x_imag,n,qam,z);
    
}


    
    
    
    
    
    
    
    
    
    
