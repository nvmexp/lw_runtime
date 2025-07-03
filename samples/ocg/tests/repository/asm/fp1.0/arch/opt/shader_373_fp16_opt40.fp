!!FP2.0
DECLARE C0 = { 0, 0, 0, 0 };
DECLARE C1 = { .1, .1, .1, .1 };
DECLARE C2 = { .2, .2, .2, .2 };
DECLARE C3 = { .3, .3, .3, .3 };
DECLARE C4 = { .4, .4, .4, .4 };
DECLARE C5 = { .5, .5, .5, .5 };
DECLARE C6 = { .6, .6, .6, .6 };
DECLARE C7 = { .7, .7, .7, .7 };
DECLARE C8 = { .8, .8, .8, .8 };
DECLARE C9 = { .9, .9, .9, .9 };
TEX     H1, f[tex1], tex4, 2D;  
MADH    H1.xyz, c9.x, H1, c9.y; 
MULH    H0.xyz, H1.y, f[tex6];  
MADH    H0.xyz, f[tex5], H1.x, H0;
MADH    H2.xyz, f[tex7], H1.z, H0;
DP3H    H0.x, H2, H2;           
DIVH    H0.w, 1, H0.x;          
DP3H    H0.x, H2, f[tex4];      
MULH_m2 H0.w, H0.w, H0.x;       
MADH    H0.xyz, H0.w, H2, -f[TEX4];
TEX     H0.xyz, H0, tex2, 2D;   
MULH    H0.xyz, H1.w, H0;       
MULH    H0.xyz, H0, c0;         
NRMH    H3.xyz, f[tex4];        
DP3H    H3.w, H2, H3;           
MADH    H3.xyz, H0, H0, -H0;    
ADDH    H2.w, -H3.w, c9.z;      
MULH    H1.w, H2.w, H2.w;       
MULH    H1.w, H1.w, H1.w;       
MULH    H1.w, H2.w, H1.w;       
MADH    H0.xyz, c2, H3, H0;     
DP3H    H3.w, H0, c9.w;         
MADH    H3.xyz, c3, -H3.w, H3.w;
MADH    H3.xyz, c3, H0, H3;     
MOVH    H3.w, c5.w;             
MADH    H1.w, H1.w, H3.w, c4.w; 
DP3H_SAT H3.w, H1, c7;          
TEX     H0.xyz, f[TEX2].zwzw, TEX1, 2D;
MULH    H0.xyz, H0, H3.w;         
DP3H_SAT H3.w, H1, c1;            
DP3H_SAT H2.w, H1, c8;            
TEX     H2.xyz, f[tex2], tex1, 2D;
MADH    H2.xyz, H3.w, H2, H0;     
TEX     H1.xyz, f[TEX3].zwzw, TEX1, 2D;
MADH    H2.xyz, H2.w, H1, H2;     
MULH    H1.xyz, H3, H1.w;         
TEX     H0, f[tex0], tex0, 2D;    
MULH    H0.xyz, H0, f[COL0];      
MULH    H0.xyz, H2, H0;           
MADH    H0.xyz, H0, c6.x, H1;     
MULH    H0.w, H0.w, f[COL0].w;    
END

# Passes = 27 

# Registers = 2 

# Textures = 8 
