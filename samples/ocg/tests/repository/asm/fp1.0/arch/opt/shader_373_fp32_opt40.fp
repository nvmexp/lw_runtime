!!FP2.0DECLARE C0 = { 0, 0, 0, 0 };
DECLARE C1 = { .1, .1, .1, .1 };
DECLARE C2 = { .2, .2, .2, .2 };
DECLARE C3 = { .3, .3, .3, .3 };
DECLARE C4 = { .4, .4, .4, .4 };
DECLARE C5 = { .5, .5, .5, .5 };
DECLARE C6 = { .6, .6, .6, .6 };
DECLARE C7 = { .7, .7, .7, .7 };
DECLARE C8 = { .8, .8, .8, .8 };
DECLARE C9 = { .9, .9, .9, .9 };
TEX     R1, f[tex1], tex4, 2D;    
MADR    R1.xyz, c9.x, R1, c9.y;   
MULR    R0.xyz, R1.y, f[tex6];    
MADR    R0.xyz, f[tex5], R1.x, R0;
MADR    R2.xyz, f[tex7], R1.z, R0;
DP3R    R0.x, R2, R2;             
DIVR    R0.w, 1, R0.x;            
DP3R    R0.x, R2, f[tex4];        
MULR_m2 R0.w, R0.w, R0.x;         
MADR    R0.xyz, R0.w, R2, -f[tex4];
TEX     R0.xyz, R0, tex2, 2D;     
MULR    R0.xyz, R1.w, R0;         
MULR    R0.xyz, R0, c0;           
NRMH    R3.xyz, f[tex4];          
DP3R    R3.w, R2, R3;             
MADR    R3.xyz, R0, R0, -R0;      
ADDR    R2.w, -R3.w, c9.z;        
MULR    R1.w, R2.w, R2.w;         
MULR    R1.w, R1.w, R1.w;         
MULR    R1.w, R2.w, R1.w;         
MADR    R0.xyz, c2, R3, R0;       
DP3R    R3.w, R0, c9.w;           
MADR    R3.xyz, c3, -R3.w, R3.w;  
MADR    R3.xyz, c3, R0, R3;       
MOVR    R3.w, c5.w;               
MADR    R1.w, R1.w, R3.w, c4.w;   
DP3R_SAT R3.w, R1, c7;            
TEX     R0.xyz, f[TEX2].zwzw, TEX1, 2D;
MULR    R0.xyz, R0, R3.w;         
DP3R_SAT R3.w, R1, c1;            
DP3R_SAT R2.w, R1, c8;            
TEX     R2.xyz, f[tex2], tex1, 2D;
MADR    R2.xyz, R3.w, R2, R0;     
TEX     R1.xyz, f[TEX3].zwzw, TEX1, 2D;
MADR    R2.xyz, R2.w, R1, R2;     
MULR    R1.xyz, R3, R1.w;         
TEX     R0, f[tex0], tex0, 2D;    
MULR    R0.xyz, R0, f[COL0];      
MULR    R0.xyz, R2, R0;           
MADR    H0.xyz, R0, c6.x, R1;     
MULR    H0.w, R0.w, f[COL0].w;    
END

# Passes = 28 

# Registers = 4 

# Textures = 8 
