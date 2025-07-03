!!FP1.0 
TEX H0, f[TEX1], TEX1, 2D;
MOVH H1.xyz, f[TEX0];
DP3H H1.w, H1, H1;
LG2H H1.w, |H1.w|;
MULH H1.w, H1, {0.5, 0, 0, 0}.x; 
EX2H H1.w, -H1.w;
MULH H1.xyz, H1, H1.w;
DP3H_SAT H1, H0, H1;
TEX H3, f[TEX2], TEX2, 2D;
MOVH H2.xyz, f[TEX0];
DP3H H2.w, H2, H2;
LG2H H2.w, |H2.w|;
MULH H2.w, H2, {0.5, 0, 0, 0}.x; 
EX2H H2.w, -H2.w;
MULH H2.xyz, H2, H2.w;
DP3H_SAT H0, H0, H2;
TXP H2, f[TEX3], TEX3, 2D;
TEX H0, H0, TEX8, 1D;
MULH H3, H3, H2;
TEX H2, f[TEX5], TEX5, 2D;
MULH H1, H3, H1;
TEX H3, f[TEX4], TEX4, 2D;
MADH H0, H2, H0, H3;
MULH H0.xyz, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 6 
