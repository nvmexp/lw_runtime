!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX3], TEX3, 2D;
TEX H4, f[TEX4], TEX4, 2D;
DP3H_SAT H3.xyz, H3, H4;
MULH H3.xyz, H3, H3;
MULH H3.xyz, H3, H1;
MULH H3.xyz, H3, H2;
MULH H3.xyz, H3, f[COL0];
ADDH H3.xyz, H3, f[COL1];
MULH H0.xyz, H3, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 3 

# Textures = 5 
