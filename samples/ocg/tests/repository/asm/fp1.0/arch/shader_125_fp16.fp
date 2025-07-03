!!FP1.0 
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H3.x, H2, f[TEX4];
TEX H0, f[TEX3], TEX3, 2D;
MULH H3, H3.x, H0;
DP3H_SAT H0, f[TEX1], f[TEX1];
ADDH H0.x, {1, 1, 1, 1}, -H0.x;
MULH H0, H0.x, H3;
MULH H0, H0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
