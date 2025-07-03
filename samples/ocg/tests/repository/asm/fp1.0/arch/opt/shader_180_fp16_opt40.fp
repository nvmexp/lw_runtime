!!FP2.0
TEX H1, f[TEX3], TEX3, 2D;
TEX H0, f[TEX4], TEX4, 2D;
DP3H_SAT H1.xyz, H1, H0;
MULH H1.xyz, H1, H1;
TEX H0, f[TEX1], TEX1, 2D;
MULH H1.xyz, H1, H0;
TEX H0, f[TEX2], TEX2, 2D;
MULH H1.xyz, H1, H0;
MULH H1.xyz, H1, f[COL0];
ADDH H1.xyz, H1, f[COL1];
TEX H0, f[TEX0], TEX0, 2D;
MULH_m2 H0.xyz, H1, H0;
END

# Passes = 8 

# Registers = 1 

# Textures = 5 
