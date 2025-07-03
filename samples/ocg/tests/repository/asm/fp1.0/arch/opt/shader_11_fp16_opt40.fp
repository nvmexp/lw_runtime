!!FP2.0 
TEX H0, f[TEX1], TEX1, 2D;
NRMH H1.xyz, f[TEX0];
DP3H_SAT H1, H0, H1;
TEX H3, f[TEX2], TEX2, 2D;
NRMH H2.xyz, f[TEX0];
DP3H_SAT H2, H0, H2;
TEX H0, H2, TEX8, 1D;
TXP H2, f[TEX3], TEX3, 2D;
MULH H3, H3, H2;
TEX H2, f[TEX5], TEX5, 2D;
MULH H1, H3, H1;
TEX H3, f[TEX4], TEX4, 2D;
MADH H0, H2, H0, H3;
MULB H0.xyz, H1_B, H0_B;
END

# Passes = 6 

# Registers = 3 

# Textures = 6 
