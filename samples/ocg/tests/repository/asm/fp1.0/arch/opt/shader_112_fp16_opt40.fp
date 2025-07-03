!!FP2.0 
TEX H3, f[TEX3], TEX3, 2D;
TEX H1, f[TEX4], TEX4, 2D;
DP3H_SAT H3.xyz, H3, H1;
MULH H3.xyz, H3, H3;
TEX H0, f[TEX0], TEX0, 2D;
MULH H3.xyz, H3, H0.w;
TEX H1, f[TEX5], TEX5, 2D;
MULH H3.xyz, H3, H1;
TEX H2, f[TEX2], TEX2, 2D;
MULH H0.xyz, H0, H2;
TEX H2, f[TEX1], TEX1, 2D;
MADH H1.xyz, H3, f[COL0], H0;
MOVH H0, f[COL0];
MULH_m2 H0.xyz, H1, H2;
END

# Passes = 8 

# Registers = 2 

# Textures = 6 
