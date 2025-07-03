!!FP2.0
TEX H1, f[TEX3], TEX3, 2D;
MOVH H0.w, f[COL0].w;
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H2.xyz, H2, H1;
TEX H0.xyz, f[TEX4], TEX4, 2D;
MULH H2.xyz, H2, H0;
TEX H1, f[TEX1], TEX1, 2D;
MADH H1.xyz, H2, f[COL0], H1;
TEX H0.xyz, f[TEX0], TEX0, 2D;
MULH H0.xyz, H0, H1;
TEX H1.xyz, f[TEX5], TEX5, 2D;
MULH_m2 H0.xyz, H0, H1;
END

# Passes = 6 

# Registers = 2 

# Textures = 6 
