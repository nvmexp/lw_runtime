!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX3], TEX3, 2D;
TEX H5, f[TEX5], TEX5, 2D;
DP3H_SAT H2.xyz, H2, H3;
TEX H4, f[TEX4], TEX4, 2D;
MULH H2.xyz, H2, H4;
MADH H1.xyz, H2, f[COL0], H1;
MULH H0.xyz, H0, H1;
MULH H0.xyz, H0, H5;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH H0.w, f[COL0].w;
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 3 

# Textures = 6 
