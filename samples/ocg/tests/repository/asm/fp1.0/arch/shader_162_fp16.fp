!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX5], TEX3, 2D;
DP3H_SAT H3.y, H2, H3;
MOVH H3.z, C0.x;
DP3H_SAT H3.x, H2, f[TEX4];
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX0], TEX1, 2D;
TEX H5, f[TEX3], TEX5, 2D;
MULH H3, H3.x, H0;
MADH H3, H5, H1, H3;
DP3H_SAT H0, f[TEX1], f[TEX1];
ADDH H0.x, {1, 1, 1, 1}, -H0.x;
MULH H0, H0.x, H3;
MULH H0, H0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 12 

# Registers = 3 

# Textures = 6 
