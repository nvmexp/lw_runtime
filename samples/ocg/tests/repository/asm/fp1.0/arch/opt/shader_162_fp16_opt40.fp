!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX2], TEX2, 2D;
TEX H2, f[TEX5], TEX3, 2D;
DP3H_SAT H2.y, H0, H2;
MOVH H2.z, C0.x;
TEX H1, f[TEX0], TEX1, 2D;
DP3H_SAT H2.x, H0, f[TEX4];
TEX H0, f[TEX0], TEX0, 2D;
MULH H2, H2.x, H0;
TEX H0, f[TEX3], TEX5, 2D;
MADH H2, H0, H1, H2;
DP3H_SAT H0, f[TEX1], f[TEX1];
ADDH H0.x, {1, 1, 1, 1}, -H0.x;
MULH H0, H0.x, H2;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 10 

# Registers = 2 

# Textures = 6 
