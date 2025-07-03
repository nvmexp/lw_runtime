!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H1.x, H2, f[TEX4];
DP3H_SAT H1.y, H2, f[TEX5];
MOVH H1.z, C0.x;
TEX H2, H1, TEX5, 2D;
TEX H0, f[TEX0], TEX0, 2D;
MULH H1, H1.x, H0;
TEX H0, f[TEX0], TEX1, 2D;
MADH H1, H2, H0, H1;
DP3H_SAT H0.x, f[TEX1], f[TEX1];
TEX H2, f[TEX3], TEX4, 2D;
ADDH H0.x, {1, 1, 1, 1}, -H0.x;
MULH H0, H2, H0.x;
MULH H0, H0, H1;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 10 

# Registers = 2 

# Textures = 6 
