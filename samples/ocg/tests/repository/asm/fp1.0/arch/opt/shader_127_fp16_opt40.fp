!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H3.x, H2, f[TEX4];
TEX H0, f[TEX0], TEX0, 2D;
DP3H_SAT H3.y, H2, f[TEX5];
MOVH H3.z, C0.x;
TEX H2, H3, TEX5, 2D;
MULH H3, H3.x, H0;
TEX H1, f[TEX0], TEX1, 2D;
MADH H3, H2, H1, H3;
DP3H_SAT H0, f[TEX1], f[TEX1];
ADDH H0, {1, 1, 1, 1}, -H0.x;
MULH H0, H0, H3;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 10 

# Registers = 2 

# Textures = 5 
