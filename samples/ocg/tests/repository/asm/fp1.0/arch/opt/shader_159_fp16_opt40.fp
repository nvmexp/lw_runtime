!!FP2.0 
DECLARE C0={10, 20, 30, 40};
TEX H2, f[TEX2], TEX2, 2D;
MOVH HC, H2.z;
TEX H0, f[TEX0], TEX0, 2D;
DP3H_SAT H3.x, H2, f[TEX4];
TEX H1, f[TEX3], TEX4, 2D;
MULH H3, H3.x, H0;
DP3H_SAT H0, f[TEX1], f[TEX1];
MOVH H2(GE), C0.x;
ADDH H0.x, {1, 1, 1, 1}, -H0;
MULH H0, H1, H0.x;
MULH H0, H0, H3;
MOVH H2(LT), C0.y;
MULH H0, H0, H2;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 9 

# Registers = 2 

# Textures = 5 
