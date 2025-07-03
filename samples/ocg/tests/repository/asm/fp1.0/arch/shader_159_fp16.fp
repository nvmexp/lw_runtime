!!FP1.0 
DECLARE C0={10, 20, 30, 40};
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H3.x, H2, f[TEX4];
TEX H0, f[TEX0], TEX0, 2D;
TEX H4, f[TEX3], TEX4, 2D;
MULH H3, H3.x, H0;
DP3H_SAT H0, f[TEX1], f[TEX1];
ADDH H0.x, {1, 1, 1, 1}, -H0;
MULH H0, H4, H0.x;
MULH H0, H0, H3;
MOVHC HC, H2.z;
MOVH H1(GE), C0.x;
MOVH H1(LT), C0.y;
MULH H0, H0, H1;
MULH H0, H0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 3 

# Textures = 5 
