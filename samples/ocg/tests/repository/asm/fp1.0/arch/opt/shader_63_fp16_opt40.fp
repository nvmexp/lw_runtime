!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H2, f[TEX0], TEX0, 2D;
DP3H H1, H2, C0;
TEX H0, f[TEX1], TEX1, 2D;
MULH H0, H0, H1;
TEX H3, f[TEX2], TEX2, 2D;
DP3H H1, H2, C1;
MADH H0, H3, H1, H0;
TEX H3, f[TEX3], TEX3, 2D;
DP3H H1, H2, C2;
MADH H0, H3, H1, H0;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
