!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H2, f[TEX0], TEX0, 2D;
TEX H3, f[TEX1], TEX1, 2D;
TEX H4, f[TEX2], TEX2, 2D;
TEX H5, f[TEX3], TEX3, 2D;
DP3H H1, H2, C0;
MULH H0, H3, H1;
DP3H H1, H2, C1;
MADH H0, H4, H1, H0;
DP3H H1, H2, C2;
MADH H0, H5, H1, H0;
MULH H0, H0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 3 

# Textures = 4 
