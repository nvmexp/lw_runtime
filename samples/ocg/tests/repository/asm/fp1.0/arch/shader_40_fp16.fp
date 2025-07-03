!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
DP3H_SAT H3, H1, C0;
MULH H2.xyz, H3, f[TEX2];
MOVH H0.w, C3;
DP3H_SAT H3, H1, C1;
MADH H2.xyz, H3, f[TEX3], H2;
DP3H_SAT H3, H1, C2;
MADH H2.xyz, H3, f[COL0], H2;
MULH H2.xyz, H2, C3;
MULH H0.xyz, H2, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 8 

# Registers = 2 

# Textures = 4 
