!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX H1, f[TEX1], TEX1, 2D;
DP3H_SAT H0, H1, C0;
MULH H2.xyz, H0, f[TEX2];
DP3H_SAT H0, H1, C1;
MADH H2.xyz, H0, f[TEX3], H2;
DP3H_SAT H0, H1, C2;
MADH H2.xyz, H0, f[COL0], H2;
TEX H0.xyz, f[TEX0], TEX0, 2D;
MULH H2.xyz, H2, C3;
MULH_m2 H0.xyz, H2, H0;
MOVH H0.w, C3;
END

# Passes = 8 

# Registers = 2 

# Textures = 4 
