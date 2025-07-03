!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX HC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEX H0, f[TEX0], TEX0, 2D;
MULH H2.xyz, H0, C2;
MOVH H2.w, C2.w;
MULH H2.xyz, f[COL0], H2;
MULH_m2 H2.xyz, C0, H2;
MADH H2.xyz, H0.w, -H2, H2;
MULH H1, H0, C1;
MADH H0.xyz, H0.w, H1, H2;
END

# Passes = 6 

# Registers = 2 

# Textures = 2 
