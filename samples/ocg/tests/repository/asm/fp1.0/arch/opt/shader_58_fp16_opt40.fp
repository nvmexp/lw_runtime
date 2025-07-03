!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C3={1, 2, 3, 4};
TEX HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, C3;
MOVH H0.w, C3.w;
MULH H0.xyz, f[COL0], H0;
MULH_m2 H0.xyz, C0, H0;
MADH H0.xyz, H1.w, -H0, H0;
MULH H1.xyz, H1, C1;
MADH H0.xyz, H1.w, H1, H0;
END

# Passes = 5 

# Registers = 1 

# Textures = 2 
