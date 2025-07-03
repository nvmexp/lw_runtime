!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX H1, f[TEX0], TEX0, 2D;
MULH H0, H1, C3;
TEX H1, f[TEX2], TEX2, 2D;
ADDH H1.w, {1, 1, 1, 1}, -H1;
TEX H1.xyz, f[TEX1], TEX1, 2D;
MULH H1, H1, H1.w;
MADH H0.xyz, H1, C2, H0;
MULH H0.xyz, f[COL0], H0;
MADH_m2 H0.xyz, C0, H0, C0.w;
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
