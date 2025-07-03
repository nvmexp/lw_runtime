!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX HC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEX HC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEX HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX H0, f[TEX0], TEX0, 2D;
MADH H0.xyz, H0.w, -C0, C0;
MADH H0.xyz, H0.w, f[COL0], H0;
MOVH H0.w, C0.w;
END

# Passes = 5 

# Registers = 2 

# Textures = 4 
