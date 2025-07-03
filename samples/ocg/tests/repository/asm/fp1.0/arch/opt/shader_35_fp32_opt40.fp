!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX RC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEX H0, f[TEX0], TEX0, 2D;
MULR R2.xyz, H0, C2;
MOVR R2.w, C2.w;
MULR R2.xyz, f[COL0], R2;
MULR_m2 R2.xyz, C0, R2;
MADR R2.xyz, H0.w, -R2, R2;
MULR R1, H0, C1;
MADR H0.xyz, H0.w, R1, R2;
END

# Passes = 6 

# Registers = 3 

# Textures = 2 
