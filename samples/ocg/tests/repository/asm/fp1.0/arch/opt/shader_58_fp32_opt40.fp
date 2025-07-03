!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C3={1, 2, 3, 4};
TEX RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX R1, f[TEX0], TEX0, 2D;
MULR R0.xyz, R1, C3;
MOVR R0.w, C3.w;
MOVR R2, f[COL0];
MULR R0.xyz, R2, R0;
MULR_m2 R0.xyz, C0, R0;
MADR R2.xyz, R1.w, -R0, R0;
MOVR R2.w, R1.w;
MULR R1.xy, R1, C1;
MULR R1.w, R1.z, C1.z;
MADR R0.xyz, R2.w, R1.xyww, R2;
END

# Passes = 5 

# Registers = 3 

# Textures = 2 
