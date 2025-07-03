!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX R0.xyz, f[TEX0], TEX0, 2D;
MOVR R2, f[COL0];
MULR R2.xyz, R0, R2;
TEX R1, f[TEX1], TEX1, 2D;
MULR R2.xyz, R1, R2;
MULR_m2 R2.xyz, C0, R2;
MADR R2.xyz, R2.w, -R2, R2;
MULR R1.xy, C1, R0;
MULR R1.w, C1.xywz, R0.xywz;
MADR R0.xyz, R2.w, R1.xywz, R2;
MOVR R0.w, R2.w;
END

# Passes = 4 

# Registers = 3 

# Textures = 2 
