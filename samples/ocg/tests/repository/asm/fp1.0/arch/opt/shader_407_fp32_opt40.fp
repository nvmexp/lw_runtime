!!FP2.0
DECLARE C5={1.000000, 0.333333, 2.000000, 0.000000};
MOVR R0.xyz, f[TEX6];
DP3R R0.w, R0, R0;
RCPR R0.w, R0.w;
MOVR R1.xyz, f[TEX3];
DP3R R0.z, R0, R1;
MULR R0.w, R0, R0.z;
MOVR R2.xy, f[TEX3];
ADDR R0.w, R0, R0;
TEX R1, f[TEX0], TEX0, 2D;
ADDR R1.w, -R1, C5.x;
MADR R0.xy, R0.w, R0, -R2;
TEX R0, R0, TEX1, 2D;
MULR R0.xyz, R0, R1.w;
MOVR R0.w, C5.x;
MULR R0.xyz, R0, C5.x;
MADR R2.xyz, R0, R0, -R0;
MADR R0.xyz, C5.x, R2, R0;
MULR R2.xyz, f[COL0], C5.x;
DP3R R1.w, R0, C5.x;
ADDR R0.xyz, R0, -R1.w;
MULR R0.xyz, C5.x, R0;
ADDR R0.xyz, R1.w, R0;
MULR R1.xyz, R1, R2;
MULR R1.xyz, R1, C5.x;
MADR R0.xyz, C5.x, R1, R0;
END

# Passes = 14 

# Registers = 3 

# Textures = 3 
