!!FP1.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
MOVR R0.xyz, f[TEX1];
MOVR R0.w, C0.x;
DP3R R2.w, f[TEX2], f[TEX2];
LG2R R2.w, |R2.w|;
MULR R2.w, R2, {0.5, 0, 0, 0}.x; 
EX2R R2.w, -R2.w;
MULR R2.xyz, R2.w, f[TEX2];
MOVR R2.w, f[COL0].x;
TEX R3.xyz, f[TEX0], TEX0, 2D;
MOVR R3.w, C0.x;
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 4 

# Textures = 3 
