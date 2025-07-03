!!FP2.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
DP3R R2.w, f[TEX2], f[TEX2];
MOVR R0.w, C0.x;
LG2R_d2 R2.w, |R2.w|;
MOVR R0.xyz, f[TEX1];
EX2R R2.w, -R2.w;
MOVR R3.w, C0.x;
MULR R2.xyz, R2.w, f[TEX2];
TEX R3.xyz, f[TEX0], TEX0, 2D;
MOVR R2.w, f[COL0].x;
END

# Passes = 8 

# Registers = 4 

# Textures = 3 
