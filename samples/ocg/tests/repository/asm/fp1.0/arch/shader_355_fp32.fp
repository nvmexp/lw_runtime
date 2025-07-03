!!FP1.0
TEX R1, f[TEX0], TEX1, 2D;
ADDR R1.xyz, R1, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3R R0.x, f[TEX2], R1;
DP3R R0.z, f[TEX4], R1;
DP3R R0.y, f[TEX3], R1;
DP3R R0.w, R0, R0;
LG2R R0.w, |R0.w|;
MULR R0.w, R0, {0.5, 0, 0, 0}.x; 
EX2R R0.w, -R0.w;
MULR R2.xyz, R0, R0.w;
MOVR R0.w, {0.855620, 0.770058, 0.693052, 0.623747}.y;
MULR R0.xyz, R2, {0.561372, 0.505235, 0.454712, 0.409240}.x;
ADDR R0.xyz, f[TEX1], R0;
MOVR R2.w, f[COL0].x;
TEX R3.xyz, f[TEX0], TEX0, 2D;
MOVR R3.w, R1.w;
MOVR o[COLR], R0; 
END

# Passes = 14 

# Registers = 4 

# Textures = 5 
