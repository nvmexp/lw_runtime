!!FP2.0
TEX R3, f[TEX0], TEX1, 2D;
ADDR R3.xyz, R3, {1.987654, 1.788889, 1.610000, 1.449000}.x;
DP3R R0.x, f[TEX2], R3;
DP3R R0.z, f[TEX4], R3;
DP3R R0.y, f[TEX3], R3;
DP3R R0.w, R0, R0;
MOVR R2.w, f[COL0].x;
LG2R_d2 R0.w, |R0.w|;
TEX R3.xyz, f[TEX0], TEX0, 2D;
EX2R R0.w, -R0.w;
MULR R2.xyz, R0, R0.w;
MOVR R0.w, {0.855620, 0.770058, 0.693052, 0.623747}.y;
MULR R0.xyz, R2, {0.561372, 0.505235, 0.454712, 0.409240}.x;
MOVR R1.xy, f[TEX1];
MOVR R1.w, f[TEX1].xywz;
ADDR R0.xyz, R1.xywz, R0;
END

# Passes = 13 

# Registers = 4 

# Textures = 5 
