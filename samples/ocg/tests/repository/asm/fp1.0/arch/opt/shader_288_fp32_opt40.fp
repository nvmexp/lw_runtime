!!FP2.0
TEX R0, f[TEX1], TEX4, 2D;
MADR R0.x, {1.987654, 1.788889, 1.610000, 1.449000}.x, R0, {1.987654, 1.788889, 1.610000, 1.449000}.y;
MULR R1.xyz, R0.x, f[TEX6];
MADR R1.xyz, f[TEX5], R0.x, R1;
MADR R1.xyz, f[TEX7], R0.x, R1;
DP3R R2.y, R1, f[TEX4];
DP3R R2.x, R1, R1;
RCPR R2.w, R2.x;
MULR R1.w, R2, R2.y;
MOVR R2.xy, f[TEX3];
MOVR R2.zw, f[TEX3].xxxy;
ADDR R4.w, R1, R1;
ADDR R2.xy, f[TEX2], R2;
ADDR R2.zw, R2.xxxy, R2;
TEX R5, R2.zwzz, TEX1, 2D;
DP3R R1.w, f[TEX4], f[TEX4];
TEX R2, R2, TEX1, 2D;
LG2R R1.w, |R1.w|;
MULR R1.w, R1, {1.304100, 1.173690, 1.056321, 0.950689}.x;
EX2R R5.w, R1.w;
TEX R3, f[TEX2], TEX1, 2D;
DP3R_SAT R1.w, R0.x, {0.855620, 0.770058, 0.693052, 0.623747};
MULR R4.xyz, R2, R1.w;
MOVR R1.w, {0.561372, 0.505235, 0.454712, 0.409240}.x;
MULR R2.xyz, R5.w, f[TEX4];
DP3R R1.z, R1, R2;
ADDR R3.w, -R1.z, {0.368316, 0.331485, 0.298336, 0.268503}.x;
DP3R_SAT R1.z, R0.x, {0.241652, 0.217487, 0.195738, 0.176165};
MADR R2.yzw, R1.z, R3.xxyz, R4.xxyz;
MULR R5.w, R3, R3;
DP3R_SAT R2.x, R0.x, {0.158548, 0.142693, 0.128424, 0.115582};
MULR R5.w, R5, R5;
MOVR R0.w, R0.x;
MADR R0.xy, R4.w, R1, -f[TEX4];
MULR R5.w, R3, R5;
MADR R5.w, R5, R1, {0.104023, 0.093621, 0.084259, 0.075833}.x;
MADR R5.xyz, R2.x, R5, R2.yzwy;
TEX R1, R0, TEX2, 2D;
MULR R0.xyz, R0.w, R1;
MULR R0.xyz, R0, {0.068250, 0.061425, 0.055282, 0.049754};
MULR R1.xyz, R5, {0.044779, 0.040301, 0.036271, 0.032644};
MULR R2.xyz, R1, {0.029379, 0.026441, 0.023797, 0.021417}.x;
MADR R1.xyz, R0, R0, -R0;
MADR R0.xyz, {0.019276, 0.017348, 0.015613, 0.014052}, R1, R0;
DP3R R1.x, R0, {0.012647, 0.011382, 0.010244, 0.009220}.x;
ADDR R0.xyz, R0, -R1.x;
MULR R1.yzw, {0.008298, 0.007468, 0.006721, 0.006049}.wxyz, R0.xxyz;
TEX R0, f[TEX0], TEX0, 2D;
ADDR R1.xyz, R1.x, R1.yzwy;
MULR R1.xyz, R1, R5.w;
MULR R0, R0, f[COL0];
MADR H0.xyz, R0, R2, R1;
MOVR H0.w, R0;
END

# Passes = 39 

# Registers = 6 

# Textures = 8 
