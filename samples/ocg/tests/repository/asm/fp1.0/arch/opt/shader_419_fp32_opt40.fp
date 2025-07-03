!!FP2.0
TEX R0, f[TEX0], TEX3, 2D;
NRMH R1.xyz, f[TEX1];
DP3R R0.z, R1, R0;
MOVR R1.xy, f[TEX2];
ADDR R0.z, -R0, {0.855620, 0.770058, 0.693052, 0.623747}.x;
MULR R1.w, R0.z, R0.z;
DP2AR R2.w, R1.x, f[TEX3], R0.xyxy;
MOVR R1.xz, f[TEX4].xxyy;
MULR R1.w, R1, R1;
MOVR R2.xy, f[TEX7];
MULR R1.w, R0.z, R1;
MOVR R0.z, f[TEX2];
DP2AR R3.w, R1.y, R1.xzxx, R0.xyxy;
RCPR R1.z, R0.z;
MULR R3.x, R2.w, R1.z;
MOVR R1.xy, f[TEX5];
MULR R3.y, R1.z, R3.w;
TEX R3, R3, TEX4, 2D;
DP2AR R2.y, R1.y, R2, {0.987, 0, 0, 0}.x;
DP2AR R2.x, R1.x, f[TEX6], R0.xyxy;
MULR R0.xy, R1.z, R2.xyyy;
MOVR R1.z, R1.w;
TEX R2, R0, TEX2, 2D;
MULR R0.xyz, R3, {0.561372, 0.505235, 0.454712, 0.409240};
MULR R1.xy, R2, {0.368316, 0.331485, 0.298336, 0.268503};
MULR R1.w, R2.z, {0.368316, 0.331485, 0.298336, 0.268503}.z;
MADR R0.xyz, R0, R1.z, R1.xyww;
END

# Passes = 14 

# Registers = 4 

# Textures = 8 
