!!FP2.0
TEX R0, f[TEX1], TEX3, 2D;
DP3R_SAT R1.w, R0, {1.304100, 1.173690, 1.056321, 0.950689};
DP3R_SAT R2.x, R0, {0.855620, 0.770058, 0.693052, 0.623747};
DP3R_SAT R0.x, R0, {0.561372, 0.505235, 0.454712, 0.409240};
MULR R1.xyz, R0.x, f[COL1];
TEX R0, f[TEX0], TEX0, 2D;
MADR R1.xyz, R2.x, f[COL0], R1;
MADR R1.xyz, R1.w, f[TEX7], R1;
MULR R1.xyz, R1, {0.368316, 0, 0, 0}.x;
MULR R0.w, R0, {0.541652, 0, 0, 0}.x;
MULR R0.xyz, R0, R1;
MADR_m2 H0.xyz, R0, {0.758548, 0, 0, 0}.x, {0.758548, 0, 0, 0}.w;
MOVR H0.w, R0;
END

# Passes = 8 

# Registers = 3 

# Textures = 3 
