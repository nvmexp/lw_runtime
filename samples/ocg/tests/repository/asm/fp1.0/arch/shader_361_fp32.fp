!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
DP3R R0.w, R0, R0;
TEX R1, f[TEX0], TEX1, 2D;
LG2R R0.w, |R0.w|;
MULR R0.w, R0, {0.5, 0, 0, 0}.x; 
EX2R R0.w, -R0.w;
MADR R0.xyz, R0, -R0.w, -{1.304100, 1.173690, 1.056321, 0.950689}.x;
DP3R R0.w, R0, R0;
LG2R R0.w, |R0.w|;
MULR R0.w, R0, {0.5, 0, 0, 0}.x; 
EX2R R0.w, -R0.w;
MULR R0.xyz, R0, R0.w;
DP3R R1.w, -R0, R1;
DP3R R1.x, -{1.304100, 1.173690, 1.056321, 0.950689}.x, R1;
TEX R0, R1.xwxx, TEX2, 2D;
MULR o[COLR], R0, {0.368316, 0.331485, 0.298336, 0.268503}.x;
END

# Passes = 10 

# Registers = 2 

# Textures = 1 
