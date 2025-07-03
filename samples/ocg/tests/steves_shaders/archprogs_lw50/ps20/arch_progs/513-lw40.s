//;; Id: 513   pixel count: 954124 lw40 ppc: 0.941176470588
ps_2_0
def c2, 0.816497, 0.000000, 0.577350, 0.000000
def c3, -0.408248, 0.707107, 0.577350, 0.000000
def c4, -0.408248, -0.707107, 0.577350, 0.000000
def c5, 2.000000, -1.000000, 16.000000, 0.062500
def c6, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl t1.rg
dcl t3.rgb
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl v0.rgb
dcl v1.rgb
dcl t7.rgb
dcl_2d s0
dcl_lwbe s1
dcl_2d s3
texld r1, t1, s3
mad r3.rgb, c5.r, r1, c5.g
dp3 r0.r, r3, t4
dp3 r0.g, r3, t5
dp3 r0.b, r3, t6
dp3 r2.r, r0, t3
dp3 r1.r, r0, r0
add r0.a, r2.r, r2.r
mul r1.rgb, r1.r, t3
mad r0.rgb, r0.a, r0, -r1
texld r2, r0, s1
texld r0, t0, s0
mul r1.rgb, r2.a, r2
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c0
mul r2.rgb, r1, c5.b
dp3_sat r1.r, r3, c3
mul r1.rgb, r1.r, v1
dp3_sat r4.r, r3, c2
dp3_sat r3.r, r3, c4
mad r1.rgb, r4.r, v0, r1
mad r1.rgb, r3.r, t7, r1
mul r1.rgb, r1, c1
mad r0.rgb, r0, r1, r2
dp3 r1.r, r0, c6
mul r0.a, r1.r, c5.a
mov oC0, r0
