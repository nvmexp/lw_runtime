//;; Id: 503   pixel count: 29124374 lw40 ppc: 1.6
ps_2_0
def c0, 2.000000, -1.000000, 16.000000, 0.062500
def c1, 0.816497, 0.000000, 0.577350, 0.000000
def c2, -0.408248, 0.707107, 0.577350, 0.000000
def c3, -0.408248, -0.707107, 0.577350, 0.000000
def c4, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl t1.rg
dcl_pp t2
dcl_pp t3
dcl v0.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s4
texld r0, t2, s1
mul r0.rgb, r0.a, r0
mul r4.rgb, r0, c0.b
mov r1.r, t2.b
mov r1.g, t2.a
mov r0.r, t3.b
mov r0.g, t3.a
texld r3, r1, s1
texld r2, t1, s4
texld r1, r0, s1
texld r0, t0, s0
mul r3.rgb, r3.a, r3
mul r5.rgb, r3, c0.b
mad r3.rgb, c0.r, r2, c0.g
dp3_sat r2.r, r3, c2
mul r2.rgb, r5, r2.r
dp3_sat r5.r, r3, c1
dp3_sat r3.r, r3, c3
mad r2.rgb, r5.r, r4, r2
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c0.b
mad r1.rgb, r3.r, r1, r2
mul r0.rgb, r0, v0
mul r0.rgb, r1, r0
dp3 r1.r, r0, c4
mul r0.a, r1.r, c0.a
mov oC0, r0
