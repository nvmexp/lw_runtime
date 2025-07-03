;; Id: 565   pixel count: 947412 lw40 ppc: 1.33333333333
ps_2_0
def c0, 16.000000, 2.000000, -1.000000, 0.062500
def c1, 0.816497, 0.000000, 0.577350, 0.000000
def c2, -0.408248, 0.707107, 0.577350, 0.000000
def c3, -0.408248, -0.707107, 0.577350, 0.000000
def c4, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0
dcl t1.rg
dcl_pp t2
dcl_pp t3
dcl v0.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s4
dcl_2d s6
mov r2.r, t3.b
mov r2.g, t3.a
mov r0.r, t2.b
mov r0.g, t2.a
mov r1.r, t0.b
mov r1.g, t0.a
texld r6, r2, s1
texld r5, r0, s1
texld r4, t1, s4
texld r0, t2, s1
texld r3, t3, s1
texld r2, r1, s6
texld r1, t0, s0
mul r6.rgb, r6.a, r6
mul r6.rgb, r6, c0.r
mul r5.rgb, r5.a, r5
mul r7.rgb, r5, c0.r
mad r5.rgb, c0.g, r4, c0.b
dp3_sat r4.r, r5, c2
mul r4.rgb, r7, r4.r
mul r0.rgb, r0.a, r0
mul r0.rgb, r0, c0.r
dp3_sat r7.r, r5, c1
dp3_sat r5.r, r5, c3
mad r0.rgb, r7.r, r0, r4
mad r0.rgb, r5.r, r6, r0
mul r3.rgb, r3.a, r3
mul r3.rgb, r3, c0.r
mul r2.rgb, r2, r3
mul r1.rgb, r1, v0
mad r0.rgb, r0, r1, -r2
mad r0.rgb, r0.a, r0, r2
dp3 r1.r, r0, c4
mul r0.a, r1.r, c0.a
mov oC0, r0
