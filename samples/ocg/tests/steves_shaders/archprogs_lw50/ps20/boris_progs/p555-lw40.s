ps_2_0
def c5, 2.000000, -1.000000, 1.000000, 0.333333
dcl t0.rg
dcl t1.rg
dcl_pp t2.rg
dcl t3.rg
dcl t4.rgb
dcl t5.rgb
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s6
texld r0, t4, s6
dp3 r2.r, t5, t5
mad r0.rgb, c5.r, r0, c5.g
dp3 r1.r, t5, r0
mul r0.rgb, r2.r, r0
add r0.a, r1.r, r1.r
add r5.a, -r1.r, c5.b
mad r0.rgb, r0.a, t5, -r0
texld r3, r0, s2
texld r4, t3, s4
texld r2, t2, s1
texld r0, t0, s0
texld r1, t1, s3
mul r3.rgb, r3, r4
mul r4.rgb, r3, c0
mad r3.rgb, r4, r4, -r4
mul r2.a, r5.a, r5.a
mad r4.rgb, c2, r3, r4
mul r2.a, r2.a, r2.a
dp3 r5.r, r4, c5.a
mul r2.a, r5.a, r2.a
lrp r3.rgb, c3, r4, r5.r
mad r2.a, r2.a, c4.b, c4.a
mul r3.rgb, r3, r2.a
mul r2.rgb, r2, c1
mul r2.rgb, r2, c6.r
mad r2.rgb, c6.b, r2, c6.g
mul r0.rgb, r0, r1
mul r0.a, r0.a, r1.a
mad r0.rgb, r0, r2, r3
mov oC0, r0
