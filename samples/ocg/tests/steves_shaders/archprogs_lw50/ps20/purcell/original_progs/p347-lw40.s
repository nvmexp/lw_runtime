ps_2_0
def c4, 2.000000, -1.000000, 0.333333, 1.000000
dcl t0.rg
dcl t1
dcl t2
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7.rgb
dcl_2d s0
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
texld r3, t0, s3
texld r0, t0, s0
mad r1.rgb, c4.r, r3, c4.g
mul r2.rgb, r1.g, t6
mad r2.rgb, t5, r1.r, r2
mad r1.rgb, t7, r1.b, r2
dp3 r2.r, r1, r1
rcp r0.a, r2.r
dp3 r2.r, r1, t4
mul r0.a, r0.a, r2.r
add r0.a, r0.a, r0.a
mad r1.rgb, r0.a, r1, -t4
rcp_pp r1.a, t1.a
rcp_pp r0.a, t2.a
mul r3.rgb, r1.a, t1
mul r2.rgb, r0.a, t2
dp3 r3.r, r3, r0
dp3 r3.g, r2, r0
texld r2, r1, s4
texld r1, r3, s2
texld r0, t0, s5
mul r2.rgb, r3.a, r2
mul r3.rgb, r2, c1
mad r2.rgb, r3, r3, -r3
mad r3.rgb, c2, r2, r3
dp3 r4.r, r3, c4.b
mul r2.rgb, r1, c0
lrp r1.rgb, c3, r3, r4.r
mul r0.rgb, r0, r2
mad r0.rgb, c4.r, r0, r1
mov r0.a, c4.a
mov oC0, r0
