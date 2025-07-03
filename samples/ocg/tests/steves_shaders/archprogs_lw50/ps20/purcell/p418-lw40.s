ps_2_0
def c4, 0.000000, -0.001953, 0.111111, 1.000000
def c5, -0.001953, 0.000000, 0.001953, 0.333333
def c6, -0.001953, 0.001953, 2.000000, -1.000000
def c7, 0.000000, 0.001953, 0.000000, 0.000000
def c8, 0.001953, -0.001953, 0.000000, 0.000000
def c9, 0.001953, 0.000000, 0.000000, 0.000000
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
texld r0, t0, s0
texld r1, t0, s3
rcp_pp r2.a, t1.a
rcp_pp r0.a, t2.a
mul r3.rgb, r2.a, t1
mul r2.rgb, r0.a, t2
dp3 r5.r, r3, r0
dp3 r5.g, r2, r0
add r9.rg, r5, c5.r
mad r0.rgb, c6.b, r1, c6.a
add r8.rg, r5, c5
mul r1.rgb, r0.g, t6
add r7.rg, r5, c6
mad r1.rgb, t5, r0.r, r1
add r6.rg, r5, c4
mad r0.rgb, t7, r0.b, r1
add r4.rg, r5, c7
dp3 r1.r, r0, r0
rcp r0.a, r1.r
dp3 r1.r, r0, t4
add r3.rg, r5, c8
mul r0.a, r0.a, r1.r
add r2.rg, r5, c9
add r0.a, r0.a, r0.a
add r1.rg, r5, c5.b
mad r0.rgb, r0.a, r0, -t4
texld r9, r9, s2
texld r10, r8, s2
texld r8, r7, s2
texld r7, r6, s2
texld r6, r5, s2
texld r5, r4, s2
texld r4, r3, s2
texld r3, r2, s2
texld r2, r1, s2
texld r0, r0, s4
add r1.rgb, r9, r10
add r1.rgb, r8, r1
add r1.rgb, r7, r1
add r1.rgb, r6, r1
add r1.rgb, r5, r1
add r1.rgb, r4, r1
add r1.rgb, r3, r1
add r1.rgb, r2, r1
mul r0.rgb, r1.a, r0
mul r2.rgb, r0, c1
mad r0.rgb, r2, r2, -r2
mad r2.rgb, c2, r0, r2
dp3 r3.r, r2, c5.a
mul r0.rgb, r1, c0
lrp r1.rgb, c3, r2, r3.r
mad r0.rgb, r0, c4.b, r1
mov r0.a, c4.a
mov oC0, r0
