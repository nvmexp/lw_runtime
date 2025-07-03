ps_2_0
def c4, -0.001953, 0.000000, 0.001953, 0.111111
def c5, -0.001953, 0.001953, 2.000000, -1.000000
def c6, 0.000000, -0.001953, 0.333333, 1.000000
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
dcl_2d s5
texld r0, t0, s0
texld r4, t0, s3
rcp_pp r1.a, t1.a
rcp_pp r0.a, t2.a
mul r2.rgb, r1.a, t1
mul r1.rgb, r0.a, t2
dp3 r5.r, r2, r0
dp3 r5.g, r1, r0
add r9.rg, r5, c4.r
add r8.rg, r5, c4
add r7.rg, r5, c5
add r6.rg, r5, c6
add r3.rg, r5, c7
add r2.rg, r5, c8
add r1.rg, r5, c9
add r0.rg, r5, c4.b
texld r9, r9, s2
texld r10, r8, s2
texld r8, r7, s2
texld r7, r6, s2
texld r6, r5, s2
texld r3, r3, s2
texld r5, r2, s2
texld r2, r1, s2
texld r1, r0, s2
texld r0, t0, s5
add r10.rgb, r9, r10
mad r4.rgb, c5.b, r4, c5.a
mul r9.rgb, r4.g, t6
mad r9.rgb, t5, r4.r, r9
mad r9.rgb, t7, r4.b, r9
add r4.rgb, r8, r10
dp3 r8.r, r9, r9
rcp r0.a, r8.r
dp3 r8.r, r9, t4
add r4.rgb, r7, r4
mul r0.a, r0.a, r8.r
add r4.rgb, r6, r4
add r0.a, r0.a, r0.a
add r4.rgb, r3, r4
mad r3.rgb, r0.a, r9, -t4
texld r3, r3, s4
add r4.rgb, r5, r4
mul r3.rgb, r4.a, r3
add r2.rgb, r2, r4
mul r3.rgb, r3, c1
add r1.rgb, r1, r2
mad r2.rgb, r3, r3, -r3
mul r1.rgb, r1, c0
mad r2.rgb, c2, r2, r3
mul r1.rgb, r1, c4.a
dp3 r3.r, r2, c6.b
mul r0.rgb, r0, r1
lrp r1.rgb, c3, r2, r3.r
mad r0.rgb, c5.b, r0, r1
mov r0.a, c6.a
mov oC0, r0
