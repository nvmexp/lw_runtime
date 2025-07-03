ps_2_0
def c0, 0.218500, 0.201300, 0.082100, 0.046100
def c1, 0.026200, 0.016200, 0.010200, 0.000000
dcl t0.rg
dcl t1.rg
dcl t2.rg
dcl t3
dcl t4
dcl t5
dcl t6
dcl t7
dcl_2d s0
texld r1, t1, s0
texld r2, t2, s0
texld r0, t0, s0
add r1, r1, r2
mul r1, r1, c0.r
mad r9, r0, c0.g, r1
mov r4.rg, t3.abgr
mov r3.rg, t4.abgr
mov r2.rg, t5.abgr
mov r1.rg, t6.abgr
mov r0.rg, t7.abgr
texld r10, r4, s0
texld r7, t3, s0
texld r8, r3, s0
texld r5, t4, s0
texld r6, r2, s0
texld r3, t5, s0
texld r4, r1, s0
texld r1, t6, s0
texld r2, r0, s0
texld r0, t7, s0
add r7, r10, r7
mad r7, r7, c0.b, r9
add r5, r8, r5
mad r5, r5, c0.a, r7
add r3, r6, r3
mad r3, r3, c1.r, r5
add r1, r4, r1
mad r1, r1, c1.g, r3
add r0, r2, r0
mad r0, r0, c1.b, r1
mov oC0, r0
