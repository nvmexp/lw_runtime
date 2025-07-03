vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
mul r0, v0.x, c[0]
mad r2, v0.y, c[1], r0
mad r4, v0.z, c[2], r2
mad r6, v0.w, c[3], r4
mul r1, r6.x, c[4]
mad r3, r6.y, c[5], r1
mad r0, r6.z, c[6], r3
mad oPos, r6.w, c[7], r0
add oT1.xyz, -v0, c[8]
mov oT0.xyz, v1
mov oT2.xy, v2
