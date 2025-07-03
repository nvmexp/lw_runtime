vs_2_0
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_color0 v5
dcl_blendweight0 v6
mul r0, v0.y, c[1]
mul r1.xyz, v1.y, c[5]
mad r0, c[0], v0.x, r0
mad r1.xyz, c[4], v1.x, r1
mad r0, c[2], v0.z, r0
mad r1.xyz, c[6], v1.z, r1
mad r0, c[3], v0.w, r0
add r5.xyz, r1, c[7]
add r1.xyz, -r0, c[12]
dp3 oT4.x, r5, r1
mul r3.xyz, v2.y, c[5]
mul r2.xyz, v3.y, c[5]
mad r3.xyz, c[4], v2.x, r3
mad r2.xyz, c[4], v3.x, r2
mad r3.xyz, c[6], v2.z, r3
mad r2.xyz, c[6], v3.z, r2
add r4.xyz, r3, c[7]
add r2.xyz, r2, c[7]
dp3 oT4.y, r4, r1
dp3 oT4.z, r2, r1
mul oT5.xyz, r1, c[12].w
mul r1, r0.y, c[9]
mad r3.xyz, r0, -c[13].w, c[13]
mad r1, c[8], r0.x, r1
dp3 oT6.x, r5, r3
mad r1, c[10], r0.z, r1
dp3 oT6.y, r4, r3
mad r0, c[11], r0.w, r1
dp3 oT6.z, r2, r3
mov oPos, r0
mov oT7, r0
mov oT0.xy, v5
mov oT1.xy, v4
mov oD0.x, v6.x
