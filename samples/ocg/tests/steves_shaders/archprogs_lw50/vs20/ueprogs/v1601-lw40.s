vs_2_0
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_color0 v5
mul r0.xyz, v1.y, c[5]
mad r0.xyz, c[4], v1.x, r0
mad r1.xyz, c[6], v1.z, r0
mul r0.xyz, v0.y, c[1]
add r1.xyz, r1, c[7]
mad r0.xyz, c[0], v0.x, r0
mad r2.xyz, c[2], v0.z, r0
mul r0.xyz, v2.y, c[5]
mad r3.xyz, c[3], v0.w, r2
mad r0.xyz, c[4], v2.x, r0
mad r2.xyz, r3, -c[12].w, c[12]
mad r0.xyz, c[6], v2.z, r0
dp3 oT6.x, r1, r2
add r0.xyz, r0, c[7]
mov oT7.x, r1.z
dp3 oT6.y, r0, r2
mov oT7.y, r0.z
mul r0.xyz, v3.y, c[5]
mad r1.xyz, c[4], v3.x, r0
mul r0, r3.y, c[9]
mad r1.xyz, c[6], v3.z, r1
mad r0, c[8], r3.x, r0
add r1.xyz, r1, c[7]
mad r0, c[10], r3.z, r0
dp3 oT6.z, r1, r2
add r0, r0, c[11]
mov oT7.z, r1.z
mov oPos, r0
mov oT0.xy, v5
mov oT1.xy, v4
mov oT5, r0
mov oT6.w, r0.w
