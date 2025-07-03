vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
dcl_color0 v2
mul r0.xyz, v0.y, c[1]
mad r0.xyz, c[0], v0.x, r0
mad r0.xyz, c[2], v0.z, r0
mad r1.xyz, c[3], v0.w, r0
mul r0, r1.y, c[5]
mad r0, c[4], r1.x, r0
mad r0, c[6], r1.z, r0
add r0, r0, c[7]
mov oPos, r0
mov oT7, r0
mov oT0.xy, v2
mov oT1.xy, v1
