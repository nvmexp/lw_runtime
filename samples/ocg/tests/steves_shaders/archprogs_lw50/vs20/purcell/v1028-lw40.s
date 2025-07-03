vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_texcoord1 v3
dcl_tangent0 v4
dcl_binormal0 v5
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp3 oT5.x, v4, c[42]
dp3 oT5.y, v4, c[43]
dp3 oT5.z, v4, c[44]
dp3 oT6.x, v5, c[42]
dp3 oT6.y, v5, c[43]
dp3 oT6.z, v5, c[44]
dp3 oT7.x, v1, c[42]
dp3 oT7.y, v1, c[43]
dp3 oT7.z, v1, c[44]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r2.y, r0, c[6]
add oT4.xyz, -r1, c[2]
add r1.xy, -r1.z, c[2].wzzw
mul r0.xy, v2, c[90]
add oT0.x, r0.y, r0.x
mul r0.xy, v2, c[91]
add oT0.y, r0.y, r0.x
mul r0.xy, v2, c[96]
add oT0.z, r0.y, r0.x
mul r0.xy, v2, c[97]
add oT0.w, r0.y, r0.x
mul r0.xy, v2, c[92]
add oT1.x, r0.y, r0.x
mul r0.xy, v2, c[93]
max r0.w, r1.x, c[0].x
rcp r1.w, r1.y
add oT1.y, r0.y, r0.x
mul r0.w, r0.w, r1.w
mul r0.w, r2.y, r0.w
mov oPos.z, r2.y
mad oFog, -r0.w, c[16].w, c[16].y
mov oT1.zw, c[0].x
mov oT2, c[0].x
mov oT3.xy, v3
mov oT3.zw, c[0].x
mov oD0, c[38]
