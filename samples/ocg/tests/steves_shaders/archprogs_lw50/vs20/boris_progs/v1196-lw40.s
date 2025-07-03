vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_texcoord1 v3
dcl_texcoord2 v4
dcl_tangent0 v5
dcl_binormal0 v6
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp3 oT5.x, v5, c[42]
dp3 oT6.x, v5, c[43]
dp3 oT7.x, v5, c[44]
dp3 oT5.y, v6, c[42]
dp3 oT6.y, v6, c[43]
dp3 oT7.y, v6, c[44]
dp3 oT5.z, v1, c[42]
dp3 oT6.z, v1, c[43]
dp3 oT7.z, v1, c[44]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r2.y, r0, c[6]
add oT4.xyz, -r1, c[2]
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
add oT1.y, r0.y, r0.x
mov r0.w, v4.x
add r1.x, r0.w, v3.x
add r0.x, r1.x, v4.x
mov r0.z, v3.y
add oT3.z, r0.x, v4.x
add oT3.w, r0.z, c[0].x
mad oFog, -r2.y, c[16].w, c[16].x
mov oPos.z, r2.y
mov oT1.zw, c[0].x
mov r1.z, v3.y
add r1.y, v3.y, c[0].x
add r0.y, r1.z, c[0].x
mov oT2.xy, r1
mov oT2.zw, r0.xyxy
mov oT3.xy, v3
mov oD0, c[38]
