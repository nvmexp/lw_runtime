vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord1 v2
dcl_texcoord2 v3
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r2.y, r0, c[6]
mul r0.xy, v1, c[90]
add oT4.xyz, -r1, c[2]
add oT0.x, r0.y, r0.x
mul r1.xy, v1, c[91]
mul r0.xy, v1, c[96]
add oT0.y, r1.y, r1.x
add oT0.z, r0.y, r0.x
mul r1.xy, v1, c[97]
mul r0.xy, v1, c[92]
add oT0.w, r1.y, r1.x
add oT1.x, r0.y, r0.x
mul r0.xy, v1, c[93]
mov r0.w, v3.x
add r1.x, r0.w, v2.x
add oT1.y, r0.y, r0.x
add r0.x, r1.x, v3.x
add oT3.z, r0.x, v3.x
mov r0.z, v2.y
add oT3.w, r0.z, c[0].x
mad oFog, -r2.y, c[16].w, c[16].x
mov oPos.z, r2.y
mov oT1.zw, c[0].x
add r1.y, v2.y, c[0].x
mov r1.z, v2.y
mov oT2.xy, r1
add r0.y, r1.z, c[0].x
mov oT2.zw, r0.xyxy
mov oT3.xy, v2
mov oT5.xyz, c[0].x
mov oT6.xyz, c[0].x
mov oT7.xyz, c[0].x
mov oD0, c[38]
