vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord1 v2
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r2.y, r0, c[6]
add oT4.xyz, -r1, c[2]
mul r1.xy, v1, c[90]
mul r0.xy, v1, c[91]
add oT0.x, r1.y, r1.x
add oT0.y, r0.y, r0.x
mul r1.xy, v1, c[96]
mul r0.xy, v1, c[97]
add oT0.z, r1.y, r1.x
add oT0.w, r0.y, r0.x
mul r1.xy, v1, c[92]
mul r0.xy, v1, c[93]
add oT1.x, r1.y, r1.x
add oT1.y, r0.y, r0.x
mad oFog, -r2.y, c[16].w, c[16].x
mov oPos.z, r2.y
mov oT1.zw, c[0].x
mov oT2, c[0].x
mov oT3.xy, v2
mov oT3.zw, c[0].x
mov oT5.xyz, c[0].x
mov oT6.xyz, c[0].x
mov oT7.xyz, c[0].x
mov oD0, c[38]
