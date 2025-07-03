vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_texcoord1 v3
dcl_texcoord2 v4
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp3 oT5.x, v1, c[42]
dp3 oT5.y, v1, c[43]
dp3 oT5.z, v1, c[44]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r2.y, r0, c[6]
add oT4.xyz, -r1, c[2]
mul r1.xy, v2, c[90]
mul r0.xy, v2, c[91]
add oT0.x, r1.y, r1.x
add oT0.y, r0.y, r0.x
mul r1.xy, v4, c[92]
mul r0.xy, v4, c[93]
add oT1.x, r1.y, r1.x
add oT1.y, r0.y, r0.x
mul r1.xy, v2, c[94]
mul r0.xy, v2, c[95]
add oT3.x, r1.y, r1.x
add oT3.y, r0.y, r0.x
mad oFog, -r2.y, c[16].w, c[16].x
mov oPos.z, r2.y
mov oT2.xy, v3
mov oD0, c[38]
