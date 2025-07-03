vs_1_1
dcl_position0 v0
dcl_texcoord0 v7
dcl_texcoord1 v8
dcl_texcoord2 v9
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r0.z, v0, c[6]
dp4 r0.w, v0, c[7]
mov oPos, r0
mad oFog, -r0.z, c[16].w, c[16].x
mov r0.xy, v9
dp4 oT0.x, v7, c[90]
dp4 oT0.y, v7, c[91]
add oT1.xy, r0, v8
mad oT2.xy, r0, c[0].z, v8
mad oT3.xy, r0, c[39].x, v8
