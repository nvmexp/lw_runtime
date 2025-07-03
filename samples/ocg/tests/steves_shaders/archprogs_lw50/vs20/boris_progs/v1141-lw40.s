vs_2_0
dcl_position0 v0
dcl_color0 v1
dcl_texcoord0 v2
dcl_texcoord1 v3
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
mov r0.w, c[0].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r0.y, r0, c[10]
mad oFog, -r0.y, c[16].w, c[16].x
mov oPos.z, r0.y
dp4 oT0.x, v2, c[90]
dp4 oT0.y, v2, c[91]
dp4 oT1.x, v2, c[92]
dp4 oT1.y, v2, c[93]
dp4 oT4.x, v2, c[94]
dp4 oT4.y, v2, c[95]
mov oT2.xy, v3
mov oD0, v1
