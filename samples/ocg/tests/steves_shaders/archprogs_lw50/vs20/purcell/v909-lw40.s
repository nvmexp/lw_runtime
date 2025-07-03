vs_2_0
def c[3], 1.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
dp3 r3.x, v4, c[42]
dp3 r3.y, v4, c[43]
dp3 r3.z, v4, c[44]
dp3 r2.x, v1, c[42]
dp3 r2.y, v1, c[43]
dp3 r2.z, v1, c[44]
mul r0.xyz, r3.yzxw, r2.zxyw
mad r0.xyz, r2.yzxw, r3.zxyw, -r0
mov oT4.xyz, r3
mul oT5.xyz, r0, v4.w
mov r0.w, c[3].x
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r4.y, r0, c[10]
add r3.xy, -r0.z, c[2].wzzw
max r0.w, r3.x, c[0].x
rcp r1.w, r3.y
mul r0.w, r0.w, r1.w
mul r0.w, r4.y, r0.w
mov oPos.z, r4.y
mad oFog, -r0.w, c[16].w, c[16].y
add oT3.xyz, -r0, c[2]
mov r5.x, c[31].w
add r4.xyz, -r0, c[29]
add r3.xyz, -r0, c[34]
dp3 r0.x, r4, r4
mad r0.w, r0.x, -r5.x, c[3].x
max r0.w, r0.w, c[3].y
min r2.w, r0.w, c[3].x
rsq r5.y, r0.x
mov r5.xz, c[3].x
mov r0.yw, c[3].x
mul r0.xyz, r0.yxxw, r5
mul r4.xyz, r4, r5.y
dp3 r0.x, r0, c[31]
rcp r1.w, r0.x
mul r1.w, r2.w, r1.w
mul r0.xyz, r1.w, c[27]
dp3 r5.x, c[28], -r4
dp3 r1.x, r2, r4
add r1.w, r5.x, -c[30].z
mul r1.w, r1.w, c[30].w
max r2.w, r1.w, c[0].x
pow r1.w, r2.w, c[30].x
min r1.w, r1.w, c[0].y
mul r0.xyz, r0, r1.w
slt r4.xyz, r2, c[3].y
mova a0.xyz, r4
mul r4.xyz, r2, r2
mul r5.xyz, r4.y, c[a0.y+23]
mad r5.xyz, r4.x, c[a0.x+21], r5
mad r4.xyz, r4.z, c[a0.z+25], r5
max r1.w, r1.x, c[0].x
mad r1.xyz, r0, r1.w, r4
dp3 r0.z, r3, r3
mov r5.xz, c[3].x
rsq r5.y, r0.z
mul r4.xyz, r0.wzzw, r5
mul r3.xyz, r3, r5.y
dp3 r0.x, r4, c[36]
rcp r0.w, r0.x
mov r0.x, c[36].w
mad r1.w, r0.z, -r0.x, c[3].x
max r1.w, r1.w, c[3].y
min r1.w, r1.w, c[3].x
mul r0.w, r0.w, r1.w
mul r0.xyz, r0.w, c[32]
dp3 r4.x, c[33], -r3
dp3 r3.x, r2, r3
mov oT6.xyz, r2
add r0.w, r4.x, -c[35].z
mul r0.w, r0.w, c[35].w
max r1.w, r0.w, c[0].x
pow r0.w, r1.w, c[35].x
min r0.w, r0.w, c[0].y
mul r0.xyz, r0, r0.w
max r0.w, r3.x, c[0].x
mad r0.xyz, r0, r0.w, r1
log r0.x, r0.x
log r0.y, r0.y
log r0.z, r0.z
mul r0.xyz, r0, c[1].x
exp r0.x, r0.x
exp r0.y, r0.y
exp r0.z, r0.z
mul r0.xyz, r0, c[1].w
max r0.w, r0.y, r0.x
max r1.w, r0.z, c[3].x
max r0.w, r0.w, r1.w
rcp r0.w, r0.w
mul oD0.xyz, r0, r0.w
dp4 oT0.x, v2, c[90]
dp4 oT0.y, v2, c[91]
dp4 oT1.x, v2, c[92]
dp4 oT1.y, v2, c[93]
dp4 oT2.x, v2, c[94]
dp4 oT2.y, v2, c[95]
mov oD0.w, c[3].y
mov oD1.xyz, c[3].y
mov oT7.xyz, c[3].y
