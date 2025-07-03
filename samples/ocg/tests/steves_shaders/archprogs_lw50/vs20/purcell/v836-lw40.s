vs_2_0
def c[3], 765.005859, 1.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
add r0.w, -v1.x, c[3].y
add r3.w, r0.w, -v1.y
mul r0.xyz, v2.zyxw, c[3].x
mova a0.xyz, r0
mul r0, v1.y, c[a0.y+42]
mad r0, c[a0.x+42], v1.x, r0
mad r0, c[a0.z+42], r3.w, r0
dp3 r4.x, v6, r0
mul r1, v1.y, c[a0.y+43]
mad r1, c[a0.x+43], v1.x, r1
mad r2, c[a0.z+43], r3.w, r1
dp3 r4.y, v6, r2
mul r1, v1.y, c[a0.y+44]
mad r1, c[a0.x+44], v1.x, r1
mad r1, c[a0.z+44], r3.w, r1
dp3 r4.z, v6, r1
dp3 r3.x, v3, r0
dp4 r0.x, v0, r0
dp3 r3.y, v3, r2
dp4 r0.y, v0, r2
dp3 r3.z, v3, r1
dp4 r0.z, v0, r1
mul r2.xyz, r4.yzxw, r3.zxyw
mad r2.xyz, r3.yzxw, r4.zxyw, -r2
mov oT4.xyz, r4
mul oT5.xyz, r2, v6.w
mov r0.w, c[3].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r2.y, r0, c[10]
mad oFog, -r2.y, c[16].w, c[16].x
mov oPos.z, r2.y
add oT3.xyz, -r0, c[2]
add r1.xyz, -r0, c[29]
add r0.xyz, -r0, c[34]
dp3 r5.x, r1, r1
rsq r4.y, r5.x
mov r5.y, c[3].y
mov r4.xz, c[3].y
mul r2.xyz, r5.yxxw, r4
mul r1.xyz, r1, r4.y
dp3 r2.x, r2, c[31]
rcp r0.w, r2.x
mov r2.x, c[31].w
mad r1.w, r5.x, -r2.x, c[3].y
max r1.w, r1.w, c[3].z
min r1.w, r1.w, c[3].y
mul r0.w, r0.w, r1.w
mul r2.xyz, r0.w, c[27]
dp3 r4.x, c[28], -r1
dp3 r5.x, r3, r1
add r0.w, r4.x, -c[30].z
mul r0.w, r0.w, c[30].w
max r1.w, r0.w, c[0].x
pow r0.w, r1.w, c[30].x
min r0.w, r0.w, c[0].y
mul r2.xyz, r2, r0.w
slt r1.xyz, r3, c[3].z
mova a0.xyz, r1
mul r1.xyz, r3, r3
mul r4.xyz, r1.y, c[a0.y+23]
mad r4.xyz, r1.x, c[a0.x+21], r4
mad r1.xyz, r1.z, c[a0.z+25], r4
max r0.w, r5.x, c[0].x
mad r1.xyz, r2, r0.w, r1
dp3 r2.z, r0, r0
rsq r0.w, r2.z
mul r0.xyz, r0, r0.w
dst r2.xy, r2.z, r0.w
dp3 r2.x, c[36], r2
rcp r0.w, r2.x
mov r2.x, c[36].w
mad r1.w, r2.z, -r2.x, c[3].y
max r1.w, r1.w, c[3].z
min r1.w, r1.w, c[3].y
mul r0.w, r0.w, r1.w
dp3 r0.x, r3, r0
mov oT6.xyz, r3
max r1.w, r0.x, c[0].x
mul r0.xyz, r1.w, c[32]
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
max r1.w, r0.z, c[3].y
max r0.w, r0.w, r1.w
rcp r0.w, r0.w
mul oD0.xyz, r0, r0.w
dp4 oT0.x, v4, c[90]
dp4 oT0.y, v4, c[91]
dp4 oT1.x, v4, c[92]
dp4 oT1.y, v4, c[93]
dp4 oT2.x, v4, c[94]
dp4 oT2.y, v4, c[95]
mov oD0.w, c[3].z
mov oD1.xyz, c[3].z
mov oT7.xyz, c[3].z
