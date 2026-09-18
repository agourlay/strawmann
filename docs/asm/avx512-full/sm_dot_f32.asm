push   rbp
mov    rbp,rsp
cmp    rdx,0x80
jae    1003304 <sm_dot_f32+0x34>
vxorps xmm1,xmm1,xmm1
xor    ecx,ecx
vxorps xmm6,xmm6,xmm6
vxorps xmm2,xmm2,xmm2
vxorps xmm3,xmm3,xmm3
vxorps xmm5,xmm5,xmm5
vxorps xmm7,xmm7,xmm7
vxorps xmm4,xmm4,xmm4
vxorps xmm0,xmm0,xmm0
jmp    10033c7 <sm_dot_f32+0xf7>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vxorps xmm4,xmm4,xmm4
vxorps xmm7,xmm7,xmm7
vxorps xmm5,xmm5,xmm5
vxorps xmm3,xmm3,xmm3
vxorps xmm2,xmm2,xmm2
vxorps xmm6,xmm6,xmm6
vxorps xmm1,xmm1,xmm1
cs nop WORD PTR [rax+rax*1+0x0]
vmovups zmm8,ZMMWORD PTR [rdi+rax*4]
vmovups zmm9,ZMMWORD PTR [rdi+rax*4+0x40]
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x80]
vmovups zmm11,ZMMWORD PTR [rdi+rax*4+0xc0]
vfmadd231ps zmm7,zmm8,ZMMWORD PTR [rsi+rax*4]
vfmadd231ps zmm5,zmm9,ZMMWORD PTR [rsi+rax*4+0x40]
vfmadd231ps zmm3,zmm10,ZMMWORD PTR [rsi+rax*4+0x80]
vfmadd231ps zmm2,zmm11,ZMMWORD PTR [rsi+rax*4+0xc0]
vmovups zmm8,ZMMWORD PTR [rdi+rax*4+0x100]
vfmadd231ps zmm6,zmm8,ZMMWORD PTR [rsi+rax*4+0x100]
vmovups zmm8,ZMMWORD PTR [rdi+rax*4+0x140]
vfmadd231ps zmm1,zmm8,ZMMWORD PTR [rsi+rax*4+0x140]
vmovups zmm8,ZMMWORD PTR [rdi+rax*4+0x180]
vfmadd231ps zmm4,zmm8,ZMMWORD PTR [rsi+rax*4+0x180]
vmovups zmm8,ZMMWORD PTR [rdi+rax*4+0x1c0]
vfmadd231ps zmm0,zmm8,ZMMWORD PTR [rsi+rax*4+0x1c0]
lea    rcx,[rax+0x80]
add    rax,0x100
cmp    rax,rdx
mov    rax,rcx
jbe    1003330 <sm_dot_f32+0x60>
mov    rax,rcx
or     rax,0x10
cmp    rax,rdx
jbe    10033e0 <sm_dot_f32+0x110>
mov    rax,rcx
jmp    10033fe <sm_dot_f32+0x12e>
nop    DWORD PTR [rax+rax*1+0x0]
vmovups zmm8,ZMMWORD PTR [rdi+rcx*4]
vfmadd231ps zmm7,zmm8,ZMMWORD PTR [rsi+rcx*4]
lea    rax,[rcx+0x10]
add    rcx,0x20
cmp    rcx,rdx
mov    rcx,rax
jbe    10033e0 <sm_dot_f32+0x110>
vaddps zmm6,zmm6,zmm7
vaddps zmm1,zmm1,zmm5
vaddps zmm3,zmm3,zmm4
vaddps zmm3,zmm3,zmm6
vaddps zmm0,zmm2,zmm0
vaddps zmm0,zmm1,zmm0
vaddps zmm0,zmm0,zmm3
vextractf64x4 ymm1,zmm0,0x1
vaddps zmm0,zmm0,zmm1
vextractf128 xmm1,ymm0,0x1
vaddps xmm0,xmm0,xmm1
vshufpd xmm1,xmm0,xmm0,0x1
vaddps xmm0,xmm0,xmm1
vmovshdup xmm1,xmm0
vaddss xmm0,xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    10034ff <sm_dot_f32+0x22f>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x7
je     1003483 <sm_dot_f32+0x1b3>
nop    DWORD PTR [rax+rax*1+0x0]
vmovss xmm1,DWORD PTR [rsi+rax*4]
vfmadd231ss xmm0,xmm1,DWORD PTR [rdi+rax*4]
inc    rax
dec    r8
jne    1003470 <sm_dot_f32+0x1a0>
cmp    rcx,0xfffffffffffffff8
ja     10034ff <sm_dot_f32+0x22f>
nop    DWORD PTR [rax+0x0]
vmovss xmm1,DWORD PTR [rsi+rax*4]
vmovss xmm2,DWORD PTR [rsi+rax*4+0x4]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4]
vfmadd231ss xmm1,xmm2,DWORD PTR [rdi+rax*4+0x4]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x8]
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x8]
vmovss xmm1,DWORD PTR [rsi+rax*4+0xc]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0xc]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x10]
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x10]
vmovss xmm1,DWORD PTR [rsi+rax*4+0x14]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0x14]
vmovss xmm2,DWORD PTR [rsi+rax*4+0x18]
vfmadd132ss xmm2,xmm1,DWORD PTR [rdi+rax*4+0x18]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x1c]
vfmadd132ss xmm0,xmm2,DWORD PTR [rdi+rax*4+0x1c]
add    rax,0x8
cmp    rdx,rax
jne    1003490 <sm_dot_f32+0x1c0>
pop    rbp
vzeroupper
ret
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
