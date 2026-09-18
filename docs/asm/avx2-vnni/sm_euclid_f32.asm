cmp    rdx,0x40
jae    10030fd <sm_euclid_f32+0x2d>
vxorps xmm1,xmm1,xmm1
xor    ecx,ecx
vxorps xmm6,xmm6,xmm6
vxorps xmm2,xmm2,xmm2
vxorps xmm3,xmm3,xmm3
vxorps xmm5,xmm5,xmm5
vxorps xmm7,xmm7,xmm7
vxorps xmm4,xmm4,xmm4
vxorps xmm0,xmm0,xmm0
jmp    10031d2 <sm_euclid_f32+0x102>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vxorps xmm4,xmm4,xmm4
vxorps xmm7,xmm7,xmm7
vxorps xmm5,xmm5,xmm5
vxorps xmm3,xmm3,xmm3
vxorps xmm2,xmm2,xmm2
vxorps xmm6,xmm6,xmm6
vxorps xmm1,xmm1,xmm1
nop
vmovups ymm8,YMMWORD PTR [rdi+rax*4]
vmovups ymm9,YMMWORD PTR [rdi+rax*4+0x20]
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0x40]
vmovups ymm11,YMMWORD PTR [rdi+rax*4+0x60]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rax*4]
vfmadd231ps ymm7,ymm8,ymm8
vsubps ymm8,ymm9,YMMWORD PTR [rsi+rax*4+0x20]
vsubps ymm9,ymm10,YMMWORD PTR [rsi+rax*4+0x40]
vfmadd231ps ymm5,ymm8,ymm8
vfmadd231ps ymm3,ymm9,ymm9
vsubps ymm8,ymm11,YMMWORD PTR [rsi+rax*4+0x60]
vfmadd231ps ymm2,ymm8,ymm8
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0x80]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rax*4+0x80]
vfmadd231ps ymm6,ymm8,ymm8
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xa0]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rax*4+0xa0]
vfmadd231ps ymm1,ymm8,ymm8
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xc0]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rax*4+0xc0]
vfmadd231ps ymm4,ymm8,ymm8
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xe0]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rax*4+0xe0]
vfmadd231ps ymm0,ymm8,ymm8
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1003120 <sm_euclid_f32+0x50>
mov    rax,rcx
or     rax,0x8
cmp    rax,rdx
jbe    10031f0 <sm_euclid_f32+0x120>
mov    rax,rcx
jmp    100320f <sm_euclid_f32+0x13f>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovups ymm8,YMMWORD PTR [rdi+rcx*4]
vsubps ymm8,ymm8,YMMWORD PTR [rsi+rcx*4]
vfmadd231ps ymm7,ymm8,ymm8
lea    rax,[rcx+0x8]
add    rcx,0x10
cmp    rcx,rdx
mov    rcx,rax
jbe    10031f0 <sm_euclid_f32+0x120>
vaddps ymm6,ymm6,ymm7
vaddps ymm1,ymm1,ymm5
vaddps ymm3,ymm3,ymm4
vaddps ymm3,ymm3,ymm6
vaddps ymm0,ymm2,ymm0
vaddps ymm0,ymm1,ymm0
vaddps ymm0,ymm0,ymm3
vextractf128 xmm1,ymm0,0x1
vaddps xmm0,xmm0,xmm1
vshufpd xmm1,xmm0,xmm0,0x1
vaddps xmm0,xmm0,xmm1
vmovshdup xmm1,xmm0
vaddss xmm0,xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    1003323 <sm_euclid_f32+0x253>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x7
je     1003277 <sm_euclid_f32+0x1a7>
xchg   ax,ax
vmovss xmm1,DWORD PTR [rdi+rax*4]
vsubss xmm1,xmm1,DWORD PTR [rsi+rax*4]
vfmadd231ss xmm0,xmm1,xmm1
inc    rax
dec    r8
jne    1003260 <sm_euclid_f32+0x190>
cmp    rcx,0xfffffffffffffff8
ja     1003323 <sm_euclid_f32+0x253>
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovss xmm1,DWORD PTR [rdi+rax*4]
vmovss xmm2,DWORD PTR [rdi+rax*4+0x4]
vsubss xmm1,xmm1,DWORD PTR [rsi+rax*4]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4+0x4]
vfmadd213ss xmm1,xmm1,xmm0
vmovss xmm0,DWORD PTR [rdi+rax*4+0x8]
vsubss xmm0,xmm0,DWORD PTR [rsi+rax*4+0x8]
vfmadd213ss xmm2,xmm2,xmm1
vmovss xmm1,DWORD PTR [rdi+rax*4+0xc]
vsubss xmm1,xmm1,DWORD PTR [rsi+rax*4+0xc]
vfmadd213ss xmm0,xmm0,xmm2
vmovss xmm2,DWORD PTR [rdi+rax*4+0x10]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4+0x10]
vfmadd213ss xmm1,xmm1,xmm0
vmovss xmm0,DWORD PTR [rdi+rax*4+0x14]
vsubss xmm3,xmm0,DWORD PTR [rsi+rax*4+0x14]
vfmadd213ss xmm2,xmm2,xmm1
vmovss xmm0,DWORD PTR [rdi+rax*4+0x18]
vsubss xmm1,xmm0,DWORD PTR [rsi+rax*4+0x18]
vfmadd213ss xmm3,xmm3,xmm2
vmovss xmm0,DWORD PTR [rdi+rax*4+0x1c]
vsubss xmm0,xmm0,DWORD PTR [rsi+rax*4+0x1c]
vfmadd213ss xmm1,xmm1,xmm3
vfmadd213ss xmm0,xmm0,xmm1
add    rax,0x8
cmp    rdx,rax
jne    1003290 <sm_euclid_f32+0x1c0>
push   rbp
mov    rbp,rsp
vbroadcastss xmm1,DWORD PTR [rip+0xffffffffffffcf14]        # 1000244 <__init_array_end+0x244>
vxorps xmm0,xmm0,xmm1
pop    rbp
vzeroupper
ret
nop    DWORD PTR [rax+0x0]
