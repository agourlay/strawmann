push   rbp
mov    rbp,rsp
cmp    rdx,0x80
jae    1002644 <sm_dot_f32u8+0x34>
vxorps xmm1,xmm1,xmm1
xor    ecx,ecx
vxorps xmm6,xmm6,xmm6
vxorps xmm2,xmm2,xmm2
vxorps xmm3,xmm3,xmm3
vxorps xmm5,xmm5,xmm5
vxorps xmm7,xmm7,xmm7
vxorps xmm4,xmm4,xmm4
vxorps xmm0,xmm0,xmm0
jmp    1002737 <sm_dot_f32u8+0x127>
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
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm7,zmm8,ZMMWORD PTR [rdi+rax*4]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x10]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm5,zmm8,ZMMWORD PTR [rdi+rax*4+0x40]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x20]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm3,zmm8,ZMMWORD PTR [rdi+rax*4+0x80]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x30]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm2,zmm8,ZMMWORD PTR [rdi+rax*4+0xc0]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x40]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm6,zmm8,ZMMWORD PTR [rdi+rax*4+0x100]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x50]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm1,zmm8,ZMMWORD PTR [rdi+rax*4+0x140]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x60]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm4,zmm8,ZMMWORD PTR [rdi+rax*4+0x180]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rax*1+0x70]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm0,zmm8,ZMMWORD PTR [rdi+rax*4+0x1c0]
lea    rcx,[rax+0x80]
add    rax,0x100
cmp    rax,rdx
mov    rax,rcx
jbe    1002670 <sm_dot_f32u8+0x60>
mov    rax,rcx
or     rax,0x10
cmp    rax,rdx
jbe    1002750 <sm_dot_f32u8+0x140>
mov    rax,rcx
jmp    1002774 <sm_dot_f32u8+0x164>
nop    DWORD PTR [rax+rax*1+0x0]
vpmovzxbd zmm8,XMMWORD PTR [rsi+rcx*1]
vcvtdq2ps zmm8,zmm8
vfmadd231ps zmm7,zmm8,ZMMWORD PTR [rdi+rcx*4]
lea    rax,[rcx+0x10]
add    rcx,0x20
cmp    rcx,rdx
mov    rcx,rax
jbe    1002750 <sm_dot_f32u8+0x140>
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
jae    100289b <sm_dot_f32u8+0x28b>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x7
je     10027f8 <sm_dot_f32u8+0x1e8>
xchg   ax,ax
movzx  r9d,BYTE PTR [rsi+rax*1]
vcvtsi2ss xmm1,xmm15,r9d
vfmadd231ss xmm0,xmm1,DWORD PTR [rdi+rax*4]
inc    rax
dec    r8
jne    10027e0 <sm_dot_f32u8+0x1d0>
cmp    rcx,0xfffffffffffffff8
ja     100289b <sm_dot_f32u8+0x28b>
data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  ecx,BYTE PTR [rsi+rax*1]
vcvtsi2ss xmm1,xmm15,ecx
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4]
movzx  ecx,BYTE PTR [rsi+rax*1+0x1]
vcvtsi2ss xmm0,xmm15,ecx
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x4]
movzx  ecx,BYTE PTR [rsi+rax*1+0x2]
vcvtsi2ss xmm1,xmm15,ecx
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0x8]
movzx  ecx,BYTE PTR [rsi+rax*1+0x3]
vcvtsi2ss xmm0,xmm15,ecx
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0xc]
movzx  ecx,BYTE PTR [rsi+rax*1+0x4]
vcvtsi2ss xmm1,xmm15,ecx
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0x10]
movzx  ecx,BYTE PTR [rsi+rax*1+0x5]
vcvtsi2ss xmm0,xmm15,ecx
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x14]
movzx  ecx,BYTE PTR [rsi+rax*1+0x6]
vcvtsi2ss xmm1,xmm15,ecx
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0x18]
movzx  ecx,BYTE PTR [rsi+rax*1+0x7]
vcvtsi2ss xmm0,xmm15,ecx
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x1c]
add    rax,0x8
cmp    rdx,rax
jne    1002810 <sm_dot_f32u8+0x200>
pop    rbp
vzeroupper
ret
