push   rbp
mov    rbp,rsp
cmp    rdx,0x40
jae    1002551 <sm_dot_f32u8+0x31>
vxorps xmm1,xmm1,xmm1
xor    ecx,ecx
vxorps xmm6,xmm6,xmm6
vxorps xmm2,xmm2,xmm2
vxorps xmm3,xmm3,xmm3
vxorps xmm5,xmm5,xmm5
vxorps xmm7,xmm7,xmm7
vxorps xmm4,xmm4,xmm4
vxorps xmm0,xmm0,xmm0
jmp    1002636 <sm_dot_f32u8+0x116>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vxorps xmm4,xmm4,xmm4
vxorps xmm7,xmm7,xmm7
vxorps xmm5,xmm5,xmm5
vxorps xmm3,xmm3,xmm3
vxorps xmm2,xmm2,xmm2
vxorps xmm6,xmm6,xmm6
vxorps xmm1,xmm1,xmm1
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm7,ymm8,YMMWORD PTR [rdi+rax*4]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x8]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm5,ymm8,YMMWORD PTR [rdi+rax*4+0x20]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x10]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm3,ymm8,YMMWORD PTR [rdi+rax*4+0x40]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x18]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm2,ymm8,YMMWORD PTR [rdi+rax*4+0x60]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x20]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm6,ymm8,YMMWORD PTR [rdi+rax*4+0x80]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x28]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm1,ymm8,YMMWORD PTR [rdi+rax*4+0xa0]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x30]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm4,ymm8,YMMWORD PTR [rdi+rax*4+0xc0]
vpmovzxbd ymm8,QWORD PTR [rsi+rax*1+0x38]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm0,ymm8,YMMWORD PTR [rdi+rax*4+0xe0]
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002580 <sm_dot_f32u8+0x60>
mov    rax,rcx
or     rax,0x8
cmp    rax,rdx
jbe    1002650 <sm_dot_f32u8+0x130>
mov    rax,rcx
jmp    1002671 <sm_dot_f32u8+0x151>
nop    WORD PTR [rax+rax*1+0x0]
vpmovzxbd ymm8,QWORD PTR [rsi+rcx*1]
vcvtdq2ps ymm8,ymm8
vfmadd231ps ymm7,ymm8,YMMWORD PTR [rdi+rcx*4]
lea    rax,[rcx+0x8]
add    rcx,0x10
cmp    rcx,rdx
mov    rcx,rax
jbe    1002650 <sm_dot_f32u8+0x130>
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
jae    100277b <sm_dot_f32u8+0x25b>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x7
je     10026d8 <sm_dot_f32u8+0x1b8>
movzx  r9d,BYTE PTR [rsi+rax*1]
vcvtsi2ss xmm1,xmm15,r9d
vfmadd231ss xmm0,xmm1,DWORD PTR [rdi+rax*4]
inc    rax
dec    r8
jne    10026c0 <sm_dot_f32u8+0x1a0>
cmp    rcx,0xfffffffffffffff8
ja     100277b <sm_dot_f32u8+0x25b>
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
jne    10026f0 <sm_dot_f32u8+0x1d0>
pop    rbp
vzeroupper
ret
