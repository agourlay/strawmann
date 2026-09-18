push   rbp
mov    rbp,rsp
cmp    rdx,0x40
jae    100286d <sm_dot_u8u8+0x5d>
pxor   xmm0,xmm0
movdqa XMMWORD PTR [rbp-0x20],xmm0
xor    r8d,r8d
movdqa XMMWORD PTR [rbp-0x30],xmm0
pxor   xmm2,xmm2
pxor   xmm14,xmm14
pxor   xmm5,xmm5
pxor   xmm11,xmm11
pxor   xmm3,xmm3
pxor   xmm10,xmm10
movdqa XMMWORD PTR [rbp-0x10],xmm0
movdqa XMMWORD PTR [rbp-0x40],xmm0
pxor   xmm9,xmm9
pxor   xmm12,xmm12
pxor   xmm13,xmm13
pxor   xmm15,xmm15
pxor   xmm1,xmm1
jmp    1002a87 <sm_dot_u8u8+0x277>
pxor   xmm4,xmm4
xor    eax,eax
pxor   xmm12,xmm12
pxor   xmm13,xmm13
pxor   xmm15,xmm15
pxor   xmm1,xmm1
pxor   xmm0,xmm0
movdqa XMMWORD PTR [rbp-0x10],xmm0
movdqa XMMWORD PTR [rbp-0x40],xmm0
pxor   xmm9,xmm9
pxor   xmm5,xmm5
pxor   xmm11,xmm11
pxor   xmm3,xmm3
pxor   xmm10,xmm10
pxor   xmm2,xmm2
movdqa XMMWORD PTR [rbp-0x20],xmm2
movdqa XMMWORD PTR [rbp-0x30],xmm2
pxor   xmm14,xmm14
xchg   ax,ax
movdqa XMMWORD PTR [rbp-0x60],xmm12
movdqa XMMWORD PTR [rbp-0x50],xmm1
movdqu xmm1,XMMWORD PTR [rdi+rax*1]
movdqa xmm6,xmm1
punpcklbw xmm6,xmm4
movdqu xmm12,XMMWORD PTR [rsi+rax*1]
movdqa xmm7,xmm12
punpcklbw xmm7,xmm4
pmullw xmm7,xmm6
movdqa xmm6,xmm7
punpcklwd xmm6,xmm4
paddd  xmm5,xmm6
movdqu xmm8,XMMWORD PTR [rdi+rax*1+0x10]
punpckhwd xmm7,xmm4
paddd  xmm11,xmm7
movdqu xmm6,XMMWORD PTR [rsi+rax*1+0x10]
punpckhbw xmm1,xmm4
punpckhbw xmm12,xmm4
pmullw xmm12,xmm1
movdqa xmm1,xmm12
punpcklwd xmm1,xmm4
paddd  xmm3,xmm1
punpckhwd xmm12,xmm4
paddd  xmm10,xmm12
movdqa xmm1,xmm8
punpcklbw xmm1,xmm4
movdqa xmm7,xmm6
punpcklbw xmm7,xmm4
pmullw xmm7,xmm1
movdqa xmm1,xmm7
punpcklwd xmm1,xmm4
movdqa xmm12,XMMWORD PTR [rbp-0x20]
paddd  xmm12,xmm1
movdqa XMMWORD PTR [rbp-0x20],xmm12
movdqu xmm1,XMMWORD PTR [rdi+rax*1+0x20]
punpckhwd xmm7,xmm4
movdqa xmm12,XMMWORD PTR [rbp-0x30]
paddd  xmm12,xmm7
movdqa XMMWORD PTR [rbp-0x30],xmm12
movdqu xmm7,XMMWORD PTR [rsi+rax*1+0x20]
punpckhbw xmm8,xmm4
punpckhbw xmm6,xmm4
pmullw xmm6,xmm8
movdqa xmm8,xmm6
punpcklwd xmm8,xmm4
paddd  xmm2,xmm8
punpckhwd xmm6,xmm4
paddd  xmm14,xmm6
movdqa xmm6,xmm1
punpcklbw xmm6,xmm4
movdqa xmm12,xmm7
punpcklbw xmm12,xmm4
pmullw xmm12,xmm6
movdqa xmm6,xmm12
punpcklwd xmm6,xmm4
movdqa xmm8,XMMWORD PTR [rbp-0x10]
paddd  xmm8,xmm6
movdqa XMMWORD PTR [rbp-0x10],xmm8
movdqu xmm8,XMMWORD PTR [rdi+rax*1+0x30]
punpckhwd xmm12,xmm4
movdqa xmm6,XMMWORD PTR [rbp-0x40]
paddd  xmm6,xmm12
movdqa XMMWORD PTR [rbp-0x40],xmm6
movdqa xmm12,XMMWORD PTR [rbp-0x60]
movdqu xmm6,XMMWORD PTR [rsi+rax*1+0x30]
punpckhbw xmm1,xmm4
punpckhbw xmm7,xmm4
pmullw xmm7,xmm1
movdqa xmm1,xmm7
punpcklwd xmm1,xmm4
paddd  xmm9,xmm1
punpckhwd xmm7,xmm4
paddd  xmm0,xmm7
movdqa xmm1,xmm8
punpcklbw xmm1,xmm4
movdqa xmm7,xmm6
punpcklbw xmm7,xmm4
pmullw xmm7,xmm1
movdqa xmm1,xmm7
punpcklwd xmm1,xmm4
paddd  xmm12,xmm1
punpckhwd xmm7,xmm4
paddd  xmm13,xmm7
punpckhbw xmm8,xmm4
punpckhbw xmm6,xmm4
pmullw xmm6,xmm8
movdqa xmm1,xmm6
punpcklwd xmm1,xmm4
paddd  xmm15,xmm1
movdqa xmm1,XMMWORD PTR [rbp-0x50]
punpckhwd xmm6,xmm4
paddd  xmm1,xmm6
lea    r8,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,r8
jbe    10028c0 <sm_dot_u8u8+0xb0>
movdqa XMMWORD PTR [rbp-0x70],xmm13
movdqa XMMWORD PTR [rbp-0x80],xmm14
movdqa XMMWORD PTR [rbp-0x50],xmm15
movdqa XMMWORD PTR [rbp-0x60],xmm12
movdqa xmm14,XMMWORD PTR [rbp-0x20]
movdqa xmm12,XMMWORD PTR [rbp-0x10]
mov    rax,r8
or     rax,0x10
movdqa xmm15,xmm1
cmp    rax,rdx
jbe    1002ac6 <sm_dot_u8u8+0x2b6>
mov    rcx,r8
movdqa xmm13,xmm14
jmp    1002b3b <sm_dot_u8u8+0x32b>
pxor   xmm4,xmm4
movdqa xmm13,xmm14
nop
movdqu xmm1,XMMWORD PTR [rdi+r8*1]
movdqu xmm6,XMMWORD PTR [rsi+r8*1]
movdqa xmm7,xmm1
punpcklbw xmm7,xmm4
movdqa xmm8,xmm6
punpcklbw xmm8,xmm4
pmullw xmm8,xmm7
movdqa xmm7,xmm8
punpcklwd xmm7,xmm4
paddd  xmm5,xmm7
punpckhwd xmm8,xmm4
paddd  xmm11,xmm8
punpckhbw xmm1,xmm4
punpckhbw xmm6,xmm4
pmullw xmm6,xmm1
movdqa xmm1,xmm6
punpcklwd xmm1,xmm4
paddd  xmm3,xmm1
punpckhwd xmm6,xmm4
paddd  xmm10,xmm6
lea    rcx,[r8+0x10]
add    r8,0x20
cmp    r8,rdx
mov    r8,rcx
jbe    1002ad0 <sm_dot_u8u8+0x2c0>
movdqa xmm1,XMMWORD PTR [rbp-0x40]
paddd  xmm1,XMMWORD PTR [rbp-0x30]
paddd  xmm1,XMMWORD PTR [rbp-0x70]
paddd  xmm1,xmm11
paddd  xmm0,XMMWORD PTR [rbp-0x80]
paddd  xmm0,xmm15
paddd  xmm0,xmm10
paddd  xmm0,xmm1
paddd  xmm12,xmm13
paddd  xmm12,XMMWORD PTR [rbp-0x60]
paddd  xmm12,xmm5
paddd  xmm9,xmm2
paddd  xmm9,XMMWORD PTR [rbp-0x50]
paddd  xmm9,xmm3
paddd  xmm9,xmm12
paddd  xmm9,xmm0
pshufd xmm1,xmm9,0xee
paddd  xmm1,xmm9
pshufd xmm0,xmm1,0x55
paddd  xmm0,xmm1
movd   eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1002c40 <sm_dot_u8u8+0x430>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x3
je     1002bdb <sm_dot_u8u8+0x3cb>
nop    DWORD PTR [rax+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movzx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
add    rcx,0x1
add    r9,0xffffffffffffffff
jne    1002bc0 <sm_dot_u8u8+0x3b0>
cmp    r8,0xfffffffffffffffc
ja     1002c40 <sm_dot_u8u8+0x430>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movzx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movzx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  r8d,BYTE PTR [rdi+rcx*1+0x3]
movzx  eax,BYTE PTR [rsi+rcx*1+0x3]
imul   eax,r8d
add    eax,r10d
add    rcx,0x4
cmp    rdx,rcx
jne    1002bf0 <sm_dot_u8u8+0x3e0>
pop    rbp
ret
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+0x0]
