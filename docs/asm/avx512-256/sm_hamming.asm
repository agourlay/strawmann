cmp    rdx,0x10
jae    1001c0e <sm_hamming+0x1e>
vpxor  xmm0,xmm0,xmm0
xor    r8d,r8d
vpxor  xmm1,xmm1,xmm1
vpxor  xmm3,xmm3,xmm3
vpxor  xmm2,xmm2,xmm2
jmp    1001d0f <sm_hamming+0x11f>
vpxor  xmm4,xmm4,xmm4
xor    eax,eax
vpbroadcastd ymm5,DWORD PTR [rip+0xffffffffffffe623]        # 1000240 <__init_array_end+0x240>
vbroadcasti128 ymm6,XMMWORD PTR [rip+0xffffffffffffe63a]        # 1000260 <__anon_4796+0x10>
vpxor  xmm2,xmm2,xmm2
vpxor  xmm3,xmm3,xmm3
vpxor  xmm1,xmm1,xmm1
vpxor  xmm0,xmm0,xmm0
cs nop WORD PTR [rax+rax*1+0x0]
vmovdqu ymm7,YMMWORD PTR [rsi+rax*8]
vmovdqu ymm8,YMMWORD PTR [rsi+rax*8+0x20]
vmovdqu ymm9,YMMWORD PTR [rsi+rax*8+0x40]
vmovdqu ymm10,YMMWORD PTR [rsi+rax*8+0x60]
vpxor  ymm7,ymm7,YMMWORD PTR [rdi+rax*8]
vpand  ymm11,ymm7,ymm5
vpshufb ymm11,ymm6,ymm11
vpsrlw ymm7,ymm7,0x4
vpand  ymm7,ymm7,ymm5
vpshufb ymm7,ymm6,ymm7
vpaddb ymm7,ymm11,ymm7
vpsadbw ymm7,ymm7,ymm4
vpaddq ymm1,ymm7,ymm1
vpxor  ymm7,ymm8,YMMWORD PTR [rdi+rax*8+0x20]
vpand  ymm8,ymm7,ymm5
vpshufb ymm8,ymm6,ymm8
vpsrlw ymm7,ymm7,0x4
vpand  ymm7,ymm7,ymm5
vpshufb ymm7,ymm6,ymm7
vpaddb ymm7,ymm8,ymm7
vpsadbw ymm7,ymm7,ymm4
vpaddq ymm0,ymm7,ymm0
vpxor  ymm7,ymm9,YMMWORD PTR [rdi+rax*8+0x40]
vpand  ymm8,ymm7,ymm5
vpshufb ymm8,ymm6,ymm8
vpsrlw ymm7,ymm7,0x4
vpand  ymm7,ymm7,ymm5
vpshufb ymm7,ymm6,ymm7
vpaddb ymm7,ymm8,ymm7
vpsadbw ymm7,ymm7,ymm4
vpxor  ymm8,ymm10,YMMWORD PTR [rdi+rax*8+0x60]
vpaddq ymm3,ymm7,ymm3
vpand  ymm7,ymm8,ymm5
vpshufb ymm7,ymm6,ymm7
vpsrlw ymm8,ymm8,0x4
vpand  ymm8,ymm8,ymm5
vpshufb ymm8,ymm6,ymm8
vpaddb ymm7,ymm8,ymm7
vpsadbw ymm7,ymm7,ymm4
vpaddq ymm2,ymm7,ymm2
lea    r8,[rax+0x10]
add    rax,0x20
cmp    rax,rdx
mov    rax,r8
jbe    1001c40 <sm_hamming+0x50>
mov    rax,r8
or     rax,0x4
cmp    rax,rdx
jbe    1001d20 <sm_hamming+0x130>
mov    rcx,r8
jmp    1001d7f <sm_hamming+0x18f>
push   rbp
mov    rbp,rsp
vpbroadcastd ymm4,DWORD PTR [rip+0xffffffffffffe513]        # 1000240 <__init_array_end+0x240>
vbroadcasti128 ymm5,XMMWORD PTR [rip+0xffffffffffffe52a]        # 1000260 <__anon_4796+0x10>
vpxor  xmm6,xmm6,xmm6
pop    rbp
nop    DWORD PTR [rax+rax*1+0x0]
vmovdqu ymm7,YMMWORD PTR [rsi+r8*8]
vpxor  ymm7,ymm7,YMMWORD PTR [rdi+r8*8]
vpand  ymm8,ymm7,ymm4
vpshufb ymm8,ymm5,ymm8
vpsrlw ymm7,ymm7,0x4
vpand  ymm7,ymm7,ymm4
vpshufb ymm7,ymm5,ymm7
vpaddb ymm7,ymm8,ymm7
vpsadbw ymm7,ymm7,ymm6
vpaddq ymm1,ymm7,ymm1
lea    rcx,[r8+0x4]
add    r8,0x8
cmp    r8,rdx
mov    r8,rcx
jbe    1001d40 <sm_hamming+0x150>
vpaddq ymm0,ymm3,ymm0
vpaddq ymm0,ymm0,ymm2
vpaddq ymm0,ymm0,ymm1
vextracti128 xmm1,ymm0,0x1
vpaddq xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddq xmm0,xmm0,xmm1
vmovq  rax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1001e95 <sm_hamming+0x2a5>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1001dd8 <sm_hamming+0x1e8>
nop    DWORD PTR [rax+rax*1+0x0]
mov    r10,QWORD PTR [rsi+rcx*8]
xor    r10,QWORD PTR [rdi+rcx*8]
popcnt r10,r10
add    rax,r10
inc    rcx
dec    r9
jne    1001dc0 <sm_hamming+0x1d0>
cmp    r8,0xfffffffffffffff8
ja     1001e95 <sm_hamming+0x2a5>
data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
mov    r8,QWORD PTR [rsi+rcx*8]
mov    r9,QWORD PTR [rsi+rcx*8+0x8]
xor    r8,QWORD PTR [rdi+rcx*8]
popcnt r8,r8
add    r8,rax
xor    r9,QWORD PTR [rdi+rcx*8+0x8]
xor    eax,eax
popcnt rax,r9
mov    r9,QWORD PTR [rsi+rcx*8+0x10]
xor    r9,QWORD PTR [rdi+rcx*8+0x10]
popcnt r9,r9
add    r9,rax
add    r9,r8
mov    rax,QWORD PTR [rsi+rcx*8+0x18]
xor    rax,QWORD PTR [rdi+rcx*8+0x18]
popcnt rax,rax
mov    r8,QWORD PTR [rsi+rcx*8+0x20]
xor    r8,QWORD PTR [rdi+rcx*8+0x20]
popcnt r8,r8
add    r8,rax
mov    rax,QWORD PTR [rsi+rcx*8+0x28]
xor    rax,QWORD PTR [rdi+rcx*8+0x28]
xor    r10d,r10d
popcnt r10,rax
add    r10,r8
add    r10,r9
mov    rax,QWORD PTR [rsi+rcx*8+0x30]
xor    rax,QWORD PTR [rdi+rcx*8+0x30]
mov    r8,QWORD PTR [rsi+rcx*8+0x38]
xor    r8,QWORD PTR [rdi+rcx*8+0x38]
xor    r9d,r9d
popcnt r9,rax
xor    eax,eax
popcnt rax,r8
add    rax,r9
add    rax,r10
add    rcx,0x8
cmp    rdx,rcx
jne    1001df0 <sm_hamming+0x200>
vzeroupper
ret
nop    DWORD PTR [rax+0x0]
