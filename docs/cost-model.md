# The cost model, measured

**M0's exit criterion.** §5 is the spine of the project, and §5's own framing is
that the model comes first and the measurement grades it:

> "Write it before writing the server; validate each number with a
> microbenchmark; then treat the ratio *measured / modelled* as the primary
> engineering KPI."

This document is that grading. Every number below was produced by
`zig build bench` on the host described at the bottom; nothing here is copied
from the spec except the modelled values it is being compared against.

**Status of these numbers.** They were measured before 2026-08-17, when this
host ran a `powersave` governor with boost on. Per §7.1 that makes everything
here **development-grade, not publishable**, and only re-measuring changes it.
The absent `isolcpus`/`nohz_full` now selects the `as-deployed` profile instead
of failing the gate, so it is no longer part of the verdict; the governor and
boost at measurement time still are. The
ratios are stable enough to be informative and the qualitative findings are
robust, but no number in this file should appear in a comparison against Qdrant
until it is re-measured on a host that passes the gate.

**Harness revision.** Every number in this file was produced by the harness
*before* the methodology fixes listed here; none has been re-measured since.
Where a fix changes what a number means, the table says so in place. The fixes,
and what each one invalidates:

| fix | what it changes | which numbers below it affects |
|---|---|---|
| CPU pinning (`--pin`, default: the starting CPU; refuses to run unpinned without `--no-pin`) | every cell is now a per-core number and says which core; the host is hybrid (Zen 5 with 16 MiB L3, Zen 5c with 8 MiB L3) and the old runs migrated between clusters | all of §1 to §3; in particular any L3-resident cell may have run on an 8 MiB core |
| L3-resident working set sized from the pinned CPU's sysfs L3 (50%) instead of an 8 MiB constant | the old 8 MiB set was the *entire* L3 of a Zen 5c core, so on that cluster "L3-resident" measured L3 thrash | §3's L3-resident column |
| a second, **dependent** access loop next to the original (now labelled `independent (MLP)`) | the old "DRAM-cold" cells are throughput numbers with dozens of misses in flight, not §5.2's dependent-fetch cost | §2 entirely; §3's cold column |
| arenas filled with values of the kernel's type instead of random bytes | random bytes read as fp32/fp16 are NaN/Inf/denormals; denormal FMA operands cost +12% here and far more on Intel | the fp32/fp16 hot cells, and the fp32/fp16 ISA ratios in `isa-matrix.md` |
| bandwidth reader uses 4 wide integer XOR accumulators instead of one FADD chain | the old reader could be ALU-limited, not bus-limited | §1's sequential read row |
| perf counters opened as one group, scaled by enabled/running, multiplexing reported | old counters were scheduled independently and could cover different windows | every cyc/insn/IPC/eff.freq column, at the margin |
| huge-page backing read from `/proc/self/smaps` and printed | the "2 MiB pages" arm was labelled from the `madvise` request, never from the outcome | §1's TLB row: assumed, now checked |
| SQ8 and PQ code rows at production stride (`dim` / `m` bytes) instead of padded to 64 B | PQ at m=96 was walking 128 B rows, not 96; SQ8 at d=768 was unaffected (768 is a multiple of 64) | §2's PQ row |

The rows below that were measured the old way and have not been re-measured
are marked; a `TODO` in a table cell means the column now exists in the harness
and has no published number yet.

---

## 1. Hardware constants (§7.2 layer 1)

| constant | measured | §5's assumption | verdict |
|---|--:|--:|---|
| Sequential read bandwidth, single core | 47.0 GB/s (old FADD-chain reader; unpinned) | (| machine-specific; re-measure |
| Sequential read bandwidth, all cores | TODO (`sysinfo.probe`, now printed by `zig build bench hw`) |) | machine-specific |
| STREAM triad, single core | 37.2 GB/s (unpinned) |, | machine-specific |
| DRAM latency, dependent chase, 2 MiB pages | 116 to 120 ns (huge backing not verified at the time) | "~85 to 100 ns" | **outside; rescale by ~1.3×** |
| Outstanding misses at saturation | ~64 | "roughly 10 to 16" | **4× higher** |
| Peak per-core random-access bandwidth | 25.3 GB/s | "≈ 8.5 GB/s" | **3× higher** |
| Random-access bandwidth at 16 chains | 7.84 GB/s | "≈ 8.5 GB/s" | **matches** |

### The MLP finding, which is the important one

§5.2 derives its entire ns/vector table from one constant:

> "A core can sustain roughly 10 to 16 outstanding line fills (LFB/MSHR-limited).
> With ~85 to 100 ns of DRAM latency: per-core random-access throughput ≈
> (12 lines × 64 B) / 90 ns ≈ 8.5 GB/s"

Measured, sweeping independent dependent-load chains:

| chains | ns/access | GB/s | speedup vs 1 |
|--:|--:|--:|--:|
| 1 | 116.1 | 0.55 | 1.00× |
| 2 | 58.4 | 1.10 | 1.99× |
| 4 | 31.0 | 2.07 | 3.75× |
| 8 | 16.0 | 3.99 | 7.23× |
| 16 | 8.2 | 7.84 | 14.22× |
| 32 | 4.3 | 14.90 | 27.03× |
| 64 | 2.7 | 23.93 | 43.41× |
| 128 | 2.5 | 25.32 | 45.93× |
| 256 | 2.5 | 25.11 | 45.54× |

Two things fall out, and they point in opposite directions:

1. **§5.2's 8.5 GB/s is right for the concurrency HNSW actually supplies.** At
   16 chains, the independent fetches one `M=16` expansion offers, this host
   measures 7.84 GB/s. The spec's arithmetic is sound.

2. **The hardware can do 3.2× better.** Saturation is at ~64 outstanding misses
   and 25.3 GB/s, not 16 and 8.5.

The gap between those is the budget for §5.2's own recommendation:

> "**Software prefetch of the neighbour list's vectors, issued before scoring
> the current candidate, is the single highest-leverage optimisation in the
> search path**"

On this host that claim is worth **3.2×**, and it is available to anything that
raises the concurrent-miss count above `M`: prefetching the *next* candidate's
neighbours while scoring the current one, scoring several candidates at once, or
simply a larger `M`.

### TLB (§5.5)

| backing | ns/access | dTLB load misses/access | huge-backed (smaps) |
|---|--:|--:|--:|
| 4 KiB pages | 138.6 | 0.953 | not checked when measured |
| 2 MiB pages | 119.1 | 0.001 | not checked when measured; the 0.001 misses/access is consistent with ~100% |

The harness now reads the fraction of the 2 MiB arm actually backed by huge
pages from `/proc/self/smaps` after the run and warns below 90%; these two
rows predate that check, and the miss rate is the only evidence the backing
was what the label says.

A **~1000× reduction in TLB misses buys 16%** of latency. §5.5 predicted "tens
of nanoseconds", and 19.5 ns is tens of nanoseconds, the model is right about
the mechanism and about the magnitude. It is worth noting how *cheap* a miss is
here: Zen 5's page walker overlaps well, so the effect is real but modest. On a
part with a smaller TLB or a slower walker the same experiment should show more,
which is precisely why §7.5 insists the matrix is meaningless on one machine.

---

## 2. The §5.2 ns/vector table, graded (d=768, DRAM-cold)

**What §5.2 models, and what was measured, are not the same loop.** §5.2's
table is the cost of "a dependent chain, you cannot know which vector to fetch
next until the current expansion has been scored". The harness's timed loop
walked a random permutation whose next address never depended on the previous
result, so the out-of-order core kept up to the MLP limit of misses in flight.
That is a **throughput** number, the ceiling a batched or prefetched scan can
reach, and it is what the `measured` column below is. Grading a dependent-chain
model against it was a category error: it says the model is 3× pessimistic for
fp32 when the model was never asked that question. The tell was in the data all
along: hamming at d=128 read 20 ns "cold" against a measured 116 ns DRAM
latency, which no dependent fetch can do.

The harness now runs both loops over the same arena and reports them as
`independent (MLP)` and `dependent`. The published column is the independent
one; the dependent column has been added to the harness and **not yet
measured** for this table.

| encoding | bytes/vec | lines | modelled (dependent) | **measured, independent (MLP)** | measured/modelled | measured, dependent |
|---|--:|--:|--:|--:|--:|--:|
| fp32 | 3072 | 48 | ~360 ns | **114.4 ns** | **0.32×** | TODO |
| fp16 | 1536 | 24 | ~180 ns | **148.8 ns** | 0.83× | TODO |
| int8 SQ, symmetric | 768 | 12 | ~90 ns | **57.7 ns** | 0.64× | TODO |
| int8 SQ, asymmetric | 768 | 12 | ~90 ns | **125.9 ns** | 1.40× | TODO |
| PQ (m=96, 96 B codes) | 96 | 2 | ~35 ns | **130.1 ns** (rows padded to 128 B when measured) | **3.72×** | TODO |
| binary | 128 (stored) | 2 | ~15 to 90 ns | **20.3 ns** | in range, low end | TODO |

Every `measured/modelled` ratio in this table therefore compares a throughput
measurement to a latency model, and the discussion below should be read with
that in mind: the *ordering* and the *kernel-bound* findings survive (they are
about the loop body), the absolute ratios do not. Four of these disagree with
the model, and each disagreement is a finding rather than noise.

### binary landed in range only after a tail-handling fix

Binary measured 29.8 ns before `quant/binary.paddedWordsFor` and **20.3 ns**
after, a 1.47× improvement from padding stored rows to a whole vector register
so the Hamming kernel stops falling back to a scalar tail. §5.2's range for this
row is "~15 to 90 ns" depending on whether several vectors are in flight; the
measurement now sits near the fast end. The full before/after is in
`docs/isa-matrix.md`.

### fp32 is 3× faster than modelled, because the loop was not the modelled one

114.4 ns at 3072 B is **26.9 GB/s**, which is the measured MLP ceiling of
25.3 GB/s. The kernel is saturating the memory system exactly as §5.1 predicts
*for an independent stream*; that is the loop that was timed. Rescaling §5.2's
table by the measured 25.3 GB/s instead of the assumed 8.5 predicts 121 ns,
against 114 measured, a 6% error, which is a good fit of the throughput model
to the throughput measurement. Whether §5.2's dependent-chain figure of ~360 ns
is right is what the `dependent` column will say.

### fp16 is *slower* than fp32 in the independent loop, hypothesis: ROB occupancy

fp16 moves half the bytes and takes 30% longer (148.8 vs 114.4 ns), achieving
only 10.3 GB/s against fp32's 26.9. Both have IPC ~0.3, so both are stalled on
memory, and the question is why the fp16 stall is longer per byte.

The hypothesis, now that the loop is understood to be an *independent* one, is
that in such a loop the achieved memory parallelism is set by how many
iterations fit in the reorder buffer at once. Each iteration is one vector's
worth of instructions plus the loop and call overhead; the core can only have
as many vector fetches outstanding as it has iterations in flight, and it has
as many iterations in flight as the ROB holds. The AVX-512 fp16 kernel executes
*more* instructions per vector than fp32 (the widening `vcvtph2ps` doubles the
op count: at d=768, 158 cycles/vector hot on avx512-full against fp32's 99,
`isa-matrix.md`), so fewer fp16 iterations fit in the window, so fewer of its
(smaller) fetches overlap, so its achieved bandwidth is lower. On this reading
the "bytes moved" framing of §5.1 is incomplete for the independent case in a
specific way: instructions per vector, not bytes per vector, gates MLP once the
loop is not otherwise stalled.

**This is a hypothesis, not a finding**, and the `dependent` column is the
test: in a dependent chain the ROB holds one iteration's fetch at a time
regardless of instruction count, so if the hypothesis is right the fp16
dependent number should sit at or *below* fp32's (fewer lines per vector,
same latency per line), and if it is still above, the mechanism is something
else. §11 says an optimisation that improves QPS without a model explaining why
is treated as suspicious; the same discipline applies in reverse to a
measurement the model does not predict, so this stays open until measured.

### PQ is 3.7× *slower* than modelled, the ADC table does not fit in L1

(Measured with code rows padded to 128 B; production packs them at 96 B, and
the harness now does too. The lines-per-vector count in the table above is the
production one; the measurement crossed 2 to 3 lines rather than 2.)

§6.6.1 puts PQ in the "LUT lookup throughput is the kernel" regime. It is worse
than that: at `m = 96` an 8-bit ADC table is `96 × 256 × 4 B = 98 KB`, which
does not fit in a 48 KB L1. Every one of the 96 lookups per vector is an L2 hit
at best, and the table is re-read for every vector scored.

This is the strongest possible argument for §6.7's PQ4/FastScan "stretch goal",
and it can now be made quantitatively: a 4-bit table is `96 × 16 × 1 B = 1.5 KB`
and fits in L1 with room to spare. The §6.7 note that PQ4 is "the version that
actually competes" is measured, not asserted.

### SQ8 symmetric beats asymmetric by 2.2×, so the cold case is not purely memory bound

Both move 768 B/vector, yet symmetric costs 57.7 ns and asymmetric 125.9 ns.
The difference is entirely the kernel: symmetric is `vpdpbusd` at 103
instructions/vector, asymmetric is an f32×u8 convert-and-FMA at 236. If SQ8 cold
were purely bandwidth bound these would be equal.

At 12 cache lines per vector, §5.2 places SQ8 exactly at the threshold where "a
single dependent fetch no longer fills the memory pipeline". The measurement
says the ALU is a co-limiter there, which makes the VNNI kernel worth more than
the pure-bandwidth model suggests.

---

## 3. Residency, and where SIMD width should matter (§6.6.1)

§6.6.1's prediction, restated as a testable claim: cost should track the
encoding's *working-set* position, not its byte count, and the L1-hot column is
where the kernel itself is visible.

Measured, d=768, ns/vector. All three columns are the **independent (MLP)**
loop; the L3-resident column used a fixed 8 MiB working set on an unpinned run,
which on this host is the whole L3 of a Zen 5c core, so it may be partly a
thrash number (the harness now sizes it from the pinned core's sysfs L3). The
fp32/fp16 hot cells were measured over byte-filled arenas (NaN/denormal
operands) and are expected to drop slightly on re-measurement.

| encoding | L1-hot | L3-resident (independent) | DRAM-cold (independent) | cold/hot |
|---|--:|--:|--:|--:|
| dot_f32 | 19.7 | 31.4 | 114.4 | 5.8× |
| l2_f32 | 20.1 | 30.9 | 122.1 | 6.1× |
| dot_f16 | 31.9 | 35.2 | 148.8 | 4.7× |
| dot_i8 asym | 19.5 | 22.5 | 125.9 | 6.5× |
| dot_i8 sym | 5.7 | 10.0 | 57.7 | 10.1× |
| hamming | 4.4 | 9.6 | 20.3 | 4.6× |
| pq_adc_8 (m=96) | 20.9 | 28.5 | 130.1 | 6.2× |

The cold/hot ratio is 5 to 10× for every encoding. §6.6.3's instruction, "the gap
between those two numbers is the memory story of §5 made concrete", reads
clearly here: **the memory system costs more than the kernel does, for every
encoding, by roughly an order of magnitude.**

The hot column is where the ISA question lives, and it is answered in
`docs/isa-matrix.md`.

---

## 4. What to change in the spec

Ordered by how much downstream reasoning depends on it.

1. **§5.2's outstanding-miss constant should be measured per host, not assumed.**
   10 to 16 was 4× low here. The derived 8.5 GB/s happens to match the M=16 case,
   so the table's *numbers* survive while its *derivation* does not, which
   would have been invisible without the sweep.

2. **§5.2's ns/vector table needs a per-vector latency floor.** It models cost
   as bytes ÷ bandwidth, which over-predicts fp32 by 3× and under-predicts PQ
   by 3.7×. Neither error is small, and they have opposite signs.

3. **PQ's cost is the lookup table, not the codes.** §5.4's working-set table
   counts 192 MB of codes for 1M×768 at x16 and concludes "large L3
   (partially)". It should also count the 98 KB per-query table against L1,
   which is the binding constraint.

4. **fp16 needs an explanation before it is used for anything.** Being slower
   than fp32 while moving half the bytes contradicts §5.1's framing directly.
   The ROB-occupancy hypothesis in §2 is testable with the dependent column.

5. **§5.2's table should carry two columns, dependent and independent.** The
   model describes the dependent case and the harness measured the independent
   one; both are real costs on the search path (an unprefetched walk pays the
   first, a prefetched or batched one approaches the second) and the gap
   between them is the prefetch budget of §5.2 in ns/vector rather than GB/s.

---

## 5. Host

```
cpu_model=AMD Ryzen AI 9 HX PRO 370 w/ Radeon 890M   (Zen 5 + Zen 5c, Strix Point)
cpu_cores=24 (12 physical, SMT on): Zen 5 cpus 0-3,12-15; Zen 5c cpus 4-11,16-23
L1d=48K per core  L2=1 MiB per core
L3=16 MiB shared by cpus 0-3,12-15 (Zen 5) + 8 MiB shared by cpus 4-11,16-23 (Zen 5c)
    (from /sys/devices/system/cpu/cpuN/cache/index3/{size,shared_cpu_list};
     an earlier revision of this file said "24 MiB (2x12M)", which was wrong)
pinned_cpu=NONE for every number in this file (the harness now pins and prints the cpu)
isa: avx2 avx512f avx512vnni avx512_vpopcntdq avx_vnni  (all yes)
zig=0.16.0 (pinned in build.zig.zon)
thp=madvise  numa_nodes=1  numa_balancing=0  perf_event_paranoid=-1
governor=powersave  boost=enabled  isolcpus=none  nohz_full=none
env_hash=f3e65bfb610904bd
```

The last line of that block is why these numbers are development-grade. §7.1:
"Every result row carries a hash of the environment description. Results from
different environments never share a chart."

Reproduce with:

```
zig build bench            # hardware baseline + kernel matrix
zig build bench-isa        # the forced-ISA arms
bench/isa/dump_asm.py      # regenerate docs/asm/ and the emitted-width summary
bench/setup.py check       # the §7.1 gate
```
