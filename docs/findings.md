# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted, and its number is not reused, since commits
and code cite items by number. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The d=1536 comparison is licensed since 2026-09-24, and the current page is
`sm/qd-dbp1m-perf-0925` (T4 with T3 passing, 1.26x to 1.44x at matched recall;
the account of the licence is in `decisions.md`). What that page publishes
wrongly, or still refuses and could not:

**1. strawmANN's SQ8 bounds now follow Qdrant's rule; the rows have not been
measured under it.** T4 on both d=1536 pages read `sq8/strawmann |Δscore|
p50=4.866e-1` against Qdrant's `1.943e-3`: `scalar.train` clipped 0.5% of the
pooled *values*, and dbpedia's component 194 sits at -0.64 in every vector, so
it and two others were clipped out of every vector. Since 2026-09-25
`scalar.train` ports Qdrant's `find_quantile_interval`, a cut counted in
vectors (`decisions.md`, 2026-09-25). Simulated, that takes `|Δ|` to 0.0010
and the stage-1 top-10 overlap from 0.810 to 0.947 on dbpedia and 0.972 to
0.985 on sift1m. The next pair on each corpus is the measurement: T4's
`|Δscore|`, the SQ8 recall sweep (which should not move, rescore being on),
and W6 and its `ef` sweep, whose throughput may move either way because the
walk navigates on the codes. Until then the published W6 rows are the old
encoder's.

**3. W11's write rate is fixed, and its search is sized to fit.** The 25 s
and 60 s spans were sized for d=128, where a 50,000-query search ends inside
them. At d=1536 it did not: the writer covered 12 to 23% of the search on every
pass of both engines, so both mixed rows measured the rebuild the append
provoked, and both were refused. 0a3de76 stretched the span to the previous
search instead, which set the write rate from a search the rate had set:
2,000 and 3,300 points/s on 0924, 200 and 500 on 0925 (W11-steady +803% on
strawmANN with no engine change), about 1,100 and 300 next. Since 2026-09-25
the spans are the constants again, so the rate is fixed (1,900 and 3,300
points/s at 1M) and hashed into the stamp, and `fullrun.resolve_w11_queries`
shortens the search instead: 4,840 and 6,965 queries for the next dbpedia pair,
read off 0924, the newest pair at that rate (`decisions.md`, 2026-09-25). The
next pair's `write overlap` says whether that holds; a search that overruns
shrinks the one after it. Item 8 is what the row shows once it measures what it
claims to.

**4. `matched` oversampling matches SQ8 and nothing else.** `W6` is licensed
at 1.29x under the policy; `W7` (0.9474 against 0.8887) and `W8` (0.9656
against 0.9428) stay refused. The sweeps carry the signature findings 42 named:
Qdrant's recall@1 climbs to 0.978 (binary) and 0.989 (PQ) at `ef` 512 while
its recall@10 flattens at 0.906 and 0.963, which is a rescore pool of `limit`
times oversampling, 40 and 20 candidates whatever `ef` is. strawmANN's pool is
`ef`-sized and its recall@10 rises to 0.986 and 0.989. The policy sends
oversampling 2 only to rows that do not already name one, and `W7` names 4, so
it never touched binary. Matching the pools means an oversampling of
`ef / limit` per row on Qdrant's side, or `ef` sweeps on `bench7` and `bench8`
so the matched-recall interpolation can cover them without touching the pool.

### P2. What the licensed numbers are made of, and what the run costs

**5. The saturating win at d=1536 is two halves, one of them narrowed.** W4
reads 1.38x, and the decomposition puts it at 0.85x less work per query times
1.61x cores busy. strawmANN's cycles per query rise from 1.66M at W3 to 4.01M
at W4 while its DRAM per query holds at 4.4 MB and its IPC halves from 0.71 to
0.29; aggregate traffic is 16 GB/s against a 73 GB/s bus, so the seven workers
are waiting on their own misses, not on the bus. The next-candidate prefetch
(715343f) was measured on sift1m, where W4's IPC is 1.08, and not at d=1536,
where the latency it exists to hide is the whole cost. That half is open.

Qdrant, on the same row and the same client parallelism of 64, fills 4.47 of
its 8 pinned cores. Read in its source at the measured commit (878843e6e) on
2026-09-25, nothing caps its searches below 8: `max_search_threads: 0` builds
a `search-io` blocking pool of `num_cpus x 4` = 32 threads, and
`AdaptiveSearchHandle` moves to the 8-thread `search-cpu` pool only above 0.9
x `num_cpus` of process CPU (7.2 cores), so at 4.46 it never does; there is no
search semaphore and no gRPC concurrency limit. The row agrees that the
threads are not waiting for a CPU but blocked: 0.2 s of run-queue wait over
the row, zero major faults, and 5.85 voluntary context switches per query
against strawmANN's 1.8. What they block on is one of two things the counters
cannot separate: the hand-off from the gRPC runtime through the search
runtime's single async worker to the blocking pool and back, or lock waits on
the segment. An off-CPU profile of Qdrant's search threads under W4 decides it,
and `max_search_threads: 8`, which removes the adaptive switch, is the A/B.

**6. Qdrant's single-query cost doubles at d=1536: its fp32 kernel is 256-bit,
which explains the instructions and not all the cycles.** W3 flips between the
tiers: 0.90x on sift1m, 1.73x here. Qdrant spends 3.13M cycles and 2.13M
instructions per query against strawmANN's 1.66M and 1.17M, at the same
recall. Qdrant has no AVX-512 path for dense fp32 (read at 878843e6e: the
only AVX-512 in `lib/` is in quantization), and the measured binary's
`dot_similarity_avx` (sha256 `dbeb0f73dea2d371`) uses `ymm` registers only,
while strawmANN's hot kernel is `Dot(16,8)` on `zmm`. Nothing else in Qdrant's
path is O(dim) per node: vectors are read in place from the mmap, cosine is a
dot on pre-normalised vectors, and each node is scored once. At 256 against
512 bits a 1536-dim dot is about 630 instructions against 255, and over the
~2,500 nodes a query scores that is the 0.96M gap; at d=128 the same arithmetic
is ~70k, inside the 5% measured there. The cycles are another matter.
strawmANN's own ISA matrix (`docs/isa-matrix.md`, measured before pinning, so
approximate) shows its AVX2 build at 1.79x the instructions of AVX-512 at
d=1536 but only 1.02x the cycles L1-hot and 1.25x DRAM-cold, which bounds the
width at about 0.6M of the 1.47M-cycle gap. Two leads on the remainder:
Qdrant's vector file carries a 4-byte header, so its vectors are 4-byte
aligned and a share of its loads split cache lines; and its server-side p50
is 0.63 ms longer (1.47 against 0.84 ms), about the whole remainder. W3 on
strawmANN's AVX2 build at d=1536 is the measurement: near Qdrant's cycles and
the ratio is the width, near its own and the rest is elsewhere. Either way the
1.73x is correct as measured, and part of what it measures is that strawmANN
uses the host's AVX-512 and Qdrant as shipped does not.

**7. At 1% selectivity strawmANN offers only the exact answer.** On 0925,
with trusted postings (202841c) and the ACORN-1 walk (15bddf7), `W12-sel10`
reads 811 q/s and leads at matched recall up to 0.988 (1.44x to 1.78x; not
established above 0.994), and `W12-sel1` reads 1,873. What is left is sel1's
sweep: strawmANN scans the ~1,900 matches at every `ef`, flat at recall
1.0000, because below `1/m0` the walk strands (0.888 recall at twice the
cost, in-process), while Qdrant's walk trades recall for speed and at `ef` 64
serves 2,383 q/s at 0.9963. That point reads 0.79x and is a trade strawmANN
has no setting for; at equal recall it leads (1.61x at 0.9999). The query is
latency-bound at the client's two in flight, 0.96 ms on one core reading 11.7
MB of scattered rows at about 12 GB/s. Two ways to offer the trade were looked
at on 2026-09-25 and not taken: an SQ8 pre-score needs codes `bench12` does not
have (it is unquantized) and a first stage that item 1 has not fixed yet;
splitting one query's scan across the idle workers would cut the latency but
is an occupancy choice, not a cheaper scan, and is the user's call. Software
prefetch in `searchSelected` was measured on 2026-09-24 and lost (0.75x to
0.80x).

**8. The exhaustive pending tail costs 15x at d=1536.** `W11-steady` is refused
(item 3), but its counters are readable: during a 5% append strawmANN's search
fell from 3,570 to 244 qps at 57.6M cycles and 59 MB of DRAM per query, which
is every query scanning 49,500 pending vectors, 304 MB at this width, before it
returns. On sift1m the same row ran at 78% of W4 because that tail is 25 MB.
The cost is tail size times dimension, and findings 31 asked for incremental
insertion on exactly this ground. Live insertion exists behind `-Dlive-insert`
(1f03ba2) and was characterised on sift1m only (7cc278d). The measurement is
`W11-steady` at d=1536 with it on, once item 3 makes the row measure a
concurrent write.

**10. Exact search loses at d=1536, on memory, not compute.** `W9` is 1.31x on
sift1m and 0.77x on dbpedia-openai-1m. strawmANN's IPC falls from 2.06 to
0.26 on 7 cores and 9.8 q/s x the 6.08 GB arena is about 60 GB/s against a
65 GB/s bus: each query streams the whole arena from DRAM, where on sift1m's
0.5 GB the concurrent scans shared lines (118 GB/s implied against 73). Qdrant
implies ~78 GB/s from 4 cores. The 2026-09-22 decision against a cross-request
gather rested on that sharing, which is gone at 6 GB, and it tested only a
gather onto one worker. `bench/harness/w9_ab.py` measures three arms (7
workers, 4 workers, and 8 queries per request as the ceiling of a gather split
across workers) plus a profile; a 100K smoke run read 101 / 136 / 470 q/s and
90% of cycles in the dot kernel. The 1M run is scheduled; it says whether
fewer streams or one shared pass is the lever, and whether 09-22 reopens for
d=1536.

**11. strawmANN's index build ran at the search workers' priority during
`W11`.** On 0925 strawmANN's `W11` served 102 q/s against Qdrant's 216, with
1,568 s of run-queue wait over the row: the append's extending builds ran
eight threads at normal priority on the search workers' eight CPUs. Since
2026-09-25 builds run at nice 10, as Qdrant's HNSW builds do (`decisions.md`).
The next pair says whether that nets out ahead once the slower build's longer
pending tail (item 8) is counted.
