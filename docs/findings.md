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

**1. strawmANN's SQ8 bounds follow Qdrant's rule; dbpedia is not measured
under it yet.** T4 on both d=1536 pages read `sq8/strawmann |Δscore|
p50=4.866e-1` against Qdrant's `1.943e-3`: `scalar.train` clipped 0.5% of the
pooled *values*, and dbpedia's component 194 sits at -0.64 in every vector.
Since 2026-09-25 it ports Qdrant's `find_quantile_interval`, a cut counted in
vectors (`decisions.md`). sift1m's 0926 pair measured it where no dimension
dominates: T4's stage-1 overlap went from 0.974 to 0.986 (simulated 0.972 to
0.985), p99 `|Δscore|` from 5.75 to 0.78, W6 unchanged within build variance,
recall untouched. dbpedia, where the old rule did its damage (simulated
`|Δ|` 0.49 to 0.001, overlap 0.810 to 0.947), is the 0927 pair.

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

**4. `matched` oversampling matches SQ8 and nothing else; `pool` should
match all three.** Under `matched`, `W6` is licensed and `W7` (0.9474
against 0.8887) and `W8` (0.9656 against 0.9428) stay refused: Qdrant's pool
is `limit` times oversampling, 40 and 20 candidates whatever `ef` is, so its
recall@10 flattens at 0.907 (binary) and 0.962 (PQ) while strawmANN's
`ef`-sized pool reaches 0.986 and 0.989. Since 2026-09-25 the `pool` policy
sends `ef / limit` on every quantized row, so both engines rescore `ef`
candidates (`decisions.md`). A night pair run with `OVERSAMPLING_POLICY=pool`
is the measurement: equal recall per row on all three encodings if the pool
was the whole gap, and any gap left is the encoders'.

**12. Every native-Qdrant pair before 2026-09-26 ran Qdrant's development
profile; sift1m is re-measured, dbpedia is not.** `fullrun.start_qdrant_binary`
never set `RUN_MODE`, Qdrant's `settings.rs` defaults it to `development`, and
started from its checkout it merged `config/development.yaml`:
`max_search_threads: 4`, audit logging of every request, `log_level: DEBUG`,
`feature_flags: all`. Fixed 2026-09-25 (3cd5588). sift1m's 0926 pair, the
first in production mode, measured what it cost: Qdrant's W4 went from 5.50 to
7.79 busy cores and the saturating ratio from 2.24x to 1.65x, W10-ef512 from
2.14x to 1.69x, W3 from 0.90x to 0.84x; W9 went the other way (item 10). The
Qdrant binary changed between those pairs too, so no single row attributes to
the profile alone. On dbpedia a production-mode W4 probe read Qdrant at
about 3,230 q/s on 7.91 cores against the published 2,580 on 4.46, which puts
that page's 1.38x nearer 1.1x; the 0927 pair replaces both dbpedia pages.

### P2. What the licensed numbers are made of, and what the run costs

**5. strawmANN's IPC halves at saturation at d=1536.** W4 read 1.38x on the
development-profile page, as 0.85x less work per query times 1.61x cores
busy. The cores half was Qdrant's 4 search threads (item 12): in production
mode it fills 7.91 of 8 on the same row (probe, 2026-09-26), `max_search_threads:
8` changes nothing, and the 0927 pair will read the ratio again. What is left
is strawmANN's half: its cycles per query rise from 1.66M at W3 to 4.01M at W4
while its DRAM per query holds at 4.4 MB and its IPC halves from 0.71 to 0.29;
aggregate traffic is 16 GB/s against a 73 GB/s bus, so the seven workers are
waiting on their own misses, not on the bus. The next-candidate prefetch
(715343f) was measured on sift1m, where W4's IPC is 1.08, and not at d=1536,
where the latency it exists to hide is the whole cost.

**6. Qdrant's single-query cost doubles at d=1536; kernel width is a quarter
of it.** W3 read 1.73x at d=1536 (0.84x on sift1m in production mode). Qdrant
has no AVX-512 path for dense fp32 (878843e6e), and its measured binary's
`dot_similarity_avx` uses `ymm` only, where strawmANN's `Dot(16,8)` uses
`zmm`. Measured on 2026-09-26, pinned, on dbpedia's W3: strawmANN built for
256-bit vectors (`avx2`, `avx512-256`) spends 2.05M cycles and 1.72M
instructions per query, and at 512 bits (`avx512-full`) 1.67M and 1.17M,
885 against 1,061 q/s. So width is 0.38M of the 1.47M-cycle gap to Qdrant's
3.13M, and at 256 bits strawmANN still spends 1.5x fewer cycles. That 3.13M
was measured in Qdrant's development profile, whose audit and debug logging
cost it about 7% of W3's cycles on sift1m; the 0927 pair gives the production
figure. Past that, the leads are Qdrant's 4-byte-aligned vectors splitting
cache lines and its longer server-side p50.

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

**10. Exact search at d=1536 is lost to scan contention, and a shared pass
wins it back.** `W9` read 0.77x on dbpedia (1.50x on sift1m). Measured on
2026-09-26 (`w9_ab.py`, pinned, three reps each, no foreign load):

| arm | q/s | cores | IPC | implied GB/s |
|---|--:|--:|--:|--:|
| 7 workers, one query a request (W9 as published) | 9.74 | 7.00 | 0.26 | 59 |
| 4 workers | 13.04 | 4.00 | 0.60 | 79 |
| 7 workers, 8 queries a request | 51.06 | 6.94 | 1.33 | 311 |

Seven concurrent 6 GB scans get less from the bus than four: 1.34x from
dropping three workers, about Qdrant's published 12.8, which it reached on the
development profile's 4 threads. Production Qdrant runs 7.83 scan threads on
sift1m's W9 and each costs 2.3x the cycles, the same contention on its side.
And one pass shared by 8 queries is 5.2x: the 2026-09-22 decision against a
cross-request gather rested on sift1m, where concurrent scans shared lines
through the cache, and tested a gather onto one worker. At 6 GB there is no
sharing, and a gather split across the workers is the lever, which reopens
that decision for d=1536. Capping concurrent exact scans below the worker
count is the smaller one.

**11. strawmANN's index build ran at the search workers' priority during
`W11`.** On 0925 strawmANN's `W11` served 102 q/s against Qdrant's 216, with
1,568 s of run-queue wait over the row: the append's extending builds ran
eight threads at normal priority on the search workers' eight CPUs. Since
2026-09-25 builds run at nice 10, as Qdrant's HNSW builds do (`decisions.md`).
The next pair says whether that nets out ahead once the slower build's longer
pending tail (item 8) is counted.
