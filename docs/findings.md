# Open work

What is still unmeasured or unexplained, in priority order, and nothing else.
An item that closes is deleted, and its number is not reused, since commits
and code cite items by number. Why a choice was made goes to
[`decisions.md`](decisions.md); everything else lives in the commit that did it
and in the test that holds it.

Ranked by what a wrong or missing number costs.

### P1. What costs a number on the headline page

The d=1536 comparison is licensed since 2026-09-24, and the current page is
`sm/qd-dbp1m-perf-0927`, the first with Qdrant in production mode (T4 with T3
passing, 1.01x to 1.32x at matched recall; the account of the licence is in
`decisions.md`). What that page publishes wrongly, or still refuses and could
not:

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
shortens the search instead: 4,840 and 6,965 queries for the 0927 pair, read
off 0924, the newest pair at that rate (`decisions.md`, 2026-09-25). It did not
hold. 0924 was measured on development-mode Qdrant, so `W11-steady`'s 4,840
queries ran at 1,944 and 1,580 q/s and covered only 10 to 12% of the append,
mostly a quiet collection. On `W11` strawmANN's search outlived its writer by
78 s (`write overlap 43%`), and Qdrant's writer missed the stamped 3,300
points/s on every pass (2,440, 2,758 and 3,300; the row drifted +53%). The
next dbpedia pair sizes both from 0927. Item 8 is what the row shows once it
measures what it claims to.

### P2. What the licensed numbers are made of, and what the run costs

**5. strawmANN's IPC halves at saturation at d=1536.** W4 read 1.38x on the
development-profile page, as 0.85x less work per query times 1.61x cores
busy. The cores half was Qdrant's 4 search threads: on the 0927 pair, in
production mode, Qdrant fills 7.79 cores against strawmANN's 7.02, and W4 reads
parity (3,549 against 3,487 q/s), and at d=2048 on h-and-m's first pair (0929) it
reads 0.82x (4,656 against 5,683), the first saturating loss. What is left is
strawmANN's half: its cycles
per query rise from 1.66M at W3 to 3.95M at W4 (Qdrant's 4.43M, at IPC 0.43)
while its DRAM per query holds at 4.4 MB and its IPC falls from 0.70 to 0.29;
aggregate traffic is 16 GB/s against a 73 GB/s bus, so the seven workers are
waiting on their own misses, not on the bus. The next-candidate prefetch
(715343f) was measured on sift1m, where W4's IPC is 1.08, and not at d=1536,
where the latency it exists to hide is the whole cost.

**6. Qdrant's single-query cost doubles at d=1536; kernel width is a quarter
of it.** W3 reads 1.75x at d=1536 on the 0927 pair (0.84x on sift1m). Qdrant
has no AVX-512 path for dense fp32 (878843e6e), and its measured binary's
`dot_similarity_avx` uses `ymm` only, where strawmANN's `Dot(16,8)` uses
`zmm`. Measured on 2026-09-26, pinned, on dbpedia's W3: strawmANN built for
256-bit vectors (`avx2`, `avx512-256`) spends 2.05M cycles and 1.72M
instructions per query, and at 512 bits (`avx512-full`) 1.67M and 1.17M,
885 against 1,061 q/s. So width is 0.38M of the 1.44M-cycle gap to Qdrant's
3.10M, and at 256 bits strawmANN still spends 1.5x fewer cycles. That 3.10M is
the 0927 pair's, in production mode; the development profile's 3.13M cost it
1% here, against about 7% of W3's cycles on sift1m. Past that, the leads are Qdrant's 4-byte-aligned vectors splitting
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
have (it is unquantized), and a first stage whose SQ8 bounds were only fixed
at this width on the 0927 pair (T4 `|Δscore|` p50 9.7e-4, from 4.9e-1);
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
wins it back.** `W9` read 0.77x on the development-profile dbpedia page and
parity on 0927 (1.50x on sift1m). Measured on
2026-09-26 (`w9_ab.py`, pinned, three reps each, no foreign load):

| arm | q/s | cores | IPC | implied GB/s |
|---|--:|--:|--:|--:|
| 7 workers, one query a request (W9 as published) | 9.74 | 7.00 | 0.26 | 59 |
| 4 workers | 13.04 | 4.00 | 0.60 | 79 |
| 7 workers, 8 queries a request | 51.06 | 6.94 | 1.33 | 311 |

Seven concurrent 6 GB scans get less from the bus than four: 1.34x from
dropping three workers, about the 12.8 Qdrant read on the development
profile's 4 threads. In production mode it runs 7.97 cores on the 0927 pair's
W9 and reads 10.0 q/s, parity with strawmANN's 9.9: the same contention on its
side.
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
On 0927 the run-queue wait fell to 867 s, and the search got slower: 50 q/s
against Qdrant's 99, at 313M cycles per query. The deprioritised build never
published within the row (`rebuild start points=1089100 pending=99100` and no
`published`, in every pass), so every query scanned the whole pending tail
(item 8). Nice 10 bought the search workers their CPUs back and spent them on
the exhaustive tail.

### P3. What the datasets offer that no row measures yet

**55. h-and-m and laion ship real filtered queries, and W12 uses synthetic
ones.** `h-and-m-2048-angular-filters` carries a 24-field product payload per
vector (`payloads.jsonl`: product type and group, colour, department, section,
garment group, a free-text description), the value sets its conditions draw
from (`filters.json`), and 10,000 queries that each hold a condition such as
`{"and": [{"product_group_name": {"match": {"value": "Shoes"}}}]}` with their
25 nearest neighbours *under that condition* (`tests.jsonl`). `laion-small-clip`
ships the same shape with a float range condition on every query, and k=10.
The harness uploads the vectors alone, from the fbin, and W12 filters bfb's
synthetic keywords over `bench12`, so neither engine is measured on a
realistic filter, a multi-field payload or a range condition, and the shipped
filtered truth goes unread. What it needs: the payload uploaded with the
collection, a W12-style row driven by each query's own condition, and recall
scored against the shipped `closest_ids` with the ε-tie rule (checked once
against our own filtered oracle, as `ground-truth.md` §6 did for the
unconditional sets). This is also the workload vector-db-benchmark runs on
these two datasets.
