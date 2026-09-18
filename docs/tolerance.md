# Calibrated tolerances (§8.4)

Derived, not chosen. Each ε is a multiple of the larger of two measured
spreads: Qdrant's disagreement with itself across its runtime-dispatched
kernels, and strawmann's across the §7.5 forced-ISA matrix.

| metric | dim | ε | kind | ×  | Qdrant spread | strawmann spread |
|---|--:|--:|---|--:|---|---|
| cosine | 1536 | 9.537e-7 | absolute | 4 | n=3000 max=0.000e0 p99.9=0.000e0 p99=0.000e0 p50=0.000e0 mean=0.000e0 (rel max=0.000e0 p99=0.000e0) | n=3000 max=2.384e-7 p99.9=2.384e-7 p99=1.788e-7 p50=5.960e-8 mean=5.708e-8 (rel max=2.987e-7 p99=2.158e-7) |
| euclid | 128 | 4.172e-7 | relative | 4 | n=30000 max=0.000e0 p99.9=0.000e0 p99=0.000e0 p50=0.000e0 mean=0.000e0 (rel max=0.000e0 p99=0.000e0) | n=30000 max=0.000e0 p99.9=0.000e0 p99=0.000e0 p50=0.000e0 mean=0.000e0 (rel max=0.000e0 p99=0.000e0) |

These cells are mirrored in `conformance/src/differ/tolerance.rs` (`CALIBRATED`),
which is what `differ` and `relevance` default to when `--epsilon` is not
given; a new calibration must update both, and each run prints which ε it used.

Only the cells above are calibrated. Every other (metric, dim) falls back
to §8.4's computed floor, `expected_summation_error(dim)`. A floor is not a
calibration: it is what summation error *should* be at that width, not what
these two engines were measured to disagree by. `calibrate` is what turns
the one into the other, and each run says which of the two it used.

### euclid at d=128: measured, and measured at zero

The euclid/128 cell is SIFT1M, the corpus the published comparison runs
on, and its ε *equals* the floor, so the row looks like a floor and is
not one. Across eight forced-ISA arms (SSE2 through AVX-512, VNNI and
VPOPCNTDQ) and two Qdrant arms, over 30,000 scores, both cross-arm spreads
came back **exactly** zero, so `calibrate`'s floor clause set the value.

That zero is arithmetic rather than luck, and it is worth knowing before
anyone re-runs the calibration expecting a number. SIFT descriptors are
uint8, so a squared euclid over d=128 is an integer of at most
128·255² = 8,323,200 (the observed maximum is 126,546) and every partial
sum is exactly representable in fp32 below 2²⁴ = 16,777,216. A sum of
exact integers is the same in any order, so no arrangement of accumulators
can disagree, and the final square root is a correctly-rounded operation on
an identical input. The distinction the cell buys is therefore the label
and not the number: `calibrated` says a measurement found the disagreement
to be nil, where `floor` says nobody looked.

## Why these are not `1e-6`

§8.1 is explicit that bit-exact equality with Qdrant is **not achievable**:
its AVX path uses 4 accumulators over 32 floats with FMA, ours uses 8 with a
different reduction tree, and different summation order gives different
last bits. That is not a defect in either engine, and it is a statement
about the general case: on a corpus whose distances are exact integers it
does not hold, which is what the euclid/128 row above measures. What *is*
achievable is value equality within a tolerance derived from how much each
engine already disagrees with itself, which is what this table records.
