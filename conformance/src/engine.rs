//! One client, both engines.
//!
//! §8.5: "`conformance/` is a Rust binary using the same `qdrant-client` crate
//! that bfb uses. This is deliberate: one client, one encoder, one decoder,
//! both engines. Any difference observed is then necessarily server-side, not
//! an artefact of how the harness talks to each engine."
//!
//! That is the entire reason this file exists as a thin wrapper rather than as
//! two engine-specific adapters. `Engine` holds a `Qdrant` client and a label;
//! nothing else differs between the two arms. If a future change adds a
//! conditional on `label`, it has broken the property the tier results depend
//! on.

use crate::oracle::Metric;
use crate::relevance::Returned;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeleteCollectionBuilder, Distance, Filter,
    OptimizersConfigDiffBuilder, PointStruct, QueryBatchPointsBuilder, QueryPointsBuilder,
    SearchParamsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};

/// The `max_segment_size` (in KB) that lets Qdrant's optimizer merge a corpus
/// of `points` x `dim` into one graph.
///
/// `workloads.max_segment_size_kb` computes exactly this for the *measured*
/// collections, which is why they are held to one graph and the differ's were
/// not: `recreate_collection_hnsw` sent no `optimizers_config` at all, so
/// Qdrant's default ceiling applied and a 1536-dimension corpus split where a
/// 128-dimension one did not.
///
/// What that cost: the differ measured Qdrant's recall@10 at
/// 0.9831 against strawmANN's 0.9667 and T3 failed on non-overlapping CIs,
/// while the recall sweep over the measured collection had the two within
/// 0.0030 at every `ef`. Qdrant was not more accurate; it was searching more
/// graphs at the same `ef`, and 0.9831 is its own ef=256 figure. The whole
/// dbpedia-openai-1m run lost its comparative licence to a collection setting.
///
/// Sized from the corpus rather than a constant, for the reason the Python
/// side gives: a threshold that does not scale with the collection it governs
/// silently stops applying at the next tier up. Four bytes per component,
/// times the points, times four for index overhead and headroom.
pub fn max_segment_size_kb(points: u64, dim: u64) -> u64 {
    let corpus_kb = points.saturating_mul(dim).saturating_mul(4) / 1024;
    corpus_kb.saturating_mul(4).max(1)
}

/// One populated graph, as strawmANN serves: the ceiling above, and the
/// segment-count target the `equal-work` policy already exports to the engine.
///
/// Both are needed. `default_segment_number` alone is a target the optimizer
/// cannot reach while a merged segment would exceed `max_segment_size`, which
/// is precisely the state the differ was in.
fn one_graph_optimizers(points: u64, dim: u64) -> OptimizersConfigDiffBuilder {
    OptimizersConfigDiffBuilder::default()
        .default_segment_number(1)
        .max_segment_size(max_segment_size_kb(points, dim))
}

pub struct Engine {
    pub label: String,
    pub client: Qdrant,
}

pub fn to_distance(m: Metric) -> Distance {
    match m {
        Metric::Dot => Distance::Dot,
        Metric::Cosine => Distance::Cosine,
        Metric::Euclid => Distance::Euclid,
        Metric::Manhattan => Distance::Manhattan,
    }
}

impl Engine {
    pub fn connect(label: &str, url: &str) -> anyhow::Result<Engine> {
        // §2: the client's compatibility check calls HealthCheck on
        // construction and compares the returned version against its own. A
        // mismatch is a warning; a failure to answer is fatal. Leaving the
        // check enabled is deliberate — it is part of what T0 verifies.
        // The default client timeout is far too short for bulk ingest at the
        // headline dimension.
        //
        // Qdrant indexes *while* it ingests, so a `wait(true)` upsert of a
        // 256 KB batch can block for many seconds once the collection is large
        // and the optimizer is busy — long enough that the client cancels the
        // request. The failure surfaces as `The operation was cancelled Timeout
        // expired`, which reads like a server fault and is not one: the server
        // was healthy and mid-optimization throughout.
        //
        // This is a property of the *comparison target*, not of our harness, so
        // the right response is to wait rather than to shrink the batches or
        // drop `wait`. Both would change what is being measured.
        let client = Qdrant::from_url(url)
            .timeout(std::time::Duration::from_secs(600))
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()?;
        Ok(Engine {
            label: label.to_string(),
            client,
        })
    }

    pub async fn recreate_collection(
        &self,
        name: &str,
        dim: u64,
        metric: Metric,
        points: u64,
    ) -> anyhow::Result<()> {
        self.recreate_collection_hnsw(name, dim, metric, None, None, points)
            .await
    }

    /// §6.5's `m` and `ef_construct`, sent explicitly.
    ///
    /// Both engines default to `m=16, ef_construct=100`, so the defaults were
    /// already matched — but "matched by coincidence of defaults" is not the
    /// same as "controlled", and a sweep needs to set them.
    pub async fn recreate_collection_hnsw(
        &self,
        name: &str,
        dim: u64,
        metric: Metric,
        m: Option<u64>,
        ef_construct: Option<u64>,
        points: u64,
    ) -> anyhow::Result<()> {
        use qdrant_client::qdrant::HnswConfigDiffBuilder;

        let _ = self
            .client
            .delete_collection(DeleteCollectionBuilder::new(name))
            .await;

        let mut b = CreateCollectionBuilder::new(name)
            .vectors_config(VectorParamsBuilder::new(dim, to_distance(metric)))
            .optimizers_config(one_graph_optimizers(points, dim));
        if m.is_some() || ef_construct.is_some() {
            let mut h = HnswConfigDiffBuilder::default();
            if let Some(v) = m {
                h = h.m(v);
            }
            if let Some(v) = ef_construct {
                h = h.ef_construct(v);
            }
            b = b.hnsw_config(h);
        }
        self.client.create_collection(b).await?;
        Ok(())
    }

    /// Drop a scratch collection; a failure to drop is not a finding.
    pub async fn delete_collection(&self, name: &str) {
        let _ = self
            .client
            .delete_collection(DeleteCollectionBuilder::new(name))
            .await;
    }

    /// §8.5 T4 needs a *quantized* collection alongside the fp32 one, so the
    /// same query can be scored both ways and the difference measured.
    ///
    /// SQ8 at bfb's 0.99 quantile (§6.7), kept in RAM so the measurement is of
    /// the encoding rather than of a disk read.
    pub async fn recreate_collection_quantized(
        &self,
        name: &str,
        dim: u64,
        metric: Metric,
        points: u64,
    ) -> anyhow::Result<()> {
        use qdrant_client::qdrant::{QuantizationType, ScalarQuantizationBuilder};

        let _ = self
            .client
            .delete_collection(DeleteCollectionBuilder::new(name))
            .await;

        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(dim, to_distance(metric)))
                    .optimizers_config(one_graph_optimizers(points, dim))
                    .quantization_config(
                        ScalarQuantizationBuilder::default()
                            .r#type(QuantizationType::Int8 as i32)
                            .quantile(0.99)
                            .always_ram(true),
                    ),
            )
            .await?;
        Ok(())
    }

    pub async fn upsert(
        &self,
        name: &str,
        base: &[f32],
        n: usize,
        dim: usize,
    ) -> anyhow::Result<()> {
        // §4.3: "point ID must equal row index". The harness enforces the
        // preconditions elsewhere; here it simply honours them.
        let order: Vec<u32> = (0..n as u32).collect();
        self.upsert_in_order(name, base, dim, &order).await
    }

    /// `upsert`, inserting the rows named by `order`, in that order.
    ///
    /// Ids stay equal to the row index (§4.3) whatever the order; only the
    /// sequence the engine sees them in changes. This is what §8.6's
    /// permutation invariance needs — "insertion order must not change
    /// exact-search results" — and it was not being tested: the check that
    /// carried the name permuted the *query batch*, which no engine has ever
    /// been observed to care about.
    pub async fn upsert_in_order(
        &self,
        name: &str,
        base: &[f32],
        dim: usize,
        order: &[u32],
    ) -> anyhow::Result<()> {
        let n = order.len();
        //
        // Chunked by **bytes**, not by point count. A fixed 512 points is 262 KB
        // at SIFT's d=128 and 3.1 MB at the headline tier's d=1536 — over any
        // reasonable per-stream request limit, and the failure arrives as a
        // transport error partway through an upload rather than as anything
        // legible. Sizing by payload keeps the request shape constant across
        // dimensions, which is also what makes ingest numbers comparable
        // between tiers.
        const TARGET_BYTES: usize = 256 * 1024;
        let per_point = dim * std::mem::size_of::<f32>() + 32; // + id and framing
        let chunk = (TARGET_BYTES / per_point).clamp(1, 1024);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let points: Vec<PointStruct> = order[start..end]
                .iter()
                .map(|&id| {
                    let i = id as usize;
                    let v: Vec<f32> = base[i * dim..(i + 1) * dim].to_vec();
                    // One key per point, so T0's payload round-trip has a
                    // payload to round-trip: with `Payload::new()` both
                    // engines returned nothing and "payload presence" was
                    // two `false`s agreeing.
                    let payload = qdrant_client::Payload::try_from(serde_json::json!({ "i": id }))
                        .expect("a one-key object is a payload");
                    PointStruct::new(u64::from(id), v, payload)
                })
                .collect();
            self.client
                .upsert_points(UpsertPointsBuilder::new(name, points).wait(true))
                .await?;
            start = end;
        }
        Ok(())
    }

    /// §2: "bfb polls `collection_info` once per second and requires
    /// `status == Green` **three consecutive times**." The harness does the
    /// same, so both engines are measured in the same state.
    pub async fn wait_green(&self, name: &str, timeout_secs: u64) -> anyhow::Result<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let mut greens = 0;
        while std::time::Instant::now() < deadline {
            let info = self.client.collection_info(name).await?;
            let green = info.result.as_ref().map(|r| r.status == 1).unwrap_or(false);
            if green {
                greens += 1;
                if greens >= 3 {
                    return Ok(());
                }
            } else {
                greens = 0;
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        anyhow::bail!(
            "{}: collection {name} did not reach Green three times in a row",
            self.label
        )
    }

    /// `points_count` from `CollectionInfo`.
    ///
    /// §8.6's idempotent-upsert property is "re-upserting identical points
    /// changes nothing observable", and the count is the first observable it
    /// would change: a re-upsert that appended instead of overwriting doubles
    /// it while leaving every query answer intact.
    pub async fn count(&self, name: &str) -> anyhow::Result<u64> {
        let info = self.client.collection_info(name).await?;
        Ok(info.result.and_then(|r| r.points_count).unwrap_or(0))
    }

    /// The segment count the engine actually built, for the pin above.
    ///
    /// "A pin nobody reads back is not a pin" (`main.rs`), and this is the pin
    /// the differ never read. Reported rather than asserted: Qdrant keeps an
    /// empty appendable segment alongside the populated one, so a count of 2
    /// can mean one graph, and the count is an upper bound on the graphs
    /// searched. A caller that sees more than one should say so loudly; it
    /// must not silently fail a run that is fine.
    pub async fn segments_count(&self, name: &str) -> anyhow::Result<u64> {
        let info = self.client.collection_info(name).await?;
        Ok(info.result.map(|r| r.segments_count).unwrap_or(0))
    }

    /// What the engine says its collection actually is, not what it was asked
    /// for.
    ///
    /// §8.9 pins the comparison target and §4 stamps the collection settings
    /// the harness *sent*; neither checks what came back. `--segments 1` sets
    /// Qdrant's `default_segment_number`, which is a target its optimizer is
    /// free to miss, and the segment count is the single biggest confound in
    /// the `ef` comparison (`docs/comparison.md` §2: eight segments searched at
    /// the requested `ef` is roughly eight times the work of one). A pin nobody
    /// reads back is not a pin.
    pub async fn collection_config(&self, name: &str) -> anyhow::Result<serde_json::Value> {
        let info = self.client.collection_info(name).await?;
        let Some(r) = info.result else {
            anyhow::bail!("{name}: collection_info returned no result");
        };
        let cfg = r.config.as_ref();
        let params = cfg.and_then(|c| c.params.as_ref());
        // The first named vector, or the unnamed one: §4's collections have a
        // single vector field, and a config that grew a second is a difference
        // the reader should see rather than a field this quietly averages.
        let vp = params
            .and_then(|p| p.vectors_config.as_ref())
            .and_then(|v| {
                use qdrant_client::qdrant::vectors_config::Config;
                match v.config.as_ref()? {
                    Config::Params(p) => Some(*p),
                    Config::ParamsMap(m) => m.map.values().next().copied(),
                }
            });
        let hnsw = cfg.and_then(|c| c.hnsw_config.as_ref());
        let opt = cfg.and_then(|c| c.optimizer_config.as_ref());
        let quant = cfg.and_then(|c| c.quantization_config.as_ref()).map(|q| {
            use qdrant_client::qdrant::quantization_config::Quantization;
            match q.quantization.as_ref() {
                Some(Quantization::Scalar(_)) => "scalar",
                Some(Quantization::Product(_)) => "product",
                Some(Quantization::Binary(_)) => "binary",
                Some(Quantization::Turboquant(_)) => "turboquant",
                None => "unset",
            }
        });
        let mut payload_indexes: Vec<&str> = r.payload_schema.keys().map(String::as_str).collect();
        payload_indexes.sort_unstable();
        Ok(serde_json::json!({
            "collection": name,
            "status": match r.status { 1 => "green", 2 => "yellow", 3 => "red",
                                       4 => "grey", _ => "unknown" },
            "segments_count": r.segments_count,
            "points_count": r.points_count,
            "indexed_vectors_count": r.indexed_vectors_count,
            "shard_number": params.map(|p| p.shard_number),
            "vector_size": vp.as_ref().map(|v| v.size),
            // Names, not tags. A report showing `distance: 2` asks its reader
            // to know the protobuf enum by heart, which is a worse failure than
            // omitting the field.
            "distance": vp.as_ref().map(|v| match v.distance {
                1 => "cosine", 2 => "euclid", 3 => "dot", 4 => "manhattan", _ => "unknown",
            }),
            // §5.5's residency, and the field both engines already speak:
            // strawmANN writes `coll.config.placement` here and Qdrant its
            // `Memory`. Capturing it is what stops an I/O or RSS comparison
            // being read as architecture when it is a placement nobody chose —
            // strawmANN defaults to `pinned` (anonymous memory) and Qdrant to
            // `cached` (mmap), and `pinned` is not available to Qdrant for
            // dense vectors at all.
            "placement": vp.as_ref().and_then(|v| v.memory).map(|m| match m {
                1 => "cold", 2 => "cached", 3 => "pinned", _ => "unknown",
            }),
            "datatype": vp.as_ref().and_then(|v| v.datatype).map(|d| match d {
                0 => "default", 1 => "float32", 2 => "uint8", 3 => "float16",
                4 => "turbo4", _ => "unknown",
            }),
            // Which payload fields the engine says it *actually* indexed, not
            // which ones the run asked for. W12 is a filtered search, and
            // Qdrant picks its search path by filter cardinality rather than by
            // whether an index exists (`read_view/dispatch.rs`): with no
            // payload index a low-cardinality filter takes
            // `iter_filtered_points` and checks every point, per query. That is
            // a full scan wearing a filtered search's name, and until this
            // field existed the only record of it was a comment in
            // `workloads.py`. `compare.py` refuses the row when this is empty.
            "payload_indexes": payload_indexes,
            // Both deprecated in 1.19 (`on_disk_payload`, `VectorParams::on_disk`)
            // and not what this capture is for: the segment count is.
            "hnsw_m": hnsw.and_then(|h| h.m),
            "hnsw_ef_construct": hnsw.and_then(|h| h.ef_construct),
            "default_segment_number": opt.and_then(|o| o.default_segment_number),
            "quantization": quant,
        }))
    }

    /// Run a batch of queries and return the results in wire order.
    ///
    /// `exact` maps to `params.exact`, which §6.5 makes the recall ground truth
    /// and W9's benchmark — it must never be silently served by an index.
    #[allow(clippy::too_many_arguments)]
    pub async fn query(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        exact: bool,
        hnsw_ef: Option<u64>,
    ) -> anyhow::Result<Vec<Returned>> {
        self.query_quant(name, queries, n_queries, dim, limit, exact, hnsw_ef, None)
            .await
    }

    /// `query` with `offset`, for §8.6's offset-consistency property.
    ///
    /// The property is `query(limit=L, offset=O) == query(limit=L+O)[O..]`,
    /// and it needs a request that actually carries an offset. It was being
    /// "checked" by slicing one offset-free list against another at `O=0`,
    /// which is `x == x`.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_offset(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        offset: u64,
        exact: bool,
    ) -> anyhow::Result<Vec<Returned>> {
        self.query_full(
            name, queries, n_queries, dim, limit, offset, exact, None, None, None, None,
        )
        .await
    }

    /// §8.5 T4: the same query scored through the quantized path and through
    /// fp32, so `|quantized_score - fp32_score|` is measurable.
    ///
    /// `quant` is `Some((ignore, rescore))`. `ignore: true` forces the fp32
    /// path on a quantized collection, which is what makes the two runs
    /// differ *only* in the encoding rather than in the collection.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_quant(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        exact: bool,
        hnsw_ef: Option<u64>,
        quant: Option<(bool, bool)>,
    ) -> anyhow::Result<Vec<Returned>> {
        self.query_quant_oversampled(
            name, queries, n_queries, dim, limit, exact, hnsw_ef, quant, None,
        )
        .await
    }

    /// The same, with `quantization.oversampling` set.
    ///
    /// §8.6's quantization-dominance property is a statement about how recall
    /// moves with oversampling, so measuring it needs the knob the property is
    /// about.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_quant_oversampled(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        exact: bool,
        hnsw_ef: Option<u64>,
        quant: Option<(bool, bool)>,
        oversampling: Option<f64>,
    ) -> anyhow::Result<Vec<Returned>> {
        self.query_full(
            name,
            queries,
            n_queries,
            dim,
            limit,
            0,
            exact,
            hnsw_ef,
            quant,
            oversampling,
            None,
        )
        .await
    }

    /// `query`, carrying a payload filter.
    ///
    /// W12's recall is only a recall if the sweep searches the way the row
    /// does *and* is scored against a truth restricted to the same condition
    /// (docs/workloads.md W12 point 3). This is the search half; the truth half
    /// is `oracle::compute_filtered`, and `relevance` refuses to pair one with
    /// the other's absence.
    #[allow(clippy::too_many_arguments)]
    pub async fn query_filtered(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        exact: bool,
        hnsw_ef: Option<u64>,
        filter: &Filter,
    ) -> anyhow::Result<Vec<Returned>> {
        self.query_full(
            name,
            queries,
            n_queries,
            dim,
            limit,
            0,
            exact,
            hnsw_ef,
            None,
            None,
            Some(filter),
        )
        .await
    }

    /// Every query knob in one place; the named entry points above are the
    /// combinations the tiers use.
    #[allow(clippy::too_many_arguments)]
    async fn query_full(
        &self,
        name: &str,
        queries: &[f32],
        n_queries: usize,
        dim: usize,
        limit: u64,
        offset: u64,
        exact: bool,
        hnsw_ef: Option<u64>,
        quant: Option<(bool, bool)>,
        oversampling: Option<f64>,
        filter: Option<&Filter>,
    ) -> anyhow::Result<Vec<Returned>> {
        let mut out = Vec::with_capacity(n_queries);

        // Batched, because §2 notes bfb searches via QueryBatch rather than
        // Search and the two exercise different server paths.
        const BATCH: usize = 32;
        let mut start = 0usize;
        while start < n_queries {
            let end = (start + BATCH).min(n_queries);
            let mut batch = Vec::with_capacity(end - start);
            for qi in start..end {
                let v: Vec<f32> = queries[qi * dim..(qi + 1) * dim].to_vec();
                let mut params = SearchParamsBuilder::default();
                params = params.exact(exact);
                if let Some(ef) = hnsw_ef {
                    params = params.hnsw_ef(ef);
                }
                if let Some((ignore, rescore)) = quant {
                    let mut qp = qdrant_client::qdrant::QuantizationSearchParamsBuilder::default();
                    qp = qp.ignore(ignore).rescore(rescore);
                    if let Some(ov) = oversampling {
                        qp = qp.oversampling(ov);
                    }
                    params = params.quantization(qp);
                }
                let mut qb = QueryPointsBuilder::new(name)
                    .query(v)
                    .limit(limit)
                    .params(params);
                if offset > 0 {
                    qb = qb.offset(offset);
                }
                // W12: the same condition the ground truth was restricted to.
                // A search that does not carry it is answering a different
                // question from the one it is about to be scored against.
                if let Some(f) = filter {
                    qb = qb.filter(f.clone());
                }
                batch.push(qb.build());
            }

            let resp = self
                .client
                .query_batch(QueryBatchPointsBuilder::new(name, batch))
                .await?;

            for br in resp.result {
                let mut ids = Vec::with_capacity(br.result.len());
                let mut scores = Vec::with_capacity(br.result.len());
                for p in br.result {
                    let id = match p.id.and_then(|i| i.point_id_options) {
                        // Not `as u32`: that truncated an id past 2^32 to some
                        // other row's id, and the tiers then judged the wrong
                        // point rather than a wrong wire.
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => {
                            u32::try_from(n).map_err(|_| {
                                anyhow::anyhow!(
                                    "{}: point id {n} does not fit u32; §4.3 ids are row indices",
                                    self.label
                                )
                            })?
                        }
                        // §4.3 requires numeric ids equal to the row index for
                        // relevance runs, so a UUID here means the run was
                        // misconfigured rather than that we should guess.
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => {
                            anyhow::bail!(
                                "{}: got UUID id {u}; relevance runs require numeric ids (§4.3)",
                                self.label
                            )
                        }
                        None => anyhow::bail!("{}: ScoredPoint with no id", self.label),
                    };
                    ids.push(id);
                    scores.push(f64::from(p.score));
                }
                out.push(Returned { ids, scores });
            }
            start = end;
        }
        Ok(out)
    }

    /// §8.5 T0: capture the response *shape*, for structural comparison.
    ///
    /// A gRPC error is a shape too. §8.5 T0: "unsupported operations must
    /// produce the same gRPC status code as Qdrant, not a hang or a
    /// wrong-shaped success." The status was hard-coded to 0 and the error
    /// path returned `Err`, so `compare_shape`'s status comparison could only
    /// ever see two zeros; a run in which one engine refused the request and
    /// the other answered it aborted instead of recording the difference.
    pub async fn wire_shape(
        &self,
        name: &str,
        query: &[f32],
        limit: u64,
    ) -> anyhow::Result<crate::differ::WireShape> {
        let resp = match self
            .client
            .query_batch(QueryBatchPointsBuilder::new(
                name,
                // `with_payload`: T0 compares payload presence, and a probe
                // that asks for none compares two `false`s.
                vec![
                    QueryPointsBuilder::new(name)
                        .query(query.to_vec())
                        .limit(limit)
                        .with_payload(true)
                        .build(),
                ],
            ))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return match grpc_status_code(&e) {
                    Some(code) => Ok(crate::differ::WireShape {
                        result_count: 0,
                        id_variant: crate::differ::IdVariant::Absent,
                        has_time: false,
                        has_payload: false,
                        has_vectors: false,
                        status_code: code,
                    }),
                    // Not a server verdict — a transport or client fault —
                    // so there is no shape to record.
                    None => Err(e.into()),
                };
            }
        };

        let first = resp.result.first();
        let points = first.map(|b| b.result.as_slice()).unwrap_or(&[]);

        let mut variant = crate::differ::IdVariant::Absent;
        for p in points {
            let v = match p.id.as_ref().and_then(|i| i.point_id_options.as_ref()) {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(_)) => {
                    crate::differ::IdVariant::Num
                }
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(_)) => {
                    crate::differ::IdVariant::Uuid
                }
                None => crate::differ::IdVariant::Absent,
            };
            variant = match variant {
                crate::differ::IdVariant::Absent => v,
                existing if existing == v => existing,
                // Spelled out rather than `_`: this is our own enum, and a
                // variant added later must be decided here rather than
                // silently folded into `Mixed`.
                crate::differ::IdVariant::Num
                | crate::differ::IdVariant::Uuid
                | crate::differ::IdVariant::Mixed => crate::differ::IdVariant::Mixed,
            };
        }

        Ok(crate::differ::WireShape {
            result_count: points.len(),
            id_variant: variant,
            // §8.5 T0 checks *presence*; the value is on the ignore-list
            // because it differs by construction on every call.
            has_time: resp.time > 0.0,
            has_payload: points.iter().any(|p| !p.payload.is_empty()),
            has_vectors: points.iter().any(|p| p.vectors.is_some()),
            // A successful reply carries `OK`, which is 0 in the gRPC numbering.
            status_code: GRPC_OK,
        })
    }
}

/// `tonic::Code::Ok as i32`.
pub const GRPC_OK: i32 = 0;

/// The gRPC status code a client error carries, if the server sent one.
///
/// `tonic::Code`'s discriminants are the wire numbering (`Ok=0`,
/// `InvalidArgument=3`, `NotFound=5`, `Unimplemented=12`, ...), so `as i32`
/// is the value T0 compares.
pub fn grpc_status_code(e: &qdrant_client::QdrantError) -> Option<i32> {
    // `QdrantError` is theirs, not ours, and the two arms below are the only
    // ones that carry a gRPC status at all. Enumerating the rest — as
    // `wildcard_enum_match_arm` asks — would pin this file to their variant
    // list, so a new error in one of their patch releases would stop the
    // conformance binary compiling for a status code it does not have. The
    // lint is right about our own enums and wrong about this one.
    #[allow(clippy::wildcard_enum_match_arm)]
    match e {
        qdrant_client::QdrantError::ResponseError { status } => Some(status.code() as i32),
        qdrant_client::QdrantError::ResourceExhaustedError { status, .. } => {
            Some(status.code() as i32)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_status_is_propagated_as_its_grpc_code_and_a_client_fault_is_not() {
        // §8.5 T0: the *code* is what must match between engines. Built from
        // the same tonic types the client hands back, so the mapping is the
        // one a live run exercises.
        let unimplemented = qdrant_client::QdrantError::ResponseError {
            status: tonic::Status::unimplemented("no such rpc"),
        };
        assert_eq!(grpc_status_code(&unimplemented), Some(12));
        let invalid = qdrant_client::QdrantError::ResponseError {
            status: tonic::Status::invalid_argument("dim mismatch"),
        };
        assert_eq!(grpc_status_code(&invalid), Some(3));
        assert_ne!(Some(GRPC_OK), grpc_status_code(&invalid));

        // A conversion error never reached the server; there is no status.
        let client_side = qdrant_client::QdrantError::ConversionError("bad vector".into());
        assert_eq!(grpc_status_code(&client_side), None);
    }

    #[test]
    fn the_segment_ceiling_clears_the_corpus_at_every_width_measured() {
        // The ceiling must exceed the corpus, or the optimizer cannot merge to
        // one graph however low `default_segment_number` is set. Checked at the
        // two widths that actually run, because the 1536 one is where the
        // engine default stopped clearing it and T3 failed.
        for &(points, dim) in &[(1_000_000u64, 128u64), (990_000, 1536)] {
            let corpus_kb = points * dim * 4 / 1024;
            assert!(
                max_segment_size_kb(points, dim) > corpus_kb,
                "ceiling must exceed the {points}x{dim} corpus of {corpus_kb} KB"
            );
        }
    }

    #[test]
    fn the_segment_ceiling_matches_what_the_measured_collections_get() {
        // `workloads.max_segment_size_kb` sends 29,700,000 KB for the
        // dbpedia-openai-1m capacity of 1,237,500 x 1536. The differ's
        // collections must be held to the same rule, or the two sides of the
        // same run are pinned differently -- which is the defect this replaces.
        assert_eq!(max_segment_size_kb(1_237_500, 1536), 29_700_000);
    }

    #[test]
    fn the_segment_ceiling_scales_with_the_corpus_rather_than_being_a_constant() {
        // A threshold that does not scale silently stops applying at the next
        // tier up. Twelve times the width must move it.
        let narrow = max_segment_size_kb(990_000, 128);
        let wide = max_segment_size_kb(990_000, 1536);
        assert_eq!(wide, narrow * 12);
        assert!(max_segment_size_kb(1, 1) >= 1, "never zero, whatever rounds down");
    }
}
