/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reuse signed weight payloads with compatible declared preparation layouts.

use std::collections::{BTreeMap, BTreeSet};

use helios_core::package::{cache, protection, reader, schema::wire as pwire};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use super::{
    CaptureRequest, RankOutput, Result, WeightPayloadGroup, WeightPayloadSourceSpan,
    WeightPayloadSources, invalid, weight_payload,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Source {
    envelope: Vec<u8>,
    publishers: Vec<helios_core::runtime_package::Publisher>,
    #[serde(default)]
    rebuild_manifest: Option<serde_json::Value>,
}

#[pyclass(module = "flare_heliopause")]
pub(crate) struct NativeWeightSource {
    pub(crate) source: Source,
    // The entry's hardlinks retain all payloads without one open file per weight.
    _entry: cache::Entry,
}

#[pymethods]
impl NativeWeightSource {
    fn metadata(&self) -> PyResult<String> {
        serde_json::to_string(&self.source)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyfunction]
fn export_reuse_weights(
    py: Python<'_>,
    cache: String,
    model: String,
    profiles: String,
) -> PyResult<NativeWeightSource> {
    py.detach(|| open_source(std::path::Path::new(&cache), &model, &profiles))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn open_source(
    cache: &std::path::Path,
    model: &str,
    profiles: &str,
) -> Result<NativeWeightSource> {
    let root = cache::Root::open(cache)?;
    let entry = root.resolve(&cache::model_key(&model)?)?.ok_or_else(|| {
        invalid(
            "--reuse-weights requires a cached default model version; export with weights first",
        )
    })?;
    let metadata = reader::Metadata::open(&root.primary(&entry)?)?;
    let publishers = super::conversion::payload_publishers(&profiles)?;
    let catalog = protection::verify(
        &metadata.envelope,
        &publishers,
        protection::Selection::Catalog,
    )?;
    if catalog.package().model_identity != model {
        return Err(invalid("cached weight source belongs to a different model"));
    }
    let mut payloads = BTreeSet::new();
    for deployment in 0..catalog.deployment_count() {
        let verified = protection::verify(
            &metadata.envelope,
            &publishers,
            protection::Selection::Deployment {
                deployment,
                rank: None,
            },
        )?;
        for rank in verified.ranks() {
            for initializer in &rank.initializers {
                let Some(source) = initializer
                    .source
                    .as_ref()
                    .filter(|source| is_weight(source.role))
                else {
                    continue;
                };
                let slice = source
                    .slice
                    .as_ref()
                    .ok_or_else(|| invalid("weight initializer has no payload slice"))?;
                if payloads.contains(&slice.payload_id) {
                    continue;
                }
                let id = helios_core::package::PayloadId::from_bytes(
                    slice.payload_id.as_slice().try_into()?,
                );
                let object = root.payload(id)?.ok_or_else(|| {
                    invalid(format!("--reuse-weights is missing cached payload {id}"))
                })?;
                if object.length() != verified.catalog().length(&slice.payload_id)? {
                    return Err(invalid(format!(
                        "cached weight payload {id} has a different length"
                    )));
                }
                payloads.insert(slice.payload_id.clone());
            }
        }
        if let Some(engram) = verified.deployment().and_then(|body| body.engram.as_ref()) {
            for source in engram_uses(engram) {
                let slice = source
                    .slice
                    .as_ref()
                    .ok_or_else(|| invalid("cached Engram plane has no slice"))?;
                let id = helios_core::package::PayloadId::from_bytes(
                    slice.payload_id.as_slice().try_into()?,
                );
                let object = root.payload(id)?.ok_or_else(|| {
                    invalid(format!(
                        "--reuse-weights is missing cached Engram payload {id}"
                    ))
                })?;
                if object.length() != verified.catalog().length(&slice.payload_id)? {
                    return Err(invalid("cached Engram payload has a different length"));
                }
                payloads.insert(slice.payload_id.clone());
            }
        }
    }
    if payloads.is_empty() {
        return Err(invalid(
            "--reuse-weights requires a cached model version containing weights",
        ));
    }
    Ok(NativeWeightSource {
        source: Source {
            envelope: metadata.envelope,
            publishers,
            rebuild_manifest: None,
        },
        _entry: entry,
    })
}

fn is_weight(role: i32) -> bool {
    role == pwire::PayloadRole::Weights as i32 || role == pwire::PayloadRole::TunedWeights as i32
}

pub(crate) fn numeric_ranges(
    request: &CaptureRequest,
    allocations: &[super::Allocation],
) -> Result<Vec<(u64, u64)>> {
    if request.options.reuse_weights.is_none() {
        return Ok(Vec::new());
    }
    let declaration: weight_payload::WeightPayloadDeclaration = serde_json::from_value(
        request
            .weight_payload
            .clone()
            .ok_or_else(|| invalid("--reuse-weights requires semantic weight declarations"))?,
    )?;
    let owners = super::capture::weight_allocations(allocations);
    let addresses: BTreeMap<_, _> = owners
        .iter()
        .map(|owner| (owner.name.as_str(), owner.address))
        .collect();
    let catalog = weight_payload::extract(
        &declaration.into_request(owners.clone(), request.zero_ranges.clone()),
    )?;
    let mut ranges = Vec::new();
    for group in &catalog.groups {
        for span in &group.spans {
            let start = addresses[span.allocation.as_str()] + span.destination_offset;
            let end = start + span.nbytes;
            for &(numeric_start, numeric_end) in &request.numeric {
                let begin = start.max(numeric_start);
                let stop = end.min(numeric_end);
                if begin < stop {
                    ranges.push((begin, stop));
                }
            }
        }
    }
    Ok(super::reachability::merge_ranges(ranges))
}

impl Source {
    pub(crate) fn engram(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
        data: &serde_json::Value,
    ) -> Result<pwire::Engram> {
        let verified = self.deployment(request, output)?;
        let previous = verified
            .deployment()
            .and_then(|body| body.engram.as_ref())
            .ok_or_else(|| invalid("--reuse-weights requires cached Engram embedding tables"))?;
        let mut metadata = data.clone();
        for table in metadata["tables"]
            .as_array_mut()
            .ok_or_else(|| invalid("Engram tables must be an array"))?
        {
            plane_lengths(table)?;
            if let Some(shards) = table
                .get_mut("shards")
                .and_then(|value| value.as_array_mut())
            {
                for shard in shards {
                    plane_lengths(shard)?;
                }
            }
        }
        let current: super::wire::Engram = super::binding::message("Engram", &metadata)?;
        rebind_engram(previous, &current)
    }

    pub(crate) fn groups(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
        catalog: &weight_payload::WeightPayloadCatalog,
        views: &WeightPayloadSources,
    ) -> Result<Vec<WeightPayloadGroup>> {
        let verified = self.deployment(request, output)?;
        map_groups(
            verified
                .rank(request.rank as u32)
                .ok_or_else(|| invalid("cached deployment lacks the preparation rank"))?,
            verified.catalog(),
            output,
            catalog,
            views,
        )
    }

    fn deployment(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
    ) -> Result<protection::VerifiedPackage> {
        let expected: super::wire::RebuildManifest = super::binding::message(
            "RebuildManifest",
            self.rebuild_manifest
                .as_ref()
                .ok_or_else(|| invalid("--reuse-weights requires preparation input metadata"))?,
        )?;
        let shared = protection::verify(
            &self.envelope,
            &self.publishers,
            protection::Selection::Catalog,
        )?;
        for deployment in 0..shared.deployment_count() {
            let body = shared.deployment_body(deployment)?;
            if body.tensor_parallel as usize != request.world_size
                || body.architecture != output.index.ranks[0].architecture
            {
                continue;
            }
            let previous = body
                .rebuild
                .as_ref()
                .ok_or_else(|| invalid("cached deployment has no preparation input metadata"))?;
            if previous.recipe != expected.recipe
                || previous.checkpoint_repository != expected.checkpoint_repository
                || previous.checkpoint_revision != expected.checkpoint_revision
                || previous.local_checkpoint != expected.local_checkpoint
                || previous.tensor_parallel != expected.tensor_parallel
            {
                return Err(invalid(
                    "--reuse-weights requires the same recipe, checkpoint revision, and tensor-parallel size",
                ));
            }
            let scope = |value: &str| if value.is_empty() { "all".to_owned() } else { value.to_owned() };
            if previous.online_mxfp8 != expected.online_mxfp8
                || scope(&previous.online_mxfp8_scope) != scope(&expected.online_mxfp8_scope)
            {
                return Err(invalid("--reuse-weights requires the same online MXFP8 conversion scope"));
            }
            return Ok(protection::verify(
                &self.envelope,
                &self.publishers,
                protection::Selection::Deployment {
                    deployment,
                    rank: Some(request.rank as u32),
                },
            )?);
        }
        Err(invalid(
            "--reuse-weights has no cached deployment with the requested architecture and tensor-parallel size",
        ))
    }
}

pub(crate) fn engram_uses(engram: &pwire::Engram) -> Vec<&pwire::PayloadUse> {
    engram
        .tables
        .iter()
        .flat_map(|table| {
            table.weights.iter().chain(table.scales.iter()).chain(
                table
                    .shards
                    .iter()
                    .flat_map(|shard| shard.weights.iter().chain(shard.scales.iter())),
            )
        })
        .collect()
}

fn plane_lengths(value: &mut serde_json::Value) -> Result<()> {
    for plane in ["weights", "scales"] {
        if let Some(parts) = value.get_mut(plane).filter(|parts| !parts.is_null()) {
            let sources: Vec<super::FileSource> = serde_json::from_value(parts.clone())?;
            let nbytes = sources.iter().try_fold(0u64, |sum, source| {
                sum.checked_add(source.nbytes)
                    .ok_or_else(|| invalid("Engram plane length overflow"))
            })?;
            *parts = serde_json::json!({"offset":0,"nbytes":nbytes});
        }
    }
    Ok(())
}

fn rebind_engram(previous: &pwire::Engram, current: &super::wire::Engram) -> Result<pwire::Engram> {
    if previous.token_map != current.token_map
        || previous.pad_token_id != current.pad_token_id
        || previous.compressed_vocabulary != current.compressed_vocabulary
        || previous.tables.len() != current.tables.len()
    {
        return Err(invalid(
            "Engram vocabulary or table catalog differs from the cached weights",
        ));
    }
    let plane = |old: &Option<pwire::PayloadUse>,
                 new: &Option<super::wire::Span>|
     -> Result<Option<pwire::PayloadUse>> {
        match (old, new) {
            (None, None) => Ok(None),
            (Some(old), Some(new))
                if old
                    .slice
                    .as_ref()
                    .is_some_and(|slice| slice.nbytes == new.nbytes) =>
            {
                Ok(Some(old.clone()))
            }
            _ => Err(invalid(
                "Engram weight or scale plane differs from the cached layout",
            )),
        }
    };
    let mut result = previous.clone();
    for (old, new) in result.tables.iter_mut().zip(&current.tables) {
        let hash_matches = match (&old.hash, &new.hash) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.method == b.method
                    && a.orders == b.orders
                    && a.multipliers == b.multipliers
                    && a.primes == b.primes
                    && a.offsets == b.offsets
            }
            _ => false,
        };
        if old.layer != new.layer
            || old.rows != new.rows
            || old.embedding_width != new.embedding_width
            || old.scale_block != new.scale_block
            || old.encoding != new.encoding
            || old.global_scale.to_bits() != new.global_scale.to_bits()
            || !hash_matches
            || old.shards.len() != new.shards.len()
        {
            return Err(invalid(
                "Engram embedding table geometry, encoding, or hashing differs from the cached weights",
            ));
        }
        old.weights = plane(&old.weights, &new.weights)?;
        old.scales = plane(&old.scales, &new.scales)?;
        for (old, new) in old.shards.iter_mut().zip(&new.shards) {
            if old.rank != new.rank || old.first_row != new.first_row || old.rows != new.rows {
                return Err(invalid(
                    "Engram embedding shard partition differs from the cached weights",
                ));
            }
            old.weights = plane(&old.weights, &new.weights)?;
            old.scales = plane(&old.scales, &new.scales)?;
        }
        old.weight_buffer = new.weight_buffer.clone();
        old.scale_buffer = new.scale_buffer.clone();
        old.hash_buffer = new.hash_buffer.clone();
        old.ready_event = new.ready_event.clone();
        old.request_buffer = new.request_buffer.clone();
    }
    Ok(result)
}

fn map_groups(
    previous: &pwire::RankBody,
    payloads: &helios_core::package::schema::Catalog,
    output: &RankOutput,
    catalog: &weight_payload::WeightPayloadCatalog,
    views: &WeightPayloadSources,
) -> Result<Vec<WeightPayloadGroup>> {
    let bundle = &output.index.ranks[0];
    let rebuild = bundle
        .rebuild_weights
        .as_ref()
        .ok_or_else(|| invalid("--reuse-weights requires declared weight views"))?;
    let current = super::package::convert_prepared(rebuild, bundle, Some(views))?;
    let prior = helios_core::package::schema::prepared_weights(previous)?.ok_or_else(|| {
        invalid("cached rank has no prepared weight catalog; export with weights first")
    })?;
    let old_views: BTreeMap<_, _> = prior
        .views
        .iter()
        .map(|view| (view.name.as_str(), view))
        .collect();
    let declarations: BTreeMap<_, _> = views.views.iter().map(|view| (view.name.as_str(), view)).collect();
    let aliases: BTreeMap<_, _> = catalog.aliases.iter().map(|alias| (alias.name.as_str(), alias.of.as_str())).collect();
    let mut placements = Vec::new();
    for view in &current.views {
        let old = old_views.get(view.name.as_str()).ok_or_else(|| {
            invalid(format!("cached weights lack view {}; export with weights", view.name))
        })?;
        let mut normalized = (*old).clone();
        normalized.allocation = view.allocation;
        normalized.offset = view.offset;
        if normalized != *view {
            return Err(invalid(format!(
                "weight view {} has a different layout or checkpoint mapping; export without --reuse-weights",
                view.name,
            )));
        }
        let prior_allocation = previous.allocations.get(old.allocation as usize)
            .ok_or_else(|| invalid("cached weight allocation is absent"))?;
        let allocation = &bundle.allocations[view.allocation as usize];
        let declaration = declarations.get(view.name.as_str()).copied();
        let owner = match aliases.get(view.name.as_str()).copied() {
            Some(name) => Some(*declarations.get(name).ok_or_else(|| invalid("weight alias owner is absent"))?),
            None => declaration,
        };
        let (owner_name, group, storage) = if let Some(owner) = owner {
            let storage = match owner.storage.as_str() {
                "stable" => weight_payload::StorageClass::Stable,
                "tuned" => weight_payload::StorageClass::Tuned,
                _ => return Err(invalid("weight view cannot declare state storage")),
            };
            let group = if storage == weight_payload::StorageClass::Tuned {
                format!("{}:tuned", owner.group)
            } else {
                owner.group.clone()
            };
            (owner.name.clone(), group, storage)
        } else {
            // Older prepared catalogs may name views without a semantic owner.
            // Such views still undergo exact layout matching above; a weight
            // group must identify their storage class before cached bytes move.
            let matches: Vec<_> = catalog.groups.iter().filter(|group|
                !view.group.is_empty() && (group.name == view.group || group.name == format!("{}:tuned", view.group)))
                .collect();
            if matches.is_empty() && view.group.is_empty() {
                continue;
            }
            if matches.len() != 1 {
                return Err(invalid(format!("weight view {} has no unique declared weight group", view.name)));
            }
            (view.name.clone(), matches[0].name.clone(), matches[0].storage)
        };
        let geometry = weight_payload::WeightViewInput {
            name: view.name.clone(), group: group.clone(), storage, alias_of: None,
            sources: Vec::new(), packing: String::new(), packing_parameters: None,
            statistics_domain: String::new(), element_bytes: u64::from(view.element_bytes),
            shape: view.shape.clone(), stride: view.stride.clone(), address: 0,
        };
        for (start, end) in weight_payload::occupied_runs(&geometry)? {
            let prior_start = old.offset.checked_add(start).ok_or_else(|| invalid("cached weight offset overflow"))?;
            let prior_end = old.offset.checked_add(end).ok_or_else(|| invalid("cached weight extent overflow"))?;
            let destination_start = view.offset.checked_add(start).ok_or_else(|| invalid("weight offset overflow"))?;
            let destination_end = view.offset.checked_add(end).ok_or_else(|| invalid("weight extent overflow"))?;
            if prior_end > prior_allocation.nbytes || destination_end > allocation.nbytes {
                return Err(invalid("retained weight view exceeds its allocation"));
            }
            placements.push(ViewPlacement {
                owner: owner_name.clone(), group: group.clone(), storage,
                allocation: allocation.name.clone(), start: destination_start, end: destination_end,
                prior_allocation: prior_allocation.name.clone(), prior_start,
            });
        }
    }
    let mut result = map_weight_spans(previous, placements, catalog)?;
    for group in &mut result {
        group.nbytes = payloads.length(&group.payload_id)?;
    }
    Ok(result)
}

#[derive(Clone, Debug)]
struct ViewPlacement {
    owner: String,
    group: String,
    storage: weight_payload::StorageClass,
    allocation: String,
    start: u64,
    end: u64,
    prior_allocation: String,
    prior_start: u64,
}

fn normalize_placements(mut placements: Vec<ViewPlacement>) -> Result<Vec<ViewPlacement>> {
    use super::reachability::{merge_ranges, subtract_ranges};
    // Any aliases of the current bytes must refer to the same cached bytes.
    // Comparing only payload contents would require reading the weight files.
    placements.sort_by(|a, b| (&a.allocation, a.start, a.end).cmp(&(&b.allocation, b.start, b.end)));
    let mut checked: Vec<ViewPlacement> = Vec::new();
    for placement in &placements {
        if let Some(last) = checked.last_mut() {
            if last.allocation == placement.allocation && placement.start < last.end {
                if last.prior_allocation != placement.prior_allocation
                    || i128::from(last.prior_start) - i128::from(last.start)
                        != i128::from(placement.prior_start) - i128::from(placement.start)
                {
                    return Err(invalid("overlapping retained weights require different cached bytes"));
                }
                last.end = last.end.max(placement.end);
                continue;
            }
        }
        checked.push(placement.clone());
    }
    // Match weight_payload::extract's named owner order. Aliases use their
    // declared owner's name and therefore cannot take ownership from it.
    placements.sort_by(|a, b| (&a.owner, &a.allocation, a.start).cmp(&(&b.owner, &b.allocation, b.start)));
    let mut claimed = BTreeMap::<String, Vec<(u64, u64)>>::new();
    let mut result = Vec::new();
    for placement in placements {
        let ranges = claimed.entry(placement.allocation.clone()).or_default();
        let pieces = subtract_ranges(placement.start, placement.end, ranges);
        for (start, end) in pieces {
            result.push(ViewPlacement {
                prior_start: placement.prior_start + (start - placement.start),
                start,
                end,
                ..placement.clone()
            });
        }
        ranges.push((placement.start, placement.end));
        *ranges = merge_ranges(std::mem::take(ranges));
    }
    Ok(result)
}

fn map_weight_spans(
    previous: &pwire::RankBody,
    placements: Vec<ViewPlacement>,
    catalog: &weight_payload::WeightPayloadCatalog,
) -> Result<Vec<WeightPayloadGroup>> {
    use super::reachability::merge_ranges;
    use weight_payload::StorageClass;
    let placements = normalize_placements(placements)?;
    type Coverage = BTreeMap<(String, StorageClass, String), Vec<(u64, u64)>>;
    let mut expected = Coverage::new();
    for placement in &placements {
        expected.entry((placement.group.clone(), placement.storage, placement.allocation.clone()))
            .or_default().push((placement.start, placement.end));
    }
    let allocations: BTreeSet<_> = placements.iter().map(|placement| placement.allocation.as_str()).collect();
    let mut actual = Coverage::new();
    for group in &catalog.groups {
        for span in &group.spans {
            if !allocations.contains(span.allocation.as_str()) {
                return Err(invalid(format!(
                    "current weight allocation {} has no matching declared views", span.allocation,
                )));
            }
            let end = span.destination_offset.checked_add(span.nbytes)
                .ok_or_else(|| invalid("current weight extent overflow"))?;
            actual.entry((group.name.clone(), group.storage, span.allocation.clone()))
                .or_default().push((span.destination_offset, end));
        }
    }
    for coverage in [&mut expected, &mut actual] {
        for ranges in coverage.values_mut() {
            *ranges = merge_ranges(std::mem::take(ranges));
        }
    }
    if expected != actual {
        return Err(invalid("retained weight bytes are not covered by one declared weight group"));
    }
    let mut cached = BTreeMap::<&str, Vec<(u64, u64, i32, &pwire::PayloadSlice)>>::new();
    for initializer in &previous.initializers {
        let Some(source) = initializer.source.as_ref().filter(|source| is_weight(source.role)) else {
            continue;
        };
        let slice = source.slice.as_ref().ok_or_else(|| invalid("cached weight initializer has no slice"))?;
        let end = initializer.destination_offset.checked_add(slice.nbytes)
            .ok_or_else(|| invalid("cached weight extent overflow"))?;
        cached.entry(&initializer.allocation).or_default()
            .push((initializer.destination_offset, end, source.role, slice));
    }
    for spans in cached.values_mut() {
        spans.sort_by_key(|span| span.0);
    }
    let mut result = BTreeMap::<String, WeightPayloadGroup>::new();
    for placement in placements {
        let prior_end = placement.prior_start.checked_add(placement.end - placement.start)
            .ok_or_else(|| invalid("cached weight view extent overflow"))?;
        let Some(spans) = cached.get(placement.prior_allocation.as_str()) else {
            // Signed cached allocations are initially zero; missing payload
            // spans inside a matched view are the zero-elided bytes.
            continue;
        };
        let first = spans.partition_point(|span| span.1 <= placement.prior_start);
        for &(start, end, role, slice) in &spans[first..] {
            if start >= prior_end {
                break;
            }
            let low = start.max(placement.prior_start);
            let high = end.min(prior_end);
            let storage = if role == pwire::PayloadRole::TunedWeights as i32 {
                StorageClass::Tuned
            } else {
                StorageClass::Stable
            };
            if storage != placement.storage {
                return Err(invalid("cached weight storage class differs from its retained view"));
            }
            let group = result.entry(placement.group.clone()).or_insert_with(|| WeightPayloadGroup {
                name: placement.group.clone(), storage: storage.as_str().into(), nbytes: 0,
                spans: Vec::new(), payload_id: slice.payload_id.clone(),
            });
            if group.payload_id != slice.payload_id {
                return Err(invalid("cached weight group uses multiple payload objects; export with weights"));
            }
            group.spans.push(WeightPayloadSourceSpan {
                allocation: placement.allocation.clone(),
                destination_offset: placement.start + (low - placement.prior_start),
                source_offset: slice.offset.checked_add(low - start)
                    .ok_or_else(|| invalid("cached payload offset overflow"))?,
                nbytes: high - low,
            });
        }
    }
    for group in result.values_mut() {
        group.spans.sort_by(|a, b| (&a.allocation, a.destination_offset).cmp(&(&b.allocation, b.destination_offset)));
        let mut merged: Vec<WeightPayloadSourceSpan> = Vec::new();
        for span in std::mem::take(&mut group.spans) {
            if let Some(last) = merged.last_mut() {
                if last.allocation == span.allocation
                    && last.destination_offset + last.nbytes == span.destination_offset
                    && last.source_offset + last.nbytes == span.source_offset
                {
                    last.nbytes += span.nbytes;
                    continue;
                }
            }
            merged.push(span);
        }
        group.spans = merged;
    }
    Ok(result.into_values().collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_reuse_weights, module)?)?;
    module.add_class::<NativeWeightSource>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use weight_payload::{PayloadSpan, StorageClass, WeightGroupPayload, WeightPayloadCatalog};

    fn placement(allocation: &str, start: u64, end: u64, prior_start: u64) -> ViewPlacement {
        ViewPlacement {
            owner: format!("owner-{allocation}-{start}"), group: "layer".into(),
            storage: StorageClass::Stable, allocation: allocation.into(), start, end,
            prior_allocation: "cached".into(), prior_start,
        }
    }

    fn previous(spans: &[(u64, u64, u64)]) -> pwire::RankBody {
        pwire::RankBody {
            initializers: spans.iter().map(|&(destination_offset, offset, nbytes)| pwire::Initializer {
                allocation: "cached".into(), destination_offset,
                source: Some(pwire::PayloadUse {
                    role: pwire::PayloadRole::Weights as i32,
                    slice: Some(pwire::PayloadSlice { payload_id: vec![9; 32], offset, nbytes }),
                    ..Default::default()
                }),
            }).collect(),
            ..Default::default()
        }
    }

    fn catalog(placements: &[ViewPlacement]) -> WeightPayloadCatalog {
        WeightPayloadCatalog {
            groups: vec![WeightGroupPayload {
                name: "layer".into(), storage: StorageClass::Stable,
                nbytes: placements.iter().map(|p| p.end-p.start).sum(),
                spans: placements.iter().map(|p| PayloadSpan {
                    payload_offset: 0, allocation: p.allocation.clone(),
                    destination_offset: p.start, nbytes: p.end-p.start,
                }).collect(),
            }],
            state: WeightGroupPayload {
                name: "initial-state".into(), storage: StorageClass::State, nbytes: 0, spans: vec![],
            },
            aliases: vec![],
        }
    }

    #[test]
    fn split_cached_initializer_across_relocated_views() -> Result<()> {
        let placements = vec![placement("left", 16, 20, 0), placement("right", 32, 36, 4)];
        let result = map_weight_spans(&previous(&[(0, 0, 8)]), placements.clone(), &catalog(&placements))?;
        assert_eq!(result.len(), 1);
        let spans = &result[0].spans;
        assert_eq!(spans.len(), 2);
        assert_eq!((&spans[0].allocation, spans[0].destination_offset, spans[0].source_offset, spans[0].nbytes),
                   (&"left".to_string(), 16, 0, 4));
        assert_eq!((&spans[1].allocation, spans[1].destination_offset, spans[1].source_offset, spans[1].nbytes),
                   (&"right".to_string(), 32, 4, 4));
        Ok(())
    }

    #[test]
    fn strided_view_reuse_excludes_holes_and_preserves_zero_elision() -> Result<()> {
        let view = weight_payload::WeightViewInput {
            name: "strided".into(), group: "layer".into(), storage: StorageClass::Stable,
            alias_of: None, sources: vec![], packing: String::new(), packing_parameters: None,
            statistics_domain: String::new(), element_bytes: 1,
            shape: vec![2, 2], stride: vec![3, 1], address: 0,
        };
        let runs = weight_payload::occupied_runs(&view)?;
        assert_eq!(runs, vec![(0, 2), (3, 5)]);
        let placements: Vec<_> = runs.into_iter().map(|(start, end)| placement("target", 16+start, 16+end, start)).collect();
        // The second occupied row is zero-elided; the intervening byte belongs
        // to no declared weight and must not be read or initialized by reuse.
        let result = map_weight_spans(&previous(&[(0, 0, 2)]), placements.clone(), &catalog(&placements))?;
        assert_eq!(result[0].spans.len(), 1);
        assert_eq!((result[0].spans[0].destination_offset, result[0].spans[0].nbytes), (16, 2));
        Ok(())
    }

    #[test]
    fn identical_aliases_deduplicate_and_conflicting_aliases_fail() -> Result<()> {
        let owner = placement("target", 16, 24, 0);
        let mut alias = owner.clone();
        alias.owner = "alias".into();
        let coverage = catalog(&[owner.clone()]);
        let result = map_weight_spans(&previous(&[(0, 0, 16)]), vec![owner.clone(), alias.clone()], &coverage)?;
        assert_eq!(result[0].spans.len(), 1);
        assert_eq!(result[0].spans[0].nbytes, 8);
        alias.prior_start = 8;
        let error = map_weight_spans(&previous(&[(0, 0, 16)]), vec![owner, alias], &coverage).unwrap_err();
        assert!(error.to_string().contains("different cached bytes"));
        Ok(())
    }

    #[test]
    fn relocated_view_rejects_incomplete_current_coverage() {
        let placements = vec![placement("target", 16, 24, 0)];
        let mut coverage = catalog(&placements);
        coverage.groups[0].spans[0].nbytes -= 1;
        let error = map_weight_spans(&previous(&[(0, 0, 8)]), placements, &coverage).unwrap_err();
        assert!(error.to_string().contains("not covered by one declared weight group"));
    }
}
