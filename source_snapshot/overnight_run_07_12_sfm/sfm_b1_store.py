"""Round-sharded B1 D/D+/D- store and exact hierarchical replay."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
import random

import numpy as np
import torch


class RoundShard:
    """One macro-round. Context payloads are stored once; queries reference context IDs."""

    VERSION = 2

    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.queries = []
        self.errors = []
        self._keys = set()

    def add_context(self, *, scenario_id, gamma, step, state, hp10, low5, hist,
                    ped_xy, ped_vel, mode=None):
        key = (int(scenario_id), round(float(gamma), 8), int(step))
        if key in self._keys:
            raise ValueError(f"context already stored: {key}")
        self._keys.add(key)
        context_id = len(self.contexts)
        self.contexts.append(dict(
            context_id=context_id, round=self.round_i, scenario_id=int(scenario_id),
            gamma=float(gamma), step=int(step), state=np.asarray(state, np.float32),
            hp10=np.asarray(hp10, np.float32), low5=np.asarray(low5, np.float32),
            hist=np.asarray(hist, np.float32), ped_xy=np.asarray(ped_xy, np.float32),
            ped_vel=np.asarray(ped_vel, np.float32), mode=mode,
        ))
        return context_id

    def add_resolved_query(self, context_id, candidate_id, controls, sigma, result,
                           *, acquisition_step, executed=False, hp_margin=None,
                           expert_cost=None, mode=None, gp_base_sigma=None):
        if not bool(result.get("resolved")):
            raise ValueError("unresolved/SOCP-error query cannot enter D")
        if int(result.get("y", -1)) not in (0, 1):
            raise ValueError("resolved query needs binary y")
        if not bool(result.get("full_h")) or int(result.get("terminal_step", -1)) != 10:
            raise ValueError("B1 query store accepts only full H=10 verifier results")
        context_id = int(context_id)
        if not 0 <= context_id < len(self.contexts):
            raise IndexError("unknown context")
        query_id = len(self.queries)
        row = dict(
            query_id=query_id, context_id=context_id, candidate_id=int(candidate_id),
            acquisition_step=int(acquisition_step), controls=np.asarray(controls, np.float32),
            sigma=float(sigma),
            gp_base_sigma=None if gp_base_sigma is None else float(gp_base_sigma),
            gp_retained=False,
            y=int(result["y"]), taskspace=bool(result["taskspace"]),
            collision_free=bool(result["collision_free"]), certificate=bool(result["certificate"]),
            full_h=bool(result["full_h"]), terminal_step=int(result["terminal_step"]),
            train_eligible=bool(result["train_eligible"]), executed=bool(executed),
            hp_margin=None if hp_margin is None else float(hp_margin),
            expert_cost=None if expert_cost is None else float(expert_cost), mode=mode,
            segment=np.asarray(result["segment"], np.float32),
            pedestrian_prediction=np.asarray(result["pedestrian_prediction"], np.float32),
            verifier_diagnostics=dict(result["diagnostics"]),
        )
        # Every resolved query is full-H, so D is exactly partitioned by y.
        row["train_eligible"] = bool(row["y"] == 1)
        self.queries.append(row)
        return query_id

    def add_error(self, *, context_key, candidate_id, error):
        self.errors.append(dict(context_key=tuple(context_key), candidate_id=int(candidate_id), error=str(error)))

    def mark_executed(self, query_id, *, hp_margin, expert_cost=None):
        row = self.queries[int(query_id)]
        row["executed"] = True
        row["hp_margin"] = float(hp_margin)
        if expert_cost is not None:
            row["expert_cost"] = float(expert_cost)

    @property
    def D(self):
        return list(self.queries)

    @property
    def Dplus(self):
        return [row for row in self.queries if row["y"] == 1 and row["full_h"]]

    @property
    def Dminus(self):
        return [row for row in self.queries if row["y"] == 0]

    def validate(self):
        for expected, context in enumerate(self.contexts):
            if context["context_id"] != expected:
                raise AssertionError("context IDs are not dense")
        for expected, query in enumerate(self.queries):
            if query["query_id"] != expected:
                raise AssertionError("query IDs are not dense")
            if not 0 <= query["context_id"] < len(self.contexts):
                raise AssertionError("query references missing context")
            if query["train_eligible"] and not (query["y"] == 1 and query["full_h"]):
                raise AssertionError("invalid D+ eligibility")
            if not bool(query["full_h"]) or int(query["terminal_step"]) != 10:
                raise AssertionError("stored B1 query is not full H=10")
            if bool(query["train_eligible"]) != bool(query["y"] == 1):
                raise AssertionError("full-H query eligibility must equal y")
            base_sigma = query.get("gp_base_sigma")
            if base_sigma is not None and (
                    not math.isfinite(float(base_sigma)) or float(base_sigma) < 0.0):
                raise AssertionError("invalid GP base sigma")
            if query.get("gp_retained", False) and not (
                    query["y"] == 1 and query["full_h"] and base_sigma is not None):
                raise AssertionError("GP memory may retain only scored full-H D+")
        if len(self.Dplus) + len(self.Dminus) != len(self.D):
            raise AssertionError("D must be partitioned exactly into D+ and D-")
        gp_retained = sum(bool(query.get("gp_retained", False)) for query in self.queries)
        if gp_retained > 256:
            raise AssertionError("one round may retain at most 256 GP records")
        return dict(
            round=self.round_i, contexts=len(self.contexts), D=len(self.D),
            Dplus=len(self.Dplus), Dminus=len(self.Dminus), errors=len(self.errors),
            gp_retained=gp_retained,
        )

    def save(self, path):
        path = os.fspath(path)
        summary = self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(dict(
            version=self.VERSION, round=self.round_i, contexts=self.contexts,
            queries=self.queries, errors=self.errors, summary=summary,
        ), temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        complete = path + ".COMPLETE.json"
        with open(complete + ".tmp", "w") as stream:
            json.dump(dict(status="COMPLETE", file=os.path.abspath(path), sha256=digest, **summary), stream, indent=2)
        os.replace(complete + ".tmp", complete)
        return dict(path=os.path.abspath(path), sha256=digest, complete=os.path.abspath(complete), **summary)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["version"]) not in (1, cls.VERSION):
            raise ValueError("unsupported round-shard version")
        value = cls(payload["round"])
        value.contexts = payload["contexts"]
        value.queries = payload["queries"]
        for query in value.queries:
            query.setdefault("gp_base_sigma", None)
            query.setdefault("gp_retained", False)
        value.errors = payload["errors"]
        value._keys = {(row["scenario_id"], round(row["gamma"], 8), row["step"]) for row in value.contexts}
        value.validate()
        return value


class RecentRounds:
    def __init__(self, directory, window=2):
        if int(window) != 2:
            raise ValueError("the frozen B1 study requires W=2")
        self.directory = os.path.abspath(directory)
        self.window = 2
        self._rounds = []

    def path(self, round_i):
        return os.path.join(self.directory, f"round_{int(round_i):02d}.pt")

    def append_and_save(self, shard):
        manifest = shard.save(self.path(shard.round_i))
        self._rounds.append(shard)
        self._rounds = self._rounds[-self.window:]
        return manifest

    def load_through(self, round_i):
        start = max(1, int(round_i) - self.window + 1)
        self._rounds = [RoundShard.load(self.path(index)) for index in range(start, int(round_i) + 1)]

    @property
    def rounds(self):
        return list(self._rounds)

    def positive_records(self):
        return [(shard, row) for shard in self._rounds for row in shard.Dplus]

    def negative_records(self):
        return [(shard, row) for shard in self._rounds for row in shard.Dminus]

    def gp_records(self):
        records = [
            (shard, row) for shard in self._rounds for row in shard.Dplus
            if bool(row.get("gp_retained", False))
        ]
        if len(records) > 512:
            raise AssertionError("two-round GP memory exceeds cap 512")
        per_round = defaultdict(int)
        for shard, _ in records:
            per_round[int(shard.round_i)] += 1
        if any(count > 256 for count in per_round.values()):
            raise AssertionError("a GP-memory round exceeds quota 256")
        return records


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_expansion_trainability(policy):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    for parameter in policy.enc_grid.parameters():
        parameter.requires_grad_(False)
    frozen = {name for name, parameter in policy.named_parameters() if not parameter.requires_grad}
    expected = {f"enc_grid.{name}" for name, _ in policy.enc_grid.named_parameters()}
    if frozen != expected:
        raise RuntimeError(f"unexpected frozen parameters: {sorted(frozen ^ expected)}")
    return sorted(frozen)


def _record_context(shard, query):
    return shard.contexts[int(query["context_id"])]


def hierarchy_mass(records):
    """Equal mass gamma -> (round,scenario) -> context -> query."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for shard, query in records:
        context = _record_context(shard, query)
        gamma = round(float(context["gamma"]), 8)
        cell = (int(context["round"]), int(context["scenario_id"]))
        context_key = (int(context["round"]), int(context["context_id"]))
        grouped[gamma][cell][context_key].append((shard, query))
    mass = {}
    n_gamma = len(grouped)
    if not n_gamma:
        return mass, dict(total=0.0, gamma={})
    for gamma, cells in grouped.items():
        for cell, contexts in cells.items():
            for context_key, values in contexts.items():
                value = 1.0 / (n_gamma * len(cells) * len(contexts) * len(values))
                for shard, query in values:
                    mass[(id(shard), int(query["query_id"]))] = value
    diagnostics = mass_accounting(records, mass)
    return mass, diagnostics


def mass_accounting(records, mass):
    gamma = defaultdict(float)
    cell = defaultdict(float)
    context_mass = defaultdict(float)
    total = 0.0
    for shard, query in records:
        context = _record_context(shard, query)
        value = float(mass[(id(shard), int(query["query_id"]))])
        total += value
        gamma[str(context["gamma"])] += value
        cell[f"r{context['round']}:s{context['scenario_id']}:g{context['gamma']}"] += value
        context_mass[f"r{context['round']}:c{context['context_id']}"] += value
    return dict(total=total, gamma=dict(gamma), cells=dict(cell), contexts=dict(context_mass))


def hierarchical_order(records, seed):
    """Deterministic stratified interleave; each record occurs exactly once."""
    groups = defaultdict(list)
    for shard, query in records:
        context = _record_context(shard, query)
        key = (round(float(context["gamma"]), 8), int(context["round"]),
               int(context["scenario_id"]), int(context["context_id"]))
        groups[key].append((shard, query))
    rng = random.Random(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered = []
    while keys:
        next_keys = []
        for key in keys:
            ordered.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    identities = [(shard.round_i, int(query["query_id"])) for shard, query in ordered]
    if len(identities) != len(set(identities)) or len(identities) != len(records):
        raise RuntimeError("hierarchical replay duplicate/omission")
    return ordered


def gp_round_quotas(round_i, gammas, total=256):
    """Rotate the unavoidable 256 mod 7 remainder without cross-gamma borrowing."""
    keys = sorted({round(float(gamma), 8) for gamma in gammas})
    if not keys:
        raise ValueError("GP retention requires at least one gamma")
    if int(round_i) < 1 or int(total) < 1:
        raise ValueError("invalid GP-retention round or total")
    base, remainder = divmod(int(total), len(keys))
    cursor = ((int(round_i) - 1) * remainder) % len(keys)
    extra = {keys[(cursor + offset) % len(keys)] for offset in range(remainder)}
    quotas = {gamma: base + int(gamma in extra) for gamma in keys}
    if sum(quotas.values()) != int(total) or max(quotas.values()) - min(quotas.values()) > 1:
        raise AssertionError("invalid rotating gamma quotas")
    return quotas


def retain_gp_upper_quartile(shard, gammas, *, seed, total=256):
    """Mark a gamma-balanced active GP set from this round's full-H positives."""
    if int(shard.round_i) < 1:
        raise ValueError("GP retention requires a completed positive round")
    for query in shard.queries:
        query["gp_retained"] = False
    quotas = gp_round_quotas(shard.round_i, gammas, total=total)
    diagnostics = {}
    retained_ids = []
    for gamma_index, (gamma, requested) in enumerate(quotas.items()):
        records = [
            (shard, query) for query in shard.Dplus
            if round(float(_record_context(shard, query)["gamma"]), 8) == gamma
        ]
        missing = [query["query_id"] for _, query in records
                   if query.get("gp_base_sigma") is None]
        if missing:
            raise RuntimeError(
                f"full-H D+ lacks pre-pending GP sigma at gamma={gamma}: {missing[:8]}"
            )
        scores = np.asarray([query["gp_base_sigma"] for _, query in records], dtype=float)
        q75 = None if not len(scores) else float(np.quantile(scores, 0.75))
        eligible = [] if q75 is None else [
            record for record in records if float(record[1]["gp_base_sigma"]) >= q75
        ]
        # Sampling within the upper quartile remains context-stratified and
        # deterministic; uncertainty defines eligibility rather than a new max rule.
        ordered = hierarchical_order(eligible, int(seed) + gamma_index * 1009)
        retained = ordered[:int(requested)]
        for _, query in retained:
            query["gp_retained"] = True
            retained_ids.append(int(query["query_id"]))
        diagnostics[str(float(gamma))] = dict(
            requested=int(requested), positives=len(records), q75=q75,
            eligible=len(eligible), eligible_fraction=(
                None if not records else len(eligible) / len(records)
            ), retained=len(retained), shortfall=int(requested) - len(retained),
        )
    retained_total = len(retained_ids)
    if retained_total > int(total):
        raise AssertionError("GP retention exceeds its per-round cap")
    return dict(
        round=int(shard.round_i), requested=int(total), retained=retained_total,
        shortfall=int(total) - retained_total, per_gamma=diagnostics,
        retained_query_ids=retained_ids,
        selection="per-gamma q75 then deterministic context-stratified without replacement",
        quota_policy="rotating floor/ceil; no cross-gamma redistribution",
    )


def _batches(records, batch):
    return [records[start:start + int(batch)] for start in range(0, len(records), int(batch))]


def _tensor_batch(records, device):
    contexts = [_record_context(shard, query) for shard, query in records]
    grid = torch.as_tensor(np.stack([row["hp10"] for row in contexts]), device=device)
    low = torch.as_tensor(np.stack([row["low5"] for row in contexts]), device=device)
    hist = torch.as_tensor(np.stack([row["hist"] for row in contexts]), device=device)
    controls = torch.as_tensor(np.stack([query["controls"] for _, query in records]), device=device)
    return grid.float(), low.float(), hist.float(), controls.float()


def _query_design(policy, records, *, seed, stream, device):
    """Stateless CFM/dropout randomness keyed to each stored verifier query."""
    x0, tau, dropout = [], [], []
    keep = 1.0 - float(policy.res_dropout)
    if not 0.0 < keep <= 1.0:
        raise ValueError("invalid residual-dropout keep probability")
    for shard, query in records:
        identity = f"{int(seed)}:{stream}:{int(shard.round_i)}:{int(query['query_id'])}"
        query_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(query_seed)
        x0.append(torch.randn(policy.d, generator=generator))
        tau.append(torch.rand(1, generator=generator).clamp(1.0e-4, 1.0))
        mask = torch.rand(
            len(policy.trunk.blocks), policy.width, generator=generator,
        ) < keep
        dropout.append(mask.float() / keep)
    return (
        torch.stack(x0).to(device), torch.cat(tau).to(device), torch.stack(dropout).to(device),
    )


def _accumulate_objective(policy, records, mass, batch, device, *, seed, stream):
    policy.train()
    visited = []
    total_loss = 0.0
    for values in _batches(records, batch):
        grid, low, hist, controls = _tensor_batch(values, device)
        context = policy.ctx_from(grid, low, hist)
        per_weights = torch.as_tensor([
            len(values) * mass[(id(shard), int(query["query_id"]))] for shard, query in values
        ], dtype=controls.dtype, device=device)
        x0, tau, dropout = _query_design(
            policy, values, seed=seed, stream=stream, device=device,
        )
        loss = policy.cfm_loss_designed(
            controls, context, x0=x0, tau=tau,
            dropout_scale=dropout, weights=per_weights,
        )
        loss.backward()
        total_loss += float(loss.detach())
        visited.extend((shard.round_i, int(query["query_id"])) for shard, query in values)
    return total_loss, visited


def _gradient_snapshot(policy):
    return {name: (None if parameter.grad is None else parameter.grad.detach().clone())
            for name, parameter in policy.named_parameters() if parameter.requires_grad}


def _gradient_norm(snapshot):
    values = [value.to(torch.float64).square().sum() for value in snapshot.values() if value is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _step_partitions(records, requested_steps):
    if int(requested_steps) < 1:
        raise ValueError("optimizer_steps must be positive")
    if not records:
        return []
    count = min(int(requested_steps), len(records))
    return [records[index::count] for index in range(count)]


def _subset_global_mass(records, global_mass):
    """Keep the declared global hierarchy mass when splitting one epoch."""
    total = sum(global_mass[(id(shard), int(query["query_id"]))] for shard, query in records)
    if total <= 0.0:
        raise RuntimeError("optimizer-step subset has zero hierarchy mass")
    return {
        (id(shard), int(query["query_id"])):
            global_mass[(id(shard), int(query["query_id"]))]
        for shard, query in records
    }, float(total)


def _replay_shape(optimizer_steps, optimizer_chunks, inner_epochs):
    """Resolve the new 16-chunk epoch protocol while reading legacy callers."""
    if optimizer_steps is not None:
        if int(inner_epochs) != 1 or int(optimizer_chunks) != 16:
            raise ValueError("legacy optimizer_steps cannot be mixed with epoch knobs")
        chunks, epochs, protocol = int(optimizer_steps), 1, "legacy_partition_once"
    else:
        chunks, epochs, protocol = int(optimizer_chunks), int(inner_epochs), "fixed_chunks_epochs"
        if chunks != 16:
            raise ValueError("scientific replay requires exactly 16 optimizer chunks")
        if epochs not in (1, 4, 16):
            raise ValueError("inner_epochs must be one of {1,4,16}")
    if chunks < 1:
        raise ValueError("optimizer chunk count must be positive")
    return chunks, epochs, protocol


def _visit_diagnostics(visited, eligible):
    counts = defaultdict(int)
    for identity in visited:
        counts[identity] += 1
    values = list(counts.values())
    return dict(
        unique_replay_coverage=(0.0 if not eligible else len(counts) / int(eligible)),
        visits_per_eligible=(0.0 if not eligible else len(visited) / int(eligible)),
        visit_min=(None if not values else min(values)),
        visit_max=(None if not values else max(values)),
    )


def _positive_multi_step(policy, optimizer, ordered, global_mass, accounting, *,
                         optimizer_chunks, inner_epochs, batch, device, seed,
                         replay_protocol):
    chunks = _step_partitions(ordered, optimizer_chunks)
    if len(chunks) != int(optimizer_chunks):
        raise RuntimeError("eligible D+ is smaller than the declared optimizer chunk count")
    visited, total_loss, step_sizes, step_global_mass = [], 0.0, [], []
    for _ in range(int(inner_epochs)):
        for values in chunks:
            mass, original_mass = _subset_global_mass(values, global_mass)
            optimizer.zero_grad(set_to_none=True)
            loss, seen = _accumulate_objective(
                policy, values, mass, batch, device, seed=seed, stream="positive",
            )
            optimizer.step()
            total_loss += loss
            visited.extend(seen)
            step_sizes.append(len(values))
            step_global_mass.append(original_mass)
    for start in range(0, len(step_global_mass), len(chunks)):
        if abs(sum(step_global_mass[start:start + len(chunks)]) - 1.0) > 1.0e-9:
            raise RuntimeError("positive replay epoch does not preserve global hierarchy mass")
    visit = _visit_diagnostics(visited, len(ordered))
    return dict(
        path="positive_only", eligible=len(ordered), visited=visited, loss=total_loss,
        mass=accounting, optimizer_steps=len(step_sizes), optimizer_chunks=len(chunks),
        optimizer_steps_requested=len(step_sizes),
        inner_epochs=int(inner_epochs), replay_protocol=replay_protocol,
        step_sizes=step_sizes, chunk_sizes=list(map(len, chunks)),
        step_global_mass=step_global_mass,
        replay_coverage=visit["unique_replay_coverage"],
        replay_visits_per_eligible=visit["visits_per_eligible"],
        visit_min=visit["visit_min"], visit_max=visit["visit_max"],
        loss_step_mean=total_loss / len(step_sizes),
    )


def positive_only_update(policy, optimizer, recent, *, batch=128, device="cpu", seed=0,
                         optimizer_steps=None, optimizer_chunks=16, inner_epochs=1):
    chunks, epochs, protocol = _replay_shape(
        optimizer_steps, optimizer_chunks, inner_epochs,
    )
    records = [(shard, query) for shard, query in recent.positive_records() if query["train_eligible"]]
    ordered = hierarchical_order(records, seed)
    mass, accounting = hierarchy_mass(ordered)
    optimizer.zero_grad(set_to_none=True)
    if not ordered:
        return dict(path="positive_only", eligible=0, visited=[], loss=0.0, mass=accounting,
                    optimizer_steps=0, optimizer_chunks=chunks, inner_epochs=epochs,
                    replay_protocol=protocol, loss_step_mean=None)
    return _positive_multi_step(
        policy, optimizer, ordered, mass, accounting, optimizer_chunks=chunks,
        inner_epochs=epochs, batch=batch, device=device, seed=seed,
        replay_protocol=protocol,
    )


def _signed_multi_step(policy, optimizer, pos_order, neg_order, pos_mass, neg_mass,
                       pos_accounting, neg_accounting, *, alpha, optimizer_chunks,
                       inner_epochs, batch, device, eps, seed, replay_protocol):
    positive_chunks = _step_partitions(pos_order, optimizer_chunks)
    if len(positive_chunks) != int(optimizer_chunks):
        raise RuntimeError("eligible D+ is smaller than the declared optimizer chunk count")
    negative_chunks = ([neg_order[index::len(positive_chunks)] for index in range(len(positive_chunks))]
                       if neg_order else [[] for _ in positive_chunks])
    pos_visited, neg_visited = [], []
    positive_losses, negative_losses = [], []
    positive_norms, negative_norms, rhos = [], [], []
    pos_step_mass, neg_step_mass = [], []
    for _ in range(int(inner_epochs)):
        for positives, negatives in zip(positive_chunks, negative_chunks):
            local_pos_mass, original_pos_mass = _subset_global_mass(positives, pos_mass)
            optimizer.zero_grad(set_to_none=True)
            positive_loss, seen_positive = _accumulate_objective(
                policy, positives, local_pos_mass, batch, device, seed=seed, stream="positive",
            )
            positive_gradient = _gradient_snapshot(policy)
            positive_norm = _gradient_norm(positive_gradient)
            if negatives:
                local_neg_mass, original_neg_mass = _subset_global_mass(negatives, neg_mass)
                optimizer.zero_grad(set_to_none=True)
                negative_loss, seen_negative = _accumulate_objective(
                    policy, negatives, local_neg_mass, batch, device, seed=seed, stream="negative",
                )
                negative_gradient = _gradient_snapshot(policy)
                negative_norm = _gradient_norm(negative_gradient)
                rho = float(alpha) * positive_norm / (negative_norm + float(eps))
            else:
                original_neg_mass = 0.0
                negative_loss, seen_negative = 0.0, []
                negative_gradient = {}
                negative_norm = rho = 0.0
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    continue
                pos = positive_gradient.get(name)
                neg = negative_gradient.get(name)
                if pos is None and neg is None:
                    parameter.grad = None
                elif pos is None:
                    parameter.grad = -rho * neg
                elif neg is None:
                    parameter.grad = pos
                else:
                    parameter.grad = pos - rho * neg
            optimizer.step()
            pos_visited.extend(seen_positive); neg_visited.extend(seen_negative)
            positive_losses.append(positive_loss); negative_losses.append(negative_loss)
            positive_norms.append(positive_norm); negative_norms.append(negative_norm); rhos.append(rho)
            pos_step_mass.append(original_pos_mass); neg_step_mass.append(original_neg_mass)
    for start in range(0, len(pos_step_mass), len(positive_chunks)):
        if abs(sum(pos_step_mass[start:start + len(positive_chunks)]) - 1.0) > 1.0e-9:
            raise RuntimeError("positive replay epoch does not preserve global hierarchy mass")
        if neg_order and abs(sum(neg_step_mass[start:start + len(positive_chunks)]) - 1.0) > 1.0e-9:
            raise RuntimeError("negative replay epoch does not preserve global hierarchy mass")
    pos_visit = _visit_diagnostics(pos_visited, len(pos_order))
    neg_visit = _visit_diagnostics(neg_visited, len(neg_order))
    return dict(
        path="signed", alpha=float(alpha), rho=rhos, positive_norm=positive_norms,
        negative_norm=negative_norms, positive_loss=sum(positive_losses),
        negative_loss=sum(negative_losses), positive_eligible=len(pos_order),
        negative_eligible=len(neg_order), positive_visited=pos_visited,
        negative_visited=neg_visited, positive_mass=pos_accounting,
        negative_mass=neg_accounting, optimizer_steps=len(positive_losses),
        optimizer_chunks=len(positive_chunks), inner_epochs=int(inner_epochs),
        optimizer_steps_requested=len(positive_losses),
        replay_protocol=replay_protocol,
        positive_step_sizes=list(map(len, positive_chunks)) * int(inner_epochs),
        negative_step_sizes=list(map(len, negative_chunks)) * int(inner_epochs),
        positive_chunk_sizes=list(map(len, positive_chunks)),
        negative_chunk_sizes=list(map(len, negative_chunks)),
        positive_step_global_mass=pos_step_mass, negative_step_global_mass=neg_step_mass,
        positive_replay_coverage=pos_visit["unique_replay_coverage"],
        negative_replay_coverage=(neg_visit["unique_replay_coverage"] if neg_order else None),
        positive_replay_visits_per_eligible=pos_visit["visits_per_eligible"],
        negative_replay_visits_per_eligible=(neg_visit["visits_per_eligible"] if neg_order else None),
        positive_visit_min=pos_visit["visit_min"], positive_visit_max=pos_visit["visit_max"],
        negative_visit_min=neg_visit["visit_min"], negative_visit_max=neg_visit["visit_max"],
        positive_loss_step_mean=sum(positive_losses) / len(positive_losses),
        negative_loss_step_mean=sum(negative_losses) / len(positive_losses),
    )


def signed_update(policy, optimizer, recent, *, alpha, batch=128, device="cpu", seed=0,
                  eps=1.0e-12, optimizer_steps=None, optimizer_chunks=16,
                  inner_epochs=1):
    # This delegation occurs before D- is touched: alpha=0 is the exact positive-only code path.
    if float(alpha) == 0.0:
        return positive_only_update(
            policy, optimizer, recent, batch=batch, device=device, seed=seed,
            optimizer_steps=optimizer_steps, optimizer_chunks=optimizer_chunks,
            inner_epochs=inner_epochs,
        )
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    chunks, epochs, protocol = _replay_shape(
        optimizer_steps, optimizer_chunks, inner_epochs,
    )
    positives = [(shard, query) for shard, query in recent.positive_records() if query["train_eligible"]]
    negatives = recent.negative_records()
    pos_order = hierarchical_order(positives, seed)
    neg_order = hierarchical_order(negatives, seed + 1)
    pos_mass, pos_accounting = hierarchy_mass(pos_order)
    neg_mass, neg_accounting = hierarchy_mass(neg_order)
    if not neg_order:
        return positive_only_update(
            policy, optimizer, recent, batch=batch, device=device, seed=seed,
            optimizer_steps=optimizer_steps, optimizer_chunks=optimizer_chunks,
            inner_epochs=inner_epochs,
        )
    if not pos_order:
        # Without g_pos the signed scale is zero; account for D- but do not update.
        optimizer.zero_grad(set_to_none=True)
        neg_loss, neg_visited = _accumulate_objective(
            policy, neg_order, neg_mass, batch, device, seed=seed, stream="negative",
        )
        negative_gradient = _gradient_snapshot(policy)
        negative_norm = _gradient_norm(negative_gradient)
        optimizer.zero_grad(set_to_none=True)
        return dict(
            path="signed_no_positive", alpha=float(alpha), rho=0.0, positive_norm=0.0,
            negative_norm=negative_norm, positive_loss=0.0, negative_loss=neg_loss,
            positive_eligible=0, negative_eligible=len(neg_order),
            positive_visited=[], negative_visited=neg_visited,
            positive_mass=pos_accounting, negative_mass=neg_accounting, optimizer_steps=0,
            optimizer_chunks=chunks, inner_epochs=epochs, replay_protocol=protocol,
            positive_loss_step_mean=None, negative_loss_step_mean=None,
        )
    return _signed_multi_step(
        policy, optimizer, pos_order, neg_order, pos_mass, neg_mass,
        pos_accounting, neg_accounting, alpha=alpha,
        optimizer_chunks=chunks, inner_epochs=epochs, batch=batch, device=device,
        eps=eps, seed=seed, replay_protocol=protocol,
    )
