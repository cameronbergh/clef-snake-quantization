#!/usr/bin/env node
'use strict';

// Offline, dependency-free tests of the exact normalization code shipped in the
// watcher. No browser, model server, DOM, network access or generated data needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'watcher/benchmark-watch.html'), 'utf8');
const script = html.match(/<script\b[^>]*\bid=["']watcher-core["'][^>]*>([\s\S]*?)<\/script\s*>/i);
assert.ok(script, 'the watcher must expose its DOM-free watcher-core script');
const sandbox = {};
vm.runInNewContext(script[1], sandbox, {filename: 'watcher-core.js', timeout: 1000});
const {MODELS, normalize: normalizeCore} = sandbox.WatcherData;
assert.equal(typeof normalizeCore, 'function');
// Remove cross-realm prototypes so strict assertions compare plain JSON values.
const plain = value => JSON.parse(JSON.stringify(value));
const normalize = (...args) => plain(normalizeCore(...args));
const clone = plain;
const fixture = JSON.parse(fs.readFileSync(path.join(root, 'data/2026-10-05-fiveway/results.json'), 'utf8'));
const ids = ['bf16', 'q6-k-l', 'q4-k-m', 'iq2-m', 'q2-k'];
const model = (result, id) => {
  const found = result.models.find(item => item.id === id);
  assert.ok(found, `missing model summary: ${id}`);
  return found;
};
const wins = (result, id) => {
  const found = result.wins.find(item => item.id === id);
  assert.ok(found, `missing comparison: ${id}`);
  return {wins: found.wins, ties: found.ties, losses: found.losses, n: found.n};
};
const row = (id, seed, score, seedRound = 1) => ({
  model: id, seed, score, seed_round: seedRound, round: seedRound,
  end_reason: 'collision', alive: false,
});
const live = (rows, aggregates = {}) => ({
  completed: rows,
  model_aggregates: aggregates,
  total_rounds: 15,
  total_episodes: 75,
  state: 'complete',
});
const quantIds = ids.filter(id => id !== 'bf16');
const tests = [];
const test = (name, run) => tests.push({name, run});

test('published fixture preserves all five models, scores, caps and paired comparisons', () => {
  assert.deepEqual(plain(MODELS).map(item => item.id), ids);
  for (const item of MODELS) {
    assert.equal(typeof item.label, 'string');
    assert.ok(item.label);
    assert.equal(typeof item.color, 'string');
    assert.ok(item.color);
  }
  const result = normalize(fixture);
  assert.equal(result.published, true);
  assert.equal(result.complete, true);
  assert.equal(result.expectedGames, 75);
  assert.equal(result.expectedSeeds, 15);
  assert.equal(result.rows.length, 75);
  assert.equal(result.models.length, 5);
  assert.equal(result.paired.length, 15);
  const expected = {
    bf16: [183 / 15, 13, 1, 29, 0],
    'q6-k-l': [185 / 15, 12, 1, 29, 0],
    'q4-k-m': [191 / 15, 12, 1, 29, 0],
    'iq2-m': [278 / 15, 20, 1, 38, 1],
    'q2-k': [257 / 15, 19, 1, 29, 0],
  };
  for (const [id, [mean, median, min, max, caps]] of Object.entries(expected)) {
    const summary = model(result, id);
    assert.equal(summary.rows.length, 15);
    assert.ok(Math.abs(summary.scores.mean - mean) < 0.000001, `${id}: mean uses accepted rows`);
    assert.equal(summary.scores.median, median);
    assert.equal(summary.scores.min, min);
    assert.equal(summary.scores.max, max);
    assert.equal(summary.caps, caps);
    assert.equal(summary.aggregateOnly, false);
    assert.equal(summary.aggregate.server_latency_ms.median, fixture.aggregate_by_model[id].server_latency_ms.median);
  }
  assert.deepEqual(wins(result, 'q6-k-l'), {wins: 1, ties: 13, losses: 1, n: 15});
  assert.deepEqual(wins(result, 'q4-k-m'), {wins: 3, ties: 8, losses: 4, n: 15});
  assert.deepEqual(wins(result, 'iq2-m'), {wins: 11, ties: 2, losses: 2, n: 15});
  assert.deepEqual(wins(result, 'q2-k'), {wins: 10, ties: 4, losses: 1, n: 15});
  assert.deepEqual(result.paired.map(item => item.seed).sort(), fixture.seed_manifest.seeds.slice().sort());
  for (const pair of result.paired) {
    assert.deepEqual(Object.keys(pair.rows).sort(), ids.slice().sort());
    for (const id of ids) assert.equal(pair.rows[id].seed, pair.seed);
  }
});

test('equivalent live and published input yield the same accepted evidence', () => {
  const published = normalize(fixture);
  const current = normalize(live(fixture.rounds, fixture.aggregate_by_model));
  assert.equal(current.published, false);
  for (const key of ['rows', 'models', 'paired', 'wins', 'expectedGames', 'expectedSeeds', 'complete']) {
    assert.deepEqual(current[key], published[key], `live/published disagreement: ${key}`);
  }
  // A live empty list is authoritative even when a stale published list exists.
  const empty = normalize({...fixture, completed: [], model_aggregates: {}});
  assert.equal(empty.rows.length, 0);
  assert.equal(empty.complete, false);
});

test('exact duplicate rows cannot inflate games, caps or comparisons', () => {
  const reorderedCopies = fixture.rounds.map(item => Object.fromEntries(Object.entries(clone(item)).reverse()));
  const result = normalize(live([...fixture.rounds, ...clone(fixture.rounds), ...reorderedCopies], fixture.aggregate_by_model));
  assert.equal(result.rows.length, 75);
  assert.equal(model(result, 'iq2-m').rows.length, 15);
  assert.equal(model(result, 'iq2-m').caps, 1);
  assert.equal(result.paired.length, 15);
  assert.deepEqual(wins(result, 'iq2-m'), {wins: 11, ties: 2, losses: 2, n: 15});
});

test('conflicting rows are excluded regardless of order and reduce the shared comparison set', () => {
  const original = fixture.rounds.find(item => item.model === 'iq2-m');
  const conflict = {...clone(original), score: original.score + 1};
  const alternatives = [
    [...fixture.rounds, conflict, clone(original)],
    [conflict, ...fixture.rounds],
  ];
  for (const rows of alternatives) {
    const result = normalize(live(rows, fixture.aggregate_by_model));
    assert.equal(result.rows.length, 74);
    assert.equal(model(result, 'iq2-m').rows.length, 14);
    assert.ok(!result.rows.some(item => item.model === original.model && item.seed === original.seed));
    assert.equal(result.paired.length, 14);
    for (const id of quantIds) assert.equal(wins(result, id).n, 14);
    assert.ok(result.warnings.length, 'ambiguous evidence must be disclosed');
    assert.equal(result.complete, false);
    assert.ok(!model(result, 'iq2-m').aggregate, '15-round latency cannot describe 14 accepted rows');
  }
});

test('contradictory move and timing evidence is a conflict even when scores agree', () => {
  const original = fixture.rounds.find(item => item.model === 'iq2-m');
  const conflicts = [
    {...clone(original), attempted_moves: original.attempted_moves + 1},
    {...clone(original), server_latency_ms: {...original.server_latency_ms, p90: original.server_latency_ms.p90 + 1}},
    {...clone(original), request_wall_latency_ms: {...original.request_wall_latency_ms, median: original.request_wall_latency_ms.median + 1}},
  ];
  for (const conflict of conflicts) {
    const result = normalize(live([...fixture.rounds, conflict], fixture.aggregate_by_model));
    assert.equal(result.rows.length, 74, 'contradictory evidence must not be treated as an exact duplicate');
    assert.equal(result.paired.length, 14);
    assert.equal(result.complete, false);
    assert.ok(!model(result, 'iq2-m').aggregate);
  }
});

test('all comparisons use the same intersection of five actual seed strings', () => {
  const rows = [
    ...ids.map((id, index) => row(id, 'shared-seed', 10 + index, 1)),
    ...ids.filter(id => id !== 'q2-k').map(id => row(id, 'partial-seed', 999, 2)),
  ];
  const result = normalize(live(rows));
  assert.equal(result.rows.length, 9);
  assert.equal(result.paired.length, 1);
  assert.equal(result.paired[0].seed, 'shared-seed');
  for (const id of quantIds) assert.deepEqual(wins(result, id), {wins: 1, ties: 0, losses: 0, n: 1});
  // Equal display round numbers do not make different environment seeds paired.
  const unpaired = normalize(live(ids.map((id, index) => row(id, `different-${id}`, index, 1))));
  assert.equal(unpaired.paired.length, 0);
  for (const id of quantIds) assert.deepEqual(wins(unpaired, id), {wins: 0, ties: 0, losses: 0, n: 0});
  const fourModels = normalize(live(fixture.rounds.filter(item => item.model !== 'q2-k')));
  assert.equal(fourModels.models.length, 5);
  assert.equal(model(fourModels, 'q2-k').rows.length, 0);
  assert.equal(model(fourModels, 'q2-k').scores, null);
  assert.equal(fourModels.paired.length, 0);
  for (const id of quantIds) assert.equal(wins(fourModels, id).n, 0);
});

test('the expected game count does not imply completion without the expected paired seeds', () => {
  const mismatched = clone(fixture.rounds);
  const displaced = mismatched.find(item => item.model === 'iq2-m');
  displaced.seed = 'unmatched-environment-seed';
  for (const input of [live(mismatched, fixture.aggregate_by_model), {...fixture, rounds: mismatched}]) {
    const result = normalize(input);
    assert.equal(result.rows.length, 75);
    assert.equal(result.expectedGames, 75);
    assert.equal(result.expectedSeeds, 15);
    assert.equal(result.paired.length, 14);
    assert.equal(result.complete, false);
    for (const id of quantIds) assert.equal(wins(result, id).n, 14);
  }
});

test('missing or malformed evidence never creates rows or finite-looking statistics', () => {
  const valid = row('bf16', 'valid-seed', 0);
  const malformed = [
    null, {},
    {...valid, seed: undefined}, {...valid, seed: ''},
    {...valid, model: undefined}, {...valid, model: 'unknown-model'},
    {...valid, score: undefined}, {...valid, score: null},
    {...valid, score: NaN}, {...valid, score: Infinity},
    {...valid, end_reason: undefined}, {...valid, end_reason: ''},
    {...valid, end_reason: 'collision', alive: true},
    {...valid, end_reason: 'move_cap', alive: false},
  ];
  const result = normalize(live([valid, ...malformed]));
  assert.equal(result.rows.length, 1);
  assert.equal(model(result, 'bf16').scores.mean, 0);
  assert.equal(model(result, 'bf16').scores.min, 0);
  assert.equal(model(result, 'bf16').caps, 0);
  assert.equal(result.complete, false);
  assert.ok(result.warnings.length);
  for (const input of [null, {}, {completed: null}, {completed: 'not rows'}]) {
    const empty = normalize(input);
    assert.equal(empty.rows.length, 0);
    assert.equal(empty.models.length, 5);
    assert.equal(empty.complete, false);
    for (const summary of empty.models) assert.equal(summary.scores, null);
  }
});

test('scores are recomputed from accepted rows and mismatched latency aggregates are suppressed', () => {
  const stale = clone(fixture.aggregate_by_model);
  stale.bf16.scores = {mean: 999, median: 999, min: 999, max: 999};
  const full = normalize(live(fixture.rounds, stale));
  assert.equal(model(full, 'bf16').scores.mean, 12.2);
  const partialRows = fixture.rounds.filter(item => item.model !== 'bf16' || item.seed !== fixture.seed_manifest.seeds[0]);
  const partial = normalize(live(partialRows, stale));
  assert.equal(model(partial, 'bf16').rows.length, 14);
  assert.ok(!model(partial, 'bf16').aggregate);
  assert.equal(model(partial, 'iq2-m').aggregate.rounds, 15);
  assert.equal(partial.complete, false);
});

test('legacy IQ2 aggregate supplementation never invents rows or replaces live IQ2 evidence', () => {
  const persisted = {model_aggregates: {'iq2-m': clone(fixture.aggregate_by_model['iq2-m'])}};
  const withoutIQ2 = fixture.rounds.filter(item => item.model !== 'iq2-m');
  const supplemented = normalize(live(withoutIQ2), persisted);
  const legacy = model(supplemented, 'iq2-m');
  assert.equal(supplemented.rows.length, 60);
  assert.equal(legacy.rows.length, 0);
  assert.equal(legacy.aggregateOnly, true);
  assert.equal(legacy.aggregate.rounds, 15);
  assert.equal(legacy.scores.mean, 18.533);
  assert.equal(legacy.caps, 1);
  assert.equal(supplemented.paired.length, 0);
  assert.equal(supplemented.complete, false);
  for (const id of quantIds) assert.equal(wins(supplemented, id).n, 0);
  const full = normalize(live(fixture.rounds, fixture.aggregate_by_model), persisted);
  assert.equal(full.rows.length, 75);
  assert.equal(model(full, 'iq2-m').rows.length, 15);
  assert.equal(model(full, 'iq2-m').aggregateOnly, false);
  assert.ok(Math.abs(model(full, 'iq2-m').scores.mean - 278 / 15) < 0.000001);
  const oneLive = normalize(live([row('iq2-m', 'new-live-seed', 7)]), persisted);
  assert.equal(model(oneLive, 'iq2-m').rows.length, 1);
  assert.equal(model(oneLive, 'iq2-m').scores.mean, 7);
  assert.equal(model(oneLive, 'iq2-m').aggregateOnly, false);
  assert.ok(!model(oneLive, 'iq2-m').aggregate);
  const liveAggregate = {...clone(fixture.aggregate_by_model['iq2-m']), rounds: 2,
    scores: {mean: 3, median: 3, min: 2, max: 4}};
  const aggregatePresent = normalize(live([], {'iq2-m': liveAggregate}), persisted);
  assert.ok(!model(aggregatePresent, 'iq2-m').aggregateOnly, 'persisted data cannot replace a present live aggregate');
  assert.ok(!model(aggregatePresent, 'iq2-m').aggregate);
  assert.equal(model(aggregatePresent, 'iq2-m').scores, null);
  const rejectedLive = normalize(live([
    row('iq2-m', 'conflicted-live-seed', 7),
    row('iq2-m', 'conflicted-live-seed', 8),
  ]), persisted);
  assert.equal(model(rejectedLive, 'iq2-m').rows.length, 0);
  assert.equal(model(rejectedLive, 'iq2-m').aggregateOnly, false, 'legacy summary cannot cover conflicting live evidence');
  assert.equal(model(rejectedLive, 'iq2-m').scores, null);
  assert.ok(!model(rejectedLive, 'iq2-m').aggregate);
});

let failures = 0;
for (const {name, run} of tests) {
  try {
    run();
    console.log(`ok - ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`not ok - ${name}\n${error.stack}`);
  }
}
if (failures) process.exitCode = 1;
else console.log(`All ${tests.length} watcher normalization checks passed.`);
