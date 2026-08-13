import assert from 'node:assert/strict';
import test from 'node:test';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const S = require('../src/demo_mini_chat/static/state.js');

function candidate(id, rank, probability, display = String(id)) {
  return {id, rank, probability, display, raw: display};
}

test('rankingRows keeps the selected token visible when it falls outside the top list', () => {
  const state = {
    candidates: [candidate(1, 1, 0.5), candidate(2, 2, 0.25), candidate(3, 3, 0.1)],
    selected: candidate(23, 23, 0.0074, 'defensa'),
  };
  const rows = S.rankingRows(state, 2, true);
  assert.deepEqual(rows.map((row) => row.id), [1, 2, 23]);
  assert.equal(rows[2].outsideTop, true);
});

test('rankingRows does not invent a selection during a probability-only preview', () => {
  const state = {
    candidates: [candidate(1, 1, 0.5), candidate(2, 2, 0.25)],
    selected: candidate(23, 23, 0.0074, 'defensa'),
  };
  const rows = S.rankingRows(state, 1, false);
  assert.deepEqual(rows.map((row) => row.id), [1]);
});

test('nextRecordedState returns the exact state computed after the clicked token', () => {
  const history = [
    {text_before: 'A', text_after: 'A defensa'},
    {text_before: 'A defensa', text_after: 'A defensa la'},
  ];
  assert.equal(S.nextRecordedState(history, 0), history[1]);
  assert.equal(S.nextRecordedState(history, 1), null);
  assert.equal(S.isContiguousHistory(history), true);
});

test('splitOrigin separates user context from generated continuation', () => {
  assert.deepEqual(
    S.splitOrigin('Actúa como experto: una red', 'Actúa como experto:'),
    {user: 'Actúa como experto:', generated: ' una red'},
  );
});

test('timingPlan makes the slider value the complete visual duration per token', () => {
  const timing = S.timingPlan(0.6);
  assert.equal(timing.totalMs, 600);
  assert.ok(Math.abs(timing.model + timing.probabilities + timing.selection - 600) < 1e-9);
});

test('sequenceStats chains conditional token probabilities in log space', () => {
  const states = [
    {selected: {probability: 0.5}},
    {selected: {probability: 0.25}},
  ];
  const stats = S.sequenceStats(states);
  assert.equal(stats.tokenCount, 2);
  assert.ok(Math.abs(stats.geometricMeanProbability - Math.sqrt(0.125)) < 1e-12);
  assert.ok(Math.abs(stats.cumulativeLog10Probability - Math.log10(0.125)) < 1e-12);
  assert.ok(Math.abs(stats.averageSurprisalBits - 1.5) < 1e-12);
});
