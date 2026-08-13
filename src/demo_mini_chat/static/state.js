(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DemoState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function rankingRows(state, limit = 12, includeSelected = true) {
    const rows = (state?.candidates || []).slice(0, limit).map((candidate) => ({
      ...candidate,
      outsideTop: false,
    }));

    if (
      includeSelected
      && state?.selected
      && !rows.some((candidate) => candidate.id === state.selected.id)
    ) {
      rows.push({...state.selected, outsideTop: true});
    }

    return rows;
  }

  function nextRecordedState(history, index) {
    if (!Array.isArray(history) || index < 0 || index >= history.length) return null;
    return history[index + 1] || null;
  }

  function splitOrigin(text, origin) {
    const value = String(text ?? '');
    const prefix = String(origin ?? '');
    if (prefix && value.startsWith(prefix)) {
      return {user: prefix, generated: value.slice(prefix.length)};
    }
    return {user: value, generated: ''};
  }

  function isContiguousHistory(history) {
    if (!Array.isArray(history)) return false;
    for (let index = 0; index < history.length - 1; index += 1) {
      if (history[index].text_after !== history[index + 1].text_before) return false;
    }
    return true;
  }

  function timingPlan(seconds) {
    const parsed = Number(seconds);
    const safeSeconds = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
    const totalMs = safeSeconds * 1000;
    return {
      model: totalMs * 0.15,
      probabilities: totalMs * 0.45,
      selection: totalMs * 0.40,
      totalMs,
    };
  }

  function sequenceStats(states) {
    const probabilities = (Array.isArray(states) ? states : [])
      .map((state) => Number(state?.selected?.probability))
      .filter((value) => Number.isFinite(value) && value > 0)
      .map((value) => Math.min(1, Math.max(1e-12, value)));

    if (!probabilities.length) {
      return {
        tokenCount: 0,
        geometricMeanProbability: 0,
        cumulativeLog10Probability: 0,
        averageSurprisalBits: 0,
      };
    }

    const logProbability = probabilities.reduce((total, probability) => total + Math.log(probability), 0);
    const surpriseBits = probabilities.reduce(
      (total, probability) => total - Math.log2(probability),
      0,
    );

    return {
      tokenCount: probabilities.length,
      geometricMeanProbability: Math.exp(logProbability / probabilities.length),
      cumulativeLog10Probability: logProbability / Math.LN10,
      averageSurprisalBits: surpriseBits / probabilities.length,
    };
  }

  return {
    rankingRows,
    nextRecordedState,
    splitOrigin,
    isContiguousHistory,
    timingPlan,
    sequenceStats,
  };
});
