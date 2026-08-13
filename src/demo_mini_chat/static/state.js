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

  return {rankingRows, nextRecordedState, splitOrigin, isContiguousHistory};
});
