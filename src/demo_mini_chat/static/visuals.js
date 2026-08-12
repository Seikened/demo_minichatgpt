(() => {
  const $ = (id) => document.getElementById(id);
  const pct = (value) => value < 0.0001 ? `${(value * 100).toExponential(2)}%` : `${(value * 100).toFixed(value > 0.01 ? 2 : 4)}%`;
  const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

  function setPhase(name) {
    document.querySelectorAll('.pipeline [data-phase]').forEach((node) => {
      node.classList.toggle('active', node.dataset.phase === name);
    });
  }

  function renderTokenStrip(tokens, targetId = 'token-strip') {
    const strip = $(targetId);
    if (!strip) return;
    strip.innerHTML = '';
    tokens.slice(-36).forEach((token) => {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      chip.textContent = token.display;
      chip.title = `id ${token.id} · raw: ${token.raw}`;
      strip.appendChild(chip);
    });
  }

  function renderRankingInto(targetId, state, selected = true, limit = 12) {
    const container = $(targetId);
    if (!container) return;
    container.innerHTML = '';
    const rows = state.candidates.slice(0, limit);
    const max = Math.max(...rows.map((c) => c.probability), 1e-12);
    rows.forEach((candidate) => {
      const row = document.createElement('div');
      const isSelected = selected && candidate.id === state.selected.id;
      row.className = `rank-row${isSelected ? ' selected' : ''}`;
      const relative = Math.sqrt(candidate.probability / max) * 100;
      row.innerHTML = `<span class="rank-number">${candidate.rank}</span>
        <span class="rank-token" title="${esc(candidate.raw)}">${esc(candidate.display)}</span>
        <span class="rank-track"><span class="rank-fill" style="width:${relative}%"></span></span>
        <span class="rank-prob">${pct(candidate.probability)}</span>`;
      container.appendChild(row);
    });
  }

  function renderRanking(state, selected = true) {
    renderRankingInto('ranking', state, selected, 12);
  }

  function showCandidateInto(targetId, candidate, vocabularySize) {
    const detail = $(targetId);
    if (!detail) return;
    if (!candidate) {
      detail.textContent = 'Haz clic en un token para inspeccionarlo.';
      return;
    }
    const rank = candidate.rank ? `ranking #${candidate.rank}` : 'resto del vocabulario';
    const raw = candidate.raw && candidate.raw !== candidate.display ? ` · raw: ${esc(candidate.raw)}` : '';
    detail.innerHTML = `<strong>${esc(candidate.display)}</strong> · ${pct(candidate.probability)} · ${rank} · token id ${candidate.id ?? '—'}${raw} · vocabulario ${vocabularySize.toLocaleString()}`;
  }

  function showCandidate(candidate, vocabularySize) {
    showCandidateInto('candidate-detail', candidate, vocabularySize);
  }

  function renderUniverseInto(svgId, metaId, detailId, state, model, selected = true, maxVisible = 58) {
    const svg = d3.select(`#${svgId}`);
    const element = svg.node();
    if (!element) return;
    const width = Math.max(element.clientWidth, 360);
    const height = Math.max(element.clientHeight, 300);
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    let candidates = state.candidates.slice(0, maxVisible).map((c) => ({...c, kind: 'token'}));
    if (selected && !candidates.some((c) => c.id === state.selected.id)) {
      candidates.push({...state.selected, kind: 'token', forced: true});
    }
    const maxP = Math.max(...candidates.map((c) => c.probability), 1e-12);
    const otherCount = Math.max(0, model.vocabulary_size - state.candidates.length);
    if (state.other_probability_mass > 0.00001 && otherCount > 0) {
      candidates.push({id: -1, raw: 'otros', display: `otros ${otherCount.toLocaleString()}`, probability: state.other_probability_mass, rank: null, kind: 'other'});
    }

    const root = d3.hierarchy({children: candidates}).sum((d) => {
      if (!d.probability) return 0;
      const capped = d.kind === 'other' ? Math.min(d.probability, maxP * 2.7) : d.probability;
      return Math.pow(capped, 0.34);
    });
    d3.pack().size([width, height]).padding(maxVisible > 40 ? 7 : 5)(root);
    const nodes = root.leaves();

    const join = svg.selectAll('g.token-node').data(nodes, (d) => `${d.data.kind}-${d.data.id}`);
    join.exit().transition().duration(260).style('opacity', 0).remove();
    const entered = join.enter().append('g').attr('class', 'token-node').style('opacity', 0);
    entered.append('circle').attr('r', 0);
    entered.append('text');

    const merged = entered.merge(join)
      .attr('class', (d) => `token-node${d.data.kind === 'other' ? ' other' : ''}${selected && d.data.id === state.selected.id ? ' selected' : ''}`)
      .style('cursor', 'pointer')
      .on('click', (_, d) => showCandidateInto(detailId, d.data, model.vocabulary_size));

    merged.transition().duration(460).ease(d3.easeCubicOut)
      .style('opacity', 1)
      .attr('transform', (d) => `translate(${d.x},${d.y})`);
    merged.select('circle').transition().duration(460).attr('r', (d) => d.r);
    merged.select('text')
      .text((d) => d.r > 15 || (selected && d.data.id === state.selected.id) ? d.data.display : '')
      .attr('font-size', (d) => Math.max(8, Math.min(17, d.r * 0.34)))
      .each(function(d) {
        const maxChars = Math.max(3, Math.floor(d.r / 4.8));
        if (d.data.display.length > maxChars * 2 && d.r < 38) d3.select(this).text(`${d.data.display.slice(0, maxChars)}…`);
      });

    const meta = $(metaId);
    if (meta) meta.textContent = `${model.vocabulary_size.toLocaleString()} tokens posibles · visibles ${Math.min(maxVisible, state.candidates.length)} + resto`;
  }

  function renderUniverse(state, model, selected = true) {
    renderUniverseInto('token-universe', 'universe-meta', 'candidate-detail', state, model, selected, 58);
  }

  function renderModelInfo(model) {
    const rows = [
      ['Modelo', model.name],
      ['Vocabulario', `${model.vocabulary_size.toLocaleString()} tokens`],
      ['Parámetros', model.parameter_count ? model.parameter_count.toLocaleString() : '—'],
      ['Tokenizer', model.tokenizer],
      ['Entrenamiento', model.training_data],
      ['Contexto máximo', model.context_window ? `${model.context_window.toLocaleString()} tokens` : '—'],
      ['Dispositivo', model.device],
    ];
    $('model-info').innerHTML = rows.map(([key, value]) => `<div class="info-line"><span>${esc(key)}</span><b>${esc(value)}</b></div>`).join('');
    $('model-badge').textContent = `${model.name} · ${model.vocabulary_size.toLocaleString()} tokens`;
  }

  window.DemoVisuals = {
    pct, esc, setPhase, renderTokenStrip, renderRanking, renderRankingInto,
    renderUniverse, renderUniverseInto, renderModelInfo, showCandidate, showCandidateInto,
  };
})();
