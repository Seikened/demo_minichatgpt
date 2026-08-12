(() => {
  const $ = (id) => document.getElementById(id);
  const pct = (value) => value < 0.0001 ? `${(value * 100).toExponential(2)}%` : `${(value * 100).toFixed(value > 0.01 ? 2 : 4)}%`;

  function setPhase(name) {
    document.querySelectorAll('.pipeline [data-phase]').forEach((node) => {
      node.classList.toggle('active', node.dataset.phase === name);
    });
  }

  function renderTokenStrip(tokens) {
    const strip = $('token-strip');
    strip.innerHTML = '';
    tokens.slice(-36).forEach((token) => {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      chip.textContent = token.display;
      chip.title = `id ${token.id} · raw: ${token.raw}`;
      strip.appendChild(chip);
    });
  }

  function renderRanking(state, selected = true) {
    const container = $('ranking');
    container.innerHTML = '';
    const rows = state.candidates.slice(0, 12);
    const max = Math.max(...rows.map((c) => c.probability), 1e-12);
    rows.forEach((candidate) => {
      const row = document.createElement('div');
      const isSelected = selected && candidate.id === state.selected.id;
      row.className = `rank-row${isSelected ? ' selected' : ''}`;
      const relative = Math.sqrt(candidate.probability / max) * 100;
      row.innerHTML = `<span class="rank-number">${candidate.rank}</span>
        <span class="rank-token" title="${candidate.raw}">${candidate.display}</span>
        <span class="rank-track"><span class="rank-fill" style="width:${relative}%"></span></span>
        <span class="rank-prob">${pct(candidate.probability)}</span>`;
      container.appendChild(row);
    });
  }

  function showCandidate(candidate, vocabularySize) {
    const detail = $('candidate-detail');
    if (!candidate) {
      detail.textContent = 'Haz clic en un token para inspeccionarlo.';
      return;
    }
    const rank = candidate.rank ? `ranking #${candidate.rank}` : 'resto del vocabulario';
    detail.innerHTML = `<strong>${candidate.display}</strong> · ${pct(candidate.probability)} · ${rank} · token id ${candidate.id ?? '—'} · vocabulario ${vocabularySize.toLocaleString()}`;
  }

  function renderUniverse(state, model, selected = true) {
    const svg = d3.select('#token-universe');
    const element = svg.node();
    const width = Math.max(element.clientWidth, 600);
    const height = Math.max(element.clientHeight, 430);
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    let candidates = state.candidates.slice(0, 58).map((c) => ({...c, kind: 'token'}));
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
    d3.pack().size([width, height]).padding(7)(root);
    const nodes = root.leaves();

    const join = svg.selectAll('g.token-node').data(nodes, (d) => `${d.data.kind}-${d.data.id}`);
    join.exit().transition().duration(300).style('opacity', 0).remove();
    const entered = join.enter().append('g').attr('class', 'token-node').style('opacity', 0);
    entered.append('circle').attr('r', 0);
    entered.append('text');

    const merged = entered.merge(join)
      .attr('class', (d) => `token-node${d.data.kind === 'other' ? ' other' : ''}${selected && d.data.id === state.selected.id ? ' selected' : ''}`)
      .style('cursor', 'pointer')
      .on('click', (_, d) => showCandidate(d.data, model.vocabulary_size));

    merged.transition().duration(520).ease(d3.easeCubicOut)
      .style('opacity', 1)
      .attr('transform', (d) => `translate(${d.x},${d.y})`);
    merged.select('circle').transition().duration(520).attr('r', (d) => d.r);
    merged.select('text')
      .text((d) => d.r > 15 || (selected && d.data.id === state.selected.id) ? d.data.display : '')
      .attr('font-size', (d) => Math.max(8, Math.min(17, d.r * 0.34)))
      .each(function(d) {
        const maxChars = Math.max(3, Math.floor(d.r / 4.8));
        if (d.data.display.length > maxChars * 2 && d.r < 38) d3.select(this).text(`${d.data.display.slice(0, maxChars)}…`);
      });

    $('universe-meta').textContent = `${model.vocabulary_size.toLocaleString()} tokens posibles · visibles ${Math.min(58, state.candidates.length)} + resto`;
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
    $('model-info').innerHTML = rows.map(([key, value]) => `<div class="info-line"><span>${key}</span><b>${value}</b></div>`).join('');
    $('model-badge').textContent = `${model.name} · ${model.vocabulary_size.toLocaleString()} tokens`;
  }

  window.DemoVisuals = {pct, setPhase, renderTokenStrip, renderRanking, renderUniverse, renderModelInfo, showCandidate};
})();
