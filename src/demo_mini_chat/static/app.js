(() => {
  const V = window.DemoVisuals;
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const app = {model:null,history:[],playing:false,busy:false,pending:null,temperature:.85,speed:2.5,mode:'sample',inspecting:null};

  function setPlaying(value) {
    app.playing = value;
    $('play-icon').textContent = value ? 'Ⅱ' : '▶';
    $('play-button').childNodes[$('play-button').childNodes.length - 1].textContent = value ? ' Pausar' : ' Reproducir';
  }

  function setBusy(value) {
    app.busy = value;
    $('next-button').disabled = value;
    $('prompt-editor').disabled = value;
  }

  async function api(path, payload) {
    const response = await fetch(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try { const body = await response.json(); message = body.detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  }

  async function loadModel(kind='transformer') {
    setPlaying(false);
    app.pending = null;
    const loading = $('loading');
    loading.classList.remove('done');
    $('load-error').classList.add('hidden');
    $('fallback-model').classList.add('hidden');
    $('loading-message').textContent = kind === 'transformer'
      ? 'La primera vez puede descargar el modelo. Después queda en la caché local.'
      : 'Entrenando o cargando los pesos del modelo de tu clase…';
    try {
      app.model = await api('/api/load', {model:kind});
      V.renderModelInfo(app.model);
      resetSession();
      await sleep(280);
      loading.classList.add('done');
    } catch (error) {
      $('load-error').textContent = error.message;
      $('load-error').classList.remove('hidden');
      if (kind === 'transformer') $('fallback-model').classList.remove('hidden');
    }
  }

  function resetSession() {
    app.history = [];
    app.pending = null;
    app.inspecting = null;
    setPlaying(false);
    $('prompt-editor').value = 'La inteligencia artificial';
    $('sentence').textContent = $('prompt-editor').value;
    $('selected-token').textContent = 'Todavía no se ha elegido ningún token.';
    $('phase').textContent = 'Listo para calcular el siguiente token';
    $('step-count').textContent = 'paso 0';
    $('manual-badge').classList.add('hidden');
    $('token-strip').innerHTML = '';
    $('ranking').innerHTML = '';
    d3.select('#token-universe').selectAll('*').remove();
    $('universe-meta').textContent = `${app.model?.vocabulary_size?.toLocaleString() || '—'} tokens posibles`;
    $('candidate-detail').textContent = 'Pulsa “Siguiente token” o “Reproducir”.';
    renderHistory();
    V.setPhase('text');
  }

  async function phaseWait() {
    const end = performance.now() + app.speed * 650;
    while (performance.now() < end) {
      if (!app.playing) return false;
      await sleep(45);
    }
    return true;
  }

  function showProbabilities(state) {
    $('sentence').textContent = state.text_before;
    $('phase').textContent = 'El modelo ya produjo una distribución sobre el siguiente token';
    $('selected-token').textContent = 'Todavía no elegimos: primero observa el pool de posibilidades.';
    V.renderTokenStrip(state.input_tokens);
    V.renderUniverse(state, app.model, false);
    V.renderRanking(state, false);
    V.setPhase('probabilities');
  }

  function showSelection(state) {
    $('phase').textContent = state.mode === 'greedy' ? 'Argmax: se toma el token con mayor probabilidad' : 'Muestreo: se toma una muestra de la distribución';
    $('selected-token').innerHTML = `Elegido: <strong>${state.selected.display}</strong> · ${V.pct(state.selected.probability)} · ranking #${state.selected.rank}`;
    V.renderUniverse(state, app.model, true);
    V.renderRanking(state, true);
    V.showCandidate(state.selected, app.model.vocabulary_size);
    V.setPhase('selection');
  }

  function commitState(state) {
    $('prompt-editor').value = state.text_after;
    $('sentence').textContent = state.text_after;
    $('manual-badge').classList.add('hidden');
    $('phase').textContent = 'El token elegido entra al contexto. El siguiente paso volverá a calcular todo.';
    V.setPhase('append');
    app.history.push(state);
    $('step-count').textContent = `paso ${app.history.length}`;
    renderHistory();
  }

  async function animateState(state, automatic, phase=0) {
    app.pending = null;
    if (phase <= 0) {
      showProbabilities(state);
      if (automatic && !(await phaseWait())) { app.pending = {state,phase:1}; return false; }
      if (!automatic) await sleep(Math.min(800, app.speed * 260));
    }
    if (phase <= 1) {
      showSelection(state);
      if (automatic && !(await phaseWait())) { app.pending = {state,phase:2}; return false; }
      if (!automatic) await sleep(Math.min(800, app.speed * 260));
    }
    if (phase <= 2) commitState(state);
    return true;
  }

  async function calculateStep(automatic=false) {
    if (app.busy || !app.model) return false;
    setBusy(true);
    try {
      if (app.pending) {
        const pending = app.pending;
        return await animateState(pending.state, automatic, pending.phase);
      }
      V.setPhase('tokens');
      $('phase').textContent = 'Tokenizando el contexto actual…';
      const text = $('prompt-editor').value.trim() || 'La inteligencia artificial';
      const state = await api('/api/step', {text,temperature:app.temperature,mode:app.mode,top_k:64});
      V.setPhase('model');
      $('phase').textContent = 'El modelo calcula un score crudo (logit) para cada token del vocabulario…';
      if (automatic) await sleep(Math.min(650, app.speed * 180));
      return await animateState(state, automatic, 0);
    } catch (error) {
      $('phase').textContent = `Error: ${error.message}`;
      setPlaying(false);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function togglePlay() {
    if (app.playing) { setPlaying(false); return; }
    setPlaying(true);
    while (app.playing && app.history.length < 100) {
      const completed = await calculateStep(true);
      if (!completed && !app.playing) break;
      if (!app.playing) break;
      await sleep(80);
    }
    if (!app.pending) setPlaying(false);
  }

  function renderHistory() {
    const history = $('history');
    history.innerHTML = '';
    if (!app.history.length) {
      history.className = 'history empty';
      history.textContent = 'Los tokens generados aparecerán aquí. Haz clic en cualquiera para volver a ver sus probabilidades.';
      return;
    }
    history.className = 'history';
    app.history.forEach((state, index) => {
      const button = document.createElement('button');
      button.className = `history-token${app.inspecting === index ? ' active' : ''}`;
      button.innerHTML = `<span class="history-index">${index + 1}</span><span>${state.selected.display}</span>`;
      button.title = `${V.pct(state.selected.probability)} · ranking #${state.selected.rank}`;
      button.onclick = () => inspectState(index);
      history.appendChild(button);
    });
  }

  function inspectState(index) {
    setPlaying(false);
    app.inspecting = index;
    const state = app.history[index];
    $('sentence').textContent = state.text_after;
    $('phase').textContent = `Inspeccionando el estado que produjo el token #${index + 1}`;
    $('selected-token').innerHTML = `<strong>${state.selected.display}</strong> nació con ${V.pct(state.selected.probability)} de probabilidad · ranking #${state.selected.rank}`;
    V.renderTokenStrip(state.input_tokens);
    V.renderUniverse(state, app.model, true);
    V.renderRanking(state, true);
    V.showCandidate(state.selected, app.model.vocabulary_size);
    V.setPhase('selection');
    $('present-button').classList.remove('hidden');
    renderHistory();
  }

  function backToPresent() {
    app.inspecting = null;
    $('present-button').classList.add('hidden');
    $('sentence').textContent = $('prompt-editor').value;
    $('phase').textContent = 'Presente de la generación';
    if (app.history.length) {
      const state = app.history[app.history.length - 1];
      V.renderUniverse(state, app.model, true);
      V.renderRanking(state, true);
      V.renderTokenStrip(state.input_tokens);
    }
    renderHistory();
  }

  function setMode(mode) {
    app.mode = mode;
    $('sample-mode').classList.toggle('active', mode === 'sample');
    $('greedy-mode').classList.toggle('active', mode === 'greedy');
  }

  $('play-button').onclick = togglePlay;
  $('next-button').onclick = () => calculateStep(false);
  $('reset-button').onclick = resetSession;
  $('present-button').onclick = backToPresent;
  $('fallback-model').onclick = () => loadModel('classroom');
  $('transformer-model').onclick = () => loadModel('transformer');
  $('classroom-model').onclick = () => loadModel('classroom');
  $('sample-mode').onclick = () => setMode('sample');
  $('greedy-mode').onclick = () => setMode('greedy');
  $('temperature').oninput = (event) => { app.temperature = Number(event.target.value); $('temperature-value').textContent = app.temperature.toFixed(2); };
  $('speed').oninput = (event) => { app.speed = Number(event.target.value); $('speed-value').textContent = `${app.speed.toFixed(1)} s`; };
  $('prompt-editor').addEventListener('input', () => {
    setPlaying(false);
    app.pending = null;
    app.inspecting = null;
    $('sentence').textContent = $('prompt-editor').value;
    $('manual-badge').classList.remove('hidden');
    $('present-button').classList.add('hidden');
    $('phase').textContent = 'Cambiaste el contexto: el siguiente paso se recalculará desde este texto.';
    V.setPhase('text');
  });
  window.addEventListener('resize', () => {
    const state = app.inspecting !== null ? app.history[app.inspecting] : app.history[app.history.length - 1];
    if (state && app.model) V.renderUniverse(state, app.model, true);
  });

  loadModel('transformer');
})();
