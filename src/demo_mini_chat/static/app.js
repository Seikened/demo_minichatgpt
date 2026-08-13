(() => {
  const V = window.DemoVisuals;
  const S = window.DemoState;
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const DEFAULT_AUTOCOMPLETE_PROMPT = 'Actúa como experto en inteligencia artificial y explica qué es una red neuronal:';
  const DEFAULT_CHAT_PROMPT = 'Actúa como un experto en inteligencia artificial y explícame con un ejemplo sencillo qué es una red neuronal.';

  const app = {
    model: null,
    history: [],
    afterSnapshots: new Map(),
    playing: false,
    busy: false,
    pending: null,
    temperature: 0.85,
    speed: 2.5,
    mode: 'sample',
    inspecting: null,
    maxTokens: 80,
    autocompleteOrigin: DEFAULT_AUTOCOMPLETE_PROMPT,
    view: 'autocomplete',
    chatMessages: [],
    chatStates: [],
    chatInspecting: null,
    chatStreaming: false,
    chatAbortController: null,
  };

  function setPlaying(value) {
    app.playing = value;
    $('play-icon').textContent = value ? 'Ⅱ' : '▶';
    $('play-button').childNodes[$('play-button').childNodes.length - 1].textContent = value
      ? ' Pausar'
      : ' Reproducir';
  }

  function setBusy(value) {
    app.busy = value;
    $('next-button').disabled = value;
    $('prompt-editor').disabled = value;
  }

  async function api(path, payload) {
    const response = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = body.detail || message;
      } catch (_) {
        // Keep the HTTP fallback when the server did not return JSON.
      }
      throw new Error(message);
    }
    return response.json();
  }

  async function streamApi(path, payload, signal, onEvent) {
    const response = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal,
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = body.detail || message;
      } catch (_) {
        // Keep the HTTP fallback when the server did not return JSON.
      }
      throw new Error(message);
    }
    if (!response.body) throw new Error('El navegador no recibió un stream de respuesta.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'error') throw new Error(event.detail || 'Error durante la generación.');
        onEvent(event);
      }
      if (done) break;
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === 'error') throw new Error(event.detail || 'Error durante la generación.');
      onEvent(event);
    }
  }

  function cleanChatCompletion(value, final = false) {
    const visible = String(value || '').split(/\n(?:Usuario|User|Humano):/)[0];
    return final ? visible.trim() : visible.trimStart();
  }

  function setChatStreaming(value) {
    app.chatStreaming = value;
    const button = $('chat-send');
    button.textContent = value ? 'Detener' : 'Enviar';
    button.classList.toggle('stop-stream', value);
    $('chat-input').disabled = value;
    $('chat-max-tokens').disabled = value;
    $('chat-temperature').disabled = value;
  }

  function stopChatStream() {
    if (app.chatAbortController) app.chatAbortController.abort();
  }

  async function loadModel(kind = 'transformer') {
    setPlaying(false);
    stopChatStream();
    app.pending = null;
    const loading = $('loading');
    loading.classList.remove('done');
    $('load-error').classList.add('hidden');
    $('fallback-model').classList.add('hidden');
    $('loading-message').textContent = kind === 'transformer'
      ? 'Cargando Spanish GPT-2 local. La primera vez puede descargar los pesos; después usa la caché.'
      : 'Entrenando o cargando los pesos del modelo de tu clase…';
    try {
      app.model = await api('/api/load', {model: kind});
      V.renderModelInfo(app.model);
      resetAutocomplete();
      resetChat();
      await sleep(220);
      loading.classList.add('done');
    } catch (error) {
      $('load-error').textContent = error.message;
      $('load-error').classList.remove('hidden');
      if (kind === 'transformer') $('fallback-model').classList.remove('hidden');
    }
  }

  function renderSentence(text = $('prompt-editor').value) {
    const sentence = $('sentence');
    const split = S.splitOrigin(text, app.autocompleteOrigin);
    sentence.innerHTML = '';

    const user = document.createElement('span');
    user.className = 'sentence-user';
    user.textContent = split.user;

    const model = document.createElement('span');
    model.className = 'sentence-model';
    model.textContent = split.generated;

    sentence.append(user, model);
    requestAnimationFrame(() => {
      sentence.scrollTop = sentence.scrollHeight;
    });
  }

  function resetAutocomplete() {
    app.history = [];
    app.afterSnapshots.clear();
    app.pending = null;
    app.inspecting = null;
    setPlaying(false);
    $('prompt-editor').value = DEFAULT_AUTOCOMPLETE_PROMPT;
    app.autocompleteOrigin = DEFAULT_AUTOCOMPLETE_PROMPT;
    renderSentence();
    $('selected-token').textContent = 'Todavía no se ha elegido ningún token.';
    $('phase').textContent = 'Listo para calcular el siguiente token';
    $('step-count').textContent = 'paso 0';
    $('manual-badge').classList.add('hidden');
    $('token-strip').innerHTML = '';
    $('ranking').innerHTML = '';
    d3.select('#token-universe').selectAll('*').remove();
    $('universe-meta').textContent = `${app.model?.vocabulary_size?.toLocaleString() || '—'} tokens posibles`;
    $('candidate-detail').textContent = 'Pulsa “Siguiente token” o “Reproducir”.';
    $('present-button').classList.add('hidden');
    renderHistory();
    V.setPhase('text');
  }

  async function phaseWait(milliseconds, automatic) {
    const end = performance.now() + Math.max(0, milliseconds);
    while (performance.now() < end) {
      if (automatic && !app.playing) return false;
      await sleep(Math.min(25, Math.max(1, end - performance.now())));
    }
    return true;
  }

  function showProbabilities(state) {
    renderSentence(state.text_before);
    $('phase').textContent = 'El modelo ya produjo una distribución sobre el siguiente token';
    $('selected-token').textContent = 'Todavía no elegimos: primero observa el pool de posibilidades.';
    V.renderTokenStrip(state.input_tokens);
    V.renderUniverse(state, app.model, false);
    V.renderRanking(state, false);
    V.setPhase('probabilities');
  }

  function showSelection(state) {
    $('phase').textContent = state.mode === 'greedy'
      ? 'Argmax: se toma el token con mayor probabilidad'
      : 'Muestreo: se toma una muestra de la distribución';
    $('selected-token').innerHTML = `Elegido: <strong>${V.esc(state.selected.display)}</strong> · ${V.pct(state.selected.probability)} · ranking #${state.selected.rank}`;
    V.renderUniverse(state, app.model, true);
    V.renderRanking(state, true);
    V.showCandidate(state.selected, app.model.vocabulary_size);
    V.setPhase('selection');
  }

  function commitState(state) {
    const previousState = app.history[app.history.length - 1] || null;
    if (previousState && previousState.text_after === state.text_before) {
      app.afterSnapshots.set(previousState.state_id, state);
    }

    $('prompt-editor').value = state.text_after;
    renderSentence(state.text_after);
    $('manual-badge').classList.add('hidden');
    $('phase').textContent = 'El token elegido entra al contexto. El siguiente paso volverá a calcular todo.';
    V.setPhase('append');
    app.history.push(state);
    $('step-count').textContent = `paso ${app.history.length}`;
    renderHistory();
  }

  async function animateState(state, automatic, phase = 0) {
    const timing = S.timingPlan(app.speed);
    app.pending = null;
    if (phase <= 0) {
      showProbabilities(state);
      if (!(await phaseWait(timing.probabilities, automatic))) {
        app.pending = {state, phase: 1};
        return false;
      }
    }
    if (phase <= 1) {
      showSelection(state);
      if (!(await phaseWait(timing.selection, automatic))) {
        app.pending = {state, phase: 2};
        return false;
      }
    }
    if (phase <= 2) commitState(state);
    return true;
  }

  async function calculateStep(automatic = false) {
    if (app.busy || !app.model) return false;
    setBusy(true);
    try {
      if (app.pending) {
        const pending = app.pending;
        return await animateState(pending.state, automatic, pending.phase);
      }
      V.setPhase('tokens');
      $('phase').textContent = 'Tokenizando el contexto actual…';
      const text = $('prompt-editor').value.trim() || DEFAULT_AUTOCOMPLETE_PROMPT;
      const state = await api('/api/step', {
        text,
        temperature: app.temperature,
        mode: app.mode,
        top_k: 64,
      });
      V.setPhase('model');
      $('phase').textContent = 'El modelo calcula un score crudo (logit) para cada token del vocabulario…';
      const timing = S.timingPlan(app.speed);
      if (!(await phaseWait(timing.model, automatic))) {
        app.pending = {state, phase: 0};
        return false;
      }
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
    if (app.playing) {
      setPlaying(false);
      return;
    }
    if (app.history.length >= app.maxTokens) {
      $('phase').textContent = `Límite de ${app.maxTokens} tokens alcanzado. Auméntalo o usa “Siguiente token”.`;
      return;
    }
    setPlaying(true);
    while (app.playing && app.history.length < app.maxTokens) {
      const completed = await calculateStep(true);
      if (!completed && !app.playing) break;
      if (!app.playing) break;
      await sleep(0);
    }
    if (app.history.length >= app.maxTokens) {
      $('phase').textContent = `Límite de ${app.maxTokens} tokens alcanzado · la demo se detuvo aquí.`;
    }
    if (!app.pending) setPlaying(false);
  }

  function renderHistory() {
    const history = $('history');
    history.innerHTML = '';
    if (!app.history.length) {
      history.className = 'history empty';
      history.textContent = 'Aquí aparecerán los tokens generados. Al pulsar uno verás qué volvió probable después.';
      return;
    }

    history.className = 'history';
    app.history.forEach((state, index) => {
      const button = document.createElement('button');
      button.className = `history-token${app.inspecting === index ? ' active' : ''}`;
      button.innerHTML = `<span class="history-index">${index + 1}</span><span>${V.esc(state.selected.display)}</span><span class="history-prob">${V.pct(state.selected.probability)}</span><span class="history-arrow">→ ?</span>`;
      button.title = `Haz clic para ver qué volvió probable después de “${state.selected.display}”`;
      button.onclick = () => inspectState(index);
      history.appendChild(button);
    });
  }

  async function resolveAfterSnapshot(index) {
    const sourceState = app.history[index];
    const recordedNext = S.nextRecordedState(app.history, index);
    if (recordedNext) return {snapshot: recordedNext, recorded: true};

    const cached = app.afterSnapshots.get(sourceState.state_id);
    if (cached) return {snapshot: cached, recorded: false};

    const preview = await api('/api/step', {
      text: sourceState.text_after,
      temperature: sourceState.temperature,
      mode: 'greedy',
      top_k: 64,
    });
    app.afterSnapshots.set(sourceState.state_id, preview);
    return {snapshot: preview, recorded: false};
  }

  async function inspectState(index) {
    if (index < 0 || index >= app.history.length || app.busy) return;
    setPlaying(false);
    app.inspecting = index;
    renderHistory();

    const sourceState = app.history[index];
    renderSentence(sourceState.text_after);
    $('phase').textContent = `Calculando qué fue probable después de “${sourceState.selected.display}”…`;
    $('selected-token').innerHTML = `<strong>${V.esc(sourceState.selected.display)}</strong> fue elegida con ${V.pct(sourceState.selected.probability)} · ranking #${sourceState.selected.rank}.`;
    $('present-button').classList.remove('hidden');

    setBusy(true);
    try {
      const {snapshot, recorded} = await resolveAfterSnapshot(index);
      const sourceLabel = V.esc(sourceState.selected.display);
      $('phase').textContent = `Después de “${sourceState.selected.display}”, estas eran las probabilidades del siguiente token`;

      if (recorded) {
        $('selected-token').innerHTML = `<strong>${sourceLabel}</strong> fue elegida con ${V.pct(sourceState.selected.probability)} · ranking #${sourceState.selected.rank}. Después, el token realmente seleccionado fue <strong>${V.esc(snapshot.selected.display)}</strong> con ${V.pct(snapshot.selected.probability)} · ranking #${snapshot.selected.rank}.`;
      } else {
        $('selected-token').innerHTML = `<strong>${sourceLabel}</strong> fue elegida con ${V.pct(sourceState.selected.probability)} · ranking #${sourceState.selected.rank}. El siguiente token aún no se había generado; esta es la distribución exacta producida por ese contexto.`;
      }

      V.renderTokenStrip(snapshot.input_tokens);
      V.renderUniverse(snapshot, app.model, recorded);
      V.renderRanking(snapshot, recorded);
      V.showCandidate(recorded ? snapshot.selected : snapshot.candidates[0], app.model.vocabulary_size);
      V.setPhase(recorded ? 'selection' : 'probabilities');
    } catch (error) {
      $('phase').textContent = `No se pudo reconstruir el siguiente estado: ${error.message}`;
    } finally {
      setBusy(false);
      renderHistory();
    }
  }

  function backToPresent() {
    app.inspecting = null;
    $('present-button').classList.add('hidden');
    renderSentence();
    $('phase').textContent = 'Presente de la generación';
    if (app.history.length) {
      const state = app.history[app.history.length - 1];
      V.renderUniverse(state, app.model, true);
      V.renderRanking(state, true);
      V.renderTokenStrip(state.input_tokens);
      V.showCandidate(state.selected, app.model.vocabulary_size);
    }
    renderHistory();
  }

  function setMode(mode) {
    app.mode = mode;
    $('sample-mode').classList.toggle('active', mode === 'sample');
    $('greedy-mode').classList.toggle('active', mode === 'greedy');
  }

  function showView(view) {
    app.view = view;
    setPlaying(false);
    $('autocomplete-view').classList.toggle('hidden', view !== 'autocomplete');
    $('chat-view').classList.toggle('hidden', view !== 'chat');
    $('autocomplete-tab').classList.toggle('active', view === 'autocomplete');
    $('chat-tab').classList.toggle('active', view === 'chat');
    if (view !== 'autocomplete') $('sentence-card').classList.remove('is-stuck');
  }

  function resetChat() {
    stopChatStream();
    app.chatMessages = [];
    app.chatStates = [];
    app.chatInspecting = null;
    $('chat-input').value = DEFAULT_CHAT_PROMPT;
    $('chat-token-history').className = 'chat-token-history empty';
    $('chat-token-history').textContent = 'Los tokens de la respuesta aparecerán aquí.';
    $('chat-ranking').innerHTML = '';
    d3.select('#chat-token-universe').selectAll('*').remove();
    $('chat-inspector-meta').textContent = 'esperando respuesta';
    $('chat-candidate-detail').textContent = 'Haz clic en un token generado para inspeccionar su estado.';
    renderChatSequenceSummary();
    renderChatThread();
  }

  function renderChatThread() {
    const thread = $('chat-thread');
    thread.innerHTML = '<div class="chat-explainer">Esta UI parece un chat, pero debajo GPT-2 sigue haciendo lo mismo: <b>predecir el siguiente token</b>. Si responde raro, eso también es parte de la demostración.</div>';
    app.chatMessages.forEach((message) => {
      const row = document.createElement('div');
      row.className = `chat-message ${message.role}${message.streaming ? ' streaming' : ''}`;
      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble';
      bubble.textContent = message.content || (message.streaming ? '' : '(sin continuación)');
      row.appendChild(bubble);
      if (message.role === 'assistant' && message.states?.length) {
        const inspect = document.createElement('button');
        inspect.className = 'inspect-response';
        inspect.textContent = `${message.states.length} tokens · ${message.streaming ? 'generando…' : 'ver detrás'}`;
        inspect.onclick = () => {
          app.chatStates = message.states;
          renderChatTokenHistory();
          inspectChatState(Math.max(0, message.states.length - 1));
        };
        row.appendChild(inspect);
      }
      thread.appendChild(row);
    });
    thread.scrollTop = thread.scrollHeight;
  }

  function buildChatPrompt() {
    const recent = app.chatMessages.slice(-8);
    const transcript = recent.map((message) => message.role === 'user'
      ? `Usuario: ${message.content}`
      : `Asistente: ${message.content}`).join('\n');
    return `${transcript}\nAsistente:`;
  }

  async function sendChat() {
    if (app.chatStreaming) {
      stopChatStream();
      return;
    }
    if (!app.model) return;

    const text = $('chat-input').value.trim();
    if (!text) return;
    app.chatMessages.push({role: 'user', content: text});
    $('chat-input').value = '';
    renderChatThread();

    const prompt = buildChatPrompt();
    const assistantMessage = {role: 'assistant', content: '', states: [], streaming: true};
    app.chatMessages.push(assistantMessage);
    app.chatStates = assistantMessage.states;
    renderChatThread();
    renderChatTokenHistory();

    const controller = new AbortController();
    app.chatAbortController = controller;
    setChatStreaming(true);

    try {
      const maxTokens = Number($('chat-max-tokens').value);
      const temperature = Number($('chat-temperature').value);
      await streamApi(
        '/api/generate-stream',
        {
          text: prompt,
          temperature,
          mode: 'sample',
          top_k: 48,
          max_tokens: maxTokens,
          stop_strings: ['\nUsuario:', '\nUser:', '\nHumano:'],
        },
        controller.signal,
        (event) => {
          if (event.type !== 'state' || !event.state) return;
          assistantMessage.states.push(event.state);
          app.chatStates = assistantMessage.states;
          assistantMessage.content = cleanChatCompletion(event.state.text_after.slice(prompt.length));
          renderChatThread();
          renderChatTokenHistory();
        },
      );

      assistantMessage.streaming = false;
      assistantMessage.content = cleanChatCompletion(assistantMessage.content, true) || '(sin continuación)';
      renderChatThread();
      renderChatTokenHistory();
      if (assistantMessage.states.length) inspectChatState(assistantMessage.states.length - 1);
    } catch (error) {
      assistantMessage.streaming = false;
      if (error.name === 'AbortError') {
        assistantMessage.content = cleanChatCompletion(assistantMessage.content, true) || '(generación detenida)';
      } else {
        assistantMessage.content = `Error: ${error.message}`;
        assistantMessage.states = [];
        app.chatStates = [];
      }
      renderChatThread();
      renderChatTokenHistory();
    } finally {
      if (app.chatAbortController === controller) app.chatAbortController = null;
      setChatStreaming(false);
      $('chat-input').focus();
    }
  }


  function renderChatSequenceSummary() {
    const summary = $('chat-sequence-summary');
    const stats = S.sequenceStats(app.chatStates);
    if (!stats.tokenCount) {
      summary.className = 'sequence-summary empty';
      summary.textContent = 'La cadena condicional aparecerá conforme se generen tokens.';
      return;
    }

    summary.className = 'sequence-summary';
    summary.innerHTML = `<div class="sequence-formula">P(respuesta | prompt) = ∏ P(tokenᵢ | prompt + tokens anteriores)</div>
      <div class="sequence-metrics">
        <span><b>${stats.tokenCount}</b> tokens</span>
        <span><b>${V.pct(stats.geometricMeanProbability)}</b> media geométrica por paso</span>
        <span><b>${stats.averageSurprisalBits.toFixed(2)}</b> bits/token</span>
        <span><b>${stats.cumulativeLog10Probability.toFixed(1)}</b> log₁₀ del producto</span>
      </div>
      <p>No es una medida de verdad ni de comprensión: resume qué tan probable fue esta secuencia para el modelo.</p>`;
  }

  function renderChatTokenHistory() {
    const container = $('chat-token-history');
    container.innerHTML = '';
    if (!app.chatStates.length) {
      container.className = 'chat-token-history empty';
      container.textContent = 'Los tokens de la respuesta aparecerán aquí.';
      return;
    }
    container.className = 'chat-token-history';
    app.chatStates.forEach((state, index) => {
      const button = document.createElement('button');
      button.className = `chat-state-token${app.chatInspecting === index ? ' active' : ''}`;
      button.innerHTML = `<span>${V.esc(state.selected.display)}</span><small>${V.pct(state.selected.probability)}</small>`;
      button.title = `#${index + 1} · ${V.pct(state.selected.probability)} · rank ${state.selected.rank} · raw ${state.selected.raw}`;
      button.onclick = () => inspectChatState(index);
      container.appendChild(button);
    });
    renderChatSequenceSummary();
  }

  function inspectChatState(index) {
    if (!app.chatStates.length) return;
    app.chatInspecting = index;
    const state = app.chatStates[index];
    $('chat-inspector-meta').textContent = `token ${index + 1}/${app.chatStates.length} · ${V.pct(state.selected.probability)} · rank #${state.selected.rank}`;
    V.renderUniverseInto(
      'chat-token-universe',
      'chat-inspector-meta',
      'chat-candidate-detail',
      state,
      app.model,
      true,
      28,
    );
    V.renderRankingInto('chat-ranking', state, true, 8);
    V.showCandidateInto('chat-candidate-detail', state.selected, app.model.vocabulary_size);
    renderChatTokenHistory();
  }

  function initStickySentenceObserver() {
    const sentinel = $('sentence-sentinel');
    const card = $('sentence-card');
    if (!sentinel || !card || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(
      ([entry]) => card.classList.toggle('is-stuck', app.view === 'autocomplete' && !entry.isIntersecting),
      {threshold: 0, rootMargin: '-92px 0px 0px 0px'},
    );
    observer.observe(sentinel);
  }

  $('autocomplete-tab').onclick = () => showView('autocomplete');
  $('chat-tab').onclick = () => showView('chat');
  $('play-button').onclick = togglePlay;
  $('next-button').onclick = () => calculateStep(false);
  $('reset-button').onclick = resetAutocomplete;
  $('present-button').onclick = backToPresent;
  $('fallback-model').onclick = () => loadModel('classroom');
  $('transformer-model').onclick = () => loadModel('transformer');
  $('classroom-model').onclick = () => loadModel('classroom');
  $('sample-mode').onclick = () => setMode('sample');
  $('greedy-mode').onclick = () => setMode('greedy');
  $('temperature').oninput = (event) => {
    app.temperature = Number(event.target.value);
    $('temperature-value').textContent = app.temperature.toFixed(2);
    $('chat-temperature').value = app.temperature.toFixed(2);
  };
  $('speed').oninput = (event) => {
    app.speed = Number(event.target.value);
    $('speed-value').textContent = `${app.speed.toFixed(1)} s`;
  };
  $('max-tokens').oninput = (event) => {
    app.maxTokens = Number(event.target.value);
    $('max-tokens-value').textContent = `${app.maxTokens} tokens`;
  };
  $('prompt-editor').addEventListener('input', () => {
    setPlaying(false);
    app.pending = null;
    app.inspecting = null;
    app.history = [];
    app.afterSnapshots.clear();
    app.autocompleteOrigin = $('prompt-editor').value;
    renderSentence();
    renderHistory();
    $('step-count').textContent = 'paso 0';
    $('manual-badge').classList.remove('hidden');
    $('present-button').classList.add('hidden');
    $('phase').textContent = 'Cambiaste el contexto: el siguiente paso se recalculará desde este texto.';
    V.setPhase('text');
  });
  $('chat-send').onclick = () => sendChat();
  $('chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });
  $('chat-temperature').oninput = (event) => {
    app.temperature = Number(event.target.value);
    $('temperature').value = app.temperature;
    $('temperature-value').textContent = app.temperature.toFixed(2);
  };
  window.addEventListener('resize', () => {
    if (app.view === 'autocomplete') {
      const state = app.inspecting !== null
        ? S.nextRecordedState(app.history, app.inspecting) || app.history[app.inspecting]
        : app.history[app.history.length - 1];
      if (state && app.model) V.renderUniverse(state, app.model, true);
    } else if (app.chatStates.length) {
      inspectChatState(app.chatInspecting ?? app.chatStates.length - 1);
    }
  });

  initStickySentenceObserver();
  showView('autocomplete');
  loadModel('transformer');
})();
