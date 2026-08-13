(() => {
  const V = window.DemoVisuals;
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const app = {
    model:null, history:[], playing:false, busy:false, pending:null,
    temperature:.85, speed:2.5, mode:'sample', inspecting:null, maxTokens:60,
    autocompleteOrigin:'La inteligencia artificial', view:'autocomplete',
    chatMessages:[], chatStates:[], chatInspecting:null,
  };

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
      ? 'Cargando Spanish GPT-2 local. La primera vez puede descargar los pesos; después usa la caché.'
      : 'Entrenando o cargando los pesos del modelo de tu clase…';
    try {
      app.model = await api('/api/load', {model:kind});
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
    sentence.innerHTML = '';
    const user = document.createElement('span');
    user.className = 'sentence-user';
    const model = document.createElement('span');
    model.className = 'sentence-model';
    if (text.startsWith(app.autocompleteOrigin)) {
      user.textContent = app.autocompleteOrigin;
      model.textContent = text.slice(app.autocompleteOrigin.length);
    } else {
      user.textContent = text;
    }
    sentence.append(user, model);
  }

  function resetAutocomplete() {
    app.history = [];
    app.pending = null;
    app.inspecting = null;
    setPlaying(false);
    $('prompt-editor').value = 'La inteligencia artificial';
    app.autocompleteOrigin = $('prompt-editor').value;
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
    renderSentence(state.text_before);
    $('phase').textContent = 'El modelo ya produjo una distribución sobre el siguiente token';
    $('selected-token').textContent = 'Todavía no elegimos: primero observa el pool de posibilidades.';
    V.renderTokenStrip(state.input_tokens);
    V.renderUniverse(state, app.model, false);
    V.renderRanking(state, false);
    V.setPhase('probabilities');
  }

  function showSelection(state) {
    $('phase').textContent = state.mode === 'greedy' ? 'Argmax: se toma el token con mayor probabilidad' : 'Muestreo: se toma una muestra de la distribución';
    $('selected-token').innerHTML = `Elegido: <strong>${V.esc(state.selected.display)}</strong> · ${V.pct(state.selected.probability)} · ranking #${state.selected.rank}`;
    V.renderUniverse(state, app.model, true);
    V.renderRanking(state, true);
    V.showCandidate(state.selected, app.model.vocabulary_size);
    V.setPhase('selection');
  }

  function commitState(state) {
    $('prompt-editor').value = state.text_after;
    renderSentence(state.text_after);
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
    if (app.history.length >= app.maxTokens) {
      $('phase').textContent = `Límite de ${app.maxTokens} tokens alcanzado. Auméntalo o usa “Siguiente token”.`;
      return;
    }
    setPlaying(true);
    while (app.playing && app.history.length < app.maxTokens) {
      const completed = await calculateStep(true);
      if (!completed && !app.playing) break;
      if (!app.playing) break;
      await sleep(60);
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
      history.textContent = 'Los tokens generados aparecerán aquí. Haz clic en cualquiera para volver a ver sus probabilidades.';
      return;
    }
    history.className = 'history';
    app.history.forEach((state, index) => {
      const button = document.createElement('button');
      button.className = `history-token${app.inspecting === index ? ' active' : ''}`;
      button.innerHTML = `<span class="history-index">${index + 1}</span><span>${V.esc(state.selected.display)}</span>`;
      button.title = `${V.pct(state.selected.probability)} · ranking #${state.selected.rank}`;
      button.onclick = () => inspectState(index);
      history.appendChild(button);
    });
  }

  function inspectState(index) {
    setPlaying(false);
    app.inspecting = index;
    const state = app.history[index];
    renderSentence(state.text_after);
    $('phase').textContent = `Inspeccionando el estado que produjo el token #${index + 1}`;
    $('selected-token').innerHTML = `<strong>${V.esc(state.selected.display)}</strong> nació con ${V.pct(state.selected.probability)} de probabilidad · ranking #${state.selected.rank}`;
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
    renderSentence();
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

  function showView(view) {
    app.view = view;
    setPlaying(false);
    $('autocomplete-view').classList.toggle('hidden', view !== 'autocomplete');
    $('chat-view').classList.toggle('hidden', view !== 'chat');
    $('autocomplete-tab').classList.toggle('active', view === 'autocomplete');
    $('chat-tab').classList.toggle('active', view === 'chat');
  }

  function resetChat() {
    app.chatMessages = [];
    app.chatStates = [];
    app.chatInspecting = null;
    $('chat-input').value = '';
    $('chat-token-history').className = 'chat-token-history empty';
    $('chat-token-history').textContent = 'Los tokens de la respuesta aparecerán aquí.';
    $('chat-ranking').innerHTML = '';
    d3.select('#chat-token-universe').selectAll('*').remove();
    $('chat-inspector-meta').textContent = 'esperando respuesta';
    $('chat-candidate-detail').textContent = 'Haz clic en un token generado para inspeccionar su estado.';
    renderChatThread();
  }

  function renderChatThread() {
    const thread = $('chat-thread');
    thread.innerHTML = '<div class="chat-explainer">Esta UI parece un chat, pero debajo GPT-2 sigue haciendo lo mismo: <b>predecir el siguiente token</b>. Si responde raro, eso también es parte de la demostración.</div>';
    app.chatMessages.forEach((message, messageIndex) => {
      const row = document.createElement('div');
      row.className = `chat-message ${message.role}`;
      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble';
      bubble.textContent = message.content;
      row.appendChild(bubble);
      if (message.role === 'assistant' && message.states?.length) {
        const inspect = document.createElement('button');
        inspect.className = 'inspect-response';
        inspect.textContent = `${message.states.length} tokens · ver detrás`;
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
    if (app.busy || !app.model) return;
    const text = $('chat-input').value.trim();
    if (!text) return;
    app.chatMessages.push({role:'user',content:text});
    $('chat-input').value = '';
    renderChatThread();

    const prompt = buildChatPrompt();
    const loadingMessage = {role:'assistant',content:'…'};
    app.chatMessages.push(loadingMessage);
    renderChatThread();
    $('chat-send').disabled = true;
    try {
      const maxTokens = Number($('chat-max-tokens').value);
      const temperature = Number($('chat-temperature').value);
      const states = await api('/api/generate', {
        text:prompt, temperature, mode:'sample', top_k:48, max_tokens:maxTokens,
        stop_strings:['\nUsuario:','\nUser:','\nHumano:'],
      });
      let completion = states.length ? states[states.length - 1].text_after.slice(prompt.length) : '';
      completion = completion.split(/\n(?:Usuario|User|Humano):/)[0].trim();
      app.chatMessages.pop();
      app.chatMessages.push({role:'assistant',content:completion || '(sin continuación)',states});
      app.chatStates = states;
      renderChatThread();
      renderChatTokenHistory();
      if (states.length) inspectChatState(states.length - 1);
    } catch (error) {
      app.chatMessages.pop();
      app.chatMessages.push({role:'assistant',content:`Error: ${error.message}`,states:[]});
      renderChatThread();
    } finally {
      $('chat-send').disabled = false;
      $('chat-input').focus();
    }
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
      button.textContent = state.selected.display;
      button.title = `#${index + 1} · ${V.pct(state.selected.probability)} · rank ${state.selected.rank} · raw ${state.selected.raw}`;
      button.onclick = () => inspectChatState(index);
      container.appendChild(button);
    });
  }

  function inspectChatState(index) {
    if (!app.chatStates.length) return;
    app.chatInspecting = index;
    const state = app.chatStates[index];
    $('chat-inspector-meta').textContent = `token ${index + 1}/${app.chatStates.length} · ${V.pct(state.selected.probability)} · rank #${state.selected.rank}`;
    V.renderUniverseInto('chat-token-universe','chat-inspector-meta','chat-candidate-detail',state,app.model,true,28);
    V.renderRankingInto('chat-ranking',state,true,8);
    V.showCandidateInto('chat-candidate-detail',state.selected,app.model.vocabulary_size);
    renderChatTokenHistory();
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
  $('temperature').oninput = (event) => { app.temperature = Number(event.target.value); $('temperature-value').textContent = app.temperature.toFixed(2); $('chat-temperature').value = app.temperature.toFixed(2); };
  $('speed').oninput = (event) => { app.speed = Number(event.target.value); $('speed-value').textContent = `${app.speed.toFixed(1)} s`; };
  $('max-tokens').oninput = (event) => { app.maxTokens = Number(event.target.value); $('max-tokens-value').textContent = `${app.maxTokens} tokens`; };
  $('prompt-editor').addEventListener('input', () => {
    setPlaying(false);
    app.pending = null;
    app.inspecting = null;
    app.history = [];
    app.autocompleteOrigin = $('prompt-editor').value;
    renderSentence();
    renderHistory();
    $('step-count').textContent = 'paso 0';
    $('manual-badge').classList.remove('hidden');
    $('present-button').classList.add('hidden');
    $('phase').textContent = 'Cambiaste el contexto: el siguiente paso se recalculará desde este texto.';
    V.setPhase('text');
  });
  $('chat-send').onclick = sendChat;
  $('chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendChat(); }
  });
  $('chat-temperature').oninput = (event) => {
    app.temperature = Number(event.target.value);
    $('temperature').value = app.temperature;
    $('temperature-value').textContent = app.temperature.toFixed(2);
  };
  window.addEventListener('resize', () => {
    if (app.view === 'autocomplete') {
      const state = app.inspecting !== null ? app.history[app.inspecting] : app.history[app.history.length - 1];
      if (state && app.model) V.renderUniverse(state, app.model, true);
    } else if (app.chatStates.length) {
      inspectChatState(app.chatInspecting ?? app.chatStates.length - 1);
    }
  });

  showView('autocomplete');
  loadModel('transformer');
})();
