const $ = id => document.getElementById(id);
const els = Object.fromEntries(['prompt','temperature','temperature-value','top-k','seed','generation-length','generate-button','stop-button','step-button','inspect-button','model-status','model-size','status-dot','error-banner','error-message','retry-button','context-count','generated-text','generation-status','probabilities','most-likely','entropy','distribution-mode','tokens','token-notice','embedding-label','embedding-strip','embedding-caption','layer','head','disable-head','ablation-status','ablation-description','attention-canvas','attention-summary','attention-detail','training-chart','parameter-count','architecture-summary','training-steps','training-device','corpus-link','corpus-size','experiment-notice'].map(id => [id, $(id)]));
const checkpointButtons = [...document.querySelectorAll('[data-checkpoint]')];
const controls = [els.prompt, els.temperature, els['top-k'], els.seed, els['generation-length'], els['generate-button'], els['step-button'], els['inspect-button'], els.layer, els.head, els['disable-head'], ...checkpointButtons];
const state = {worker:null, request:0, loadRequest:0, inspectRequest:0, generateRequest:0, checkpoint:'trained', manifest:null, selectedCheckpoint:null, ready:false, generating:false, stepping:false, generationCount:80, stepPrompt:'', inspection:null, selectedToken:-1, inputChanged:true, timer:null, canvasGeometry:null};
const source = 'https://github.com/elliottbarnes/ellmiott/blob/main/';
const lessonPaths = ['docs/lessons/01-tokenization.md','docs/lessons/02-embeddings.md','docs/lessons/03-attention.md','docs/lessons/04-transformer-block.md','docs/lessons/05-training.md','docs/lessons/06-sampling-and-ablation.md'];
document.querySelectorAll('[data-lesson]').forEach(link => { link.href = source + lessonPaths[Number(link.dataset.lesson)-1]; });
els.prompt.value = 'ROMEO:\n';

function characterLabel(char) { return char === ' ' ? '␣' : char === '\n' ? '↵' : char === '\t' ? '⇥' : char; }
function characterName(char) { return char === ' ' ? 'space' : char === '\n' ? 'newline' : char === '\t' ? 'tab' : char; }
function number(value) { return Number(value).toLocaleString('en-US'); }
function element(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
function options() { return {layer:Number(els.layer.value), head:Number(els.head.value), disabledHead:els['disable-head'].checked ? {layer:Number(els.layer.value),head:Number(els.head.value)} : null, temperature:Number(els.temperature.value), topK:Number(els['top-k'].value)}; }
function post(type, data = {}) { const requestId = ++state.request; state.worker.postMessage({type, requestId, ...data}); return requestId; }

function syncControls() {
  const disabled = !state.ready || state.generating;
  controls.forEach(control => { control.disabled = disabled; });
  els['stop-button'].hidden = !state.generating;
  els['generate-button'].hidden = state.generating;
  document.querySelectorAll('[data-experiment]').forEach(button => { button.disabled = !state.ready || state.generating; });
  checkpointButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.checkpoint === state.checkpoint)));
}

function showError(message) {
  state.generating = false;
  els['error-message'].textContent = message;
  els['error-banner'].hidden = false;
  els['status-dot'].className = 'status-dot error';
  els['model-status'].textContent = 'The model needs a moment';
  els['generated-text'].classList.remove('streaming');
  els['generation-status'].textContent = 'INTERRUPTED';
  syncControls();
}

function createWorker() {
  state.worker?.terminate();
  state.worker = new Worker(new URL('./worker.js', import.meta.url), {type:'module'});
  state.worker.addEventListener('message', handleMessage);
  state.worker.addEventListener('error', () => { state.ready = false; showError('The model worker could not start. Please retry, or reload this page in a modern browser.'); });
}

function clearContinuation(message = 'Inputs changed. Generate a new continuation.') {
  state.generateRequest = 0;
  els['generated-text'].textContent = message;
  els['generated-text'].className = 'generated-text empty';
  els['generation-status'].textContent = 'READY WHEN YOU ARE';
}

function clearInspection() {
  state.inspection = null;
  state.selectedToken = -1;
  state.inputChanged = true;
  state.canvasGeometry = null;
  els.probabilities.replaceChildren(element('p','loading-copy','Waiting for this checkpoint’s prediction…'));
  els['most-likely'].textContent = '—';
  els['most-likely'].setAttribute('aria-label','No prediction');
  els.entropy.textContent = '—';
  els.tokens.replaceChildren(element('span','loading-copy','Input tokens will appear when this checkpoint is ready.'));
  els['token-notice'].textContent = 'Waiting for the selected checkpoint.';
  els['embedding-label'].textContent = 'Token embedding';
  els['embedding-strip'].replaceChildren();
  els['embedding-caption'].textContent = 'First 8 learned dimensions';
  els['attention-detail'].replaceChildren();
  els['attention-summary'].textContent = 'Attention will appear when this checkpoint is ready.';
  const canvas = els['attention-canvas'];
  canvas.width = canvas.width;
  canvas.setAttribute('aria-label','Attention is unavailable while this checkpoint loads.');
  canvas.setAttribute('tabindex','-1');
}

function loadCheckpoint(id) {
  clearTimeout(state.timer);
  state.checkpoint = id;
  state.ready = false;
  state.generating = false;
  state.stepping = false;
  state.selectedCheckpoint = null;
  state.inspectRequest = 0;
  state.generateRequest = 0;
  clearInspection();
  els['error-banner'].hidden = true;
  els['model-status'].textContent = `Loading ${id === 'initial' ? 'untrained' : id} weights…`;
  els['status-dot'].className = 'status-dot loading';
  clearContinuation('One character at a time. See where it goes.');
  if (state.manifest) renderTrainingChart(state.manifest.history,state.manifest.checkpoints);
  syncControls();
  state.loadRequest = post('load', {id});
}

function inspect({immediate = false, resetToken = false} = {}) {
  if (!state.ready || state.generating) return;
  clearTimeout(state.timer);
  if (resetToken) state.inputChanged = true;
  const run = () => {
    if (!state.ready || state.generating) return;
    els['temperature-value'].textContent = Number(els.temperature.value).toFixed(2);
    els['distribution-mode'].textContent = `T = ${Number(els.temperature.value).toFixed(2)}${Number(els['top-k'].value) ? ` / K = ${els['top-k'].value}` : ''}`;
    const ablated = els['disable-head'].checked;
    els['ablation-status'].hidden = !ablated;
    els['ablation-status'].textContent = `Layer ${Number(els.layer.value)+1}, head ${Number(els.head.value)+1} disabled`;
    els['ablation-description'].textContent = ablated ? 'This head’s output is zeroed before projection. The grid still shows its computed attention; it no longer contributes to the prediction.' : 'Remove this head’s contribution and inspect how the next-token probabilities change.';
    state.inspectRequest = post('inspect', {text:els.prompt.value, ...options()});
  };
  if (immediate) run(); else state.timer = setTimeout(run, 180);
}

function handleMessage({data}) {
  if (data.type === 'error') {
    if ([state.loadRequest,state.inspectRequest,state.generateRequest].includes(data.requestId)) showError(data.message);
    return;
  }
  if (data.type === 'ready' && data.requestId === state.loadRequest) {
    state.manifest = data.meta;
    state.selectedCheckpoint = data.checkpoint;
    state.ready = true;
    els['model-status'].textContent = `${data.checkpoint.label || data.checkpoint.id} · step ${number(data.checkpoint.step)}`;
    els['status-dot'].className = 'status-dot';
    els['model-size'].textContent = `${number(data.meta.parameter_count)} parameters · ${data.meta.config.block_size}-token context`;
    populateSelect(els.layer, data.meta.config.n_layer);
    populateSelect(els.head, data.meta.config.n_head);
    [...els['top-k'].options].forEach(option => { option.disabled = Number(option.value) > data.meta.config.vocab_size; });
    renderManifest(data.meta);
    syncControls();
    inspect({immediate:true,resetToken:true});
  } else if (data.type === 'inspection' && data.requestId === state.inspectRequest) {
    state.inspection = data;
    if (state.inputChanged || state.selectedToken >= data.tokens.ids.length || state.selectedToken < 0) state.selectedToken = data.tokens.ids.length - 1;
    state.inputChanged = false;
    renderProbabilities(data);
    renderTokens(data);
    renderSelectedToken();
  } else if (data.type === 'token' && data.requestId === state.generateRequest) {
    els['generated-text'].textContent = data.text;
    els['generated-text'].className = 'generated-text streaming';
    els['generation-status'].textContent = `${data.index} / ${state.generationCount} CHARACTERS`;
  } else if (data.type === 'done' && data.requestId === state.generateRequest) {
    state.generating = false;
    els['generated-text'].textContent = data.text;
    els['generated-text'].className = 'generated-text';
    els['generation-status'].textContent = state.stepping ? '1 CHARACTER ADDED TO PROMPT' : 'COMPLETE';
    syncControls();
    if(state.stepping) { els.prompt.value=state.stepPrompt+data.text; state.stepping=false; inspect({immediate:true,resetToken:true}); }
  }
}

function populateSelect(select, count) {
  if (select.options.length === count) return;
  const selected = Math.min(Number(select.value), count-1);
  select.replaceChildren(...Array.from({length:count}, (_, index) => { const option = element('option', '', String(index+1).padStart(2,'0')); option.value = String(index); return option; }));
  select.value = String(selected);
}

function renderProbabilities(data) {
  const ranked = data.probabilities.map((probability,id) => ({probability, id, char:state.manifest.vocab[id]})).sort((a,b) => b.probability-a.probability);
  const top = ranked.filter(item => item.probability > 0).slice(0,8);
  const max = top[0]?.probability || 1;
  els.probabilities.replaceChildren(...top.map(item => {
    const row = element('div','probability-row');
    row.setAttribute('aria-label', `${characterName(item.char)}: ${(item.probability*100).toFixed(2)} percent`);
    const token = element('span','probability-token',characterLabel(item.char)); token.title = characterName(item.char);
    const track = element('div','probability-track'); const fill = element('div','probability-fill');
    fill.style.width = `${item.probability/max*100}%`; track.append(fill); track.setAttribute('aria-hidden','true');
    const percent = element('span','probability-percent', `${(item.probability*100).toFixed(1)}%`);
    row.append(token,track,percent); return row;
  }));
  els['most-likely'].textContent = top.length ? characterLabel(top[0].char) : '—';
  els['most-likely'].setAttribute('aria-label', top.length ? characterName(top[0].char) : 'No prediction');
  const entropy = -data.probabilities.reduce((sum,p) => sum+(p>0 ? p*Math.log2(p) : 0),0);
  els.entropy.textContent = `${entropy.toFixed(2)} bits`;
}

function renderTokens(data) {
  const count = Array.from(els.prompt.value).length;
  els['context-count'].textContent = `${number(count)} character${count===1?'':'s'}`;
  const notices = [];
  if (data.tokens.truncated) notices.push(`Using the last ${data.tokens.ids.length} characters; ${data.tokens.truncated} earlier characters are outside the context.`);
  if (data.tokens.unknown) notices.push(`${data.tokens.unknown} unsupported character${data.tokens.unknown===1?' was':'s were'} replaced by a vocabulary fallback.`);
  if (!count) notices.push('Empty input uses a newline to start the model.');
  els['token-notice'].textContent = notices.join(' ') || 'Select a token to inspect its embedding and attention row.';
  els.tokens.replaceChildren(...data.tokens.labels.map((char,index) => {
    const button = element('button','token',characterLabel(char)); button.type = 'button'; button.dataset.index = index;
    button.setAttribute('aria-label', `Token ${index+1}: ${characterName(char)}, vocabulary ID ${data.tokens.ids[index]}`);
    button.setAttribute('aria-pressed',String(index===state.selectedToken));
    button.title = `Position ${index+1} · ID ${data.tokens.ids[index]} · ${characterName(char)}`;
    button.addEventListener('click',() => { state.selectedToken = index; renderSelectedToken(); });
    return button;
  }));
}

function renderSelectedToken() {
  const data = state.inspection;
  if (!data) return;
  const i = state.selectedToken;
  els.tokens.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed',String(Number(button.dataset.index)===i)));
  els['embedding-label'].textContent = `${characterLabel(data.tokens.labels[i])} · token ID ${data.tokens.ids[i]}`;
  const values = data.embeddings[i] || [];
  const max = Math.max(.001,...values.map(Math.abs));
  els['embedding-strip'].replaceChildren(...values.map((value,dim) => {
    const cell = element('span','embedding-value',value.toFixed(2));
    const strength = Math.abs(value)/max;
    cell.style.background = value >= 0 ? `rgba(34,82,219,${.08+strength*.72})` : `rgba(63,66,58,${.07+strength*.63})`;
    const base=value>=0?[34,82,219]:[63,66,58],alpha=value>=0?.08+strength*.72:.07+strength*.63;
    const blended=base.map((channel,index)=>channel*alpha+[255,254,250][index]*(1-alpha));
    const linear=blended.map(channel=>{const v=channel/255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;});
    const luminance=.2126*linear[0]+.7152*linear[1]+.0722*linear[2];
    cell.style.color = (1.05/(luminance+.05))>=4.5 ? '#ffffff' : '#000000';
    cell.title = `Dimension ${dim+1}: ${value.toFixed(6)}`;
    cell.setAttribute('aria-label',cell.title);
    return cell;
  }));
  els['embedding-caption'].textContent = `First ${values.length} of ${state.manifest.config.n_embd} dimensions · blue + / graphite −`;
  drawAttention();
  const row = data.attention[i] || [];
  const ranked = row.map((weight,index) => ({weight,index})).filter(item => item.index<=i).sort((a,b) => b.weight-a.weight).slice(0,5);
  const top = ranked[0];
  els['attention-summary'].textContent = `Token ${i+1} (${characterName(data.tokens.labels[i])}) attends most to ${top ? `token ${top.index+1} (${characterName(data.tokens.labels[top.index])}), ${(top.weight*100).toFixed(1)}%` : '—'}. Colour uses a square-root scale; future positions are masked.`;
  els['attention-detail'].replaceChildren(...ranked.map(item => element('span','',`${item.index+1} ${characterLabel(data.tokens.labels[item.index])} · ${(item.weight*100).toFixed(1)}%`)));
}

function drawAttention() {
  const data = state.inspection;
  if (!data) return;
  const canvas = els['attention-canvas'];
  const width = Math.max(180,canvas.getBoundingClientRect().width);
  const height = Math.min(300,Math.max(230,data.tokens.ids.length*11));
  const ratio = Math.min(window.devicePixelRatio || 1,2);
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(ratio,ratio);
  const n = data.tokens.ids.length;
  const cell = Math.min((width-35)/n,(height-31)/n,39);
  const x = (width-cell*n+15)/2, y = (height-cell*n+19)/2;
  state.canvasGeometry = {x,y,cell,n};
  ctx.fillStyle = '#fffefa'; ctx.fillRect(0,0,width,height);
  const stride = Math.ceil(n/14);
  for (let r=0;r<n;r++) {
    for (let c=0;c<n;c++) {
      const value = data.attention[r][c];
      ctx.fillStyle = c>r ? '#f0f1e9' : `rgba(34,82,219,${.05+Math.sqrt(Math.max(0,value))*.95})`;
      const gap = n>35 ? .7 : 1.5;
      ctx.fillRect(x+c*cell,y+r*cell,cell-gap,cell-gap);
    }
    if (r%stride===0 || r===n-1 || r===state.selectedToken) {
      ctx.font = `${n>20?8:10}px ui-monospace, SFMono-Regular, monospace`;
      ctx.fillStyle = r===state.selectedToken ? '#2252db' : '#686b64';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(characterLabel(data.tokens.labels[r]),x+(r+.5)*cell,y-10);
      ctx.textAlign = 'right';
      ctx.fillText(characterLabel(data.tokens.labels[r]),x-8,y+(r+.5)*cell);
    }
  }
  ctx.strokeStyle = '#242622'; ctx.lineWidth = 1.2;
  ctx.strokeRect(x-2,y+state.selectedToken*cell-1,(state.selectedToken+1)*cell+2,cell+1);
  canvas.setAttribute('aria-label', `Attention for layer ${data.layer+1}, head ${data.head+1}. ${n} tokens. Row ${state.selectedToken+1} selected. Use up and down arrow keys to select a row; exact weights are listed below.`);
  canvas.setAttribute('tabindex','0');
}

function renderManifest(meta) {
  els['parameter-count'].textContent = number(meta.parameter_count);
  const config = meta.config;
  els['architecture-summary'].textContent = `${config.n_layer} layers · ${config.n_head} heads · ${config.n_embd} embedding dimensions`;
  els['training-steps'].textContent = number(meta.training.steps);
  els['training-device'].textContent = `${String(meta.training.device).toUpperCase()} · seed ${meta.training.seed}${Number.isFinite(meta.training.seconds) ? ` · ${(meta.training.seconds/60).toFixed(1)} minutes` : ''}`;
  els['corpus-link'].textContent = meta.corpus.title;
  if (typeof meta.corpus.url === 'string' && /^https:\/\//.test(meta.corpus.url)) els['corpus-link'].href = meta.corpus.url;
  els['corpus-size'].textContent = `${number(meta.corpus.train_characters)} training / ${number(meta.corpus.validation_characters)} held-out characters`;
  renderTrainingChart(meta.history,meta.checkpoints);
}

function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg',tag);
  Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key,String(value)));
  if (text!==undefined) node.textContent = text;
  return node;
}

function renderTrainingChart(history, checkpoints) {
  const rows = history.filter(row => Number.isFinite(row.step) && Number.isFinite(row.train_loss) && Number.isFinite(row.validation_loss));
  if (!rows.length) { els['training-chart'].replaceChildren(element('p','loading-copy','No recorded training measurements are available.')); return; }
  const width=560,height=223,pad={l:37,r:12,t:20,b:30};
  const all=rows.flatMap(row => [row.train_loss,row.validation_loss]);
  const min=Math.floor((Math.min(...all)-.15)*2)/2,max=Math.ceil((Math.max(...all)+.15)*2)/2;
  const maxStep=Math.max(1,...rows.map(row => row.step));
  const x=step => pad.l+step/maxStep*(width-pad.l-pad.r),y=loss => pad.t+(max-loss)/(max-min)*(height-pad.t-pad.b);
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-labelledby':'loss-chart-title loss-chart-description'});
  svg.append(svgElement('title',{id:'loss-chart-title'},'Recorded training and validation loss'),svgElement('desc',{id:'loss-chart-description'},`${rows.length} measurements across ${number(maxStep)} steps. Initial validation loss ${rows[0].validation_loss.toFixed(3)}; final ${rows.at(-1).validation_loss.toFixed(3)}. The solid blue curve is training loss and the dashed graphite curve is validation loss.`));
  for(let tick=0;tick<5;tick++) {
    const value=min+(max-min)*tick/4,py=y(value);
    svg.append(svgElement('line',{x1:pad.l,y1:py,x2:width-pad.r,y2:py,stroke:'#e3e5dc','stroke-width':1}),svgElement('text',{x:pad.l-10,y:py+3,fill:'#686b64','font-family':'ui-monospace,monospace','font-size':9,'text-anchor':'end'},value.toFixed(1)));
  }
  for(let tick=0;tick<5;tick++) { const step=maxStep*tick/4;svg.append(svgElement('text',{x:x(step),y:height-12,fill:'#686b64','font-family':'ui-monospace,monospace','font-size':9,'text-anchor':'middle'},number(Math.round(step)))); }
  for(const [key,colour,dash] of [['train_loss','#2252db',''],['validation_loss','#727864','5 4']]) {
    const path=rows.map((row,index)=>`${index?'L':'M'}${x(row.step).toFixed(2)},${y(row[key]).toFixed(2)}`).join(' ');
    svg.append(svgElement('path',{d:path,fill:'none',stroke:colour,'stroke-width':2,'stroke-dasharray':dash,'stroke-linejoin':'round'}));
  }
  const selected=checkpoints.find(checkpoint=>checkpoint.id===state.checkpoint);
  if(selected && Number.isFinite(selected.validation_loss)) {
    const px=x(selected.step),py=y(selected.validation_loss);
    svg.append(svgElement('line',{x1:px,y1:pad.t,x2:px,y2:height-pad.b,stroke:'#2252db','stroke-width':1,opacity:.3,'stroke-dasharray':'3 4'}),svgElement('circle',{cx:px,cy:py,r:4,fill:'#fffefa',stroke:'#2252db','stroke-width':2}));
  }
  els['training-chart'].replaceChildren(svg);
}

function generate(step = false) {
  if (!state.ready || state.generating) return;
  if (!els.seed.reportValidity()) return;
  clearTimeout(state.timer);
  inspect({immediate:true});
  state.generating=true; state.stepping=step; state.stepPrompt=els.prompt.value || '\n';
  state.generationCount=step?1:Number(els['generation-length'].value);
  els['error-banner'].hidden=true;
  els['generated-text'].textContent=''; els['generated-text'].className='generated-text streaming';
  els['generation-status'].textContent='GENERATING…';
  syncControls();
  state.generateRequest=post('generate',{text:els.prompt.value,count:state.generationCount,seed:Number(els.seed.value),...options()});
}

function stop() {
  if(!state.generating) return;
  post('cancel'); state.generateRequest=0; state.generating=false; state.stepping=false;
  els['generated-text'].classList.remove('streaming'); els['generation-status'].textContent='STOPPED';
  if(!els['generated-text'].textContent) { els['generated-text'].textContent='Stopped before the first character.'; els['generated-text'].classList.add('empty'); }
  syncControls();
}

function runExperiment(id) {
  if(!state.ready || state.generating) return;
  clearContinuation();
  let message='',target='lab',checkpoint='trained';
  els['disable-head'].checked=false; els.temperature.value='0.8'; els['top-k'].value='0'; els.seed.value='42';
  if(id==='tokens') { els.prompt.value='The '; message='Experiment 01: select T, h, e, and the space in the token strip. Repeated characters share a token embedding; position embeddings are added separately inside the model.'; }
  if(id==='embeddings') { els.prompt.value='letter'; message='Experiment 02: select the e at positions 2 and 5. Their token embeddings are identical. The model separately adds a position embedding so those two positions can behave differently.'; }
  if(id==='context') { els.prompt.value='The king '; message='Experiment 03: inspect this prompt, then shorten it to “The ”. The final token is still a space, but the surrounding context changes the prediction.'; }
  if(id==='temperature') { els.prompt.value='ROMEO:\n'; els.temperature.value='0.35'; message='Experiment 06: generate at temperature 0.35. Then raise it to 1.50 and generate again. Keep seed 42 so the change comes from the sampling distribution.'; }
  if(id==='ablation') { els.prompt.value='The king '; els.layer.value='0'; els.head.value='0'; els['disable-head'].checked=true; target='attention'; message='Experiment 04: layer 1, head 1 now contributes zero to the model. Toggle it back on and compare the probabilities above. Its attention weights remain visible for inspection.'; }
  if(id==='learning') { els.prompt.value='ROMEO:\n'; checkpoint='early'; target='training'; message='Experiment 05: the early checkpoint is selected. Compare its validation loss and generated text with Untrained and Trained. The chart records the actual training run.'; }
  els['experiment-notice'].textContent=message; els['experiment-notice'].hidden=false;
  els['temperature-value'].textContent=Number(els.temperature.value).toFixed(2);
  if(state.checkpoint!==checkpoint) loadCheckpoint(checkpoint); else inspect({immediate:true,resetToken:true});
  $(target).scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
}

checkpointButtons.forEach(button=>button.addEventListener('click',()=>loadCheckpoint(button.dataset.checkpoint)));
els.prompt.addEventListener('input',()=>{clearContinuation();inspect({resetToken:true});});
els.prompt.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();generate();}});
els.temperature.addEventListener('input',()=>{clearContinuation();els['temperature-value'].textContent=Number(els.temperature.value).toFixed(2);inspect();});
for(const input of [els['top-k'],els.layer,els.head,els['disable-head']]) input.addEventListener('change',()=>{clearContinuation();inspect({immediate:true});});
for(const input of [els.seed,els['generation-length']]) input.addEventListener('input',()=>clearContinuation());
els['inspect-button'].addEventListener('click',()=>inspect({immediate:true}));
els['generate-button'].addEventListener('click',()=>generate());
els['step-button'].addEventListener('click',()=>generate(true));
els['stop-button'].addEventListener('click',stop);
els['retry-button'].addEventListener('click',()=>{createWorker();loadCheckpoint(state.checkpoint);});
document.querySelectorAll('[data-experiment]').forEach(button=>button.addEventListener('click',()=>runExperiment(button.dataset.experiment)));
els['attention-canvas'].addEventListener('click',event=>{
  if(!state.canvasGeometry||!state.inspection)return;
  const rect=els['attention-canvas'].getBoundingClientRect(),{y,cell,n}=state.canvasGeometry;
  const row=Math.floor((event.clientY-rect.top-y)/cell);
  if(row>=0&&row<n){state.selectedToken=row;renderSelectedToken();}
});
els['attention-canvas'].addEventListener('keydown',event=>{
  if(!state.inspection)return;
  const max=state.inspection.tokens.ids.length-1;
  if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
    event.preventDefault();
    state.selectedToken=event.key==='Home'?0:event.key==='End'?max:Math.max(0,Math.min(max,state.selectedToken+(['ArrowUp','ArrowLeft'].includes(event.key)?-1:1)));
    renderSelectedToken();
  }
});
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawAttention,80);});
window.addEventListener('pagehide',()=>state.worker?.terminate());
window.addEventListener('pageshow',event=>{if(event.persisted){state.generating=false;state.stepping=false;createWorker();loadCheckpoint(state.checkpoint);}});
syncControls();
try { createWorker();loadCheckpoint('trained'); }
catch { showError('This browser could not start the model. Please use a modern browser with JavaScript and module workers enabled.'); }
