import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {TinyGPT} from '../site/engine.js';

// Exercise the real app event handlers at their DOM/worker boundary. This small
// DOM double deliberately does not claim to test browser layout or rendering.
const html = await readFile(new URL('../site/index.html', import.meta.url), 'utf8');
const manifest = JSON.parse(await readFile(new URL('../site/models/manifest.json', import.meta.url), 'utf8'));
const initial = new TinyGPT(JSON.parse(await readFile(new URL('../site/models/initial.json', import.meta.url), 'utf8')));
let harnessNumber = 0;

class NodeDouble {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.style = {};
    this.listeners = new Map();
    this.className = '';
    this.value = '';
    this.checked = false;
    this.hidden = false;
    this.width = 620;
    this.height = 360;
    this.classList = {
      add: name => { this.className = `${this.className} ${name}`.trim(); },
      remove: name => { this.className = this.className.split(' ').filter(item => item !== name).join(' '); },
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
  get options() { return this.children; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  fire(type, event = {}) { for (const listener of this.listeners.get(type) || []) listener({preventDefault() {}, ...event}); }
  querySelectorAll(tag) { return this.children.flatMap(child => [...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag)]); }
  reportValidity() { return true; }
  scrollIntoView() {}
  getBoundingClientRect() { return {width: 480, height: 260, left: 0, top: 0}; }
  getContext() { return {scale() {}, fillRect() {}, fillText() {}, strokeRect() {}}; }
}

async function harness(t) {
  const saved = Object.fromEntries(['document', 'window', 'Worker'].map(key => [key, globalThis[key]]));
  const nodes = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new NodeDouble()]));
  for (const [id, value] of Object.entries({temperature:'0.8', 'top-k':'0', seed:'42', 'generation-length':'80', layer:'0', head:'0'})) nodes[id].value = value;
  for (const id of ['layer','head']) nodes[id].append(new NodeDouble('option'));
  for (const value of ['0','5','10','20']) { const option = new NodeDouble('option'); option.value = value; nodes['top-k'].append(option); }
  const checkpoints = ['initial','early','trained','custom'].map(id => { const node = new NodeDouble('button'); node.dataset.checkpoint = id; return node; });
  const lessons = Array.from({length:6}, (_,i) => { const node = new NodeDouble('a'); node.dataset.lesson = String(i+1); return node; });
  const experiments = ['tokens','embeddings','context','ablation','learning','temperature'].map(id => { const node = new NodeDouble('button'); node.dataset.experiment = id; return node; });
  globalThis.document = {
    getElementById: id => nodes[id],
    createElement: tag => new NodeDouble(tag),
    createElementNS: (_,tag) => new NodeDouble(tag),
    querySelectorAll: selector => ({'[data-checkpoint]':checkpoints,'[data-lesson]':lessons,'[data-experiment]':experiments}[selector] || []),
  };
  const windowEvents = new Map();
  globalThis.window = {devicePixelRatio:1, addEventListener(type,listener) {windowEvents.set(type,listener);}, matchMedia:() => ({matches:true})};
  let worker;
  globalThis.Worker = class {
    constructor() { this.messages = []; this.listeners = {}; worker = this; }
    addEventListener(type, listener) { this.listeners[type] = listener; }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
    respond(data) { this.listeners.message({data}); }
  };
  await import(`../site/app.js?ui-state-test=${++harnessNumber}`);
  const latest = type => worker.messages.findLast(message => message.type === type);
  worker.respond({type:'ready', requestId:latest('load').requestId, meta:manifest, checkpoint:manifest.checkpoints.find(item => item.id === 'trained')});
  const inspection = initial.inspect('ROMEO:\n', {temperature:.8});
  worker.respond({type:'inspection', requestId:latest('inspect').requestId, ...inspection});
  t.after(() => {
    // Cancels any pending input debounce before removing the DOM double.
    checkpoints[0].fire('click');
    for (const [key,value] of Object.entries(saved)) {
      if (value === undefined) delete globalThis[key]; else globalThis[key] = value;
    }
  });
  return {nodes, checkpoints, experiments, worker, latest, inspection, windowEvents};
}

test('stepping from an empty prompt retains the implicit newline context', async t => {
  const {nodes, worker, latest} = await harness(t);
  nodes.prompt.value = '';
  nodes['step-button'].fire('click');
  const request = latest('generate');
  assert.equal(request.text, '');
  assert.equal(request.count, 1);
  worker.respond({type:'done', requestId:request.requestId, text:'A'});
  assert.equal(nodes.prompt.value, '\nA');
  assert.equal(latest('inspect').text, '\nA');
  assert.equal(nodes['generation-status'].textContent, '1 CHARACTER ADDED TO PROMPT');
});

test('a checkpoint switch clears old inference, ignores stale replies, and stays clear on load failure', async t => {
  const {nodes, checkpoints, worker, latest, inspection} = await harness(t);
  assert.ok(nodes.tokens.querySelectorAll('button').length > 0);
  const oldRequest = latest('inspect');
  checkpoints[0].fire('click');
  assert.match(nodes.probabilities.textContent, /Waiting for this checkpoint/);
  assert.equal(nodes.tokens.querySelectorAll('button').length, 0);
  assert.equal(nodes['embedding-strip'].children.length, 0);
  assert.equal(nodes['attention-detail'].children.length, 0);
  assert.equal(nodes['attention-canvas'].attributes.tabindex, '-1');
  worker.respond({type:'inspection', requestId:oldRequest.requestId, ...inspection});
  assert.equal(nodes.tokens.querySelectorAll('button').length, 0);
  worker.respond({type:'error', requestId:latest('load').requestId, message:'Download failed'});
  assert.match(nodes['error-message'].textContent, /Download failed/);
  assert.equal(nodes['generate-button'].disabled, true);
  assert.equal(nodes['most-likely'].textContent, '—');
  assert.equal(nodes['attention-detail'].children.length, 0);
  assert.match(nodes['attention-summary'].textContent, /when this checkpoint is ready/);
});

test('editing prompt, sampling, or head settings clears the previous continuation', async t => {
  const {nodes, worker, latest, experiments} = await harness(t);
  const completeGeneration = () => {
    nodes['generate-button'].fire('click');
    worker.respond({type:'done', requestId:latest('generate').requestId, text:'previous output'});
    assert.equal(nodes['generated-text'].textContent, 'previous output');
  };
  for (const [id,event] of [['prompt','input'],['temperature','input'],['top-k','change'],['layer','change'],['head','change'],['disable-head','change'],['seed','input'],['generation-length','input']]) {
    completeGeneration();
    nodes[id].fire(event);
    assert.match(nodes['generated-text'].textContent, /Inputs changed/, id);
    assert.match(nodes['generated-text'].className, /empty/, id);
  }
  completeGeneration();
  experiments[1].fire('click');
  assert.match(nodes['generated-text'].textContent, /Inputs changed/);
});


test('custom checkpoint survives stage changes and uses its own empty-prompt character', async t => {
  const {nodes, checkpoints, worker, latest, windowEvents}=await harness(t);
  assert.equal(checkpoints[3].disabled,true);
  const checkpoint={vocab:['α','β'],step:10};
  windowEvents.get('ellmiott:custom-model')({detail:{checkpoint}});
  assert.equal(latest('load-custom').checkpoint,checkpoint);
  const meta={...manifest,vocab:checkpoint.vocab,config:{...manifest.config,n_layer:1,vocab_size:2},history:[],checkpoints:[{id:'custom',step:10}],corpus:{title:'Your custom text',url:''}};
  worker.respond({type:'ready',requestId:latest('load-custom').requestId,meta,checkpoint:meta.checkpoints[0]});
  assert.equal(nodes['corpus-link'].href,'#train-your-own');
  assert.equal(checkpoints[3].disabled,false);
  nodes.prompt.value='';nodes['step-button'].fire('click');
  worker.respond({type:'done',requestId:latest('generate').requestId,text:'β'});
  assert.equal(nodes.prompt.value,'αβ');
  checkpoints[0].fire('click');
  worker.respond({type:'ready',requestId:latest('load').requestId,meta:manifest,checkpoint:manifest.checkpoints[0]});
  checkpoints[3].fire('click');assert.equal(latest('load-custom').checkpoint,checkpoint);
});
