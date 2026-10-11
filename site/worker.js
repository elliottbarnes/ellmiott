import {TinyGPT, distribution, randomSource, sample} from './engine.js';

import {customManifest} from './trainer-core.js';

let model = null;
let meta = null;
let activeJob = 0;
const cache = new Map();
const reply = (requestId, type, data = {}) => self.postMessage({requestId, type, ...data});

async function readJSON(path) {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status}). Try reloading the page.`);
  return response.json();
}

self.onmessage = async ({data}) => {
  const {requestId, type} = data;
  try {
    if (type === 'cancel') { activeJob++; return; }
    if (type === 'load-custom') {
      ++activeJob;
      const customMeta = customManifest(data.checkpoint);
      model = new TinyGPT(data.checkpoint);
      reply(requestId, 'ready', {meta:customMeta, checkpoint:customMeta.checkpoints[0]});
      return;
    }
    if (type === 'load') {
      const job = ++activeJob;
      meta ??= await readJSON('./models/manifest.json');
      const checkpoint = meta.checkpoints.find(item => item.id === data.id);
      if (!checkpoint || !/^[a-z]+\.json$/.test(checkpoint.file)) throw new Error('Unknown checkpoint.');
      if (!cache.has(data.id)) cache.set(data.id, new TinyGPT(await readJSON(`./models/${checkpoint.file}`)));
      if (job !== activeJob) return;
      model = cache.get(data.id);
      reply(requestId, 'ready', {meta, checkpoint});
      return;
    }
    if (!model) throw new Error('Wait for the model to finish loading.');
    if (type === 'inspect') {
      const inspection = model.inspect(data.text, data);
      reply(requestId, 'inspection', inspection);
    } else if (type === 'generate') {
      if (!Number.isInteger(data.count) || data.count < 1 || data.count > 300)
        throw new Error('Generate between 1 and 300 characters at a time.');
      const job = ++activeJob, random = randomSource(data.seed);
      let context = model.tokenize(data.text).labels.join(''), text = '';
      const selectedModel = model;
      for (let i = 0; i < data.count; i++) {
        if (job !== activeJob) return;
        const {logits} = selectedModel.forward(selectedModel.tokenize(context).ids, {disabledHead: data.disabledHead});
        const id = sample(distribution(logits, data.temperature ?? 0.8, data.topK ?? 0), random);
        const token = selectedModel.vocab[id]; text += token; context += token;
        reply(requestId, 'token', {text, token, index: i+1});
        // Yield so a stop or checkpoint change can interrupt a long continuation.
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      reply(requestId, 'done', {text});
    } else throw new Error('Unknown model operation.');
  } catch (error) { reply(requestId, 'error', {message: error.message || 'The model could not complete this operation.'}); }
};
