import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {TinyGPT, distribution, randomSource, sample, softmax} from '../site/engine.js';

const json = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const manifest = await json('../site/models/manifest.json');
const models = new Map();
for (const checkpoint of manifest.checkpoints) models.set(checkpoint.id, new TinyGPT(await json(`../site/models/${checkpoint.file}`)));
const model = models.get('trained');

test('all published model shapes match the declared parameter count', () => {
  assert.deepEqual([...models.keys()], ['initial','early','trained']);
  for (const current of models.values()) assert.equal(current.parameterCount, manifest.parameter_count);
});

test('JavaScript logits and attention match independently computed PyTorch fixtures', async t => {
  const fixture = await json('./fixtures/parity.json');
  for (const entry of fixture.cases) await t.test(`${entry.checkpoint}: ${JSON.stringify(entry.prompt)} ${entry.disabledHead ? 'ablated' : 'intact'}`, () => {
    const current = models.get(entry.checkpoint);
    assert.deepEqual(current.tokenize(entry.prompt).ids, entry.ids);
    const actual = current.forward(entry.ids, entry);
    assert.equal(actual.logits.length, entry.logits.length);
    const logitError = Math.max(...actual.logits.map((v,i) => Math.abs(v-entry.logits[i])));
    assert.ok(logitError < 2e-4, `Maximum logit error ${logitError}`);
    assert.equal(actual.attention.length, entry.attention.length);
    let attentionError = 0;
    for (let r=0;r<actual.attention.length;r++) for (let c=0;c<actual.attention[r].length;c++)
      attentionError = Math.max(attentionError, Math.abs(actual.attention[r][c]-entry.attention[r][c]));
    assert.ok(attentionError < 2e-5, `Maximum attention error ${attentionError}`);
  });
});

test('attention obeys the causal mask and each row sums to one', () => {
  const ids = model.tokenize('ROMEO:\nWhat').ids;
  for (let layer=0;layer<model.config.n_layer;layer++) for (let head=0;head<model.config.n_head;head++) {
    const {attention} = model.forward(ids,{layer,head});
    for (let r=0;r<ids.length;r++) {
      assert.ok(Math.abs(attention[r].reduce((a,b)=>a+b,0)-1)<1e-6);
      for (let c=r+1;c<ids.length;c++) assert.equal(attention[r][c],0);
    }
  }
});

test('future tokens cannot change earlier attention rows', () => {
  const prefix = model.tokenize('ROMEO:').ids;
  const extended = model.tokenize('ROMEO: Hello').ids;
  for (let layer=0;layer<model.config.n_layer;layer++) {
    const a=model.forward(prefix,{layer}).attention, b=model.forward(extended,{layer}).attention;
    for (let r=0;r<prefix.length;r++) for(let c=0;c<prefix.length;c++) assert.ok(Math.abs(a[r][c]-b[r][c])<1e-6);
  }
});

test('head ablation changes predictions without mutating the model', () => {
  const ids=model.tokenize('ROMEO:\nWhat is').ids;
  const original=model.forward(ids).logits;
  const altered=model.forward(ids,{disabledHead:{layer:0,head:0}}).logits;
  assert.ok(Math.max(...original.map((v,i)=>Math.abs(v-altered[i])))>1e-4);
  assert.deepEqual(model.forward(ids).logits,original);
});

test('context truncation, unsupported characters and empty prompts are explicit', () => {
  assert.equal(model.tokenize('a'.repeat(80)).ids.length,64);
  assert.equal(model.tokenize('a'.repeat(80)).truncated,16);
  assert.equal(model.tokenize('🧠').unknown,1);
  assert.equal(model.tokenize('').ids.length,1);
  assert.ok(model.tokenize('🧠').ids.every(Number.isInteger));
});

test('sampling is stable, seeded, respects top-k, and supports greedy decoding', () => {
  const logits=[1000,1001,999];
  assert.ok(Math.abs(softmax(logits).reduce((a,b)=>a+b,0)-1)<1e-12);
  assert.deepEqual(distribution(logits,0),[0,1,0]);
  assert.deepEqual(distribution(logits,0.8,1),[0,1,0]);
  assert.deepEqual(distribution(logits,Number.MIN_VALUE),[0,1,0]);
  const topTwo=distribution(logits,0.8,2); assert.equal(topTwo[2],0);
  const a=randomSource(42), b=randomSource(42);
  assert.deepEqual(Array.from({length:100},()=>sample(topTwo,a)),Array.from({length:100},()=>sample(topTwo,b)));
  assert.throws(()=>distribution(logits,-1),/Temperature/);
  assert.throws(()=>distribution(logits,1,4),/Top-k/);
});

test('invalid dimensions, weights, tokens, and heads fail with useful errors', async () => {
  const raw=await json('../site/models/initial.json');
  assert.throws(()=>new TinyGPT({...raw,format:'unknown'}),/format/);
  assert.throws(()=>new TinyGPT({...raw,config:{...raw.config,n_embd:1e9}}),/dimensions/);
  assert.throws(()=>new TinyGPT({...raw,tensors:{}}),/tensors/);
  const invalid=structuredClone(raw); invalid.tensors['ln_f.bias'].data[0]=NaN;
  assert.throws(()=>new TinyGPT(invalid),/tensor/);
  assert.throws(()=>model.forward([999]),/token/);
  assert.throws(()=>model.forward([1],{head:999}),/head/);
  assert.throws(()=>model.forward([1],{disabledHead:{layer:999,head:0}}),/ablation/);
});
