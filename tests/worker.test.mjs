import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';

// Exercise the production worker protocol with Node's isolated worker context.
// Only transport is adapted: same-site fetch reads the shipped local assets.
function startWorker() {
  const worker = new Worker(`
    const {parentPort} = require('node:worker_threads');
    const {readFile} = require('node:fs/promises');
    globalThis.self = {postMessage: data => parentPort.postMessage(data)};
    globalThis.fetch = async url => ({ok:true,json:async()=>JSON.parse(await readFile(url,'utf8'))});
    import(${JSON.stringify(new URL('../site/worker.js', import.meta.url).href)}).then(()=>{
      parentPort.on('message',data=>self.onmessage({data}));
      parentPort.postMessage({type:'booted'});
    });
  `, {eval:true});
  return worker;
}
function waitFor(worker, predicate) {
  return new Promise((resolve,reject) => {
    const timer=setTimeout(()=>{cleanup();reject(new Error('Worker did not respond'));},10000);
    const onError=error=>{cleanup();reject(error);};
    const onMessage=message=>{if(predicate(message)){cleanup();resolve(message);}};
    function cleanup(){clearTimeout(timer);worker.off('message',onMessage);worker.off('error',onError);}
    worker.on('message',onMessage);worker.on('error',onError);
  });
}
async function request(worker, payload, resultType) {
  const result=waitFor(worker,message=>message.requestId===payload.requestId && (message.type===resultType || message.type==='error'));
  worker.postMessage(payload);return result;
}

test('production worker loads, inspects, generates reproducibly, switches stages and reports errors', async () => {
  const worker=startWorker();
  try {
    await waitFor(worker,message=>message.type==='booted');
    const before=await request(worker,{requestId:0,type:'inspect',text:'ROMEO:'},'inspection');
    assert.equal(before.type,'error');
    const ready=await request(worker,{requestId:1,type:'load',id:'trained'},'ready');
    assert.equal(ready.type,'ready');assert.equal(ready.checkpoint.id,'trained');
    const insight=await request(worker,{requestId:2,type:'inspect',text:'ROMEO:',layer:1,head:3,temperature:0.8,topK:5},'inspection');
    assert.equal(insight.type,'inspection');assert.equal(insight.tokens.ids.length,6);
    assert.equal(insight.probabilities.filter(p=>p>0).length,5);
    const options={type:'generate',text:'ROMEO:\n',count:16,temperature:0.8,topK:10,seed:42};
    const first=await request(worker,{requestId:3,...options},'done');
    const second=await request(worker,{requestId:4,...options},'done');
    assert.equal(first.type,'done');assert.equal(first.text.length,16);assert.equal(first.text,second.text);
    const early=await request(worker,{requestId:5,type:'load',id:'early'},'ready');
    assert.equal(early.checkpoint.id,'early');
    const bad=await request(worker,{requestId:6,type:'generate',count:301},'done');assert.equal(bad.type,'error');
    const missing=await request(worker,{requestId:7,type:'load',id:'missing'},'ready');assert.equal(missing.type,'error');
  } finally {await worker.terminate();}
});

test('cancel interrupts a long continuation and leaves the worker usable', async () => {
  const worker=startWorker(); const received=[];
  try {
    await waitFor(worker,message=>message.type==='booted');
    await request(worker,{requestId:1,type:'load',id:'trained'},'ready');
    worker.on('message',message=>received.push(message));
    const token=waitFor(worker,message=>message.requestId===2 && message.type==='token');
    worker.postMessage({requestId:2,type:'generate',text:'ROMEO:\n',count:300});await token;
    worker.postMessage({requestId:3,type:'cancel'});
    const inspection=await request(worker,{requestId:4,type:'inspect',text:'The king'},'inspection');
    assert.equal(inspection.type,'inspection');
    assert.ok(received.filter(m=>m.requestId===2 && m.type==='token').length<300);
    assert.ok(!received.some(m=>m.requestId===2 && m.type==='done'));
  } finally {await worker.terminate();}
});

test('custom import switches inference and preserves the published checkpoint cache', async()=>{
  const {readFile}=await import('node:fs/promises');
  const cp=JSON.parse(await readFile(new URL('../site/models/initial.json',import.meta.url),'utf8'));
  cp.vocab=cp.vocab.map((_,i)=>String.fromCodePoint(0x3b1+i));
  const worker=startWorker();
  try{
    await waitFor(worker,m=>m.type==='booted');
    await request(worker,{requestId:1,type:'load',id:'trained'},'ready');
    const custom=await request(worker,{requestId:2,type:'load-custom',checkpoint:cp},'ready');
    assert.equal(custom.meta.corpus.title,'Your custom text');assert.equal(custom.meta.corpus.url,'');assert.equal(custom.meta.history.length,0);
    const result=await request(worker,{requestId:3,type:'generate',text:cp.vocab[0],count:8},'done');
    assert.equal(Array.from(result.text).length,8);assert.ok(Array.from(result.text).every(c=>cp.vocab.includes(c)));
    const published=await request(worker,{requestId:4,type:'load',id:'trained'},'ready');
    assert.equal(published.meta.corpus.title,'Tiny Shakespeare');
    cp.tensors['lm_head.weight'].data[0]=null;
    const bad=await request(worker,{requestId:5,type:'load-custom',checkpoint:cp},'ready');assert.equal(bad.type,'error');
  }finally{await worker.terminate();}
});
