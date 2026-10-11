import {FORMAT, randomSource, TinyGPT} from './engine.js';

export function prepareCorpus(text) {
  const chars = Array.from(text);
  if (chars.length < 400 || chars.length > 200000) throw new Error('Use 400–200,000 characters of UTF-8 text.');
  if (text.includes('\0')) throw new Error('This looks like a binary file. Choose plain UTF-8 text.');
  const vocab = [...new Set(chars)].sort();
  if (vocab.length < 2 || vocab.length > 256) throw new Error('Use between 2 and 256 distinct characters.');
  const lookup = new Map(vocab.map((c,i) => [c,i]));
  const ids = chars.map(c => lookup.get(c)), split = Math.floor(ids.length * .9);
  return {vocab, train:ids.slice(0,split), validation:ids.slice(split), characters:ids.length};
}

// The same pre-norm, causal decoder used by engine.js, with automatic gradients.
export function createTrainer(tf, corpus, seed = 42) {
  const config = {vocab_size:corpus.vocab.length, block_size:32, n_embd:32, n_head:4, n_layer:1};
  const weights = {}, C = config.n_embd, V = config.vocab_size, T = config.block_size;
  let serial = 0;
  const add = (name, shape, kind='normal') => {
    const initial = kind === 'one' ? tf.ones(shape) : kind === 'zero' ? tf.zeros(shape) : tf.randomNormal(shape,0,.02,'float32',seed + serial++);
    weights[name] = tf.variable(initial); initial.dispose();
  };
  add('token_embedding.weight',[V,C]); add('position_embedding.weight',[T,C]);
  for (const name of ['blocks.0.ln1','blocks.0.ln2','ln_f']) { add(name+'.weight',[C],'one'); add(name+'.bias',[C],'zero'); }
  for (const [name,out,input] of [['attn.qkv',3*C,C],['attn.proj',C,C],['mlp.fc',4*C,C],['mlp.proj',C,4*C]]) {
    add('blocks.0.'+name+'.weight',[out,input]); add('blocks.0.'+name+'.bias',[out],'zero');
  }
  add('lm_head.weight',[V,C]);
  const optimizer = tf.train.adam(.003);
  const linear = (x,name) => {
    const shape=x.shape, w=weights[name+'.weight'];
    const y=tf.matMul(x.reshape([-1,shape.at(-1)]),w,false,true).reshape([...shape.slice(0,-1),w.shape[0]]);
    return weights[name+'.bias'] ? y.add(weights[name+'.bias']) : y;
  };
  const norm = (x,name) => { const {mean,variance}=tf.moments(x,[-1],true); return x.sub(mean).mul(variance.add(1e-5).rsqrt()).mul(weights[name+'.weight']).add(weights[name+'.bias']); };
  function forward(ids) {
    const [B,N]=ids.shape, H=config.n_head, D=C/H;
    let x=tf.gather(weights['token_embedding.weight'],ids).add(tf.gather(weights['position_embedding.weight'],tf.range(0,N,1,'int32')));
    const [q,k,v]=tf.split(linear(norm(x,'blocks.0.ln1'),'blocks.0.attn.qkv'),3,-1).map(a=>a.reshape([B,N,H,D]).transpose([0,2,1,3]));
    const mask=tf.tensor2d(Array.from({length:N*N},(_,i)=>i%N>Math.floor(i/N)?-1e9:0),[N,N]);
    const attention=tf.softmax(tf.matMul(q,k,false,true).mul(1/Math.sqrt(D)).add(mask),-1);
    const combined=tf.matMul(attention,v).transpose([0,2,1,3]).reshape([B,N,C]);
    x=x.add(linear(combined,'blocks.0.attn.proj'));
    const z=linear(norm(x,'blocks.0.ln2'),'blocks.0.mlp.fc');
    const gelu=z.mul(.5).mul(tf.tanh(z.add(z.pow(tf.scalar(3)).mul(.044715)).mul(Math.sqrt(2/Math.PI))).add(1));
    x=x.add(linear(gelu,'blocks.0.mlp.proj'));
    return linear(norm(x,'ln_f'),'lm_head');
  }
  function batch(data,random) {
    const input=[],target=[];
    for(let b=0;b<4;b++) { const start=Math.floor(random()*(data.length-T)); input.push(...data.slice(start,start+T)); target.push(...data.slice(start+1,start+T+1)); }
    return [tf.tensor2d(input,[4,T],'int32'),tf.tensor2d(target,[4,T],'int32')];
  }
  function loss(data,random) {
    const [x,y]=batch(data,random);
    return tf.logSoftmax(forward(x).reshape([-1,V])).mul(tf.oneHot(y.flatten(),V)).sum(-1).neg().mean();
  }
  return {
    config,
    logits(ids) { return tf.tidy(()=>Array.from(forward(tf.tensor2d([ids],[1,ids.length],'int32')).dataSync())); },
    step(random) { return tf.tidy(()=> { const cost=optimizer.minimize(()=>loss(corpus.train,random),true,Object.values(weights)); return cost.dataSync()[0]; }); },
    evaluate() { return tf.tidy(()=>({train_loss:loss(corpus.train,randomSource(seed+2)).dataSync()[0],validation_loss:loss(corpus.validation,randomSource(seed+3)).dataSync()[0]})); },
    export(step) { return {format:FORMAT,config,vocab:corpus.vocab,step,tensors:Object.fromEntries(Object.entries(weights).map(([name,w])=>[name,{shape:w.shape,data:Array.from(w.dataSync())}]))}; },
    dispose() { optimizer.dispose(); Object.values(weights).forEach(w=>w.dispose()); }
  };
}

export function customManifest(checkpoint) {
  new TinyGPT(checkpoint); // Independently validate every tensor before use.
  const history=(checkpoint.training_run?.history || []).filter(r=>Number.isInteger(r.step)&&r.step>=0&&r.step<=100000000&&Number.isFinite(r.train_loss)&&r.train_loss>=0&&r.train_loss<1000000&&Number.isFinite(r.validation_loss)&&r.validation_loss>=0&&r.validation_loss<1000000).slice(0,300);
  const step=Number.isInteger(checkpoint.step)&&checkpoint.step>=0?checkpoint.step:0;
  const selected={id:'custom',label:'Your model',step,validation_loss:history.at(-1)?.validation_loss};
  const characters=checkpoint.training_run?.characters;
  return {config:checkpoint.config,vocab:checkpoint.vocab,parameter_count:Object.values(checkpoint.tensors).reduce((n,t)=>n+t.data.length,0),checkpoints:[selected],history,
    training:{device:checkpoint.training_run?.device==='browser CPU'?'browser CPU':'imported checkpoint',seed:Number.isInteger(checkpoint.training_run?.seed)?checkpoint.training_run.seed:'unknown',steps:step},
    corpus:{title:'Your custom text',characters:Number.isInteger(characters)&&characters>0?characters:0,train_characters:Number.isInteger(characters)&&characters>0?Math.floor(characters*.9):undefined,validation_characters:Number.isInteger(characters)&&characters>0?characters-Math.floor(characters*.9):undefined,url:''}};
}
