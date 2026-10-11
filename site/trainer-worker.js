// Classic worker: the vendored UMD bundle exposes tf only inside this worker.
importScripts('./vendor/tf.min.js');
let cancelled=false, running=false;
self.onmessage=async ({data})=>{
  if(data.type==='cancel') {cancelled=true;return;}
  if(data.type!=='train'||running)return;
  running=true;cancelled=false;
  let trainer;
  try {
    const {prepareCorpus,createTrainer}=await import('./trainer-core.js');
    const {randomSource}=await import('./engine.js');
    if(!Number.isInteger(data.steps)||data.steps<1||data.steps>2000||!Number.isInteger(data.seed)||data.seed<0||data.seed>4294967295)throw Error('Invalid training settings.');
    await tf.setBackend('cpu');await tf.ready();
    const corpus=prepareCorpus(data.text);trainer=createTrainer(tf,corpus,data.seed);
    const history=[],random=randomSource(data.seed+1),start=performance.now();
    let step=0;
    const report=()=>{const metrics={step,...trainer.evaluate()};if(!Number.isFinite(metrics.train_loss)||!Number.isFinite(metrics.validation_loss))throw Error('Training diverged. Try a new run.');history.push(metrics);self.postMessage({type:'progress',...metrics,steps:data.steps,seconds:(performance.now()-start)/1000});};
    report();
    for(;step<data.steps&&!cancelled;){trainer.step(random);step++;if(step%10===0||step===data.steps)report();await new Promise(r=>setTimeout(r,0));}
    if(history.at(-1).step!==step)report();
    const checkpoint=trainer.export(step);
    checkpoint.training_run={history,characters:corpus.characters,seed:data.seed,device:'browser CPU'};
    self.postMessage({type:'complete',checkpoint,cancelled});
  }catch(error){self.postMessage({type:'error',message:error.message});}
  finally{trainer?.dispose();running=false;}
};
