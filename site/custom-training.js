import {prepareCorpus,customManifest} from './trainer-core.js';
const $=id=>document.getElementById(id);
let worker=null, checkpoint=null, revision=0, fileRevision=0;
const inputs=['custom-text','custom-file','custom-steps','custom-seed','custom-example','custom-import'];
function status(text){$('custom-status').textContent=text;}
function busy(value){inputs.forEach(id=>$(id).disabled=value);$('custom-start').disabled=value;$('custom-stop').hidden=!value;$('custom-use').disabled=value||!checkpoint;$('custom-download').disabled=value||!checkpoint;}
function inspect(){
  try{const c=prepareCorpus($('custom-text').value);$('custom-stats').textContent=`${c.characters.toLocaleString()} characters · ${c.vocab.length} distinct · ${c.train.length.toLocaleString()} training / ${c.validation.length.toLocaleString()} held out`;$('custom-vocab').textContent=c.vocab.map(c=>c==='\n'?'↵':c===' '?'␣':c).join(' ');return c;}
  catch(error){$('custom-stats').textContent=error.message;$('custom-vocab').textContent='';return null;}
}
function download(){const url=URL.createObjectURL(new Blob([JSON.stringify(checkpoint)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='ellmiott-custom.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function showProgress(data){
  $('custom-progress').max=data.steps;$('custom-progress').value=data.step;
  $('custom-train-loss').textContent=data.train_loss.toFixed(3);$('custom-validation-loss').textContent=data.validation_loss.toFixed(3);
  status(`Step ${data.step} / ${data.steps} · ${data.seconds.toFixed(0)} seconds · training locally`);
  const row=document.createElement('tr');for(const value of [data.step,data.train_loss.toFixed(3),data.validation_loss.toFixed(3)]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}$('custom-history').append(row);
  $('custom-history-scroll').scrollTop=$('custom-history-scroll').scrollHeight;
}
$('custom-text').addEventListener('input',()=>{fileRevision++;inspect();});
$('custom-file').addEventListener('change',async()=>{
  const ticket=++fileRevision,file=$('custom-file').files[0];if(!file)return;
  try{if(file.size>1000000)throw Error('Choose a text file smaller than 1 MB.');const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());if(ticket!==fileRevision)return;$('custom-text').value=text;inspect();}
  catch(error){if(ticket===fileRevision)status(error.message);}
});
$('custom-example').addEventListener('click',()=>{fileRevision++;$('custom-text').value=('the moon follows the sea.\nthe sea follows the moon.\nthe tide rises and the tide falls.\n').repeat(30);inspect();status('Original practice text loaded. Repetition makes a small experiment easier; it is not a test of general language ability.');});
$('custom-start').addEventListener('click',()=>{
  if(!inspect()){status('Check your text before training.');return;}
  if(!$('custom-seed').reportValidity())return;
  const ticket=++revision;fileRevision++;checkpoint=null;busy(true);$('custom-history').replaceChildren();$('custom-progress').value=0;$('custom-train-loss').textContent='—';$('custom-validation-loss').textContent='—';status('Starting the local training engine…');
  const finish=()=>{worker?.terminate();worker=null;busy(false);};
  try{worker=new Worker(new URL('./trainer-worker.js',import.meta.url));
    worker.addEventListener('message',({data})=>{if(ticket!==revision)return;
      if(data.type==='progress')showProgress(data);
      else if(data.type==='complete'){checkpoint=data.checkpoint;finish();status(`${data.cancelled?'Stopped':'Finished'} at step ${checkpoint.step}. Use your model below, or download its weights. The loss chart follows it into the lab.`);}
      else if(data.type==='error'){finish();status(data.message);}
    });
    worker.addEventListener('error',()=>{if(ticket!==revision)return;finish();status('The training worker could not run. Try reloading, or use the Python training guide.');});
    worker.postMessage({type:'train',text:$('custom-text').value,steps:Number($('custom-steps').value),seed:Number($('custom-seed').value)});
  }catch(error){finish();status(error.message);}
});
$('custom-stop').addEventListener('click',()=>{worker?.postMessage({type:'cancel'});$('custom-stop').disabled=true;status('Finishing this update and saving the current weights…');});
$('custom-start').addEventListener('click',()=>{$('custom-stop').disabled=false;});
$('custom-use').addEventListener('click',()=>{
  if(!checkpoint)return;
  window.dispatchEvent(new CustomEvent('ellmiott:custom-model',{detail:{checkpoint}}));
  document.getElementById('lab').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
});
$('custom-download').addEventListener('click',download);
$('custom-import').addEventListener('change',async()=>{
  const ticket=++revision,file=$('custom-import').files[0];if(!file)return;
  try{if(file.size>10000000)throw Error('Choose a checkpoint smaller than 10 MB.');const imported=JSON.parse(await file.text());customManifest(imported);if(ticket!==revision)return;checkpoint=imported;$('custom-history').replaceChildren();$('custom-progress').value=0;$('custom-train-loss').textContent='—';$('custom-validation-loss').textContent='—';for(const row of customManifest(imported).history)showProgress({...row,steps:Math.max(1,imported.step||1),seconds:0});busy(false);status('Checkpoint validated. Select “Use my model” to inspect it. Recorded loss is available only if included in the file.');}
  catch(error){if(ticket===revision)status(`Could not import: ${error.message}`);}
  finally{$('custom-import').value='';}
});
window.addEventListener('pagehide',()=>{if(worker){revision++;worker.terminate();worker=null;busy(false);status('Training ended when you left the page. Start a new run to train again.');}});
inspect();busy(false);
