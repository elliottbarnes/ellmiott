import {cp, mkdir, readdir, readFile, rm, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, relative, join} from 'node:path';
const root = resolve('site'), destination = resolve('dist');
const manifest = JSON.parse(await readFile(join(root,'models/manifest.json')));
for (const checkpoint of manifest.checkpoints) {
  if (!/^[a-z]+\.json$/.test(checkpoint.file)) throw new Error('Unsafe checkpoint path');
  const content = await readFile(join(root, 'models', checkpoint.file));
  if (content.length !== checkpoint.bytes || createHash('sha256').update(content).digest('hex') !== checkpoint.sha256)
    throw new Error(`Checkpoint provenance mismatch: ${checkpoint.id}`);
}
const allowed = new Set(['.html','.css','.js','.json','.svg','.txt']);
const files = [];
async function walk(dir) {
  for (const entry of await readdir(dir, {withFileTypes:true})) {
    const path = join(dir,entry.name);
    if (entry.isSymbolicLink()) throw new Error('Symlinks are not allowed in the site');
    if (entry.isDirectory()) await walk(path);
    else {
      const extension = entry.name.slice(entry.name.lastIndexOf('.'));
      if (!allowed.has(extension)) throw new Error(`Unexpected site asset: ${path}`);
      files.push(relative(root,path));
    }
  }
}
await walk(root);
await rm(destination,{recursive:true,force:true}); await mkdir(destination,{recursive:true});
await cp(root,destination,{recursive:true});
await writeFile(join(destination,'.nojekyll'),'');
console.log(`Verified checkpoint hashes; built ${files.length} site assets.`);
