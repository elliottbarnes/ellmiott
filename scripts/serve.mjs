import {createServer} from 'node:http';
import {readFile, stat} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
const root = resolve('site');
const port = Number(process.env.PORT || 4196);
const mime = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml'};
const server = createServer(async (req,res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    let filename = resolve(root, `.${path}`);
    if (filename !== root && !filename.startsWith(root+sep)) throw new Error('Invalid path');
    if ((await stat(filename)).isDirectory()) filename = resolve(filename, 'index.html');
    res.writeHead(200, {'Content-Type':mime[extname(filename)] || 'application/octet-stream','Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'});
    res.end(await readFile(filename));
  } catch { res.writeHead(404); res.end('Not found'); }
});
server.listen(port, '127.0.0.1', () => console.log(`Ellmiott GPT: http://127.0.0.1:${port}`));
