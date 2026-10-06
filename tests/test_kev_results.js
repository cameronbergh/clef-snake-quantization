'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'watcher/kev-campaign-results.html'),'utf8');
const code=html.match(/<script id="kev-results-core">([\s\S]*?)<\/script>/)[1];
const context={};vm.createContext(context);vm.runInContext(code,context);
const prepare=context.KevResults.prepare;
const raw=JSON.parse(fs.readFileSync(path.join(root,'data/2026-10-06-kev4b-campaign-v1/results.json')));
const d=prepare(raw);assert.equal(d.rows.length,30);
assert.deepEqual(Array.from(d.summary,s=>s.mean),[19.9,19.9,17.8]);
assert.deepEqual(Array.from(d.comparisons,c=>[c.wins,c.ties,c.losses]),[[5,19,6],[7,9,14]]);
function reject(mutate){const x=JSON.parse(JSON.stringify(raw));mutate(x);assert.throws(()=>prepare(x));}
reject(x=>x.games.pop());reject(x=>x.games[1]=x.games[0]);reject(x=>x.games[0].seed='f'.repeat(32));
reject(x=>x.games[0].score=-1);reject(x=>x.games[0].alive=true);reject(x=>x.games[0].attempts++);
reject(x=>x.state='running');reject(x=>x.games[0].variant='q2-k');
assert(!/setInterval|XMLHttpRequest|WebSocket|\/v1\//.test(html));
assert.equal((html.match(/read\('\.\.\/[^']+\.json'\)/g)||[]).length,2);
console.log('Kev published viewer: paired data, malformed/duplicate rows and passive request checks passed.');
