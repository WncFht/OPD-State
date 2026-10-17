// Validate local links, footnotes, math syntax, and pseudocode structure.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const katex = require(require.resolve('katex', {paths:[path.resolve(root,'../MirrorOPD')]}));
const file = path.join(root,'2026-10-17-OPD访问测度与收敛机制研究.md');
const raw = fs.readFileSync(file,'utf8');
const issues=[];
let mathCount=0, links=0;
const prose=raw.replace(/```[\s\S]*?```/g,'');
for (const m of prose.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)) {
  const target=m[1];
  if (/^(https?:|#)/.test(target)) continue;
  links++;
  if(!fs.existsSync(path.resolve(root,target.split('#')[0]))) issues.push(`missing link: ${target}`);
}
const defs=new Set([...raw.matchAll(/^\[\^([a-z-]+)\]:/gm)].map(m=>m[1]));
for (const m of raw.matchAll(/\[\^([^\]]+)\](?!:)/g)) {
  if(!defs.has(m[1])) issues.push(`missing footnote: ${m[1]}`);
}
const block=[...prose.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1]);
const inline=[...prose.replace(/\$\$[\s\S]*?\$\$/g,'').matchAll(/(?<!\\)\$([^$\n]+?)\$/g)].map(m=>m[1]);
for(const math of [...block,...inline]) {
  mathCount++;
  try {katex.renderToString(math,{throwOnError:true,strict:'ignore',displayMode:block.includes(math)});}
  catch(e) {issues.push(`math ${mathCount}: ${e.message}`);}
}
const pseudo=raw.match(/```pseudocode\n([\s\S]*?)```/)[1];
if(!pseudo.includes('\\\\{\\small\\textit{Semantics:')) issues.push('caption semantics line break missing');
for(const [open,close] of [['FOR','ENDFOR'],['IF','ENDIF']]) {
  const count=x=>(pseudo.match(new RegExp('\\\\'+x+'(?:\\{|\\n)','g'))||[]).length;
  if(count(open)!==count(close)) issues.push(`${open} block mismatch`);
}
if(raw.split('\n').length>1000) issues.push('main document over 1000 lines');
if(raw.includes('相当具体')) issues.push('forbidden phrase');
const result={math_expressions:mathCount,local_links:links,footnotes:defs.size,issues};
fs.writeFileSync(path.join(root,'results/document-check.json'),JSON.stringify(result,null,2)+'\n');
console.log(result);
if(issues.length)process.exit(1);
