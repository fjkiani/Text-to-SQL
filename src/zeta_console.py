"""
Zeta Clearance operator console — a same-origin single-page UI served by the
engine itself at /zeta/console.

Why this exists: the React portal in openclaw-saas needs the Express API and a
Postgres migration before it can run. This console talks straight to the live
/zeta/* endpoints, so the full four-layer workflow is usable in a browser today
with no additional infrastructure. Same-origin, so no CORS; the engine token is
entered by the operator and kept in sessionStorage rather than baked into HTML.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

console_router = APIRouter(prefix="/zeta", tags=["zeta"])

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Zeta Clearance — Institutional KYB</title>
<style>
:root{--ink:#000;--paper:#FAF9F3;--bone:#ECE9E2;--lime:#E9ED4C;--amber:#FF9400;
--moss:#75A025;--rose:#FD9BED;--blue:#0279EE;--line:#d8d5cc}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.5 "Liberation Sans",Arimo,"DejaVu Sans",system-ui,sans-serif}
header{background:var(--ink);color:var(--paper);padding:14px 22px;display:flex;
gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;letter-spacing:.14em;text-transform:uppercase}
header .sub{opacity:.62;font-size:12px}
header input{margin-left:auto;background:#1b1b1b;border:1px solid #3a3a3a;
color:var(--paper);padding:7px 10px;border-radius:5px;width:290px;font-size:12px}
.wrap{max-width:1280px;margin:0 auto;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
.card{background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden;
display:flex;flex-direction:column}
.card h2{margin:0;padding:11px 15px;font-size:11.5px;letter-spacing:.11em;
text-transform:uppercase;border-bottom:1px solid var(--line);background:var(--bone);
display:flex;align-items:center;gap:9px}
.tag{font-size:9.5px;padding:2px 7px;border-radius:20px;background:var(--ink);
color:var(--paper);letter-spacing:.06em}
.body{padding:15px;flex:1}
textarea{width:100%;min-height:132px;font-family:ui-monospace,Menlo,monospace;
font-size:11.5px;border:1px solid var(--line);border-radius:6px;padding:10px;resize:vertical}
button{background:var(--ink);color:var(--paper);border:0;border-radius:6px;
padding:9px 15px;font-size:12.5px;font-weight:600;cursor:pointer}
button:hover{background:#2c2c2c} button:disabled{opacity:.4;cursor:not-allowed}
button.alt{background:#fff;color:var(--ink);border:1px solid var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:11px}
.out{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;background:#0d0d0d;
color:#e6e6e6;border-radius:6px;padding:11px;white-space:pre-wrap;word-break:break-word;
max-height:270px;overflow:auto;margin-top:11px}
.pill{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;
font-weight:700;margin:3px 4px 3px 0}
.ok{background:var(--moss);color:#fff} .no{background:#c0392b;color:#fff}
.warn{background:var(--amber);color:#fff} .info{background:var(--blue);color:#fff}
.ubo{background:var(--lime);color:#000} .muted{background:var(--bone);color:#555}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:9px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:#666}
.flow{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px}
.step{flex:1;min-width:132px;background:#fff;border:1px solid var(--line);
border-left:3px solid var(--line);border-radius:6px;padding:9px 11px}
.step.done{border-left-color:var(--moss)} .step.run{border-left-color:var(--amber)}
.step .n{font-size:9.5px;letter-spacing:.1em;color:#777;text-transform:uppercase}
.step .t{font-weight:700;font-size:12.5px;margin-top:2px}
.step .s{font-size:11px;color:#666;margin-top:2px}
.hint{font-size:11.5px;color:#666;margin-top:8px}
</style></head><body>
<header>
  <h1>Zeta Clearance</h1>
  <span class="sub">Institutional KYB &middot; agentic intake &rarr; zero-knowledge vault &rarr; Canton attestation &rarr; permissioned liquidity</span>
  <input id="tok" type="password" placeholder="engine token"/>
</header>
<div class="wrap">
<div class="flow" id="flow"></div>
<div class="grid">

  <div class="card"><h2><span class="tag">L1</span> Data room &amp; interrogator</h2><div class="body">
    <textarea id="doc">ACME OPCO LTD - SHAREHOLDER REGISTER (31 Dec 2025)
Cayman HoldCo Ltd .... 40% of Acme OpCo Ltd
Maria Garcia (individual) .... 35% of Acme OpCo Ltd
Redwood Nominees Ltd .... 25% of Acme OpCo Ltd

CAYMAN HOLDCO LTD - REGISTER OF MEMBERS
John Smith (natural person) holds 60% of Cayman HoldCo Ltd.
Blue Harbour Trust holds 40% of Cayman HoldCo Ltd.</textarea>
    <div class="row">
      <input id="ent" value="acme_opco" style="padding:8px;border:1px solid var(--line);border-radius:6px;width:150px"/>
      <button onclick="runL1()">Ingest &amp; extract</button>
      <button class="alt" onclick="ask()">Ask for missing docs</button>
    </div>
    <div class="hint">The model only proposes edges with page citations. It never computes ownership.</div>
    <div class="out" id="o1">idle</div>
  </div></div>

  <div class="card"><h2><span class="tag">L2</span> Deterministic UBO graph</h2><div class="body">
    <div class="row">
      <label style="font-size:12px">threshold %
        <input id="thr" type="number" value="25" step="1" style="width:66px;padding:7px;border:1px solid var(--line);border-radius:6px"/>
      </label>
      <button onclick="runUbo()">Compute UBOs</button>
      <button class="alt" onclick="sweep()">Threshold sweep</button>
    </div>
    <div id="ubobox"></div>
    <div class="out" id="o2">idle</div>
  </div></div>

  <div class="card"><h2><span class="tag">L2</span> Privacy vault</h2><div class="body">
    <textarea id="pii" style="min-height:74px">PASSPORT X1234567
NAME John Smith
DOB 1980-01-01</textarea>
    <div class="row"><button onclick="vault()">Encrypt &amp; tokenise</button></div>
    <div class="hint">Raw PII is AES-256-GCM encrypted at rest. Only a token and an evidence hash cross this boundary.</div>
    <div class="out" id="o3">idle</div>
  </div></div>

  <div class="card"><h2><span class="tag">L3</span> Canton attestation &amp; credential</h2><div class="body">
    <div class="row">
      <select id="dec" style="padding:8px;border:1px solid var(--line);border-radius:6px">
        <option>approved</option><option>review_required</option><option>rejected</option></select>
      <select id="tier" style="padding:8px;border:1px solid var(--line);border-radius:6px">
        <option>low</option><option>medium</option><option>high</option></select>
      <button onclick="attest()">Attest &amp; issue VC</button>
    </div>
    <div class="hint">Exactly six non-PII fields reach the ledger. Anything else is rejected at the boundary.</div>
    <div class="out" id="o4">idle</div>
  </div></div>

  <div class="card"><h2><span class="tag">L4</span> Liquidity gate</h2><div class="body">
    <div class="row">
      <button onclick="verify('aave_arc')">Verify as relying party</button>
      <button class="alt" onclick="verify('stranger')">Verify as stranger</button>
    </div>
    <div class="row">
      <button onclick="relay()">Relay to EVM allowlist</button>
      <button class="alt" onclick="revoke()">Revoke attestation</button>
    </div>
    <div class="hint">The counterparty checks clearance without ever touching underlying documents.</div>
    <div class="out" id="o5">idle</div>
  </div></div>

  <div class="card"><h2>Activity</h2><div class="body">
    <div class="out" id="log" style="max-height:330px">ready</div>
  </div></div>

</div></div>
<script>
const S={edges:[],contract:null,evidence:null,ubo:null};
const STEPS=[["L1","Intake","pending"],["L1","Extraction","pending"],["L2","UBO graph","pending"],
             ["L2","Vault","pending"],["L3","Attestation","pending"],["L4","Liquidity","pending"]];
const $=i=>document.getElementById(i);
$('tok').value=sessionStorage.getItem('zt')||'';
$('tok').oninput=e=>sessionStorage.setItem('zt',e.target.value);
function flow(){$('flow').innerHTML=STEPS.map(([l,t,s])=>
 `<div class="step ${s==='done'?'done':s==='run'?'run':''}"><div class="n">${l}</div>
  <div class="t">${t}</div><div class="s">${s}</div></div>`).join('')}
function mark(i,s){STEPS[i][2]=s;flow()}
function log(m){const e=$('log');e.textContent=new Date().toLocaleTimeString()+"  "+m+"\n"+e.textContent}
function show(id,o){$(id).textContent=typeof o==='string'?o:JSON.stringify(o,null,2)}
async function api(p,b){
  const t=$('tok').value.trim();
  if(!t){alert("Enter the engine token (top right).");throw new Error("no token")}
  const r=await fetch('/zeta'+p,{method:'POST',headers:{'Content-Type':'application/json',
    'Authorization':'Bearer '+t},body:JSON.stringify(b)});
  const txt=await r.text();
  if(!r.ok)throw new Error(r.status+" "+txt.slice(0,240));
  return JSON.parse(txt);
}
const b64=s=>btoa(unescape(encodeURIComponent(s)));

async function runL1(){
  try{mark(0,'run');show('o1','ingesting...');log('L1 ingest started');
    const ing=await api('/ingest',{filename:'doc.txt',doc_b64:b64($('doc').value)});
    mark(0,'done');mark(1,'run');
    show('o1','parsed '+ing.chunk_count+' chunk(s) ('+ing.parse_mode+')\nextracting ownership edges via LLM...');
    const t0=Date.now();
    const ed=await api('/extract_edges',{chunks:ing.chunks});
    S.edges=ed.edges;S.conflicts=ed.conflicts||[];mark(1,'done');
    const secs=((Date.now()-t0)/1000).toFixed(0);
    let h='<table><tr><th>Owner</th><th>Owns</th><th>%</th><th>Type</th><th>Pg</th></tr>';
    ed.edges.forEach(e=>h+=`<tr><td>${e.owner_id}</td><td>${e.owned_entity_id}</td>
      <td>${e.direct_pct}</td><td>${e.owner_type}</td><td>${e.page}</td></tr>`);
    let cw='';
    if(S.conflicts.length){
      cw='<div class="hint" style="color:#FF9400"><b>'+S.conflicts.length+
         ' conflicting record(s)</b> — the documents disagree; both readings kept, largest retained, review forced:</div>'+
         S.conflicts.map(c=>`<div class="hint">${c.owner_id} → ${c.owned_entity_id}: `+
           `${c.values.join('% vs ')}% (pages ${c.pages.join(', ')})</div>`).join('');
    }
    show('o1','');$('o1').innerHTML=h+'</table>'+cw+
      `<div class="hint">${ed.edges.length} edges · ${ed.backend}${ed.model?' / '+ed.model:''}`+
      ` · ${ed.batches||1} batch(es) · ${secs}s</div>`;
    log('L1 extracted '+ed.edges.length+' edges in '+secs+'s'+
        (S.conflicts.length?(' ('+S.conflicts.length+' CONFLICT)'):''));
    runUbo();
  }catch(e){show('o1','ERROR '+e.message);log('L1 failed: '+e.message);mark(1,'pending')}
}
async function runUbo(){
  try{mark(2,'run');
    const u=await api('/ubo',{entity_id:$('ent').value,edges:S.edges,threshold_pct:parseFloat($('thr').value),conflicts:S.conflicts||[]});
    S.ubo=u;mark(2,'done');
    let p=u.ubos.length?u.ubos.map(x=>`<span class="pill ubo">${x.person_id} ${x.aggregate_pct}%</span>`).join('')
                       :'<span class="pill muted">no UBO at this threshold</span>';
    p+=(u.flags||[]).map(f=>`<span class="pill warn">${f}</span>`).join('');
    if(u.review_required)p+='<span class="pill no">review required</span>';
    $('ubobox').innerHTML=p+`<div class="hint">resolved entity: <b>${u.entity_id}</b></div>`;
    show('o2',u);log('L2 UBOs: '+u.ubos.map(x=>x.person_id+' '+x.aggregate_pct+'%').join(', ')||'none');
  }catch(e){show('o2','ERROR '+e.message);log('L2 failed: '+e.message)}
}
async function sweep(){
  try{show('o2','sweeping thresholds...');const rows=[];
    for(const t of [10,20,25,30,50]){
      const u=await api('/ubo',{entity_id:$('ent').value,edges:S.edges,threshold_pct:t,conflicts:S.conflicts||[]});
      rows.push(t+'% -> '+(u.ubos.map(x=>x.person_id+' '+x.aggregate_pct+'%').join(', ')||'none'));
    }
    show('o2',rows.join('\n'));log('L2 threshold sweep complete');
  }catch(e){show('o2','ERROR '+e.message)}
}
async function ask(){
  try{const r=await api('/interrogate',{entity_id:$('ent').value,edges:S.edges,have_docs:['cap_table']});
    show('o1',r.action==='request_doc'?('AGENT: '+r.message):'AGENT: all required documents present');
    log('L1 interrogator: '+r.action);
  }catch(e){show('o1','ERROR '+e.message)}
}
async function vault(){
  try{mark(3,'run');const v=await api('/vault/store',
      {subject_id:'subject',record_type:'identity_document',doc_b64:b64($('pii').value)});
    S.evidence=v.evidence_hash;mark(3,'done');
    show('o3',v);log('L2 vault token '+v.token.slice(0,8)+' (no PII returned)');
  }catch(e){show('o3','ERROR '+e.message);log('L2 vault failed')}
}
async function attest(){
  try{
    if(!S.evidence){show('o4','Vault a document first — the attestation anchors its evidence hash.');return}
    mark(4,'run');
    const a=await api('/attest',{legal_entity_name:$('ent').value,decision:$('dec').value,
      risk_tier:$('tier').value,ubo_verified:(S.ubo&&S.ubo.ubos.length>0),
      evidence_hash:S.evidence,subject:'applicant',relying_parties:['aave_arc']});
    S.contract=a.contract_id;mark(4,'done');
    show('o4',{contract_id:a.contract_id,on_ledger:a.payload,credential_id:a.credential.id,
               credential_type:a.credential.type});
    log('L3 attested '+a.contract_id.slice(0,8)+' · VC issued');
  }catch(e){show('o4','ERROR '+e.message);log('L3 attest failed: '+e.message)}
}
async function verify(who){
  try{if(!S.contract){show('o5','Attest first.');return}
    const r=await api('/verify',{contract_id:S.contract,relying_party:who});
    $('o5').innerHTML=`<span class="pill ${r.verified?'ok':'no'}">${who}: ${r.verified?'CLEARED':'DENIED'}</span>`+
      '<div style="margin-top:8px">'+JSON.stringify(r.claim||r.reason,null,2)+'</div>';
    log('L3 verify '+who+' -> '+(r.verified?'cleared':'denied'));
  }catch(e){show('o5','ERROR '+e.message)}
}
async function relay(){
  try{if(!S.contract){show('o5','Attest first.');return}
    mark(5,'run');const r=await api('/relay',{contract_id:S.contract,relying_party:'aave_arc'});
    mark(5,'done');
    $('o5').innerHTML=`<span class="pill ${r.is_cleared?'ok':'no'}">on-chain: ${r.is_cleared?'ALLOWLISTED':'BLOCKED'}</span>`+
      '<div style="margin-top:8px">entity key '+r.entity_key+'</div>';
    log('L4 relayed to EVM oracle · cleared='+r.is_cleared);
  }catch(e){show('o5','ERROR '+e.message);log('L4 relay failed')}
}
async function revoke(){
  try{if(!S.contract){show('o5','Attest first.');return}
    await api('/revoke',{contract_id:S.contract,relying_party:'zeta_issuer'});
    const r=await api('/verify',{contract_id:S.contract,relying_party:'aave_arc'});
    $('o5').innerHTML=`<span class="pill no">revoked</span><span class="pill ${r.verified?'ok':'no'}">
      relying party now: ${r.verified?'CLEARED':'DENIED'}</span>`;
    mark(5,'pending');log('L3 revoked · downstream verify='+r.verified);
  }catch(e){show('o5','ERROR '+e.message)}
}
flow();
</script></body></html>"""


@console_router.get("/console", response_class=HTMLResponse)
def zeta_console():
    return HTMLResponse(PAGE)
