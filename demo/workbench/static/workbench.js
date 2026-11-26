"use strict";
const $ = id => document.getElementById(id);
const order = ["input", "local", "community", "reasoning", "unseen", "drift", "branches"];
const states = {pending:"等待", running:"进行中", done:"已完成", complete:"已结束", empty:"无有效证据",
  skipped:"未执行", missing:"未记录或不可用", partial:"部分结果", error:"失败", failed:"未完成"};
const labels = {archive:"归档回放", local:"快速本地", full:"完整识别", upload:"上传指纹", censys:"Censys 观测",
  candidates:"近邻候选", total_compared:"比较条数", missing_perspectives:"缺失视角", device_type:"设备类型",
  similarity_score:"相似度", cluster_id:"社区编号", cluster_key:"路径所属社区", cluster_size:"成员数",
  community_key:"社区标识", is_outlier:"单成员噪声社区", matches:"通过门槛的社区", min_similarity:"匹配门槛",
  score_source:"评分来源",
  unavailable_reasons:"不可用原因及次数", observed:"归档事实", interpretation:"原因与边界", action:"修复建议",
  report_source:"报告来源", path_matching_score:"路径匹配分数", evidence_coverage:"证据覆盖权重",
  important_features:"关键视角", importance_score:"重要性", feature_name:"视角", num_clusters:"簇数",
  classified_type:"分类类型", classified_vendor:"分类厂商", predicted_type:"新类型候选", predicted_vendor:"新厂商候选",
  new_type_probability:"新类型概率", new_vendor_probability:"新厂商概率", is_unseen:"未见设备",
  drift_score:"漂移分数", threshold:"阈值", threshold_scope:"阈值范围", drifted_perspectives:"偏移视角",
  gemini:"GEMINI 分支", claude:"CLAUDE 分支", vendor:"厂商", confidence:"置信度", type_confidence:"类型置信度",
  vendor_confidence:"厂商置信度", device_type_reason:"判断依据",
  joint_vote:"双路联合判定", adapter_fallback:"本地 adapter 回退", local_candidate:"仅局部候选（非联合判定）"};
let setup, run, activeJob = "", lastReport = "", timer = 0, exampleFile = null, submitting = false;

function el(name, text, cls) {
  const node = document.createElement(name);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function page() {
  const key = location.hash.slice(1);
  const name = ["intake","stages","verdict","archive","runtime"].includes(key) ? key : "intake";
  document.querySelectorAll(".page").forEach(p => {p.hidden = p.id !== `page-${name}`;});
  document.querySelectorAll("nav [data-page]").forEach(a => {
    if (a.dataset.page === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
}
async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {const e = new Error(data.error || `服务返回 HTTP ${res.status}`); e.status = res.status; throw e;}
  return data;
}
function alertUser(msg) {
  $("global-error").textContent = msg; $("global-error").hidden = !msg;
}
function scalar(v) {
  if (v == null) return "未提供";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)));
  if (typeof v === "boolean") return v ? "是" : "否";
  return String(v);
}
function values(v) {
  if (Array.isArray(v)) {
    if (v.every(x => x === null || typeof x !== "object")) return el("span", v.map(scalar).join("、") || "无");
    const box = el("div");
    v.forEach(item => {const row=el("div",null,"rowdata"); row.append(values(item)); box.append(row);});
    return box;
  }
  if (v !== null && typeof v === "object") {
    const dl = el("dl",null,"values");
    Object.entries(v).forEach(([k,item]) => {const dd = el("dd"); dd.append(values(item)); dl.append(el("dt",labels[k] || k),dd);});
    return dl;
  }
  return el("span", scalar(v));
}
function empty(text) {
  const box=el("div",null,"empty"), link=el("a","前往资产识别 →"); link.href="#intake";
  box.append(el("h2","尚无判定报告"),el("p",text),link); return box;
}
function refreshButton() {
  const busy = submitting || !!activeJob;
  $("submit").disabled = busy || !setup?.unlocked;
  $("replay").disabled = busy || !$("sample").value;
}
function collection() {
  const groups=setup?.archive_groups;
  if(!groups) return;
  const options=groups[$("collection").value] || [];
  $("sample").replaceChildren(...options.map(key=>{const o=el("option",key);o.value=key;return o;}));
  $("collection-count").textContent=`当前范围 ${options.length} 条 / 全部 ${groups.all.length} 条；按参考标签复核的最终误判 ${groups.errors.length} 条。`;
  refreshButton();
}
function paint() {
  refreshButton();
  if (!run || !setup) {
    $("trace").replaceChildren(empty("提交单个设备，或先回放一个归档案例。"));
    $("verdict").replaceChildren(empty("识别完成后，这里会单独展示类型、厂商和判定来源。"));
    return;
  }
  const opened = new Set([...$("trace").querySelectorAll("details[open]")].map(x=>x.dataset.key));
  const trace = el("article",null,"trace"), head = el("div",null,"tracehead");
  head.append(el("h2",run.ip),el("span",`${labels[run.mode]} · ${states[run.state]}`,"tag "+run.state)); trace.append(head);
  order.forEach((key,i) => {
    const v=run.stages[key], d=el("details",null,"stage"), summary=el("summary");
    d.dataset.key=key; d.open=opened.has(key);
    summary.append(el("span",String(i+1).padStart(2,"0"),"number"),el("strong",setup.stages[key]),el("span",states[v.state],"tag "+v.state));
    const body=el("div",null,"stagebody"); body.append(el("p",v.note));
    if (v.data != null) body.append(values(v.data));
    d.append(summary,body); trace.append(d);
  });
  const info=run.comparison;
  const intro=el("aside",null,"sheet");
  if(info) {
    intro.append(el("h2",info.verified ? "局部与最终类型均命中" : "归档对照"),
      el("p",`参考标签：${info.reference} · 局部 Top-1：${info.local_top1 || "未记录"} · 最终类型：${run.final?.device_type || "未判定"}`),
      el("p",info.linked_version ? `检索来自完全相同指纹的 ${info.linked_version} 附档；旧决策未绑定检索版本，不证明同次运行。`
        : `检索与判定按本条原始记录回放，版本：${info.version || "未记录"}。`,"hint"));
    if(run.review) intro.append(el("h2","误判复盘"),values(run.review));
  }
  $("trace").replaceChildren(...(info ? [intro,trace] : [trace]));
  const f=run.final, card=el("article",null,"result");
  card.append(el("span",`${labels[run.mode]} / ${run.ip}`,"kicker"));
  const subtitle = run.state==="failed" ? "任务未完成" : f ? (labels[f.method] || f.method) : "等待判定";
  card.append(el("h2",subtitle));
  const pair=el("div",null,"pair");
  [["设备类型",f?.device_type,f?.confidence],["设备厂商",f?.vendor,f?.vendor_confidence]].forEach(([k,v,c])=>{
    const col=el("div"); col.append(el("small",k),el("strong",v || (f ? "未判定" : "—")));
    if (c != null) col.append(el("span",`置信度 ${scalar(c)}`));
    pair.append(col);
  });
  card.append(pair);
  if(info) card.append(el("p",`参考标签 ${info.reference} · 局部 Top-1 ${info.local_top1 || "未记录"}。`));
  if(run.review) card.append(values(run.review));
  if (f?.winning_llm) card.append(el("p",`类型获选分支：${f.winning_llm}`));
  if (f?.reason || run.error) card.append(el("p",run.error || f.reason));
  const back=el("a","返回阶段证据 →"); back.href="#stages"; card.append(back);
  $("verdict").replaceChildren(card);
  $("verdict-preview").textContent = f ? `${subtitle}：${f.device_type || "未判定"} / ${f.vendor || "厂商未判定"}` : run.error || "任务进行中，尚无最终判定。";
  $("runbar").hidden=false;
  $("run-status").textContent=`${labels[run.mode]} · ${run.ip} · ${states[run.state]}`;
}
function updateForm() {
  const isCensys=document.querySelector("input[name=source]:checked").value==="censys";
  $("upload-pane").hidden=isCensys; $("censys-pane").hidden=!isCensys;
  $("censys-consent").required=isCensys;
  $("pat").required=isCensys && !setup?.censys_ready;
  $("file").required=!isCensys && !exampleFile;
  const full=$("mode").value==="full";
  $("llm-pane").hidden=!full; $("llm-consent").required=full;
  $("mode-note").textContent=full ? "执行局部、社区、路径检索，Llama 未见检测、条件漂移检测和双路决策。Llama 使用 CPU，可能耗时数分钟以上。"
    : "仅使用 CPU 和本地知识库，不调用 LLM API，不判定厂商。";
}
async function poll(id) {
  if(activeJob!==id) return;
  try {
    const data=await api(`/api/runs/${encodeURIComponent(id)}`);
    if(activeJob!==id) return;
    const encoded=JSON.stringify(data);
    if(data.state!=="running") activeJob="";
    if(encoded!==lastReport) {lastReport=encoded; run=data; paint();}
    alertUser(data.error || "");
    if(activeJob) timer=setTimeout(()=>poll(id),1500);
  } catch(e) {
    if(e.status===404 || e.status===403) {activeJob=""; sessionStorage.removeItem("wb-job"); refreshButton();}
    else timer=setTimeout(()=>poll(id),4000);
    alertUser(`${e.message}${activeJob ? "；正在重试读取状态。" : ""}`);
  }
}
function newView() {
  clearTimeout(timer); lastReport=""; $("trace").replaceChildren(); alertUser(""); $("form-msg").textContent="";
}
$("intake-form").addEventListener("submit", async event=>{
  event.preventDefault();
  if(!setup || submitting || activeJob) return;
  submitting=true; refreshButton(); alertUser(""); $("form-msg").textContent="正在验证输入…";
  const source=document.querySelector("input[name=source]:checked").value;
  const data=new FormData();
  for(const [k,v] of Object.entries({ip:$("ip").value,source,mode:$("mode").value})) data.set(k,v);
  if(source==="upload") {
    const file=exampleFile || $("file").files[0]; if(file) data.set("file",file,file.name);
  } else {
    data.set("pat",$("pat").value); data.set("org",$("org").value);
    if($("censys-consent").checked) data.set("censys_consent","yes");
  }
  if($("mode").value==="full" && $("llm-consent").checked) data.set("llm_consent","yes");
  try {
    const got=await api("/api/runs",{method:"POST",headers:{"X-Workbench-Token":setup.token},body:data});
    activeJob=got.id; sessionStorage.setItem("wb-job",got.id);
    $("pat").value=""; $("censys-consent").checked=false; $("llm-consent").checked=false;
    newView(); run=null; paint(); location.hash="stages"; await poll(got.id);
  } catch(e) {alertUser(e.message); $("form-msg").textContent=e.message;}
  finally {submitting=false; refreshButton();}
});
$("use-example").addEventListener("click",async()=>{
  try {
    const data=await api("/static/example.json");
    exampleFile=new File([JSON.stringify(data)],"example.json",{type:"application/json"});
    $("file").value=""; $("ip").value=data.ip; $("form-msg").textContent="已填入合成示例 example.json（也可重新选文件）。";updateForm();
  } catch(e) {alertUser(e.message);}
});
$("file").addEventListener("change",()=>{exampleFile=null;$("form-msg").textContent="";updateForm();});
$("unlock").addEventListener("click",async()=>{
  if(!setup) return;
  try {
    const data=new FormData();data.set("key",$("access-key").value.trim());
    await api("/api/unlock",{method:"POST",headers:{"X-Workbench-Token":setup.token},body:data});
    setup.unlocked=true;$("unlock-pane").hidden=true;$("access-key").value="";alertUser("");refreshButton();
  } catch(e) {alertUser(e.message);}
});
$("access-key").addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();$("unlock").click();}});
$("replay").addEventListener("click",async()=>{
  if(!setup || submitting || activeJob) return;
  submitting=true; refreshButton();
  try {const data=await api(`/api/archive/${encodeURIComponent($("sample").value)}`);
    sessionStorage.removeItem("wb-job"); newView();run=data;paint();location.hash="stages";
  } catch(e) {alertUser(e.message);} finally {submitting=false;refreshButton();}
});
document.querySelectorAll("[name=source],#mode").forEach(x=>x.addEventListener("change",updateForm));
$("collection").addEventListener("change",collection);
window.addEventListener("hashchange",()=>{page();window.scrollTo(0,0);});
(async()=>{
  page();paint();updateForm();
  try {
    setup=await api("/api/setup"); $("unlock-pane").hidden=setup.unlocked;
    $("pat-hint").textContent=setup.censys_ready?"服务器已配置，可留空":"服务器未配置，本次查询须填写";
    $("censys-status").textContent="Censys PAT："+$("pat-hint").textContent+"。不会把凭据返回到页面或写入文件。";
    collection();
    updateForm(); activeJob=sessionStorage.getItem("wb-job") || ""; paint();
    if(activeJob) await poll(activeJob);
  } catch(e) {alertUser(e.message);}
})();
