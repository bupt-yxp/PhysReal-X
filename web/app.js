const state = { modules: [], selected: null, jobId: null, token: sessionStorage.getItem("prx_token") || "" };
const $ = (id) => document.getElementById(id);

function node(tag, className = "", text = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text) item.textContent = text;
  return item;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  const body = await response.json();
  if (response.status === 401) $("token-dialog").hidden = false;
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function showError(message) {
  $("empty-state").hidden = false;
  $("empty-state").querySelector("strong").textContent = "操作未完成";
  $("empty-state").querySelector("p").textContent = message;
}

function renderModules() {
  const target = $("modules");
  target.replaceChildren();
  $("panel-index").textContent = `${String(state.modules.findIndex((mod) => mod.slug === state.selected) + 1).padStart(2, "0")} / 06`;
  state.modules.forEach((mod, index) => {
    const button = node("button", `module-button${state.selected === mod.slug ? " active" : ""}`);
    button.type = "button";
    button.append(node("span", "module-number", String(index + 1).padStart(2, "0")));
    const label = node("span");
    label.append(node("span", "module-title", mod.title), node("span", "module-sub", mod.dependencies.length ? `前置：${mod.dependencies.length} 个模块` : "独立输入"));
    button.append(label, node("span", "module-arrow", "→"));
    button.addEventListener("click", () => { state.selected = mod.slug; renderModules(); renderForm(); });
    target.append(button);
  });
  $("job-state").textContent = state.jobId ? $("job-state").textContent : "待提交";
}

async function upload(kind, file, targetInput, progress) {
  if (!file) return;
  progress.textContent = "正在上传…";
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api/uploads?kind=${kind}`);
    request.setRequestHeader("Content-Type", "application/zip");
    if (state.token) request.setRequestHeader("Authorization", `Bearer ${state.token}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) progress.textContent = `上传中 ${Math.round(event.loaded * 100 / event.total)}%`;
    };
    request.onload = () => {
      try {
        const result = JSON.parse(request.responseText);
        if (request.status !== 201) throw new Error(result.error || `HTTP ${request.status}`);
        targetInput.value = result.directory;
        progress.textContent = "上传完成，已填入服务器目录";
        resolve(result);
      } catch (error) { progress.textContent = error.message; reject(error); }
    };
    request.onerror = () => { progress.textContent = "上传失败"; reject(new Error("上传失败")); };
    request.send(file);
  });
}

function renderForm() {
  const selected = state.modules.find((mod) => mod.slug === state.selected);
  if (!selected) return;
  $("modules").setAttribute("aria-label", `当前：${selected.title}`);
  $("form-fields").replaceChildren();
  const chain = selected.chain.map((slug) => state.modules.find((mod) => mod.slug === slug));
  const note = node("div", "chain-note", chain.length === 1 ? "此模块直接运行。" : `本次将依次运行：${chain.map((mod) => mod.title).join(" → ")}。上游输入会影响下游结果。`);
  $("form-fields").append(note);
  for (const mod of chain) {
    const group = node("fieldset", "field-group");
    group.append(node("legend", "", `${String(state.modules.indexOf(mod) + 1).padStart(2, "0")} · ${mod.title}`));
    for (const field of mod.fields) {
      const row = node("div", "field");
      const id = `${mod.slug}-${field.name}`;
      const label = node("label", "", field.label);
      label.htmlFor = id;
      row.append(label);
      let input;
      if (field.kind === "choice") {
        input = node("select");
        field.options.forEach((choice) => {
          const option = node("option", "", choice);
          option.value = choice;
          input.append(option);
        });
      } else {
        input = node("input");
        input.type = field.kind === "integer" ? "number" : "text";
        if (field.kind === "integer") { input.min = field.minimum; input.max = field.maximum; }
      }
      input.id = id;
      input.dataset.module = mod.slug;
      input.dataset.field = field.name;
      input.dataset.kind = field.kind;
      input.value = field.default;
      row.append(input, node("p", "", field.help));
      if (field.kind === "directory" || field.kind === "directory_optional") {
        const uploadRow = node("div", "upload-row");
        const fileInput = node("input");
        fileInput.type = "file";
        fileInput.accept = ".zip,application/zip";
        fileInput.setAttribute("aria-label", `${field.label} ZIP 上传`);
        const button = node("button", "", "上传 ZIP");
        button.type = "button";
        const progress = node("div", "upload-progress");
        button.addEventListener("click", async () => {
          try { await upload(field.name === "capture_dir" ? "capture" : "video", fileInput.files[0], input, progress); }
          catch (error) { showError(error.message); }
        });
        uploadRow.append(fileInput, button);
        row.append(uploadRow, progress);
      }
      group.append(row);
    }
    $("form-fields").append(group);
  }
}

function gatherInputs() {
  const inputs = {};
  $("form-fields").querySelectorAll("[data-module][data-field]").forEach((item) => {
    inputs[item.dataset.module] ||= {};
    inputs[item.dataset.module][item.dataset.field] = item.dataset.kind === "integer" ? Number(item.value) : item.value;
  });
  return inputs;
}

function statusLabel(status) {
  return ({ PENDING: "待运行", QUEUED: "排队中", RUNNING: "运行中", CANCELLING: "取消中", COMPLETED: "已完成", FAILED: "失败", CANCELLED: "已取消", INTERRUPTED: "已中断" })[status] || status;
}

function artifactUrl(jobId, item) { return `/api/jobs/${jobId}/artifacts/${item.artifact_id}`; }

function renderJob(job) {
  $("empty-state").hidden = true;
  $("job-detail").hidden = false;
  const badge = $("job-state");
  badge.className = `state-pill ${job.status}`;
  badge.textContent = statusLabel(job.status);
  const target = $("job-detail");
  target.replaceChildren();
  const body = node("div", "result-body");
  const meta = node("div", "job-meta");
  const title = node("div");
  title.append(node("strong", "", state.modules.find((mod) => mod.slug === job.module)?.title || job.module), node("p", "", `任务 ${job.id.slice(0, 12)} · ${new Date(job.created_at).toLocaleString("zh-CN")}`));
  meta.append(title);
  if (["QUEUED", "RUNNING"].includes(job.status)) {
    const cancel = node("button", "cancel-button", "取消任务");
    cancel.addEventListener("click", async () => { try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); await loadJob(job.id); } catch (error) { showError(error.message); } });
    meta.append(cancel);
  }
  body.append(meta);
  const progress = node("div", "progress-list");
  const statusByModule = Object.fromEntries((job.progress || job.result?.acceptance?.notebooks || []).map((item) => [item.module, item]));
  job.chain.forEach((slug) => {
    const item = statusByModule[slug] || { status: "PENDING" };
    const cells = item.cells_total ? ` · ${item.cells_completed}/${item.cells_total} 步` : "";
    progress.append(node("span", `progress-item ${item.status}`, `${state.modules.find((mod) => mod.slug === slug)?.title || slug} · ${statusLabel(item.status)}${cells}`));
  });
  body.append(progress);
  if (job.error) body.append(node("div", "error-box", job.error));
  if (job.result) {
    const media = job.result.artifacts.filter((item) => item.media_type?.startsWith("image/")).slice(0, 24);
    if (media.length) {
      const section = node("section", "result-section");
      section.append(node("h3", "", "本次生成的可视化"));
      const grid = node("div", "media-grid");
      media.forEach((item) => {
        const card = node("div", "media-card");
        const image = node("img");
        image.src = artifactUrl(job.id, item);
        image.alt = item.name;
        image.loading = "lazy";
        const caption = node("div", "caption");
        caption.append(node("span", "", item.name));
        const link = node("a", "", "查看 ↗");
        link.href = image.src;
        link.target = "_blank";
        link.rel = "noopener";
        caption.append(link);
        card.append(image, caption);
        grid.append(card);
      });
      section.append(grid);
      body.append(section);
    }
    const artifacts = node("section", "result-section");
    artifacts.append(node("h3", "", `本次产物 · ${job.result.artifacts.length} 项`));
    const list = node("div", "artifact-list");
    job.result.artifacts.slice(0, 60).forEach((item) => {
      const link = node("a", "", `${item.name} ↗`);
      link.href = artifactUrl(job.id, item);
      link.target = "_blank";
      link.rel = "noopener";
      list.append(link);
    });
    artifacts.append(list);
    body.append(artifacts);
    const detail = node("details", "result-section");
    detail.append(node("summary", "", "查看完整结果索引"), node("pre", "json-panel", JSON.stringify(job.result.results, null, 2)));
    body.append(detail);
  }
  if (job.log_tail) {
    const detail = node("details", "result-section");
    detail.append(node("summary", "", "查看执行日志"), node("pre", "log-panel", job.log_tail));
    body.append(detail);
  }
  target.append(body);
}

async function loadJob(id) {
  const job = await api(`/api/jobs/${id}`);
  if (state.jobId === id) renderJob(job);
}

async function refresh() {
  const records = await api("/api/jobs");
  if (!state.jobId && records.length) state.jobId = records[0].id;
  const list = $("history");
  list.replaceChildren();
  records.forEach((job) => {
    const button = node("button");
    button.type = "button";
    button.append(node("span", "", `${state.modules.find((mod) => mod.slug === job.module)?.title || job.module} · ${job.id.slice(0, 8)}`), node("span", "", statusLabel(job.status)));
    button.addEventListener("click", () => { state.jobId = job.id; loadJob(job.id).catch((error) => showError(error.message)); });
    list.append(button);
  });
  if (state.jobId) await loadJob(state.jobId);
}

async function initialize() {
  try {
    state.modules = await api("/api/modules");
    state.selected ||= state.modules[0]?.slug;
    renderModules();
    renderForm();
    const health = await api("/api/health");
    const free = Math.max(0, ...(health.gpu_devices || []).map((device) => device.free_mib));
    $("health").className = `health ${free >= 8192 ? "ready" : "warn"}`;
    $("health").textContent = health.gpu_available ? `服务在线 · GPU 最大空闲 ${(free / 1024).toFixed(1)} GB` : "服务在线 · GPU 不可用";
    await refresh();
  } catch (error) { showError(error.message); }
}

$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ module: state.selected, inputs: gatherInputs() }) });
    state.jobId = job.id;
    await refresh();
  } catch (error) { showError(error.message); }
});
$("refresh").addEventListener("click", () => refresh().catch((error) => showError(error.message)));
$("token-save").addEventListener("click", async () => {
  const token = $("token-input").value;
  try {
    const response = await fetch("/api/session", { method: "POST", headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new Error("令牌错误");
    state.token = token;
    sessionStorage.setItem("prx_token", token);
    $("token-dialog").hidden = true;
    await initialize();
  } catch (error) { showError(error.message); }
});
initialize();
setInterval(() => refresh().catch(() => {}), 2500);
