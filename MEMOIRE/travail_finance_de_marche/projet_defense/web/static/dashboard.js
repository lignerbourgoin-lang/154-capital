/* ============================================================
   State
   ============================================================ */
let currentPanel = null;   // which panel is open in overlay

/* ============================================================
   Overlay — open
   ============================================================ */
function openOverlay(panelId) {
  currentPanel = panelId;
  const cfg    = CHARTS[panelId];

  document.getElementById("sidebar-title").textContent = cfg.label;
  document.getElementById("overlay").classList.remove("hidden");

  _buildSidebar(cfg);
  _renderCharts();
}

/* ============================================================
   Overlay — close
   ============================================================ */
function closeOverlay() {
  document.getElementById("overlay").classList.add("hidden");
  currentPanel = null;
}

/* ============================================================
   Sidebar builder
   ============================================================ */
function _buildSidebar(cfg) {
  const body = document.getElementById("sidebar-body");
  body.innerHTML = "";

  // Company checkboxes
  if (cfg.hasCompany) {
    const sec = _sidebarSection("Entreprise");
    COMPANIES.forEach((c, idx) => {
      sec.append(_checkbox(`co_${c.slug}`, c.name, idx === 0, _renderCharts));
    });
    body.append(sec);
  }

  // Analysis type checkboxes
  if (cfg.hasType && cfg.types) {
    const sec = _sidebarSection("Analyse");
    cfg.types.forEach((t, idx) => {
      sec.append(_checkbox(`ty_${t.id}`, t.label, idx === 0, _renderCharts));
    });
    body.append(sec);
  }
}

function _sidebarSection(title) {
  const wrap = document.createElement("div");
  wrap.className = "sidebar-section";
  const h3 = document.createElement("h3");
  h3.textContent = title;
  wrap.append(h3);
  return wrap;
}

function _checkbox(id, label, checked, onChange) {
  const lbl   = document.createElement("label");
  const input = document.createElement("input");
  input.type    = "checkbox";
  input.id      = id;
  input.checked = checked;
  input.addEventListener("change", onChange);
  lbl.append(input, label);
  return lbl;
}

/* ============================================================
   Chart renderer
   ============================================================ */
function _renderCharts() {
  const cfg       = CHARTS[currentPanel];
  const container = document.getElementById("overlay-charts");
  container.innerHTML = "";

  const paths = _collectPaths(cfg);
  if (paths.length === 0) return;

  container.classList.toggle("single", paths.length === 1);

  paths.forEach(src => {
    const wrap   = document.createElement("div");
    wrap.className = "overlay-iframe-wrap";
    const iframe = document.createElement("iframe");
    iframe.src   = src;
    wrap.append(iframe);
    container.append(wrap);
  });
}

function _collectPaths(cfg) {
  const paths = [];

  // Panels with no selectors
  if (!cfg.hasCompany && !cfg.hasType) {
    paths.push(cfg.path());
    return paths;
  }

  // Selected companies
  const selectedCompanies = cfg.hasCompany
    ? COMPANIES.filter(c => {
        const el = document.getElementById(`co_${c.slug}`);
        return el && el.checked;
      })
    : [null];

  // Selected types
  const selectedTypes = cfg.hasType && cfg.types
    ? cfg.types.filter(t => {
        const el = document.getElementById(`ty_${t.id}`);
        return el && el.checked;
      })
    : [null];

  selectedCompanies.forEach((company, idx) => {
    const companyIdx = company ? COMPANIES.indexOf(company) : 0;

    selectedTypes.forEach(type => {
      const typeId = type ? type.id : null;
      paths.push(cfg.path(typeId, company, companyIdx));
    });
  });

  return paths;
}

/* ============================================================
   Close overlay on Escape key
   ============================================================ */
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && currentPanel) closeOverlay();
});
