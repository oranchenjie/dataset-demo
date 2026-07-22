/**
 * 数据集处理Demo - 前端交互
 * 支持混合检索（FAISS向量 + 关键词）
 */
(function () {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const CAT_EMOJI = { pdf: '\u{1F4C4}', video: '\u{1F3AC}', image: '\u{1F5BC}', text: '\u{1F4DD}',
    archive: '\u{1F4E6}', other: '\u{1F4CE}', dir: '\u{1F4C1}' };
  const CAT_LABEL = { pdf: 'PDF文档', video: '视频', image: '图片', text: '文本', archive: '压缩包', other: '其他' };

  // ========== Tabs ==========
  function initTabs() {
    $$('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        $$('.tab-panel').forEach(p => p.classList.remove('active'));
        $('#tab-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'overview') loadOverview();
        if (btn.dataset.tab === 'video') loadVideoList();
      });
    });
  }

  // ========== Overview ==========
  let ovState = { forceRefresh: false };
  async function loadOverview() {
    const loading = $('#overview-loading');
    const content = $('#overview-content');
    loading.style.display = 'block';
    content.style.display = 'none';
    try {
      const res = await fetch('/api/stats?force=' + (ovState.forceRefresh ? '1' : '0'));
      ovState.forceRefresh = false;
      const data = await res.json();
      if (data.error) { loading.textContent = data.error; return; }
      renderCatCards(data);
      renderDirTree(data.tree);
      loading.style.display = 'none';
      content.style.display = 'block';
    } catch (err) { loading.textContent = '加载失败: ' + err.message; }
  }

  function renderCatCards(data) {
    const c = $('#cat-cards');
    c.innerHTML = '';
    // 关键词索引
    const idxDocs = data.index_docs !== undefined ? data.index_docs : 0;
    const idxReady = data.index_ready === true;
    c.innerHTML += `<div class="stat-card"><div class="stat-icon">\u{1F50D}</div><div class="stat-info">
      <div class="stat-number">${idxDocs}</div>
      <div class="stat-label">关键词索引</div>
      <div class="stat-detail">${idxReady ? '就绪' : '未就绪'}</div></div></div>`;
    // 向量索引
    const fVectors = data.faiss_vectors !== undefined ? data.faiss_vectors : 0;
    const vecAvail = data.vector_search_available === true;
    c.innerHTML += `<div class="stat-card"><div class="stat-icon">\u{1F9EA}</div><div class="stat-info">
      <div class="stat-number">${fVectors}</div>
      <div class="stat-label">向量索引 (FAISS)</div>
      <div class="stat-detail">${vecAvail ? '\u{2705} 混合检索可用' : '\u{274C} 仅关键词'}</div></div></div>`;
    // 文件类型
    const cats = data.categories || {};
    for (const [cat, info] of Object.entries(cats)) {
      const count = info.count !== undefined ? info.count : 0;
      const pages = info.total_pages || 0;
      let detail = count + ' 个文件';
      if (cat === 'pdf' && pages > 0) detail += ' / ' + pages + ' 页';
      c.innerHTML += `<div class="stat-card"><div class="stat-icon">${CAT_EMOJI[cat] || '\u{1F4CE}'}</div><div class="stat-info">
        <div class="stat-number">${count}</div>
        <div class="stat-label">${CAT_LABEL[cat] || cat}</div>
        <div class="stat-detail">${detail}</div></div></div>`;
    }
  }

  // ========== Dir Tree ==========
  function renderDirTreeHTML(node, depth) {
    if (!node) return '';
    depth = depth || 0;
    const isRoot = depth === 0;
    const indent = isRoot ? '' : 'margin-left:' + (depth * 20) + 'px;';
    const catTags = node.categories ? Object.entries(node.categories).map(([c, n]) =>
      `<span class="cat-tag cat-${c}">${n}</span>`).join('') : '';
    const fc = node.file_count || 0;
    const sz = node.total_size_mb || 0;
    let html = `<div class="tree-dir" style="${indent}">
      <div class="tree-dir-header">
        <span class="tree-icon">${isRoot ? '\u{1F4C2}' : '\u{1F4C1}'}</span>
        <span class="tree-name">${esc(node.name || '(根目录)')}</span>
        <span class="tree-meta">${fc} 文件 · ${sz} MB</span>
        <span class="tree-cats">${catTags}</span>
      </div>`;
    if (node.files && node.files.length > 0) {
      html += '<div class="tree-files">';
      for (const f of node.files) {
        const cat = f.category || 'other';
        const emoji = CAT_EMOJI[cat] || '\u{1F4CE}';
        const name = esc(f.name || '');
        const mb = (f.size_mb !== undefined ? f.size_mb : 0) + ' MB';
        let meta = mb;
        if (f.pages) meta = f.pages + '页 / ' + meta;
        if (f.video_info && f.video_info.width) meta = f.video_info.width + 'x' + f.video_info.height + ' / ' + meta;
        html += `<div class="tree-file"><span>${emoji} ${name}</span><span class="file-meta">${meta}</span></div>`;
      }
      html += '</div>';
    }
    html += '</div>';
    if (node.subdirs) for (const sub of node.subdirs) html += renderDirTreeHTML(sub, depth + 1);
    return html;
  }

  function renderDirTree(node, depth) {
    $('#dir-tree').innerHTML = renderDirTreeHTML(node, depth) || '<div class="loading">暂无文件</div>';
  }

  // ========== Search ==========
  let curSearch = { page: 1, query: '' };

  function initSearch() {
    $('#btn-search').addEventListener('click', () => doSearch(1));
    $('#search-input').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(1); });
  }

  async function doSearch(page) {
    const q = $('#search-input').value.trim();
    if (!q) { $('#search-status').textContent = '请输入搜索关键词'; return; }
    const mode = $('#search-mode').value;
    curSearch = { page, query: q, mode };

    $('#search-status').textContent = '';
    $('#search-results').innerHTML = '<div class="loading">搜索中...</div>';
    $('#search-results-info').style.display = 'none';
    $('#search-pagination').innerHTML = '';

    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&page=${page}&mode=${mode}`);
      const d = await r.json();
      if (d.error) { $('#search-status').textContent = d.error; $('#search-results').innerHTML = ''; return; }

      $('#search-results-info').style.display = 'block';
      $('#result-total').textContent = d.total;

      const modeLabel = { hybrid: '混合检索（向量+关键词）', vector: '语义检索（FAISS向量）', keyword: '关键词检索' };
      $('#result-mode-info').textContent = ' · ' + (modeLabel[d.mode] || d.mode);

      renderSearchResults(d.results);
      renderPagination(d.total_pages, d.page, doSearch);
    } catch (err) {
      $('#search-status').textContent = '请求失败: ' + err.message;
      $('#search-results').innerHTML = '';
    }
  }

  function renderSearchResults(results) {
    const container = $('#search-results');
    if (results.length === 0) {
      container.innerHTML = '<div class="loading">未找到匹配结果</div>';
      return;
    }
    container.innerHTML = results.map(r => {
      if (r.type === 'video') {
        const ts = r.timestamp ? r.timestamp.toFixed(1) + 's' : '-';
        return `<div class="search-result search-result-video" data-file="${esc(r.file)}" data-frame="${r.frame}">
          <div class="result-header">
            <span class="result-file"><span class="badge badge-video">视频帧</span> ${esc(r.filename)}</span>
            <span class="result-meta">相似度 ${r.score}</span>
          </div>
          <div class="result-body">
            <span class="result-video-preview">帧 ${r.frame} · ${ts}</span>
            <span class="result-mode-tag">${r.mode || '语义匹配'}</span>
          </div>
        </div>`;
      }
      // PDF - 带匹配方式标记
      const modeClass = { '语义匹配': 'mode-vec', '关键词匹配': 'mode-kw', '混合匹配': 'mode-hybrid' };
      const modeLabel = r.mode || '语义匹配';
      const modeIcon = { '语义匹配': '🧠', '关键词匹配': '🔤', '混合匹配': '⚡' };
      return `<div class="search-result" data-file="${esc(r.file)}" data-page="${r.page}">
        <div class="result-header">
          <span class="result-file"><span class="badge badge-pdf">PDF</span> ${esc(r.filename)}</span>
          <span class="result-meta"><span class="mode-badge ${modeClass[modeLabel] || 'mode-vec'}">${modeIcon[modeLabel]||''} ${modeLabel}</span> · ${r.score}</span>
        </div>
        <div class="result-snippet">${r.snippet}</div>
      </div>`;
    }).join('');

    container.querySelectorAll('.search-result').forEach(el => {
      const file = el.dataset.file;
      const page = el.dataset.page;
      const frame = el.dataset.frame;
      if (frame !== undefined) {
        el.addEventListener('click', () => jumpToVideoFrame(file, parseInt(frame)));
      } else if (page) {
        el.addEventListener('click', () => openPdfPage(file, parseInt(page)));
      }
    });
  }

  function renderPagination(tp, cp, cb) {
    const c = $('#search-pagination');
    if (tp <= 1) { c.innerHTML = ''; return; }
    let h = '';
    for (let i = 1; i <= Math.min(tp, 15); i++) h += `<button class="btn btn-sm${i === cp ? ' btn-primary' : ''}">${i}</button>`;
    if (tp > 15) h += `<span style="line-height:32px;color:#94a3b8;margin-left:8px;">... ${tp}</span>`;
    c.innerHTML = h;
    c.querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => cb(i + 1)));
  }

  // ========== PDF Modal ==========
  let pdfModal = { file: '', page: 1, total: 1 };
  function initPdfModal() {
    const m = $('#pdf-modal');
    $('.modal-overlay', m).addEventListener('click', closePdf);
    $('.modal-close', m).addEventListener('click', closePdf);
    $('#pdf-prev-page').addEventListener('click', () => { if (pdfModal.page > 1) openPdfPage(pdfModal.file, pdfModal.page - 1); });
    $('#pdf-next-page').addEventListener('click', () => { if (pdfModal.page < pdfModal.total) openPdfPage(pdfModal.file, pdfModal.page + 1); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closePdf(); });
  }
  async function openPdfPage(file, page) {
    $('#pdf-modal').style.display = 'flex';
    $('#pdf-modal-title').textContent = file.split('/').pop();
    $('#pdf-page-text').textContent = '加载中...';
    try {
      const r = await fetch('/api/pdf/page?file=' + encodeURIComponent(file) + '&page=' + page);
      if (!r.ok) { const e = await r.json(); $('#pdf-page-text').textContent = '错误: ' + (e.error || r.statusText); return; }
      const d = await r.json();
      pdfModal = { file, page, total: d.total_pages };
      $('#pdf-page-info').textContent = '第 ' + page + ' 页 / 共 ' + d.total_pages + ' 页';
      const textEl = $('#pdf-page-text');
      if (d.is_image && d.image_url) {
        textEl.innerHTML = '<img src="' + d.image_url + '&_=' + Date.now() + '" style="max-width:100%;height:auto;border-radius:4px;" alt="PDF页面渲染图" />';
      } else {
        textEl.textContent = d.text || '(无文本)';
      }
    } catch (err) { $('#pdf-page-text').textContent = '加载失败: ' + err.message; }
  }
  function closePdf() { $('#pdf-modal').style.display = 'none'; }

  // ========== 跳转到视频帧（从搜索结果点击） ==========
  function jumpToVideoFrame(file, frame) {
    // 切换到视频tab
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('[data-tab="video"]').classList.add('active');
    $$('.tab-panel').forEach(p => p.classList.remove('active'));
    $('#tab-video').classList.add('active');

    // 选择视频
    const sel = $('#video-select');
    for (const opt of sel.options) {
      if (opt.value === file) { sel.value = file; break; }
    }
    // 加载并跳转
    loadVideoWithFrame(file, frame);
  }

  async function loadVideoWithFrame(file, frame) {
    vid.file = file;
    vid.pinned = null;
    $('#pixel-detail').style.display = 'none';
    $('#video-error').style.display = 'none';
    try {
      const r = await fetch('/api/videos');
      const d = await r.json();
      const v = d.videos.find(x => x.path === file);
      if (v && v.video_info) {
        vid.total = v.video_info.total_frames;
        vid.fps = v.video_info.fps;
        $('#vinfo-resolution').textContent = v.video_info.width + ' x ' + v.video_info.height;
        $('#vinfo-fps').textContent = v.video_info.fps;
        $('#vinfo-frames').textContent = v.video_info.total_frames;
        $('#vinfo-duration').textContent = v.video_info.duration;
        $('#video-info-bar').style.display = 'flex';
        $('#frame-nav').style.display = 'flex';
        $('#frame-input').max = vid.total - 1;
        $('#frame-total-label').textContent = '/ ' + (vid.total - 1);
        showFrame(frame);
      } else {
        showVideoError('无法读取该视频信息');
      }
    } catch (err) { showVideoError('加载失败: ' + err.message); }
  }

  // ========== Video ==========
  let vid = { file: '', total: 0, frame: 0, fps: 0, pinned: null };

  async function loadVideoList() {
    try {
      const r = await fetch('/api/videos');
      const d = await r.json();
      const sel = $('#video-select');
      sel.innerHTML = '<option value="">-- 请选择视频 --</option>';
      const groups = {};
      d.videos.forEach(v => { const g = v.parent_dir || '(根目录)'; if (!groups[g]) groups[g] = []; groups[g].push(v); });
      for (const [dir, videos] of Object.entries(groups)) {
        if (Object.keys(groups).length > 1) {
          const og = document.createElement('optgroup');
          og.label = dir;
          videos.forEach(v => { const o = document.createElement('option'); o.value = v.path; o.textContent = v.name + (v.video_info ? ` (${v.video_info.total_frames}帧, ${v.video_info.width}x${v.video_info.height})` : ''); og.appendChild(o); });
          sel.appendChild(og);
        } else {
          videos.forEach(v => { const o = document.createElement('option'); o.value = v.path; o.textContent = v.name + (v.video_info ? ` (${v.video_info.total_frames}帧, ${v.video_info.width}x${v.video_info.height})` : ''); sel.appendChild(o); });
        }
      }
    } catch (err) { console.error(err); }
  }

  function initVideo() {
    $('#btn-load-video').addEventListener('click', () => loadVideoWithFrame($('#video-select').value, 0));
    $('#btn-jump-frame').addEventListener('click', () => { const n = parseInt($('#frame-input').value); if (!isNaN(n)) showFrame(n); });
    $('#frame-input').addEventListener('keydown', e => { if (e.key === 'Enter') { const n = parseInt($('#frame-input').value); if (!isNaN(n)) showFrame(n); } });
    $('#btn-frame-prev').addEventListener('click', () => showFrame(vid.frame - 1));
    $('#btn-frame-next').addEventListener('click', () => showFrame(vid.frame + 1));
    $('#btn-frame-start').addEventListener('click', () => showFrame(0));
    $('#btn-frame-end').addEventListener('click', () => showFrame(vid.total - 1));
    document.addEventListener('keydown', e => {
      if (!$('#tab-video').classList.contains('active')) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowLeft') showFrame(vid.frame - 1);
      if (e.key === 'ArrowRight') showFrame(vid.frame + 1);
    });
  }

  function showFrame(n) {
    if (!vid.file) return;
    n = Math.max(0, Math.min(n, vid.total - 1));
    vid.frame = n;
    $('#frame-input').value = n;
    $('#finfo-frame').textContent = n;
    $('#finfo-time').textContent = vid.fps > 0 ? (n / vid.fps).toFixed(2) + ' s' : '-';
    $('#video-loading').style.display = 'block';
    $('#video-viewer').style.display = 'none';
    $('#video-error').style.display = 'none';
    const img = new Image();
    img.onload = () => {
      $('#video-loading').style.display = 'none';
      $('#video-viewer').style.display = 'block';
      const c = $('#frame-canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      $('#finfo-size').textContent = img.naturalWidth + ' x ' + img.naturalHeight;
      if (vid.pinned) displayPinned(vid.pinned);
    };
    img.onerror = () => { $('#video-loading').style.display = 'none'; showVideoError('帧加载失败'); };
    img.src = '/api/video/frame?file=' + encodeURIComponent(vid.file) + '&frame=' + n + '&_=' + Date.now();
  }

  function showVideoError(msg) {
    $('#video-loading').style.display = 'none';
    $('#video-viewer').style.display = 'none';
    const el = $('#video-error');
    el.textContent = msg;
    el.style.display = 'block';
  }

  // ========== Pixel ==========
  function initPixelInspector() {
    const canvas = $('#frame-canvas');
    const tooltip = $('#pixel-tooltip');
    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
      const x = Math.floor((e.clientX - rect.left) * sx);
      const y = Math.floor((e.clientY - rect.top) * sy);
      if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) { tooltip.style.display = 'none'; return; }
      const p = canvas.getContext('2d').getImageData(x, y, 1, 1).data;
      const hex = '#' + [p[0], p[1], p[2]].map(c => c.toString(16).padStart(2, '0')).join('');
      $('#tooltip-swatch').style.backgroundColor = hex;
      $('#tooltip-x').textContent = x;
      $('#tooltip-y').textContent = y;
      $('#tooltip-r').textContent = p[0];
      $('#tooltip-g').textContent = p[1];
      $('#tooltip-b').textContent = p[2];
      $('#tooltip-hex').textContent = hex;
      let tx = e.clientX - rect.left + 18, ty = e.clientY - rect.top + 18;
      if (tx + 180 > rect.width) tx -= 200;
      if (ty + 110 > rect.height) ty -= 120;
      if (tx < 0) tx = 5;
      if (ty < 0) ty = 5;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
      tooltip.style.display = 'block';
    });
    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    canvas.addEventListener('click', e => {
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
      const x = Math.floor((e.clientX - rect.left) * sx);
      const y = Math.floor((e.clientY - rect.top) * sy);
      if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;
      const p = canvas.getContext('2d').getImageData(x, y, 1, 1).data;
      vid.pinned = { x, y, r: p[0], g: p[1], b: p[2], hex: '#' + [p[0], p[1], p[2]].map(c => c.toString(16).padStart(2, '0')).join('') };
      displayPinned(vid.pinned);
    });
  }

  function displayPinned(p) {
    $('#pixel-detail').style.display = 'block';
    $('#pdetail-pos').textContent = '(' + p.x + ', ' + p.y + ')';
    $('#pdetail-r').textContent = p.r;
    $('#pdetail-g').textContent = p.g;
    $('#pdetail-b').textContent = p.b;
    $('#pdetail-hex').textContent = p.hex;
    $('#pdetail-swatch').style.backgroundColor = p.hex;
  }

  // ========== Util ==========
  function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  // ========== Init ==========
  function init() {
    initTabs();
    initSearch();
    initPdfModal();
    initVideo();
    initPixelInspector();
    loadOverview();
    $('#btn-refresh').addEventListener('click', () => { ovState.forceRefresh = true; loadOverview(); });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
