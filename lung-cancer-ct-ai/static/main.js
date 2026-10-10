/* ── main.js — LungCare AI ── */
'use strict';

/* ═══════════════════════════════════════════
   NAVBAR — scroll shadow + active link
═══════════════════════════════════════════ */
const navbar   = document.getElementById('navbar');
const navItems = document.querySelectorAll('.nav-item');
const sections = document.querySelectorAll('section[id]');
const hamburger= document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 30);

  // Active nav item on scroll
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  });
  navItems.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
});

hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));

// Smooth-close on link click (mobile)
navItems.forEach(a => a.addEventListener('click', () => navLinks.classList.remove('open')));

/* ═══════════════════════════════════════════
   ANIMATED SCAN LINE (hero card)
═══════════════════════════════════════════ */
const scanPreview = document.querySelector('.scan-preview');
if (scanPreview) {
  const line = document.createElement('div');
  line.className = 'scan-line';
  scanPreview.appendChild(line);
}

/* ═══════════════════════════════════════════
   HERO STATS  counter-up
═══════════════════════════════════════════ */
function animateCounter(el, target, suffix, duration=1800) {
  const isFloat = target % 1 !== 0;
  const start   = performance.now();
  const update  = (now) => {
    const t   = Math.min((now - start) / duration, 1);
    const ease= t < .5 ? 2*t*t : -1+(4-2*t)*t;
    const val = isFloat
      ? (ease * target).toFixed(2)
      : Math.round(ease * target).toLocaleString();
    el.textContent = val + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

const statsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const nums = [
      { el: document.querySelector('.stat:nth-child(1) .stat-num'), val: 81.87, suffix: '%' },
      { el: document.querySelector('.stat:nth-child(3) .stat-num'), val: 5,     suffix: ''  },
      { el: document.querySelector('.stat:nth-child(5) .stat-num'), val: 1226,  suffix: ''  },
    ];
    nums.forEach(n => { if (n.el) animateCounter(n.el, n.val, n.suffix); });
    statsObserver.disconnect();
  });
}, { threshold:.5 });
const heroStats = document.querySelector('.hero-stats');
if (heroStats) statsObserver.observe(heroStats);

/* Analytics counters */
const analyticsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const counters = [
      { sel: '.analytics-card.primary .analytics-num', val: 1226, suffix: '' },
      { sel: '.analytics-card.success .analytics-num', val: 81.87,suffix: '%' },
      { sel: '.analytics-card.danger  .analytics-num', val: 99.45,suffix: '%' },
      { sel: '.analytics-card.warning .analytics-num', val: 5,    suffix: '' },
    ];
    counters.forEach(c => {
      const el = document.querySelector(c.sel);
      if (el) animateCounter(el, c.val, c.suffix);
    });
    analyticsObserver.disconnect();
  });
}, { threshold:.4 });
const analyticsGrid = document.querySelector('.analytics-grid');
if (analyticsGrid) analyticsObserver.observe(analyticsGrid);

/* ═══════════════════════════════════════════
   FILE UPLOAD — drag & drop + preview
═══════════════════════════════════════════ */
const uploadArea     = document.getElementById('uploadArea');
const ctFile         = document.getElementById('ctFile');
const previewWrap    = document.getElementById('uploadPreviewWrap');
const previewImg     = document.getElementById('uploadPreview');
const previewName    = document.getElementById('uploadFileName');

if (uploadArea) {
  uploadArea.addEventListener('click', () => ctFile.click());

  ['dragover','dragenter'].forEach(evt =>
    uploadArea.addEventListener(evt, e => { e.preventDefault(); uploadArea.classList.add('dragover'); })
  );
  ['dragleave','dragend'].forEach(evt =>
    uploadArea.addEventListener(evt, () => uploadArea.classList.remove('dragover'))
  );
  uploadArea.addEventListener('drop', e => {
    e.preventDefault(); uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) showPreview(file);
    // Manually assign to file input (some browsers support DataTransfer)
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      ctFile.files = dt.files;
    } catch(_) {}
  });

  ctFile.addEventListener('change', () => {
    if (ctFile.files[0]) showPreview(ctFile.files[0]);
  });
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewWrap.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

/* ═══════════════════════════════════════════
   PREDICTION FORM SUBMIT
═══════════════════════════════════════════ */
const predForm        = document.getElementById('predForm');
const resultsPanel    = document.getElementById('resultsPanel');
const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsContent  = document.getElementById('resultsContent');
const loadingOverlay  = document.getElementById('loadingOverlay');
let   lastFormData    = null;   // kept for PDF download

if (predForm) {
  predForm.addEventListener('submit', async e => {
    e.preventDefault();

    const file = ctFile.files[0];
    if (!file) { showAlert('danger', 'fa-exclamation-circle', 'Please upload a CT scan image first.'); return; }

    // Show loader
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display     = 'none';
    loadingOverlay.style.display     = 'flex';

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior:'smooth', block:'nearest' });

    const fd = new FormData(predForm);

    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });

      // Parse JSON regardless of status code
      let data;
      try { data = await res.json(); }
      catch(_) { throw new Error('Server returned an unreadable response (status ' + res.status + ')'); }

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Server error ' + res.status);
      }

      // Store for PDF download
      lastFormData = true;
      renderResults(data);

    } catch(err) {
      loadingOverlay.style.display     = 'none';
      resultsPlaceholder.style.display = 'flex';
      alert('Prediction failed: ' + err.message +
            '\n\nMake sure server.py is running and the model file exists.');
    }
  });
}

/* ═══════════════════════════════════════════
   RENDER RESULTS
═══════════════════════════════════════════ */
function renderResults(data) {
  loadingOverlay.style.display = 'none';
  resultsContent.style.display = 'block';

  const isMal = data.is_malignant;
  const isUnc = data.is_uncertain;

  // Banner class
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (isUnc ? 'unc' : isMal ? 'mal' : 'safe');

  document.getElementById('verdictIcon').innerHTML =
    isUnc ? '<i class="fas fa-question-circle"></i>'
    : isMal ? '<i class="fas fa-triangle-exclamation"></i>'
    : '<i class="fas fa-check-circle"></i>';

  document.getElementById('verdictClass').textContent = data.pred_class_fmt.toUpperCase();
  document.getElementById('verdictType').textContent  = isUnc ? 'UNCERTAIN'
    : isMal ? 'MALIGNANT' : 'NON-MALIGNANT';

  // Confidence ring
  const pct   = data.confidence;
  const circ  = 2 * Math.PI * 35;                // r=35
  const offset= circ - (pct / 100) * circ;
  document.getElementById('ringFill').style.strokeDashoffset = offset;
  document.getElementById('confPct').textContent = pct.toFixed(1) + '%';

  // Images
  document.getElementById('origImg').src  = 'data:image/jpeg;base64,' + data.orig_b64;
  document.getElementById('gradImg').src  = 'data:image/jpeg;base64,' + data.gradcam_b64;

  // Probability bars
  const probBars  = document.getElementById('probBars');
  probBars.innerHTML = '';
  const sorted = Object.entries(data.probabilities)
                       .sort((a,b) => b[1]-a[1]);

  const colors = {
    'adenocarcinoma'        : '#ef4444',
    'large cell carcinoma'  : '#f97316',
    'squamous cell carcinoma':'#dc2626',
    'normal cases'          : '#10b981',
    'benign cases'          : '#3b82f6',
  };

  sorted.forEach(([cls, prob]) => {
    const color = colors[cls] || '#3b82f6';
    const isBest= cls === data.pred_class;
    const div   = document.createElement('div');
    div.className = 'prob-item';
    div.innerHTML = `
      <div class="prob-row">
        <span class="prob-name" style="${isBest ? 'font-weight:800;color:var(--navy)' : ''}">${cls.replace(/\b\w/g,c=>c.toUpperCase())}</span>
        <span class="prob-val">${prob.toFixed(2)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="background:${color}" data-width="${prob}"></div>
      </div>`;
    probBars.appendChild(div);
  });

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.prob-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    });
  });

  // Alert box
  const alertBox = document.getElementById('alertBox');
  if (isUnc) {
    alertBox.style.display = 'flex';
    alertBox.className = 'alert-box warning';
    alertBox.innerHTML = '<i class="fas fa-triangle-exclamation"></i> Confidence is below 60%. Treat this result as UNCERTAIN. Please consult a radiologist.';
  } else if (isMal) {
    alertBox.style.display = 'flex';
    alertBox.className = 'alert-box danger';
    alertBox.innerHTML = '<i class="fas fa-circle-exclamation"></i> Malignant classification detected. This is a research prototype — always consult a qualified medical professional.';
  } else {
    alertBox.style.display = 'flex';
    alertBox.className = 'alert-box success';
    alertBox.innerHTML = '<i class="fas fa-circle-check"></i> Non-malignant classification. Continue with regular medical check-ups.';
  }

  // Scroll to results
  resultsContent.scrollIntoView({ behavior:'smooth', block:'nearest' });
}

/* ═══════════════════════════════════════════
   PDF DOWNLOAD
═══════════════════════════════════════════ */
const downloadBtn = document.getElementById('downloadBtn');
if (downloadBtn) {
  downloadBtn.addEventListener('click', async () => {
    if (!lastFormData || !ctFile.files[0]) {
      alert('Please run an analysis first.'); return;
    }
    downloadBtn.disabled  = true;
    downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF…';

    // Re-build FormData with original file
    const fd = new FormData(predForm);

    try {
      const res = await fetch('/download_pdf', { method:'POST', body:fd });
      if (!res.ok) throw new Error('PDF generation failed.');
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = 'LungCare_Report_' + Date.now() + '.pdf';
      a.click();
      URL.revokeObjectURL(url);
    } catch(err) {
      alert('Error generating PDF: ' + err.message);
    }

    downloadBtn.disabled  = false;
    downloadBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Download PDF Report';
  });
}

/* ═══════════════════════════════════════════
   CONTACT FORM
═══════════════════════════════════════════ */
const contactForm = document.getElementById('contactForm');
if (contactForm) {
  contactForm.addEventListener('submit', e => {
    e.preventDefault();
    const btn = contactForm.querySelector('button[type=submit]');
    btn.innerHTML = '<i class="fas fa-check"></i> Message Sent!';
    btn.style.background = 'linear-gradient(135deg,#10b981,#34d399)';
    btn.disabled = true;
    setTimeout(() => {
      btn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Message';
      btn.style.background = '';
      btn.disabled = false;
      contactForm.reset();
    }, 3000);
  });
}

/* ═══════════════════════════════════════════
   INTERSECTION OBSERVER — fade-in on scroll
═══════════════════════════════════════════ */
const fadeEls = document.querySelectorAll('.feature-card, .class-card, .analytics-card, .model-info-card, .contact-item, .step');
const fadeObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity  = '1';
      e.target.style.transform= 'translateY(0)';
      fadeObs.unobserve(e.target);
    }
  });
}, { threshold:.15 });

fadeEls.forEach((el, i) => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(30px)';
  el.style.transition= `opacity .5s ease ${i * 60}ms, transform .5s ease ${i * 60}ms`;
  fadeObs.observe(el);
});
