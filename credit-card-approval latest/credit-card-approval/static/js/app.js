/**
 * CreditAI — Frontend JavaScript
 * Handles: multi-step form, scroll effects, summary updates, form submission
 */

// ── Current step tracker ────────────────────────────────────────────
let currentStep = 1;
const TOTAL_STEPS = 3;

// ── DOM References ──────────────────────────────────────────────────
const progressFill = document.getElementById('progressFill');

// ══════════════════════════════════════════════════════════════════
// MULTI-STEP FORM
// ══════════════════════════════════════════════════════════════════

function goToStep(step) {
  if (step < 1 || step > TOTAL_STEPS) return;
  if (step > currentStep && !validateStep(currentStep)) return;

  // Hide current, show target
  document.getElementById(`step-${currentStep}`)?.classList.remove('active');
  document.getElementById(`step-${step}`)?.classList.add('active');

  // Update step indicators
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    const indicator = document.getElementById(`fsi-${i}`);
    if (!indicator) continue;
    indicator.classList.remove('active', 'done');
    if (i === step)       indicator.classList.add('active');
    else if (i < step)    indicator.classList.add('done');
  }

  // Update progress bar
  if (progressFill) {
    progressFill.style.width = `${(step / TOTAL_STEPS) * 100}%`;
  }

  currentStep = step;

  // Scroll to form smoothly
  const formSection = document.getElementById('apply') || document.getElementById('applicationForm');
  if (formSection) {
    formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Update summary on step 3
  if (step === 3) updateSummary();
}

function nextStep(step) { goToStep(step); }

// ── Form Error Helpers ────────────────────────────────────────────
function showFormError(message) {
  const banner = document.getElementById('formErrorBanner');
  const text   = document.getElementById('formErrorText');
  if (!banner || !text) return;
  text.textContent = message;
  banner.style.display = 'flex';
}

function hideFormError() {
  const banner = document.getElementById('formErrorBanner');
  if (!banner) return;
  banner.style.display = 'none';
}

// ── Form Validation ──────────────────────────────────────────────
function validateStep(step) {
  hideFormError();
  const stepEl = document.getElementById(`step-${step}`);
  if (!stepEl) return true;

  let valid = true;
  let firstInvalidInput = null;
  const inputs = stepEl.querySelectorAll('input[required], select[required]');

  inputs.forEach(input => {
    input.style.borderColor = '';
    input.style.boxShadow   = '';
    const value = input.value?.toString().trim();
    if (!value) {
      input.style.borderColor = '#ff6b8a';
      input.style.boxShadow   = '0 0 0 3px rgba(255,107,138,0.15)';
      valid = false;
      if (!firstInvalidInput) {
        firstInvalidInput = input;
        const label = stepEl.querySelector(`label[for="${input.id}"]`);
        const fieldName = label ? label.innerText.replace(/\s*\(.*\)$/,'').trim() : 'This field';
        showFormError(`Please fill in ${fieldName}.`);
      }
      // Clear error on change
      input.addEventListener('input', () => {
        input.style.borderColor = '';
        input.style.boxShadow   = '';
      }, { once: true });
    }
  });

  // Age-based eligibility guard
  if (step === 1) {
    const ageInput = document.getElementById('age_years');
    const ageValue = parseFloat(ageInput?.value);
    if (!isNaN(ageValue) && ageValue < 18) {
      valid = false;
      if (ageInput) {
        ageInput.style.borderColor = '#ff6b8a';
        ageInput.style.boxShadow = '0 0 0 3px rgba(255,107,138,0.18)';
        ageInput.focus();
      }
      showFormError('Rejected: applicants must be at least 18 years old.');
    }
  }

  // Numeric range validation
  const numInputs = stepEl.querySelectorAll('input[type="number"]');
  numInputs.forEach(input => {
    const val = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    // Don't auto-clamp the age field on step 1 so underage values can report rejection.
    if (input.id === 'age_years' && step === 1) return;
    if (!isNaN(min) && !Number.isNaN(val) && val < min) { input.value = min; }
    if (!isNaN(max) && !Number.isNaN(val) && val > max) { input.value = max; }
  });

  if (!valid && firstInvalidInput) {
    firstInvalidInput.focus();
  }

  return valid;
}

// ── Summary Update ───────────────────────────────────────────────
function updateSummary() {
  const income  = document.getElementById('annual_income')?.value   || '0';
  const age     = document.getElementById('age_years')?.value       || '?';
  const emp     = document.getElementById('employment_years')?.value || '?';
  const prop    = document.getElementById('own_property')?.value     || 'N';

  const si = document.getElementById('sum-income');
  const sa = document.getElementById('sum-age');
  const se = document.getElementById('sum-emp');
  const sp = document.getElementById('sum-prop');

  if (si) si.textContent = '₹ ' + Number(income).toLocaleString('en-IN');
  if (sa) sa.textContent = age + ' years';
  if (se) se.textContent = emp + ' years';
  if (sp) sp.textContent = prop === 'Y' ? 'Yes' : 'No';
}

// ══════════════════════════════════════════════════════════════════
// INCOME DISPLAY
// ══════════════════════════════════════════════════════════════════

function initIncomeDisplay() {
  const incomeInput   = document.getElementById('annual_income');
  const incomeDisplay = document.getElementById('incomeDisplay');
  if (!incomeInput || !incomeDisplay) return;

  function formatIncome(val) {
    const n = parseFloat(val) || 0;
    if (n >= 10000000) return '₹ ' + (n / 10000000).toFixed(1) + ' Cr';
    if (n >= 100000)   return '₹ ' + (n / 100000).toFixed(1) + ' L';
    return '₹ ' + n.toLocaleString('en-IN');
  }

  incomeInput.addEventListener('input', () => {
    incomeDisplay.textContent = formatIncome(incomeInput.value);
  });

  incomeDisplay.textContent = formatIncome(incomeInput.value);
}

// ══════════════════════════════════════════════════════════════════
// FORM SUBMISSION LOADING
// ══════════════════════════════════════════════════════════════════

function initFormSubmission() {
  const form       = document.getElementById('creditForm');
  const submitBtn  = document.getElementById('submitBtn');
  const submitText = document.getElementById('submitText');
  const submitLoad = document.getElementById('submitLoading');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    if (!validateStep(3)) {
      e.preventDefault();
      return;
    }

    if (submitText) submitText.style.display = 'none';
    if (submitLoad) submitLoad.style.display = 'inline-flex';
    if (submitBtn)  submitBtn.disabled = true;
  });
}

// ══════════════════════════════════════════════════════════════════
// NAVBAR SCROLL EFFECT
// ══════════════════════════════════════════════════════════════════

function initNavbar() {
  const navbar = document.getElementById('navbar');
  if (!navbar) return;

  const onScroll = () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll(); // initialize
}

// ══════════════════════════════════════════════════════════════════
// SMOOTH ANCHOR SCROLL
// ══════════════════════════════════════════════════════════════════

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const href = anchor.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        const offset = 80; // navbar height
        const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  });
}

// ══════════════════════════════════════════════════════════════════
// INTERSECTION OBSERVER — Fade-in animations
// ══════════════════════════════════════════════════════════════════

function initScrollAnimations() {
  const targets = document.querySelectorAll(
    '.step-card, .model-card, .glass-card, .detail-card'
  );

  if (!('IntersectionObserver' in window)) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animation = 'fadeSlideIn 0.6s ease both';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  targets.forEach((el, i) => {
    el.style.opacity = '0';
    el.style.animationDelay = `${i * 0.06}s`;
    observer.observe(el);
  });
}

// ══════════════════════════════════════════════════════════════════
// HERO APPLY BUTTON — smooth scroll + highlight form
// ══════════════════════════════════════════════════════════════════

function initHeroCTA() {
  const heroBtn = document.getElementById('hero-apply-btn');
  if (!heroBtn) return;

  heroBtn.addEventListener('click', (e) => {
    e.preventDefault();
    const applySection = document.getElementById('apply');
    if (applySection) {
      const top = applySection.getBoundingClientRect().top + window.pageYOffset - 80;
      window.scrollTo({ top, behavior: 'smooth' });

      // Pulse the form card
      setTimeout(() => {
        const formCard = document.getElementById('applicationForm');
        if (formCard) {
          formCard.style.transition = 'box-shadow 0.3s ease';
          formCard.style.boxShadow  = '0 0 60px rgba(108,99,255,0.4)';
          setTimeout(() => { formCard.style.boxShadow = ''; }, 1500);
        }
      }, 800);
    }
  });
}

// ══════════════════════════════════════════════════════════════════
// FLOATING CARD HOVER (hero visual)
// ══════════════════════════════════════════════════════════════════

function initFloatCardInteraction() {
  const card = document.querySelector('.float-card-main');
  if (!card) return;

  card.addEventListener('mousemove', (e) => {
    const rect = card.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width  - 0.5;
    const y = (e.clientY - rect.top)  / rect.height - 0.5;
    card.style.transform = `perspective(600px) rotateX(${-y * 8}deg) rotateY(${x * 8}deg) translateY(-12px)`;
  });

  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
    card.style.transition = 'transform 0.5s ease';
  });
}

// ══════════════════════════════════════════════════════════════════
// NUMBER INPUT RIPPLE EFFECT
// ══════════════════════════════════════════════════════════════════

function initInputEffects() {
  document.querySelectorAll('.form-input, .form-select').forEach(el => {
    el.addEventListener('focus', () => {
      el.parentElement?.querySelector('label')?.style.setProperty('color', 'var(--cyan)');
    });
    el.addEventListener('blur', () => {
      el.parentElement?.querySelector('label')?.style.setProperty('color', '');
    });
  });
}

// ══════════════════════════════════════════════════════════════════
// INITIALISE
// ══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initSmoothScroll();
  initScrollAnimations();
  initHeroCTA();
  initFloatCardInteraction();
  initIncomeDisplay();
  initFormSubmission();
  initInputEffects();

  // Initialize progress bar
  if (progressFill) progressFill.style.width = '33.33%';
});

// Expose goToStep & nextStep globally (called from HTML onclick)
window.goToStep = goToStep;
window.nextStep = nextStep;
