/**
 * TopicPulse v2 — Client-side JS
 * Handles brief generation via fetch() and Chart.js initialization.
 */

// ======================
// Gap Data (from server)
// ======================
let gapsData = [];

document.addEventListener('DOMContentLoaded', function () {
    // Parse gap data from embedded JSON
    const gapsEl = document.getElementById('gapsData');
    if (gapsEl) {
        try {
            gapsData = JSON.parse(gapsEl.textContent);
        } catch (e) {
            console.error('Failed to parse gaps data:', e);
        }
    }

    // Initialize sentiment chart
    initSentimentChart();
});


// ==============================
// Brief Generation via fetch()
// ==============================

async function loadBrief(index) {
    const gap = gapsData[index];
    if (!gap) {
        console.error('No gap data at index', index);
        return;
    }

    const overlay = document.getElementById('brief-modal-overlay');
    const content = document.getElementById('brief-content');

    // Show modal with loading state
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';

    content.innerHTML = `
        <div class="brief-loading">
            <div class="spinner"></div>
            <p>Generating content brief with AI...</p>
            <p class="text-muted">This usually takes 5-8 seconds</p>
        </div>
    `;

    try {
        const res = await fetch('/brief', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gap)
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        const brief = await res.json();
        renderBrief(brief, gap, content);

    } catch (err) {
        content.innerHTML = `
            <div class="brief-error">
                <h3>⚠ Brief Generation Failed</h3>
                <p>${err.message}</p>
                <button class="btn btn-primary" onclick="loadBrief(${index})">Try Again</button>
            </div>
        `;
    }
}

function renderBrief(brief, gap, container) {
    // Difficulty label with color
    const diffColors = ['', '#10b981', '#34d399', '#fbbf24', '#f97316', '#ef4444'];
    const diffColor = diffColors[brief.difficulty] || '#94a3b8';

    // Questions HTML
    let questionsHtml = '';
    if (brief.questions_to_address && brief.questions_to_address.length > 0) {
        questionsHtml = brief.questions_to_address.map(q => `
            <li class="brief-question">
                <a href="${q.url || '#'}" target="_blank">${q.title || 'Untitled'}</a>
                <span class="brief-q-stats">${(q.views || 0).toLocaleString()} views · ${q.answers || 0} answers</span>
            </li>
        `).join('');
    } else {
        questionsHtml = '<li class="text-muted">No questions available</li>';
    }

    // Key points HTML
    let keyPointsHtml = '';
    if (brief.key_points && brief.key_points.length > 0) {
        keyPointsHtml = brief.key_points.map(p => `<li>${p}</li>`).join('');
    }

    container.innerHTML = `
        <div class="brief-header">
            <div class="brief-diff" style="color: ${diffColor}">
                Difficulty: ${brief.difficulty}/5
                <span class="brief-diff-label">${brief.difficulty_label || ''}</span>
            </div>
            <h2 class="brief-title">${brief.suggested_title || 'Untitled Brief'}</h2>
        </div>

        <div class="brief-section">
            <h4>Why Now</h4>
            <p>${brief.why_now || 'No information available.'}</p>
        </div>

        <div class="brief-section">
            <h4>Target Audience</h4>
            <p>${brief.target_audience || 'General developers'}</p>
        </div>

        <div class="brief-section">
            <h4>Key Points to Cover</h4>
            <ul class="brief-points">${keyPointsHtml || '<li>No key points generated</li>'}</ul>
        </div>

        <div class="brief-section">
            <h4>Stack Overflow Questions to Address</h4>
            <ul class="brief-questions">${questionsHtml}</ul>
        </div>

        <div class="brief-actions">
            <button class="btn btn-secondary btn-sm" onclick="copyBrief()">Copy as Markdown</button>
        </div>
    `;

    // Store for copying
    container._briefData = brief;
}

function copyBrief() {
    const container = document.getElementById('brief-content');
    const brief = container._briefData;
    if (!brief) return;

    const md = `# ${brief.suggested_title || 'Untitled'}

## Why Now
${brief.why_now || ''}

## Target Audience
${brief.target_audience || ''}

## Key Points
${(brief.key_points || []).map(p => `- ${p}`).join('\n')}

## Questions to Address
${(brief.questions_to_address || []).map(q => `- [${q.title}](${q.url}) — ${(q.views || 0).toLocaleString()} views, ${q.answers || 0} answers`).join('\n')}

## Difficulty: ${brief.difficulty}/5
${brief.difficulty_label || ''}
`;

    navigator.clipboard.writeText(md).then(() => {
        const btn = container.querySelector('.brief-actions .btn');
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = original, 2000);
    });
}


// =====================
// Modal Controls
// =====================

function closeBriefModal() {
    const overlay = document.getElementById('brief-modal-overlay');
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
}

function closeModal(event) {
    if (event.target === event.currentTarget) {
        closeBriefModal();
    }
}

// Close on Escape key
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeBriefModal();
    }
});


// ========================
// Sentiment Donut Chart
// ========================

function initSentimentChart() {
    const dataEl = document.getElementById('chartData');
    if (!dataEl) return;

    try {
        const data = JSON.parse(dataEl.textContent);
        const ctx = document.getElementById('sentimentChart');
        if (!ctx) return;

        new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [data.positive || 0, data.negative || 0, data.neutral || 0],
                    backgroundColor: ['#10b981', '#ef4444', '#64748b'],
                    borderWidth: 0,
                    borderRadius: 4,
                    spacing: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 15, 15, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        titleFont: { family: 'Inter' },
                        bodyFont: { family: 'Inter' },
                        padding: 12,
                        cornerRadius: 8
                    }
                }
            }
        });
    } catch (e) {
        console.error('Chart init error:', e);
    }
}
