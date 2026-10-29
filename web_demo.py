"""
AGENT02 Interactive Local Web Demo
==================================
Chạy trực tiếp bằng Python chuẩn (không cần cài thêm framework ngoài):
    python web_demo.py
Mở trình duyệt tại:
    http://localhost:8050
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.parse
from pathlib import Path
import numpy as np

from agent02_experiment import (
    Config,
    ECommerceEnvironment,
    QLearningAgent,
    RuleBasedAgent,
    train_rl_against_rule,
    evaluate_pair,
    moving_average,
)

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "agent02_results"
PORT = int(os.environ.get("PORT", 8050))

# Global trained baseline agent for quick step-by-step arena
GLOBAL_CONFIG = Config()
GLOBAL_AGENT, _ = train_rl_against_rule(GLOBAL_CONFIG, seed=GLOBAL_CONFIG.seed)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AGENT02 Demo — RL Q-Learning vs Rule-Based Agent</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --blue: #3b82f6;
    --orange: #f97316;
    --green: #10b981;
    --red: #ef4444;
    --purple: #8b5cf6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding-bottom: 40px;
  }
  header {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
    border-bottom: 1px solid var(--panel-border);
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.2px; }
  header p { color: var(--muted); font-size: 0.9rem; }
  .badge-student {
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid #3b82f6;
    color: #93c5fd;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .container { max-width: 1320px; margin: 0 auto; padding: 20px 24px; }
  .tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 12px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: var(--panel);
    color: var(--muted);
    border: 1px solid var(--panel-border);
    padding: 10px 18px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.92rem;
    transition: all 0.15s;
  }
  .tab-btn.active, .tab-btn:hover {
    background: var(--blue);
    color: white;
    border-color: var(--blue);
  }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .grid-sidebar {
    display: grid;
    grid-template-columns: 330px 1fr;
    gap: 20px;
  }
  @media (max-width: 960px) {
    .grid-sidebar { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 18px;
  }
  .card h2 {
    font-size: 1.05rem;
    margin-bottom: 14px;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .form-group { margin-bottom: 12px; }
  .form-group label {
    display: flex;
    justify-content: space-between;
    font-size: 0.84rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .form-group label span { color: var(--text); font-weight: 600; }
  .form-group input, .form-group select {
    width: 100%;
    padding: 7px 10px;
    border-radius: 6px;
    border: 1px solid var(--panel-border);
    background: #0f172a;
    color: var(--text);
    font-size: 0.9rem;
  }
  .btn-primary {
    width: 100%;
    padding: 11px 16px;
    background: var(--blue);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    cursor: pointer;
    transition: background 0.15s;
    margin-top: 6px;
  }
  .btn-primary:hover { background: #2563eb; }
  .btn-secondary {
    padding: 8px 14px;
    background: #334155;
    color: white;
    border: 1px solid #475569;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .btn-secondary:hover { background: #475569; }
  .preset-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }
  .preset-btn {
    flex: 1;
    font-size: 0.78rem;
    padding: 6px 8px;
    background: rgba(59, 130, 246, 0.12);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 6px;
    cursor: pointer;
  }
  .preset-btn:hover { background: rgba(59, 130, 246, 0.28); }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
  }
  .kpi-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .kpi-title { font-size: 0.8rem; color: var(--muted); }
  .kpi-val { font-size: 1.55rem; font-weight: 700; margin: 4px 0; }
  .kpi-sub { font-size: 0.8rem; color: var(--muted); }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  th, td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--panel-border);
    text-align: center;
  }
  th { background: rgba(15, 23, 42, 0.8); color: #cbd5e1; font-weight: 600; }
  td:first-child, th:first-child { text-align: left; }
  .pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
  }
  .pill-no { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid #10b981; }
  .pill-disc { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid #f97316; }
  .charts-row {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 18px;
  }
  @media (max-width: 960px) {
    .charts-row { grid-template-columns: 1fr; }
  }
  .chart-wrap { position: relative; height: 310px; width: 100%; }
  .img-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 18px;
  }
  .img-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 14px;
  }
  .img-card img {
    width: 100%;
    border-radius: 8px;
    background: white;
    padding: 6px;
  }
  .img-card p {
    margin-top: 8px;
    font-size: 0.88rem;
    color: var(--muted);
    text-align: center;
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>AGENT02: Reinforcement-Learning Agent (Q-Learning) vs Rule-Based Agent</h1>
    <p>Mô phỏng Cạnh tranh Khuyến mãi trên Sàn Thương mại Điện tử — Học phần Phát triển Hệ thống Thông minh</p>
  </div>
  <div class="badge-student">SV: Cao Tuấn Đinh Vũ • B23DCCN939 • Nhóm 05</div>
</header>

<div class="container">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-sim', this)">1. Huấn Luyện & Đánh Giá Tương Tác</button>
    <button class="tab-btn" onclick="switchTab('tab-arena', this)">2. Đấu Trường Từng Bước (Step-by-Step Arena)</button>
    <button class="tab-btn" onclick="switchTab('tab-report', this)">3. Kết Quả & Biểu Đồ 4 Thí Nghiệm Chuẩn</button>
  </div>

  <!-- TAB 1: INTERACTIVE SIMULATION -->
  <div id="tab-sim" class="tab-content active">
    <div class="grid-sidebar">
      <div class="card">
        <h2>Cấu hình Siêu tham số (Config)</h2>
        <div style="font-size:0.8rem;color:var(--muted);margin-bottom:6px;">Chọn nhanh kịch bản mẫu:</div>
        <div class="preset-row">
          <button class="preset-btn" onclick="applyPreset('baseline')">Chuẩn (Baseline)</button>
          <button class="preset-btn" onclick="applyPreset('high_lr')">α cao (0.50)</button>
          <button class="preset-btn" onclick="applyPreset('promo_market')">Khuyến mãi lời (δ=8%)</button>
        </div>

        <div class="form-group">
          <label>Số tập huấn luyện (episodes) <span id="lbl-episodes">5000</span></label>
          <input type="range" id="inp-episodes" min="500" max="5000" step="500" value="5000" oninput="document.getElementById('lbl-episodes').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Tốc độ học α (alpha) <span id="lbl-alpha">0.10</span></label>
          <select id="inp-alpha" onchange="document.getElementById('lbl-alpha').innerText=this.value">
            <option value="0.01">0.01 (Thận trọng - Low LR)</option>
            <option value="0.10" selected>0.10 (Chuẩn - Baseline)</option>
            <option value="0.30">0.30 ( Trung bình cao)</option>
            <option value="0.50">0.50 (Cập nhật mạnh - High LR)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Hệ số chiết khấu γ (gamma) <span id="lbl-gamma">0.90</span></label>
          <input type="number" id="inp-gamma" min="0.1" max="0.99" step="0.05" value="0.90" oninput="document.getElementById('lbl-gamma').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Khám phá ban đầu (epsilon_start) <span id="lbl-eps-start">0.20</span></label>
          <input type="number" id="inp-eps-start" min="0.01" max="1.0" step="0.05" value="0.20" oninput="document.getElementById('lbl-eps-start').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Khám phá tối thiểu (epsilon_min) <span id="lbl-eps-min">0.02</span></label>
          <input type="number" id="inp-eps-min" min="0.0" max="0.5" step="0.01" value="0.02" oninput="document.getElementById('lbl-eps-min').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Tốc độ suy giảm (epsilon_decay) <span id="lbl-eps-decay">0.995</span></label>
          <input type="number" id="inp-eps-decay" min="0.900" max="1.000" step="0.001" value="0.995" oninput="document.getElementById('lbl-eps-decay').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Tỷ lệ giảm giá δ (discount_rate) <span id="lbl-disc">0.20</span></label>
          <input type="number" id="inp-disc" min="0.05" max="0.40" step="0.01" value="0.20" oninput="document.getElementById('lbl-disc').innerText=this.value">
        </div>
        <div class="form-group">
          <label>Hạt giống ngẫu nhiên (seed)</label>
          <input type="number" id="inp-seed" value="42" step="1">
        </div>

        <button class="btn-primary" id="btn-run" onclick="runSimulation()">▶ Chạy Huấn Luyện & Đánh Giá</button>
      </div>

      <div>
        <div class="kpi-grid">
          <div class="kpi-box">
            <div class="kpi-title">RL Agent — Phần thưởng Đánh giá / Bước</div>
            <div class="kpi-val" id="kpi-rl-eval" style="color:#60a5fa;">--</div>
            <div class="kpi-sub" id="kpi-rl-disc">Tỷ lệ giảm giá: --</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-title">Rule-Based — Phần thưởng Đánh giá / Bước</div>
            <div class="kpi-val" id="kpi-rule-eval" style="color:#fb923c;">--</div>
            <div class="kpi-sub" id="kpi-rule-disc">Tỷ lệ giảm giá: --</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-title">RL Agent — Trung bình Huấn luyện / Bước</div>
            <div class="kpi-val" id="kpi-rl-train" style="color:#34d399;">--</div>
            <div class="kpi-sub">Có khám phá ε-greedy</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-title">Chênh lệch Đánh giá (RL − Rule)</div>
            <div class="kpi-val" id="kpi-diff">--</div>
            <div class="kpi-sub" id="kpi-winner">Đang chờ chạy...</div>
          </div>
        </div>

        <div class="charts-row">
          <div class="card">
            <h2>Đường cong Học tập (Moving Average 100 Episodes)</h2>
            <div class="chart-wrap"><canvas id="chart-learning"></canvas></div>
          </div>
          <div class="card">
            <h2>So sánh Huấn luyện vs. Đánh giá Độc lập (ε = 0)</h2>
            <div class="chart-wrap"><canvas id="chart-eval-bar"></canvas></div>
          </div>
        </div>

        <div class="card">
          <h2>Bảng Giá Trị Hành Động Học Được (Learned 8-Entry Q-Table)</h2>
          <table>
            <thead>
              <tr>
                <th>Trạng thái s = (d, c)</th>
                <th>Ý nghĩa trạng thái</th>
                <th>Q(s, 0) — Không giảm giá</th>
                <th>Q(s, 1) — Giảm giá</th>
                <th>Chênh lệch ΔQ</th>
                <th>Chính sách Tham lam π*(s)</th>
              </tr>
            </thead>
            <tbody id="qtable-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: STEP-BY-STEP ARENA -->
  <div id="tab-arena" class="tab-content">
    <div class="card">
      <h2>
        <span>Đấu Trường Tương Tác Từng Bước (1 Episode = 20 Steps)</span>
        <div style="display:flex;gap:8px;">
          <button class="btn-secondary" onclick="arenaStep()">Bước Tiếp Theo (+1 Step)</button>
          <button class="btn-secondary" onclick="arenaAutoPlay()" id="btn-autoplay">Tự Động Chạy (Auto)</button>
          <button class="btn-secondary" onclick="arenaReset()">Khởi Tạo Lại (Reset)</button>
        </div>
      </h2>
      <p style="font-size:0.88rem;color:var(--muted);margin-bottom:14px;">
        Quan sát trực tiếp cách <b>QLearningAgent</b> (sử dụng bảng Q vừa huấn luyện ở Tab 1, chế độ tham lam ε = 0) và <b>RuleBasedAgent</b> ra quyết định đồng thời ở từng bước thời gian t.
      </p>

      <div class="kpi-grid">
        <div class="kpi-box">
          <div class="kpi-title">Bước hiện tại (Step)</div>
          <div class="kpi-val" id="ar-step">0 / 20</div>
          <div class="kpi-sub" id="ar-demand">Nhu cầu hiện tại: --</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-title">Tổng Lợi Nhuận RL Agent</div>
          <div class="kpi-val" id="ar-rl-total" style="color:#60a5fa;">0.0</div>
          <div class="kpi-sub" id="ar-rl-avg">TB/bước: 0.00</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-title">Tổng Lợi Nhuận Rule-Based</div>
          <div class="kpi-val" id="ar-rule-total" style="color:#fb923c;">0.0</div>
          <div class="kpi-sub" id="ar-rule-avg">TB/bước: 0.00</div>
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Bước t</th>
            <th>Cầu d_t (Khách N_t)</th>
            <th>Trạng thái RL</th>
            <th>Hành động RL a_1(t)</th>
            <th>Hành động Rule a_2(t)</th>
            <th>Xác suất mua (p_1, p_2)</th>
            <th>Doanh số (S_1, S_2)</th>
            <th>Lợi nhuận (r_1, r_2)</th>
          </tr>
        </thead>
        <tbody id="arena-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- TAB 3: PRECOMPUTED REPORT RESULTS -->
  <div id="tab-report" class="tab-content">
    <div class="card">
      <h2>Tổng Hợp Kết Quả 4 Thí Nghiệm Chuẩn từ agent02_experiment.py</h2>
      <div id="report-tables-container">Đang tải dữ liệu...</div>
    </div>
    <div class="img-gallery">
      <div class="img-card">
        <img src="/results/episode_rewards.png" alt="Episode Rewards">
        <p>Hình 1 (Experiment 1): Đường cong học tập Baseline (episode_rewards.png)</p>
      </div>
      <div class="img-card">
        <img src="/results/learning_rate_comparison.png" alt="Learning Rate">
        <p>Hình 2 (Experiment 2): So sánh tốc độ học α ∈ {0.01, 0.10, 0.50} (learning_rate_comparison.png)</p>
      </div>
      <div class="img-card">
        <img src="/results/exploration_comparison.png" alt="Exploration">
        <p>Hình 3 (Experiment 3): So sánh lịch trình khám phá ε-greedy (exploration_comparison.png)</p>
      </div>
      <div class="img-card">
        <img src="/results/modification_comparison.png" alt="Modification">
        <p>Hình 4 (Câu hỏi 9): Khả năng tự thích nghi của RL khi thị trường thay đổi (modification_comparison.png)</p>
      </div>
    </div>
  </div>
</div>

<script>
let learningChart = null;
let evalBarChart = null;
let lastQTable = null;
let lastConfig = null;

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  if (tabId === 'tab-report') loadReportSummary();
}

function applyPreset(name) {
  if (name === 'baseline') {
    document.getElementById('inp-episodes').value = 5000;
    document.getElementById('inp-alpha').value = "0.10";
    document.getElementById('inp-gamma').value = 0.90;
    document.getElementById('inp-eps-start').value = 0.20;
    document.getElementById('inp-eps-min').value = 0.02;
    document.getElementById('inp-eps-decay').value = 0.995;
    document.getElementById('inp-disc').value = 0.20;
    document.getElementById('inp-seed').value = 42;
  } else if (name === 'high_lr') {
    document.getElementById('inp-episodes').value = 5000;
    document.getElementById('inp-alpha').value = "0.50";
    document.getElementById('inp-gamma').value = 0.90;
    document.getElementById('inp-eps-start').value = 0.20;
    document.getElementById('inp-eps-min').value = 0.02;
    document.getElementById('inp-eps-decay').value = 0.995;
    document.getElementById('inp-disc').value = 0.20;
    document.getElementById('inp-seed').value = 1042;
  } else if (name === 'promo_market') {
    document.getElementById('inp-episodes').value = 5000;
    document.getElementById('inp-alpha').value = "0.10";
    document.getElementById('inp-gamma').value = 0.90;
    document.getElementById('inp-eps-start').value = 0.20;
    document.getElementById('inp-eps-min').value = 0.02;
    document.getElementById('inp-eps-decay').value = 0.995;
    document.getElementById('inp-disc').value = 0.08;
    document.getElementById('inp-seed').value = 42;
  }
  document.getElementById('lbl-episodes').innerText = document.getElementById('inp-episodes').value;
  document.getElementById('lbl-alpha').innerText = document.getElementById('inp-alpha').value;
  document.getElementById('lbl-gamma').innerText = document.getElementById('inp-gamma').value;
  document.getElementById('lbl-eps-start').innerText = document.getElementById('inp-eps-start').value;
  document.getElementById('lbl-eps-min').innerText = document.getElementById('inp-eps-min').value;
  document.getElementById('lbl-eps-decay').innerText = document.getElementById('inp-eps-decay').value;
  document.getElementById('lbl-disc').innerText = document.getElementById('inp-disc').value;
  runSimulation();
}

async function runSimulation() {
  const btn = document.getElementById('btn-run');
  btn.innerText = '⏳ Đang huấn luyện...';
  btn.disabled = true;

  const payload = {
    episodes: parseInt(document.getElementById('inp-episodes').value),
    alpha: parseFloat(document.getElementById('inp-alpha').value),
    gamma: parseFloat(document.getElementById('inp-gamma').value),
    epsilon_start: parseFloat(document.getElementById('inp-eps-start').value),
    epsilon_min: parseFloat(document.getElementById('inp-eps-min').value),
    epsilon_decay: parseFloat(document.getElementById('inp-eps-decay').value),
    discount_rate: parseFloat(document.getElementById('inp-disc').value),
    seed: parseInt(document.getElementById('inp-seed').value)
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    lastQTable = data.q_table;
    lastConfig = payload;

    const rlEval = data.eval_summary[0];
    const ruleEval = data.eval_summary[1];
    document.getElementById('kpi-rl-eval').innerText = rlEval.mean_reward_per_step.toFixed(4);
    document.getElementById('kpi-rl-disc').innerText = `Tỷ lệ giảm giá: ${(rlEval.discount_action_rate * 100).toFixed(1)}%`;
    document.getElementById('kpi-rule-eval').innerText = ruleEval.mean_reward_per_step.toFixed(4);
    document.getElementById('kpi-rule-disc').innerText = `Tỷ lệ giảm giá: ${(ruleEval.discount_action_rate * 100).toFixed(1)}%`;
    document.getElementById('kpi-rl-train').innerText = data.train_rl_mean.toFixed(4);

    const diff = rlEval.mean_reward_per_step - ruleEval.mean_reward_per_step;
    const diffEl = document.getElementById('kpi-diff');
    diffEl.innerText = (diff >= 0 ? '+' : '') + diff.toFixed(4);
    diffEl.style.color = diff >= 0 ? '#34d399' : '#f87171';
    document.getElementById('kpi-winner').innerText =
      Math.abs(diff) < 0.35 ? 'Hai tác tử hội tụ tương đương' : (diff > 0 ? 'RL Agent vượt trội' : 'Rule-Based ổn định hơn');

    // Populate Q-table
    const meanings = {
      "(0, 0)": "Cầu thấp (d=0), Đối thủ vừa giữ giá (c=0)",
      "(0, 1)": "Cầu thấp (d=0), Đối thủ vừa giảm giá (c=1)",
      "(1, 0)": "Cầu cao (d=1), Đối thủ vừa giữ giá (c=0)",
      "(1, 1)": "Cầu cao (d=1), Đối thủ vừa giảm giá (c=1)"
    };
    const tbody = document.getElementById('qtable-body');
    tbody.innerHTML = '';
    data.q_table.forEach(row => {
      const badge = row.greedy_action === 0
        ? '<span class="pill pill-no">0 — KHÔNG GIẢM GIÁ</span>'
        : '<span class="pill pill-disc">1 — GIẢM GIÁ</span>';
      tbody.innerHTML += `
        <tr>
          <td><b>${row.state}</b></td>
          <td>${meanings[row.state]}</td>
          <td>${row.q0.toFixed(4)}</td>
          <td>${row.q1.toFixed(4)}</td>
          <td>${(row.q0 - row.q1).toFixed(4)}</td>
          <td>${badge}</td>
        </tr>`;
    });

    // Update Charts
    renderCharts(data);
    arenaReset();
  } finally {
    btn.innerText = '▶ Chạy Huấn Luyện & Đánh Giá';
    btn.disabled = false;
  }
}

function renderCharts(data) {
  const ctx1 = document.getElementById('chart-learning').getContext('2d');
  if (learningChart) learningChart.destroy();
  learningChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: data.curve_episodes,
      datasets: [
        {
          label: 'RL Moving Average (100)',
          data: data.curve_rl_ma,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'Rule-Based Moving Average (100)',
          data: data.curve_rule_ma,
          borderColor: '#f97316',
          backgroundColor: 'rgba(249, 115, 22, 0.1)',
          borderWidth: 2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#cbd5e1' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
      }
    }
  });

  const ctx2 = document.getElementById('chart-eval-bar').getContext('2d');
  if (evalBarChart) evalBarChart.destroy();
  evalBarChart = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: ['RL (Train)', 'Rule (Train)', 'RL (Eval ε=0)', 'Rule (Eval)'],
      datasets: [{
        label: 'Mean Reward / Step',
        data: [
          data.train_rl_mean,
          data.train_rule_mean,
          data.eval_summary[0].mean_reward_per_step,
          data.eval_summary[1].mean_reward_per_step
        ],
        backgroundColor: ['#60a5fa88', '#fb923c88', '#3b82f6', '#f97316'],
        borderColor: ['#60a5fa', '#fb923c', '#3b82f6', '#f97316'],
        borderWidth: 1.5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#cbd5e1' }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
      }
    }
  });
}

// TAB 2: ARENA LOGIC
let arenaState = { step: 0, demand: 0, prev: [0, 0], rlTotal: 0, ruleTotal: 0, timer: null };

function arenaReset() {
  if (arenaState.timer) { clearInterval(arenaState.timer); arenaState.timer = null; }
  document.getElementById('btn-autoplay').innerText = 'Tự Động Chạy (Auto)';
  arenaState = {
    step: 0,
    demand: Math.random() < 0.5 ? 0 : 1,
    prev: [0, 0],
    rlTotal: 0,
    ruleTotal: 0,
    timer: null
  };
  document.getElementById('ar-step').innerText = '0 / 20';
  document.getElementById('ar-demand').innerText =
    `Nhu cầu bước tới: d = ${arenaState.demand} (${arenaState.demand === 0 ? '10 khách' : '20 khách'})`;
  document.getElementById('ar-rl-total').innerText = '0.0';
  document.getElementById('ar-rl-avg').innerText = 'TB/bước: 0.00';
  document.getElementById('ar-rule-total').innerText = '0.0';
  document.getElementById('ar-rule-avg').innerText = 'TB/bước: 0.00';
  document.getElementById('arena-tbody').innerHTML = '';
}

async function arenaStep() {
  if (arenaState.step >= 20) return;
  const res = await fetch('/api/arena_step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      demand: arenaState.demand,
      prev_actions: arenaState.prev,
      discount_rate: lastConfig ? lastConfig.discount_rate : 0.20
    })
  });
  const d = await res.json();
  arenaState.step += 1;
  arenaState.rlTotal += d.rewards[0];
  arenaState.ruleTotal += d.rewards[1];
  arenaState.demand = d.next_demand;
  arenaState.prev = d.actions;

  document.getElementById('ar-step').innerText = `${arenaState.step} / 20`;
  document.getElementById('ar-demand').innerText =
    `Nhu cầu bước tới: d = ${arenaState.demand} (${arenaState.demand === 0 ? '10 khách' : '20 khách'})`;
  document.getElementById('ar-rl-total').innerText = arenaState.rlTotal.toFixed(1);
  document.getElementById('ar-rl-avg').innerText = `TB/bước: ${(arenaState.rlTotal / arenaState.step).toFixed(2)}`;
  document.getElementById('ar-rule-total').innerText = arenaState.ruleTotal.toFixed(1);
  document.getElementById('ar-rule-avg').innerText = `TB/bước: ${(arenaState.ruleTotal / arenaState.step).toFixed(2)}`;

  const rlBadge = d.actions[0] === 0 ? '<span class="pill pill-no">0 (Giữ giá)</span>' : '<span class="pill pill-disc">1 (Giảm giá)</span>';
  const ruleBadge = d.actions[1] === 0 ? '<span class="pill pill-no">0 (Giữ giá)</span>' : '<span class="pill pill-disc">1 (Giảm giá)</span>';

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><b>#${arenaState.step}</b></td>
    <td>d=${d.demand} (${d.customers} khách)</td>
    <td>(${d.rl_state[0]}, ${d.rl_state[1]})</td>
    <td>${rlBadge}</td>
    <td>${ruleBadge}</td>
    <td>${(d.conversion[0]*100).toFixed(0)}% vs ${(d.conversion[1]*100).toFixed(0)}%</td>
    <td>${d.sales[0]} đơn vs ${d.sales[1]} đơn</td>
    <td><b style="color:#60a5fa">${d.rewards[0].toFixed(1)}</b> vs <b style="color:#fb923c">${d.rewards[1].toFixed(1)}</b></td>
  `;
  document.getElementById('arena-tbody').prepend(tr);
}

function arenaAutoPlay() {
  if (arenaState.timer) {
    clearInterval(arenaState.timer);
    arenaState.timer = null;
    document.getElementById('btn-autoplay').innerText = 'Tự Động Chạy (Auto)';
    return;
  }
  if (arenaState.step >= 20) arenaReset();
  document.getElementById('btn-autoplay').innerText = '⏸ Tạm Dừng';
  arenaState.timer = setInterval(() => {
    if (arenaState.step >= 20) {
      clearInterval(arenaState.timer);
      arenaState.timer = null;
      document.getElementById('btn-autoplay').innerText = 'Tự Động Chạy (Auto)';
    } else {
      arenaStep();
    }
  }, 450);
}

async function loadReportSummary() {
  const res = await fetch('/api/report_summary');
  const data = await res.json();
  let html = `
    <table>
      <thead>
        <tr>
          <th>Thí nghiệm / Cấu hình</th>
          <th>Phần thưởng Đánh giá TB / Bước</th>
          <th>Tỷ lệ Giảm giá Đánh giá</th>
        </tr>
      </thead>
      <tbody>
  `;
  data.rows.forEach(r => {
    html += `<tr><td><b>${r.label}</b></td><td>${r.reward}</td><td>${r.disc}</td></tr>`;
  });
  html += `</tbody></table>`;
  document.getElementById('report-tables-container').innerHTML = html;
}

window.addEventListener('DOMContentLoaded', () => {
  runSimulation();
});
</script>
</body>
</html>
"""


class DemoRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path.startswith("/results/"):
            fname = parsed.path.split("/results/", 1)[1]
            fpath = (RESULTS_DIR / fname).resolve()
            if fpath.parent == RESULTS_DIR.resolve() and fpath.exists():
                raw = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_error(404)
            return

        if parsed.path == "/api/report_summary":
            rows = [
                {"label": "Experiment 1 — Baseline RL (α=0.10, seed=42)", "reward": "35.1275", "disc": "0.0%"},
                {"label": "Experiment 1 — Rule-Based Agent (seed=42)", "reward": "34.9685", "disc": "0.0%"},
                {"label": "Experiment 2 — Learning Rate α = 0.01", "reward": "35.2475", "disc": "0.0%"},
                {"label": "Experiment 2 — Learning Rate α = 0.10", "reward": "35.2475", "disc": "0.0%"},
                {"label": "Experiment 2 — Learning Rate α = 0.50 (Mất ổn định)", "reward": "28.5085", "disc": "40.5%"},
                {"label": "Experiment 3 — Exploration Low / Baseline / High", "reward": "35.1045", "disc": "0.0%"},
                {"label": "Experiment 4 — Multi-Seed (5 seeds) RL Mean ± Std", "reward": "34.8919 ± 0.3132", "disc": "0.0%"},
                {"label": "Experiment 4 — Multi-Seed (5 seeds) Rule-Based Mean ± Std", "reward": "34.9321 ± 0.2496", "disc": "0.0%"},
            ]
            self._send_json({"rows": rows})
            return

        self.send_error(404)

    def do_POST(self):
        global GLOBAL_AGENT, GLOBAL_CONFIG
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if self.path == "/api/simulate":
            cfg = Config(
                episodes=int(body.get("episodes", 5000)),
                alpha=float(body.get("alpha", 0.10)),
                gamma=float(body.get("gamma", 0.90)),
                epsilon_start=float(body.get("epsilon_start", 0.20)),
                epsilon_min=float(body.get("epsilon_min", 0.02)),
                epsilon_decay=float(body.get("epsilon_decay", 0.995)),
                discount_rate=float(body.get("discount_rate", 0.20)),
                seed=int(body.get("seed", 42)),
            )
            agent, train_rows = train_rl_against_rule(cfg, seed=cfg.seed)
            summary, _ = evaluate_pair(
                cfg, agent, episodes=cfg.evaluation_episodes, seed=cfg.seed + 500_000
            )
            GLOBAL_AGENT = agent
            GLOBAL_CONFIG = cfg

            rl_tr = [r["rl_mean_reward_per_step"] for r in train_rows]
            rule_tr = [r["rule_mean_reward_per_step"] for r in train_rows]
            win = min(100, max(10, len(train_rows) // 20))
            rl_ma = moving_average(rl_tr, win)
            rule_ma = moving_average(rule_tr, win)
            ep_ma = list(range(win, len(train_rows) + 1))

            # Downsample curve to ~120 points for fast rendering
            stride = max(1, len(ep_ma) // 120)
            q_table = []
            for d in range(2):
                for c in range(2):
                    q0 = float(agent.q[d, c, 0])
                    q1 = float(agent.q[d, c, 1])
                    q_table.append({
                        "state": f"({d}, {c})",
                        "q0": q0,
                        "q1": q1,
                        "greedy_action": int(np.argmax(agent.q[d, c])),
                    })

            self._send_json({
                "train_rl_mean": float(np.mean(rl_tr)),
                "train_rule_mean": float(np.mean(rule_tr)),
                "eval_summary": summary,
                "q_table": q_table,
                "curve_episodes": ep_ma[::stride],
                "curve_rl_ma": [round(float(x), 4) for x in rl_ma[::stride]],
                "curve_rule_ma": [round(float(x), 4) for x in rule_ma[::stride]],
            })
            return

        if self.path == "/api/arena_step":
            demand = int(body.get("demand", 0))
            prev_actions = list(body.get("prev_actions", [0, 0]))
            disc_rate = float(body.get("discount_rate", GLOBAL_CONFIG.discount_rate))

            cfg = Config(discount_rate=disc_rate)
            env = ECommerceEnvironment(cfg, seed=np.random.SeedSequence().entropy)
            env.demand = demand
            env.previous_actions = prev_actions

            states = env.get_states()
            rl_act = GLOBAL_AGENT.act(states[0], explore=False)
            rule_act = RuleBasedAgent().act(states[1])
            next_states, rewards, info = env.step([rl_act, rule_act])

            self._send_json({
                "demand": demand,
                "customers": info["customers"],
                "rl_state": list(states[0]),
                "actions": [rl_act, rule_act],
                "conversion": info["conversion"],
                "sales": info["sales"],
                "rewards": rewards,
                "next_demand": env.demand,
            })
            return

        self.send_error(404)


def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DemoRequestHandler)
    print("=" * 60)
    print(f"AGENT02 Web Demo is running at: http://localhost:{PORT}")
    print("=" * 60)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
