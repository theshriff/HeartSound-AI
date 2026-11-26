const input = document.getElementById('audioFile');
const dropzone = document.getElementById('dropzone');
const selectedFileBox = document.getElementById('selectedFile');
const audioPlayer = document.getElementById('audioPlayer');
const analyzeBtn = document.getElementById('analyzeBtn');
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');
let currentFile = null;
let audioBuffer = null;
let audioContext = null;

// مسیر API بک‌اند Flask را در صورت نیاز تغییر بده.
const API_URL = 'https://heartsound-ai.onrender.com/predict';

drawPlaceholder();

input.addEventListener('change', () => {
  if (input.files[0]) handleFile(input.files[0]);
});

['dragenter','dragover'].forEach(eventName => dropzone.addEventListener(eventName, e => {
  e.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave','drop'].forEach(eventName => dropzone.addEventListener(eventName, e => {
  e.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

document.getElementById('removeFile').addEventListener('click', () => {
  currentFile = null; audioBuffer = null; input.value = '';
  selectedFileBox.classList.add('hidden');
  document.getElementById('audioPreview').classList.add('hidden');
  analyzeBtn.disabled = true;
  setText('resultTitle', 'آماده دریافت سیگنال');
  setText('resultDescription', 'برای شروع، یک فایل صوتی انتخاب کن.');
  setText('resultFile', '—'); setText('resultDuration', '—'); setText('resultRate', '—');
  setText('systemStatus', 'آماده'); setText('waveStatus', 'در انتظار فایل صوتی');
  drawPlaceholder();
});

async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.wav')) {
    alert('لطفاً فایل با فرمت WAV انتخاب کن.');
    return;
  }
  currentFile = file;
  selectedFileBox.classList.remove('hidden');
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  document.getElementById('audioPreview').classList.remove('hidden');
  audioPlayer.src = URL.createObjectURL(file);
  analyzeBtn.disabled = false;
  setText('resultTitle', 'فایل با موفقیت بارگذاری شد');
  setText('resultDescription', 'سیگنال آماده ارسال برای تحلیل است.');
  setText('resultFile', file.name);
  setText('systemStatus', 'فایل بارگذاری شد');
  setText('waveStatus', 'در حال خواندن سیگنال...');

  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioContext = audioContext || new AudioContextClass();
    audioBuffer = await audioContext.decodeAudioData(await file.arrayBuffer());
    const duration = audioBuffer.duration;
    setText('audioDuration', formatTime(duration));
    setText('resultDuration', `${duration.toFixed(2)} ثانیه`);
    setText('resultRate', `${audioBuffer.sampleRate.toLocaleString('en-US')} Hz`);
    setText('waveStatus', 'سیگنال بارگذاری شد');
    drawWaveform(audioBuffer.getChannelData(0));
  } catch (error) {
    setText('waveStatus', 'خواندن شکل موج ممکن نشد');
    setText('resultDescription', 'فایل انتخاب شده، اما مرورگر نتوانست پیش‌نمایش صوتی بسازد.');
  }
}

analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  // این بخش فایل را به Flask می‌فرستد. تا زمان آماده شدن API، پیام اتصال نمایش داده می‌شود.
  analyzeBtn.disabled = true;
  setText('systemStatus', 'در حال ارسال...');
  setText('resultTitle', 'در حال تحلیل سیگنال');
  setText('resultDescription', 'در حال ارسال فایل به سرور پردازش...');
  try {
    const formData = new FormData();
    formData.append('file', currentFile);
    const response = await fetch(API_URL, { method: 'POST', body: formData });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    // قرارداد پیشنهادی پاسخ: { prediction: "Normal" یا "Abnormal", confidence: 0.87 }
    const prediction = result.prediction || result.label || 'نامشخص';
    const confidence = typeof result.confidence === 'number'
      ? ` · اطمینان ${(result.confidence * 100).toFixed(1)}٪` : '';
    setText('resultTitle', prediction === 'Normal' ? 'الگوی طبیعی' : prediction === 'Abnormal' ? 'الگوی غیرطبیعی' : prediction);
    setText('resultDescription', `نتیجه مدل${confidence}`);
    setText('systemStatus', 'تحلیل انجام شد');
  } catch (error) {
    setText('resultTitle', 'اتصال به مدل برقرار نشد');
    setText('resultDescription', 'رابط آماده است؛ برای تحلیل واقعی باید API بک‌اند روی آدرس تنظیم‌شده اجرا شود.');
    setText('systemStatus', 'در انتظار اتصال بک‌اند');
    console.error(error);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function drawPlaceholder() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#20383b'; ctx.lineWidth = 1;
  for (let y = 20; y < 170; y += 35) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1000, y); ctx.stroke();
  }
  ctx.strokeStyle = '#4bbba8'; ctx.lineWidth = 2; ctx.beginPath();
  for (let x = 0; x < 1000; x++) {
    const y = 85 + Math.sin(x * .055) * 5 + Math.sin(x * .17) * 2;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawWaveform(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#20383b'; ctx.lineWidth = 1;
  for (let y = 20; y < 170; y += 35) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1000, y); ctx.stroke();
  }
  const step = Math.max(1, Math.floor(data.length / canvas.width));
  ctx.strokeStyle = '#66d7c3'; ctx.lineWidth = 1.4; ctx.beginPath();
  for (let x = 0; x < canvas.width; x++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const value = data[x * step + j] ?? 0;
      min = Math.min(min, value); max = Math.max(max, value);
    }
    ctx.moveTo(x, 85 + min * 75);
    ctx.lineTo(x, 85 + max * 75);
  }
  ctx.stroke();
}

function setText(id, value) { document.getElementById(id).textContent = value; }
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
}
