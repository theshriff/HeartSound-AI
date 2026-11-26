
console.log("ONNX Runtime:", typeof ort);

const input = document.getElementById('audioFile');
const dropzone = document.getElementById('dropzone');
const selectedFileBox = document.getElementById('selectedFile');
const audioPlayer = document.getElementById('audioPlayer');
const analyzeBtn = document.getElementById('analyzeBtn');
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');

let currentFile = null;
let audioBuffer = null;
let audioObjectURL = null;

drawPlaceholder();


// File selection
input.addEventListener('change', () => {
  if (input.files[0]) {
    handleFile(input.files[0]);
  }
});


// Drag and drop
['dragenter', 'dragover'].forEach(eventName =>
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.add('drag');
  })
);

['dragleave', 'drop'].forEach(eventName =>
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.remove('drag');
  })
);

dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];

  if (file) {
    handleFile(file);
  }
});


// Remove selected file
document.getElementById('removeFile').addEventListener('click', () => {
  currentFile = null;
  audioBuffer = null;
  input.value = '';

  if (audioObjectURL) {
    URL.revokeObjectURL(audioObjectURL);
    audioObjectURL = null;
  }

  audioPlayer.pause();
  audioPlayer.removeAttribute('src');
  audioPlayer.load();

  selectedFileBox.classList.add('hidden');
  document.getElementById('audioPreview').classList.add('hidden');

  analyzeBtn.disabled = true;

  setText('resultTitle', 'آماده دریافت سیگنال');
  setText('resultDescription', 'برای شروع، یک فایل صوتی انتخاب کن.');
  setText('resultFile', '—');
  setText('resultDuration', '—');
  setText('resultRate', '—');
  setText('systemStatus', 'آماده');
  setText('waveStatus', 'در انتظار فایل صوتی');
  setText('audioDuration', '00:00');

  drawPlaceholder();
});


// Handle WAV file
async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.wav')) {
    alert('لطفاً فایل با فرمت WAV انتخاب کن.');
    return;
  }

  currentFile = file;
  audioBuffer = null;
  analyzeBtn.disabled = true;

  selectedFileBox.classList.remove('hidden');

  document.getElementById('fileName').textContent = file.name;

  document.getElementById('fileSize').textContent =
    `${(file.size / 1024 / 1024).toFixed(2)} MB`;

  document.getElementById('audioPreview').classList.remove('hidden');

  if (audioObjectURL) {
    URL.revokeObjectURL(audioObjectURL);
    audioObjectURL = null;
  }

  audioPlayer.pause();
  audioPlayer.removeAttribute('src');
  audioPlayer.load();

  setText('resultTitle', 'فایل با موفقیت بارگذاری شد');
  setText('resultDescription', 'سیگنال آماده تحلیل است.');
  setText('resultFile', file.name);
  setText('systemStatus', 'فایل بارگذاری شد');
  setText('waveStatus', 'در حال خواندن سیگنال...');

  try {
    // Read original WAV file
    audioBuffer = await readWavFile(file);

    const duration = audioBuffer.duration;

    setText('audioDuration', formatTime(duration));

    setText(
      'resultDuration',
      `${duration.toFixed(2)} ثانیه`
    );

    setText(
      'resultRate',
      `${audioBuffer.sampleRate.toLocaleString('en-US')} Hz`
    );

    // Convert original samples to browser-compatible PCM WAV
    const playableWav = createPlayableWav(
      audioBuffer.samples,
      audioBuffer.sampleRate
    );

    // Create audio URL
    audioObjectURL = URL.createObjectURL(playableWav);

    audioPlayer.src = audioObjectURL;
    audioPlayer.load();

    // Draw waveform from original samples
    drawWaveform(audioBuffer.samples);

    analyzeBtn.disabled = false;

    setText('waveStatus', 'شکل موج با موفقیت رسم شد');

    setText(
      'resultDescription',
      'سیگنال آماده تحلیل و پخش است.'
    );

  } catch (error) {
    audioBuffer = null;
    currentFile = null;

    analyzeBtn.disabled = true;

    setText('waveStatus', 'خواندن شکل موج ممکن نشد');

    setText(
      'resultDescription',
      error.message || 'فرمت WAV پشتیبانی نمی‌شود.'
    );

    drawPlaceholder();

    console.error('WAV reading error:', error);
  }
}


// Read WAV directly without AudioContext
async function readWavFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const view = new DataView(arrayBuffer);

  if (
    view.byteLength < 44 ||
    readString(view, 0, 4) !== 'RIFF' ||
    readString(view, 8, 4) !== 'WAVE'
  ) {
    throw new Error('Invalid WAV file');
  }

  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readString(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;

    if (chunkStart + chunkSize > view.byteLength) {
      throw new Error('Invalid WAV chunk size');
    }

    if (chunkId === 'fmt ') {
      if (chunkSize < 16) {
        throw new Error('Invalid WAV format chunk');
      }

      format = {
        audioFormat: view.getUint16(chunkStart, true),
        channels: view.getUint16(chunkStart + 2, true),
        sampleRate: view.getUint32(chunkStart + 4, true),
        blockAlign: view.getUint16(chunkStart + 12, true),
        bitsPerSample: view.getUint16(chunkStart + 14, true)
      };

      // WAVE_FORMAT_EXTENSIBLE
      if (format.audioFormat === 0xFFFE && chunkSize >= 40) {
        format.audioFormat = view.getUint16(
          chunkStart + 24,
          true
        );
      }
    }

    if (chunkId === 'data' && dataOffset === -1) {
      dataOffset = chunkStart;
      dataSize = chunkSize;
    }

    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (!format || dataOffset < 0) {
    throw new Error('WAV format or audio data not found');
  }

  const {
    audioFormat,
    channels,
    sampleRate,
    blockAlign,
    bitsPerSample
  } = format;

  if (!channels || !sampleRate || !blockAlign) {
    throw new Error('Invalid WAV parameters');
  }

  if (![1, 3].includes(audioFormat)) {
    throw new Error('Unsupported WAV encoding');
  }

  if (
    audioFormat === 1 &&
    ![8, 16, 24, 32].includes(bitsPerSample)
  ) {
    throw new Error('Unsupported PCM bit depth');
  }

  if (
    audioFormat === 3 &&
    ![32, 64].includes(bitsPerSample)
  ) {
    throw new Error('Unsupported float bit depth');
  }

  const bytesPerSample = bitsPerSample / 8;

  if (blockAlign < channels * bytesPerSample) {
    throw new Error('Invalid WAV block alignment');
  }

  const frameCount = Math.floor(dataSize / blockAlign);

  if (frameCount === 0) {
    throw new Error('Empty WAV audio data');
  }

  const samples = new Float32Array(frameCount);

  for (let i = 0; i < frameCount; i++) {
    let sum = 0;

    for (let ch = 0; ch < channels; ch++) {
      const pos =
        dataOffset + i * blockAlign + ch * bytesPerSample;

      let value;

      if (audioFormat === 3) {
        value = bitsPerSample === 32
          ? view.getFloat32(pos, true)
          : view.getFloat64(pos, true);

      } else if (bitsPerSample === 8) {
        value = (view.getUint8(pos) - 128) / 128;

      } else if (bitsPerSample === 16) {
        value = view.getInt16(pos, true) / 32768;

      } else if (bitsPerSample === 24) {
        let raw =
          view.getUint8(pos) |
          (view.getUint8(pos + 1) << 8) |
          (view.getUint8(pos + 2) << 16);

        if (raw & 0x800000) {
          raw |= 0xFF000000;
        }

        value = raw / 8388608;

      } else {
        value = view.getInt32(pos, true) / 2147483648;
      }

      if (!Number.isFinite(value)) {
        value = 0;
      }

      sum += value;
    }

    samples[i] = sum / channels;
  }

  return {
    samples,
    sampleRate,
    duration: frameCount / sampleRate,
    channels,
    bitsPerSample
  };
}


// Convert Float32 samples to standard PCM 16-bit WAV
function createPlayableWav(samples, sampleRate) {
  const dataSize = samples.length * 2;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, text) {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  // RIFF header
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');

  // Format chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  // Data chunk
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // Float32 to signed PCM 16-bit
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(
      -1,
      Math.min(1, samples[i])
    );

    const value = sample < 0
      ? sample * 32768
      : sample * 32767;

    view.setInt16(44 + i * 2, value, true);
  }

  return new Blob([buffer], {
    type: 'audio/wav'
  });
}


// Read string from DataView
function readString(view, offset, length) {
  let result = '';

  for (let i = 0; i < length; i++) {
    result += String.fromCharCode(
      view.getUint8(offset + i)
    );
  }

  return result;
}


// ONNX browser inference
analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  analyzeBtn.disabled = true;

  setText('systemStatus', 'در حال تحلیل...');
  setText('resultTitle', 'در حال تحلیل سیگنال');

  setText(
    'resultDescription',
    'در حال استخراج ویژگی‌ها و اجرای مدل هوش مصنوعی...'
  );

  try {
    // IMPORTANT: Send original file to ONNX pipeline
    const result = await window.predictHeartSound(currentFile);

    console.log('Heart Sound Analysis:', result);

    const prediction = result.prediction;

    let confidenceValue = Number(result.confidence);

    if (
      !Number.isFinite(confidenceValue) &&
      result.probabilities
    ) {
      const probabilities = Array.from(result.probabilities);
      const classIndex = prediction === 'Normal' ? 1 : 0;

      confidenceValue = Number(probabilities[classIndex]);
    }

    if (prediction === 'Normal') {
      setText('resultTitle', 'الگوی طبیعی');

    } else if (prediction === 'Abnormal') {
      setText('resultTitle', 'الگوی غیرطبیعی');

    } else {
      throw new Error('مدل نتیجه معتبری برنگرداند.');
    }

    if (
      Number.isFinite(confidenceValue) &&
      confidenceValue >= 0 &&
      confidenceValue <= 1
    ) {
      const confidence = (
        confidenceValue * 100
      ).toFixed(1);

      setText(
        'resultDescription',
        `نتیجه مدل · اطمینان ${confidence}٪`
      );

    } else {
      setText(
        'resultDescription',
        'نتیجه مدل دریافت شد؛ میزان اطمینان قابل محاسبه نیست.'
      );

      console.error('Invalid confidence:', result);
    }

    setText('systemStatus', 'تحلیل انجام شد');

  } catch (error) {
    console.error('Prediction error:', error);

    setText('resultTitle', 'خطا در تحلیل سیگنال');

    setText(
      'resultDescription',
      error.message || 'اجرای مدل با مشکل مواجه شد.'
    );

    setText('systemStatus', 'خطا در تحلیل');

  } finally {
    analyzeBtn.disabled = false;
  }
});


// Draw placeholder waveform
function drawPlaceholder() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawGrid();

  ctx.strokeStyle = '#4bbba8';
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let x = 0; x < canvas.width; x++) {
    const y =
      canvas.height / 2 +
      Math.sin(x * 0.055) * 5 +
      Math.sin(x * 0.17) * 2;

    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}


// Draw waveform grid
function drawGrid() {
  ctx.strokeStyle = '#20383b';
  ctx.lineWidth = 1;

  for (let y = 20; y < canvas.height; y += 35) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}


// Draw actual waveform
function drawWaveform(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawGrid();

  const width = canvas.width;
  const height = canvas.height;
  const center = height / 2;
  const amplitude = height * 0.44;

  ctx.strokeStyle = '#66d7c3';
  ctx.lineWidth = 1.4;
  ctx.beginPath();

  for (let x = 0; x < width; x++) {
    const start = Math.floor(
      x * data.length / width
    );

    const end = Math.max(
      start + 1,
      Math.floor((x + 1) * data.length / width)
    );

    let min = 1;
    let max = -1;

    for (
      let i = start;
      i < end && i < data.length;
      i++
    ) {
      min = Math.min(min, data[i]);
      max = Math.max(max, data[i]);
    }

    ctx.moveTo(
      x,
      center - max * amplitude
    );

    ctx.lineTo(
      x,
      center - min * amplitude
    );
  }

  ctx.stroke();
}


// Update text elements
function setText(id, value) {
  document.getElementById(id).textContent = value;
}


// Format audio duration
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);

  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}
