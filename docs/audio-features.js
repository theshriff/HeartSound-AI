console.log("Audio Features module loaded.");

async function loadWavFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const view = new DataView(arrayBuffer);

    if (
        view.getUint32(0, false) !== 0x52494646 ||
        view.getUint32(8, false) !== 0x57415645
    ) {
        throw new Error("Invalid WAV file.");
    }

    let offset = 12;
    let format = null;
    let channels = null;
    let sampleRate = null;
    let bitsPerSample = null;
    let audioData = null;
    let blockAlign = null;

    while (offset + 8 <= view.byteLength) {
        const chunkId = view.getUint32(offset, false);
        const chunkSize = view.getUint32(offset + 4, true);
        const start = offset + 8;

        if (chunkId === 0x666d7420) {
            format = view.getUint16(start, true);
            channels = view.getUint16(start + 2, true);
            sampleRate = view.getUint32(start + 4, true);
            blockAlign = view.getUint16(start + 12, true);
            bitsPerSample = view.getUint16(start + 14, true);
        }

        if (chunkId === 0x64617461) {
            audioData = { start, size: chunkSize };
        }

        offset = start + chunkSize + (chunkSize % 2);
    }

    if (!audioData || !sampleRate || !channels) {
        throw new Error("WAV format not supported.");
    }

    if (![1, 3].includes(format)) {
        throw new Error("Only PCM and IEEE float WAV files are supported.");
    }

    const bytesPerSample = bitsPerSample / 8;
    const frames = Math.floor(audioData.size / blockAlign);
    const audio = new Float64Array(frames);

    for (let i = 0; i < frames; i++) {
        let sum = 0;

        for (let ch = 0; ch < channels; ch++) {
            const pos = audioData.start + i * blockAlign + ch * bytesPerSample;
            let value;

            if (format === 3 && bitsPerSample === 32) {
                value = view.getFloat32(pos, true);
            } else if (format === 1 && bitsPerSample === 16) {
                value = view.getInt16(pos, true);
            } else if (format === 1 && bitsPerSample === 24) {
                let raw = view.getUint8(pos) |
                    (view.getUint8(pos + 1) << 8) |
                    (view.getUint8(pos + 2) << 16);

                if (raw & 0x800000) raw |= 0xFF000000;
                value = raw;
            } else if (format === 1 && bitsPerSample === 32) {
                value = view.getInt32(pos, true);
            } else if (format === 1 && bitsPerSample === 8) {
                value = view.getUint8(pos) - 128;
            } else {
                throw new Error("Unsupported WAV bit depth: " + bitsPerSample);
            }

            sum += value;
        }

        audio[i] = sum / channels;
    }

    if (sampleRate <= 800 || audio.length < 1024) {
        throw new Error("Audio sample rate or duration is invalid.");
    }

    return {
        audio,
        sampleRate,
        channels,
        bitsPerSample
    };
}

// Butterworth 4th-order bandpass filter (25-400 Hz, 2000 Hz)
const BUTTERWORTH_SOS = [
    [0.03797304141947671, 0.07594608283895342, 0.03797304141947671, 1, -0.4309758062025911, 0.09241225285118021],
    [1, 2, 1, 1, -0.48298360428383835, 0.5047338773310682],
    [1, -2, 1, 1, -1.8480081559656902, 0.8549577693610224],
    [1, -2, 1, 1, -1.9400651139716438, 0.9462258657927407]
];

function getSosInitialConditions(sos) {
    let scale = 1;

    return sos.map(section => {
        const [b0, b1, b2, , a1, a2] = section;

        const gain = (b0 + b1 + b2) / (1 + a1 + a2);
        const y = gain;

        const zi = [
            (y - b0) * scale,
            (b2 - a2 * y) * scale
        ];

        scale *= gain;
        return zi;
    });
}

function sosFilter(data, sos, initialValue) {
    let output = Float64Array.from(data);
    const zi = getSosInitialConditions(sos);

    for (let s = 0; s < sos.length; s++) {
        const [b0, b1, b2, , a1, a2] = sos[s];

        let z0 = zi[s][0] * initialValue;
        let z1 = zi[s][1] * initialValue;

        const next = new Float64Array(output.length);

        for (let i = 0; i < output.length; i++) {
            const x = output[i];
            const y = b0 * x + z0;

            z0 = b1 * x - a1 * y + z1;
            z1 = b2 * x - a2 * y;

            next[i] = y;
        }

        output = next;
    }

    return output;
}

function butterworthFilter(data) {
    const edge = 27;

    if (data.length <= edge) {
        throw new Error("Audio is too short for filtering.");
    }

    const extended = new Float64Array(data.length + 2 * edge);

    // Odd extension at the beginning
    for (let i = 0; i < edge; i++) {
        extended[i] = 2 * data[0] - data[edge - i];
    }

    extended.set(data, edge);

    // Odd extension at the end
    for (let i = 0; i < edge; i++) {
        extended[edge + data.length + i] =
            2 * data[data.length - 1] - data[data.length - 2 - i];
    }

    // Forward filtering
    const forward = sosFilter(
        extended,
        BUTTERWORTH_SOS,
        extended[0]
    );

    // Reverse
    forward.reverse();

    // Backward filtering
    const backward = sosFilter(
        forward,
        BUTTERWORTH_SOS,
        forward[0]
    );

    // Reverse and remove padding
    backward.reverse();

    return backward.slice(edge, edge + data.length);
}

console.log("Butterworth filter loaded.");

function extractBasicFeatures(filteredAudio, sampleRate) {
    const n = filteredAudio.length;

    // 1. RMS
    let sumSquares = 0;

    for (let i = 0; i < n; i++) {
        sumSquares += filteredAudio[i] ** 2;
    }

    const rms = Math.sqrt(sumSquares / n);

    // 2. Zero Crossing Rate
    let crossings = 0;

    for (let i = 1; i < n; i++) {
        if (
            (filteredAudio[i] < 0 && filteredAudio[i - 1] >= 0) ||
            (filteredAudio[i] >= 0 && filteredAudio[i - 1] < 0)
        ) {
            crossings++;
        }
    }

    const zcr = crossings / n;

    // 3. FFT
    const fftSize = n;
    const real = new Float64Array(fftSize);
    const imag = new Float64Array(fftSize);

    for (let i = 0; i < n; i++) {
        real[i] = filteredAudio[i];
    }

    fftInPlace(real, imag);

    const half = Math.floor(fftSize / 2) + 1;
    const power = new Float64Array(half);
    const frequencies = new Float64Array(half);

    let totalPower = 0;

    for (let i = 0; i < half; i++) {
        frequencies[i] = i * sampleRate / fftSize;
        power[i] = real[i] ** 2 + imag[i] ** 2;

        if (i === 0) power[i] = 0;

        totalPower += power[i];
    }

    if (totalPower === 0) {
        throw new Error("Audio contains no usable signal.");
    }

    // 4. Spectral Centroid
    let weightedSum = 0;

    for (let i = 0; i < half; i++) {
        weightedSum += frequencies[i] * power[i];
    }

    const centroid = weightedSum / totalPower;

    // 5. Spectral Bandwidth
    let variance = 0;

    for (let i = 0; i < half; i++) {
        variance += ((frequencies[i] - centroid) ** 2) * power[i];
    }

    const bandwidth = Math.sqrt(variance / totalPower);

    // 6. Dominant Frequency
    let maxIndex = 0;

    for (let i = 1; i < half; i++) {
        if (power[i] > power[maxIndex]) {
            maxIndex = i;
        }
    }

    const dominantFrequency = frequencies[maxIndex];

    return {
        RMS: rms,
        ZCR: zcr,
        Spectral_Centroid: centroid,
        Spectral_Bandwidth: bandwidth,
        Dominant_Frequency: dominantFrequency
    };
}

function fftInPlace(real, imag) {
    const n = real.length;

    // Bluestein FFT: supports any signal length
    let m = 1;
    while (m < 2 * n - 1) m *= 2;

    const aR = new Float64Array(m);
    const aI = new Float64Array(m);
    const bR = new Float64Array(m);
    const bI = new Float64Array(m);

    for (let i = 0; i < n; i++) {
        const angle = Math.PI * (i * i % (2 * n)) / n;

        const c = Math.cos(angle);
        const s = Math.sin(angle);

        aR[i] = real[i] * c + imag[i] * s;
        aI[i] = imag[i] * c - real[i] * s;

        bR[i] = c;
        bI[i] = s;

        if (i !== 0) {
            bR[m - i] = c;
            bI[m - i] = s;
        }
    }

    radix2FFT(aR, aI, false);
    radix2FFT(bR, bI, false);

    for (let i = 0; i < m; i++) {
        const r = aR[i] * bR[i] - aI[i] * bI[i];
        const im = aR[i] * bI[i] + aI[i] * bR[i];
        aR[i] = r;
        aI[i] = im;
    }

    radix2FFT(aR, aI, true);

    for (let i = 0; i < n; i++) {
        const angle = Math.PI * (i * i % (2 * n)) / n;
        const c = Math.cos(angle);
        const s = Math.sin(angle);

        real[i] = aR[i] * c + aI[i] * s;
        imag[i] = aI[i] * c - aR[i] * s;
    }
}

function radix2FFT(real, imag, inverse) {
    const n = real.length;

    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;

        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;

        if (i < j) {
            [real[i], real[j]] = [real[j], real[i]];
            [imag[i], imag[j]] = [imag[j], imag[i]];
        }
    }

    for (let size = 2; size <= n; size *= 2) {
        const angle = 2 * Math.PI / size * (inverse ? 1 : -1);
        const wr0 = Math.cos(angle);
        const wi0 = Math.sin(angle);

        for (let start = 0; start < n; start += size) {
            let wr = 1;
            let wi = 0;

            for (let j = 0; j < size / 2; j++) {
                const even = start + j;
                const odd = even + size / 2;

                const tr = wr * real[odd] - wi * imag[odd];
                const ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] += tr;
                imag[even] += ti;

                const nextWr = wr * wr0 - wi * wi0;
                wi = wr * wi0 + wi * wr0;
                wr = nextWr;
            }
        }
    }

    if (inverse) {
        for (let i = 0; i < n; i++) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

console.log("FFT module loaded.");

window.testAudioFeatures = async function(file) {
    const wav = await loadWavFile(file);
    const filtered = butterworthFilter(wav.audio);
    const features = extractBasicFeatures(filtered, wav.sampleRate);

    console.table(features);
    return features;
};

let melFiltersCache = null;

async function loadMelFilters() {
    if (melFiltersCache) return melFiltersCache;

    const response = await fetch("mel_filters.json");

    if (!response.ok) {
        throw new Error("Could not load mel_filters.json");
    }

    melFiltersCache = await response.json();
    return melFiltersCache;
}

function extractMFCCFeatures(filteredAudio, sampleRate, melFilters) {
    const nFFT = 512;
    const hopLength = 128;
    const nMels = 128;
    const nMFCC = 13;
    const padding = nFFT / 2;

    const frameCount = 1 + Math.floor(filteredAudio.length / hopLength);

    const melEnergies = Array.from(
        { length: nMels },
        () => new Float64Array(frameCount)
    );

    // Hann window
    const hann = new Float64Array(nFFT);

    for (let i = 0; i < nFFT; i++) {
        hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFFT);
    }

    // STFT and Mel spectrogram
    for (let frame = 0; frame < frameCount; frame++) {
        const real = new Float64Array(nFFT);
        const imag = new Float64Array(nFFT);

        const start = frame * hopLength - padding;

        for (let i = 0; i < nFFT; i++) {
            const index = start + i;

            const sample =
                index >= 0 && index < filteredAudio.length
                    ? filteredAudio[index]
                    : 0;

            real[i] = sample * hann[i];
        }

        fftInPlace(real, imag);

        const power = new Float64Array(nFFT / 2 + 1);

        for (let k = 0; k <= nFFT / 2; k++) {
            power[k] = real[k] ** 2 + imag[k] ** 2;
        }

        for (let m = 0; m < nMels; m++) {
            let energy = 0;

            for (let k = 0; k <= nFFT / 2; k++) {
                energy += melFilters[m][k] * power[k];
            }

            melEnergies[m][frame] = energy;
        }
    }

    // Power to decibels
    const amin = 1e-10;
    const db = Array.from(
        { length: nMels },
        () => new Float64Array(frameCount)
    );

    let maxDB = -Infinity;

    for (let m = 0; m < nMels; m++) {
        for (let t = 0; t < frameCount; t++) {
            const value = 10 * Math.log10(
                Math.max(amin, melEnergies[m][t])
            );

            db[m][t] = value;
            maxDB = Math.max(maxDB, value);
        }
    }

    // librosa top_db = 80
    for (let m = 0; m < nMels; m++) {
        for (let t = 0; t < frameCount; t++) {
            db[m][t] = Math.max(db[m][t], maxDB - 80);
        }
    }

    // DCT-II with orthonormal normalization
    const mfcc = Array.from(
        { length: nMFCC },
        () => new Float64Array(frameCount)
    );

    for (let k = 0; k < nMFCC; k++) {
        const scale = k === 0
            ? Math.sqrt(1 / nMels)
            : Math.sqrt(2 / nMels);

        for (let t = 0; t < frameCount; t++) {
            let sum = 0;

            for (let m = 0; m < nMels; m++) {
                sum += db[m][t] *
                    Math.cos(Math.PI * k * (m + 0.5) / nMels);
            }

            mfcc[k][t] = scale * sum;
        }
    }

    // Mean and standard deviation
    const features = {};

    for (let i = 0; i < nMFCC; i++) {
        const values = mfcc[i];

        let mean = 0;

        for (const value of values) {
            mean += value;
        }

        mean /= frameCount;

        let variance = 0;

        for (const value of values) {
            variance += (value - mean) ** 2;
        }

        const std = Math.sqrt(variance / frameCount);

        features[`MFCC_${i + 1}_mean`] = mean;
        features[`MFCC_${i + 1}_std`] = std;
    }

    return features;
}

// Full feature extraction test
window.testAudioFeatures = async function(file) {
    if (!file) {
        throw new Error("Please select a WAV file first.");
    }

    const wav = await loadWavFile(file);

    if (wav.sampleRate !== 2000) {
        throw new Error("This version currently supports 2000 Hz audio only.");
    }

    // First five features
    const filtered = butterworthFilter(wav.audio);
    const basic = extractBasicFeatures(filtered, wav.sampleRate);

    // MFCC branch: normalize original audio first
    const normalized = Float64Array.from(wav.audio);

    let maxValue = 0;

    for (const value of normalized) {
        maxValue = Math.max(maxValue, Math.abs(value));
    }

    if (maxValue > 0) {
        for (let i = 0; i < normalized.length; i++) {
            normalized[i] /= maxValue;
        }
    }

    const mfccFiltered = butterworthFilter(normalized);
    const melFilters = await loadMelFilters();

    const mfcc = extractMFCCFeatures(
        mfccFiltered,
        wav.sampleRate,
        melFilters
    );

    const allFeatures = { ...basic, ...mfcc };

    console.table(allFeatures);

    return allFeatures;
};

console.log("MFCC module loaded.");

let onnxSession = null;

async function loadONNXModel() {
    if (onnxSession) return onnxSession;

    console.log("Loading ONNX model...");

    onnxSession = await ort.InferenceSession.create(
        "heart_sound_model.onnx",
        {
            executionProviders: ["wasm"]
        }
    );

    console.log("ONNX model loaded!");
    console.log("Inputs:", onnxSession.inputNames);
    console.log("Outputs:", onnxSession.outputNames);

    return onnxSession;
}

window.predictHeartSound = async function(file) {
    if (!file) {
        throw new Error("Please select a WAV file.");
    }

    // Extract 31 features
    const features = await window.testAudioFeatures(file);

    const featureOrder = [
        "RMS",
        "ZCR",
        "Spectral_Centroid",
        "Spectral_Bandwidth",
        "Dominant_Frequency"
    ];

    for (let i = 1; i <= 13; i++) {
        featureOrder.push(`MFCC_${i}_mean`);
        featureOrder.push(`MFCC_${i}_std`);
    }

    const inputData = new Float32Array(
        featureOrder.map(name => features[name])
    );

    if (inputData.length !== 31) {
        throw new Error("Expected 31 features.");
    }

    // Load model
    const session = await loadONNXModel();

    // Prepare ONNX input
    const inputName = session.inputNames[0];

    const tensor = new ort.Tensor(
        "float32",
        inputData,
        [1, 31]
    );

    // Run prediction
    const outputs = await session.run({
        [inputName]: tensor
    });

    console.log("Raw ONNX outputs:", outputs);

    const outputValues = Object.values(outputs);

    const labelOutput = outputValues.find(
        output => output.data.length === 1
    );

    const probabilityOutput = outputValues.find(
        output => output.data.length === 2
    );

    if (!labelOutput || !probabilityOutput) {
        throw new Error("Unexpected ONNX output format.");
    }

    const prediction = Number(labelOutput.data[0]);
    const probabilities = Array.from(probabilityOutput.data);

    const confidence = probabilities[prediction];

    const result = {
        prediction: prediction === 1 ? "Normal" : "Abnormal",
        confidence,
        probabilities
    };

    console.log("Final prediction:", result);

    return result;
};

console.log("MFCC module loaded.");