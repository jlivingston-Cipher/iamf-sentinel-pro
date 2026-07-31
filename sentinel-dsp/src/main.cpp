// sentinel-dsp — static C++ kernel of the Sentinel Pro numeric core (ADR-6b).
//
// Clean-room implementation written from ITU-R BS.1770-4 (K-weighting
// pre-filter + RLB high-pass, 400 ms gated blocks at 75% overlap, absolute
// -70 LKFS and relative -10 LU gates, LFE excluded) and Annex 2 (4x
// oversampled true peak), matching the documented conventions of the Python
// reference `dsp.py` (Phase 2, doc 19): BS.1770-4 Tables 4/5 channel
// weighting (sides M±090 / 5.1 surrounds M±110 at 1.41; rears M±135 and all
// heights 1.00 — F33 fix, doc 65), FFMPEG_WEIGHTS variant for cross-checks
// (ffmpeg's BACK_MASK over-weighting, F29), 48-tap Kaiser(12)
// windowed-sinc polyphase true peak with per-phase DC normalization.
//
// Subprocess/JSON boundary (ADR-4): reads one WAV, emits one JSON document.
// No third-party dependencies; C++17; double precision throughout.
//
// Usage:
//   sentinel-dsp FILE.wav [--layout L] [--weights w1,w2,...] [--variant primary|ffmpeg]
//                         [--win-s SECONDS]
//
// Output JSON fields mirror dsp.py's function surface:
//   sample_rate, channels, frames,
//   integrated_lufs, digital_peak_dbfs, true_peak_dbtp,
//   channel_stats[{rms_dbfs,peak_dbfs}],
//   correlation: {duplicate_pairs[[i,j,corr]], silent_channels[], mean_offdiag},
//   energy_profile{windows,median_dev,p95_dev}   (when --layout has positions)

#include <algorithm>
#include <array>
#include <cmath>

#ifndef M_PI                       // not standard C++; absent on MSVC
#define M_PI 3.14159265358979323846
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void die(const std::string& msg) {
    std::fprintf(stderr, "sentinel-dsp: error: %s\n", msg.c_str());
    std::exit(2);
}

// ---------------------------------------------------------------------------
// WAV reading (WAVE_FORMAT_EXTENSIBLE-tolerant) — mirrors dsp.read_wav
// ---------------------------------------------------------------------------

struct Audio {
    uint32_t sr = 0;
    size_t frames = 0, channels = 0;
    std::vector<double> x;  // interleaved-deinterleaved: x[ch*frames + n]
};

static uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static Audio read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4))
        die("not a RIFF/WAVE file: " + path);
    size_t pos = 12, end = d.size();
    const uint8_t *fmt = nullptr, *payload = nullptr;
    size_t fmt_len = 0, payload_len = 0;
    while (pos + 8 <= end) {
        const uint8_t* cid = d.data() + pos;
        uint32_t csize = rd_u32(d.data() + pos + 4);
        if (pos + 8 + csize > end) csize = (uint32_t)(end - pos - 8);
        if (!std::memcmp(cid, "fmt ", 4)) { fmt = d.data() + pos + 8; fmt_len = csize; }
        else if (!std::memcmp(cid, "data", 4)) { payload = d.data() + pos + 8; payload_len = csize; }
        pos += 8 + csize + (csize & 1);
    }
    if (!fmt || !payload) die("missing fmt/data chunk: " + path);
    uint16_t wformat = rd_u16(fmt);
    uint16_t ch = rd_u16(fmt + 2);
    uint32_t sr = rd_u32(fmt + 4);
    uint16_t bits = rd_u16(fmt + 14);
    if (wformat == 0xFFFE && fmt_len >= 26) wformat = rd_u16(fmt + 24);

    std::vector<double> interleaved;
    if (wformat == 3 && bits == 32) {
        size_t n = payload_len / 4;
        interleaved.resize(n);
        for (size_t i = 0; i < n; i++) {
            float v; std::memcpy(&v, payload + 4 * i, 4);
            interleaved[i] = (double)v;
        }
    } else if (wformat == 3 && bits == 64) {
        size_t n = payload_len / 8;
        interleaved.resize(n);
        for (size_t i = 0; i < n; i++) {
            double v; std::memcpy(&v, payload + 8 * i, 8);
            interleaved[i] = v;
        }
    } else if (bits == 16) {
        size_t n = payload_len / 2;
        interleaved.resize(n);
        for (size_t i = 0; i < n; i++) {
            int16_t v = (int16_t)rd_u16(payload + 2 * i);
            interleaved[i] = (double)v / 32768.0;
        }
    } else if (bits == 24) {
        size_t n = payload_len / 3;
        interleaved.resize(n);
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)payload[3 * i] | ((int32_t)payload[3 * i + 1] << 8)
                      | ((int32_t)payload[3 * i + 2] << 16);
            if (v >= (1 << 23)) v -= (1 << 24);
            interleaved[i] = (double)v / 8388608.0;
        }
    } else if (bits == 32) {
        size_t n = payload_len / 4;
        interleaved.resize(n);
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)rd_u32(payload + 4 * i);
            interleaved[i] = (double)v / 2147483648.0;
        }
    } else {
        die("unhandled bit depth " + std::to_string(bits) + ": " + path);
    }
    Audio a;
    a.sr = sr; a.channels = ch; a.frames = interleaved.size() / ch;
    a.x.resize(a.frames * ch);
    for (size_t n = 0; n < a.frames; n++)
        for (size_t c = 0; c < ch; c++)
            a.x[c * a.frames + n] = interleaved[n * ch + c];
    return a;
}

// ---------------------------------------------------------------------------
// BS.1770-4 K-weighting — mirrors dsp._k_weighting_coeffs / _biquad
// ---------------------------------------------------------------------------

struct Biquad { double b0, b1, b2, a1, a2; };

static void k_weighting_coeffs(double sr, Biquad& shelf, Biquad& hp) {
    {   // Stage 1: high shelf
        double f0 = 1681.9744509555319, G = 3.999843853973347, Q = 0.7071752369554193;
        double K = std::tan(M_PI * f0 / sr);
        double Vh = std::pow(10.0, G / 20.0);
        double Vb = std::pow(Vh, 0.499666774155);
        double a0 = 1.0 + K / Q + K * K;
        shelf.b0 = (Vh + Vb * K / Q + K * K) / a0;
        shelf.b1 = 2.0 * (K * K - Vh) / a0;
        shelf.b2 = (Vh - Vb * K / Q + K * K) / a0;
        shelf.a1 = 2.0 * (K * K - 1.0) / a0;
        shelf.a2 = (1.0 - K / Q + K * K) / a0;
    }
    {   // Stage 2: RLB high-pass — numerator kept as exactly [1, -2, 1]
        double f0 = 38.13547087602444, Q = 0.5003270373238773;
        double K = std::tan(M_PI * f0 / sr);
        double a0 = 1.0 + K / Q + K * K;
        hp.b0 = 1.0; hp.b1 = -2.0; hp.b2 = 1.0;
        hp.a1 = 2.0 * (K * K - 1.0) / a0;
        hp.a2 = (1.0 - K / Q + K * K) / a0;
    }
}

static void biquad_inplace(std::vector<double>& x, size_t frames, size_t ch, const Biquad& q) {
    for (size_t c = 0; c < ch; c++) {
        double z1 = 0.0, z2 = 0.0;
        double* p = x.data() + c * frames;
        for (size_t n = 0; n < frames; n++) {
            double xn = p[n];
            double yn = q.b0 * xn + z1;
            z1 = q.b1 * xn - q.a1 * yn + z2;
            z2 = q.b2 * xn - q.a2 * yn;
            p[n] = yn;
        }
    }
}

// ---------------------------------------------------------------------------
// Channel weight tables — mirror dsp.CHANNEL_WEIGHTS / FFMPEG_WEIGHTS
// ---------------------------------------------------------------------------

static const std::map<std::string, std::vector<double>> CHANNEL_WEIGHTS = {
    {"2.0",   {1, 1}},
    {"5.1",   {1, 1, 1, 0, 1.41, 1.41}},
    {"5.1.2", {1, 1, 1, 0, 1.41, 1.41, 1, 1}},
    {"5.1.4", {1, 1, 1, 0, 1.41, 1.41, 1, 1, 1, 1}},
    // 7.1.x: Lss/Rss (M±090) 1.41; Lrs/Rrs (M±135) 1.00 per BS.1770-4
    // Tables 4/5 (F33 fix — rears were 1.41 in earlier revisions).
    {"7.1",   {1, 1, 1, 0, 1.41, 1.41, 1, 1}},
    {"7.1.2", {1, 1, 1, 0, 1.41, 1.41, 1, 1, 1, 1}},
    {"7.1.4", {1, 1, 1, 0, 1.41, 1.41, 1, 1, 1, 1, 1, 1}},
    {"3.1.2", {1, 1, 1, 0, 1, 1}},
};
static const std::map<std::string, std::vector<double>> FFMPEG_WEIGHTS = {
    {"2.0",   {1, 1}},
    {"5.1",   {1, 1, 1, 0, 1.41, 1.41}},
    {"7.1",   {1, 1, 1, 0, 1.41, 1.41, 1.41, 1.41}},
    {"7.1.4", {1, 1, 1, 0, 1.41, 1.41, 1.41, 1.41, 1, 1, 1.41, 1.41}},
};

// ---------------------------------------------------------------------------
// Integrated loudness — mirrors dsp.integrated_loudness
// ---------------------------------------------------------------------------

static double integrated_loudness(const Audio& a, const std::vector<double>& weights) {
    size_t nch = a.channels;
    std::vector<double> w(nch, 1.0);
    for (size_t c = 0; c < nch && c < weights.size(); c++) w[c] = weights[c];

    Biquad shelf, hp;
    k_weighting_coeffs((double)a.sr, shelf, hp);
    std::vector<double> y = a.x;                       // copy; filtered in place
    biquad_inplace(y, a.frames, nch, shelf);
    biquad_inplace(y, a.frames, nch, hp);

    size_t block = (size_t)std::llround(0.400 * a.sr);
    size_t hop = (size_t)std::llround(0.100 * a.sr);
    size_t frames = a.frames;
    if (frames < block) {                              // zero-pad to one block
        std::vector<double> y2(block * nch, 0.0);
        for (size_t c = 0; c < nch; c++)
            std::copy(y.begin() + c * frames, y.begin() + c * frames + frames,
                      y2.begin() + c * block);
        y.swap(y2);
        frames = block;
    }
    size_t n_blocks = 1 + (frames - block) / hop;

    // Per-block weighted mean-square power, then BS.1770 gating.
    std::vector<double> zsum(n_blocks);                // sum_ch w_c * z_cb
    for (size_t b = 0; b < n_blocks; b++) {
        size_t s = b * hop;
        double acc = 0.0;
        for (size_t c = 0; c < nch; c++) {
            if (w[c] == 0.0) continue;
            const double* p = y.data() + c * frames + s;
            double ss = 0.0;
            for (size_t n = 0; n < block; n++) ss += p[n] * p[n];
            acc += w[c] * (ss / (double)block);
        }
        zsum[b] = acc;
    }
    auto lk_of = [](double z) { return -0.691 + 10.0 * std::log10(std::max(z, 1e-12)); };

    double abs_sum = 0.0; size_t abs_n = 0;
    for (size_t b = 0; b < n_blocks; b++)
        if (lk_of(zsum[b]) > -70.0) { abs_sum += zsum[b]; abs_n++; }
    if (abs_n == 0) return -70.0;
    double rel_thr = lk_of(abs_sum / (double)abs_n) - 10.0;

    double g_sum = 0.0; size_t g_n = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        double lk = lk_of(zsum[b]);
        if (lk > -70.0 && lk > rel_thr) { g_sum += zsum[b]; g_n++; }
    }
    if (g_n == 0) return -70.0;
    return lk_of(g_sum / (double)g_n);
}

// ---------------------------------------------------------------------------
// Peaks — mirror dsp.digital_peak_dbfs / _tp_fir / true_peak_dbtp
// ---------------------------------------------------------------------------

static double digital_peak_dbfs(const Audio& a) {
    double p = 0.0;
    for (double v : a.x) p = std::max(p, std::fabs(v));
    return p > 0.0 ? 20.0 * std::log10(p) : -120.0;
}

static double bessel_i0(double x) {
    // Series expansion; converges quickly for the Kaiser beta range used.
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 64; k++) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

static std::vector<double> tp_fir(int taps = 48, int os = 4) {
    std::vector<double> h(taps);
    double beta = 12.0, denom = bessel_i0(beta);
    double half = (taps - 1) / 2.0;
    for (int i = 0; i < taps; i++) {
        double n = i - half;
        double s = n / (double)os;                     // np.sinc(n/os)
        double sinc = (s == 0.0) ? 1.0 : std::sin(M_PI * s) / (M_PI * s);
        double r = (i - half) / half;                  // np.kaiser(taps, 12)
        double kais = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;
        h[i] = sinc * kais;
    }
    for (int p = 0; p < os; p++) {                     // per-phase DC normalization
        double s = 0.0;
        for (int i = p; i < taps; i += os) s += h[i];
        if (std::fabs(s) > 1e-12)
            for (int i = p; i < taps; i += os) h[i] /= s;
    }
    return h;
}

static double true_peak_dbtp(const Audio& a, int os = 4) {
    std::vector<double> h = tp_fir(48, os);
    int K = (int)h.size();
    int off = (K - 1) / 2;                             // np.convolve 'same' start
    double peak = 0.0;
    for (size_t c = 0; c < a.channels; c++) {
        const double* x = a.x.data() + c * a.frames;
        long N = (long)a.frames * os;                  // zero-stuffed length
        // y[i] = sum_k h[k] * up[i + off - k]; up[j] = x[j/os] iff j%os==0
        for (long i = 0; i < N; i++) {
            double acc = 0.0;
            long base = i + off;
            // j = base - k must satisfy j % os == 0, 0 <= j < N
            long k0 = base % os;                       // smallest k with (base-k)%os==0
            for (long k = k0; k < K; k += os) {
                long j = base - k;
                if (j < 0) break;
                if (j >= N) continue;
                acc += h[k] * x[j / os];
            }
            peak = std::max(peak, std::fabs(acc));
        }
    }
    return peak > 0.0 ? 20.0 * std::log10(peak) : -120.0;
}

// ---------------------------------------------------------------------------
// Channel identity — mirror dsp.channel_stats / correlation_matrix /
// duplicate_pairs / silent_channels / mean_offdiag_correlation
// ---------------------------------------------------------------------------

struct ChanStat { double rms_dbfs, peak_dbfs; };

static std::vector<ChanStat> channel_stats(const Audio& a) {
    std::vector<ChanStat> out(a.channels);
    for (size_t c = 0; c < a.channels; c++) {
        const double* x = a.x.data() + c * a.frames;
        double peak = 0.0, ss = 0.0;
        for (size_t n = 0; n < a.frames; n++) {
            peak = std::max(peak, std::fabs(x[n]));
            ss += x[n] * x[n];
        }
        double rms = a.frames ? std::sqrt(ss / (double)a.frames) : 0.0;
        out[c].rms_dbfs = rms > 1e-9 ? 20.0 * std::log10(rms) : -120.0;
        out[c].peak_dbfs = peak > 1e-9 ? 20.0 * std::log10(peak) : -120.0;
    }
    return out;
}

static std::vector<std::vector<double>> correlation_matrix(const Audio& a) {
    size_t nch = a.channels, N = a.frames;
    std::vector<double> mean(nch, 0.0), norm(nch, 0.0);
    for (size_t c = 0; c < nch; c++) {
        const double* x = a.x.data() + c * N;
        double m = 0.0;
        for (size_t n = 0; n < N; n++) m += x[n];
        mean[c] = N ? m / (double)N : 0.0;
    }
    for (size_t c = 0; c < nch; c++) {
        const double* x = a.x.data() + c * N;
        double s = 0.0;
        for (size_t n = 0; n < N; n++) { double v = x[n] - mean[c]; s += v * v; }
        norm[c] = std::sqrt(s);
        if (norm[c] < 1e-12) norm[c] = 1.0;
    }
    std::vector<std::vector<double>> corr(nch, std::vector<double>(nch, 0.0));
    for (size_t i = 0; i < nch; i++) {
        const double* xi = a.x.data() + i * N;
        for (size_t j = i; j < nch; j++) {
            const double* xj = a.x.data() + j * N;
            double s = 0.0;
            for (size_t n = 0; n < N; n++)
                s += (xi[n] - mean[i]) * (xj[n] - mean[j]);
            corr[i][j] = corr[j][i] = s / (norm[i] * norm[j]);
        }
    }
    return corr;
}

// ---------------------------------------------------------------------------
// Energy-vector profile — mirrors dsp.energy_vector_profile
// ---------------------------------------------------------------------------

static const std::map<std::string, std::vector<std::pair<double, double>>> SPEAKER_POSITIONS = {
    {"2.0", {{30, 0}, {-30, 0}}},
    {"5.1", {{30, 0}, {-30, 0}, {0, 0}, {0, -30}, {110, 0}, {-110, 0}}},
    {"7.1.4", {{30, 0}, {-30, 0}, {0, 0}, {0, -30}, {90, 0}, {-90, 0},
               {135, 0}, {-135, 0}, {45, 45}, {-45, 45}, {135, 45}, {-135, 45}}},
};

struct EnergyProfile { long windows = 0; double median_dev = 0, p95_dev = 0; bool valid = false; };

static double percentile_linear(std::vector<double> v, double q) {
    // numpy default 'linear' interpolation
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    double idx = q / 100.0 * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
    double fr = idx - (double)lo;
    return v[lo] * (1.0 - fr) + v[hi] * fr;
}

static EnergyProfile energy_vector_profile(const Audio& a, const std::string& layout,
                                           double win_s = 0.5) {
    EnergyProfile ep;
    auto it = SPEAKER_POSITIONS.find(layout);
    if (it == SPEAKER_POSITIONS.end() || a.channels != it->second.size()) return ep;
    size_t nch = a.channels;
    std::vector<std::array<double, 3>> dirs(nch);
    double e0[3] = {0, 0, 0};
    for (size_t c = 0; c < nch; c++) {
        double az = it->second[c].first * M_PI / 180.0;
        double el = it->second[c].second * M_PI / 180.0;
        dirs[c] = {std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
        for (int k = 0; k < 3; k++) e0[k] += dirs[c][k];
    }
    for (int k = 0; k < 3; k++) e0[k] /= (double)nch;
    size_t win = std::max((size_t)1, (size_t)(win_s * a.sr));
    std::vector<double> devs;
    for (size_t s = 0; s + win <= a.frames; s += win) {
        double v[3] = {0, 0, 0}, tot = 0.0;
        for (size_t c = 0; c < nch; c++) {
            const double* x = a.x.data() + c * a.frames + s;
            double e = 0.0;
            for (size_t n = 0; n < win; n++) e += x[n] * x[n];
            tot += e;
            for (int k = 0; k < 3; k++) v[k] += e * dirs[c][k];
        }
        if (tot < 1e-9) continue;
        double dev = 0.0;
        for (int k = 0; k < 3; k++) {
            double d = v[k] / tot - e0[k];
            dev += d * d;
        }
        devs.push_back(std::sqrt(dev));
    }
    if (devs.empty()) return ep;
    ep.valid = true;
    ep.windows = (long)devs.size();
    ep.median_dev = percentile_linear(devs, 50.0);
    ep.p95_dev = percentile_linear(devs, 95.0);
    return ep;
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

static std::string jd(double v) {                      // JSON double, repr-ish
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.12g", v);
    return buf;
}

static std::string js(const std::string& v) {          // JSON string escape
    std::string out;
    for (char c : v) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if ((unsigned char)c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        }
        else out += c;
    }
    return out;
}

static double parse_double(const std::string& s, const std::string& what) {
    try {
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) die("invalid number for " + what + ": " + s);
        return v;
    } catch (const std::logic_error&) {
        die("invalid number for " + what + ": " + s);
    }
    return 0.0;  // unreachable
}

int main(int argc, char** argv) {
    std::string path, layout, variant = "primary";
    std::vector<double> weights;
    double win_s = 0.5;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--layout") layout = next();
        else if (a == "--variant") variant = next();
        else if (a == "--win-s") win_s = parse_double(next(), "--win-s");
        else if (a == "--weights") {
            std::string s = next(); size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                weights.push_back(parse_double(s.substr(p, q - p), "--weights"));
                p = q + 1;
            }
        }
        else if (a == "--help" || a == "-h") {
            std::puts("usage: sentinel-dsp FILE.wav [--layout L] [--weights w1,w2,...] "
                      "[--variant primary|ffmpeg] [--win-s S]");
            return 0;
        }
        else if (!a.empty() && a[0] == '-') die("unknown option " + a);
        else path = a;
    }
    if (path.empty()) die("no input file (see --help)");

    Audio a = read_wav(path);
    if (weights.empty() && !layout.empty()) {
        const auto& table = (variant == "ffmpeg") ? FFMPEG_WEIGHTS : CHANNEL_WEIGHTS;
        auto it = table.find(layout);
        if (it != table.end()) weights = it->second;
    }
    if (weights.empty()) weights.assign(a.channels, 1.0);

    double il = integrated_loudness(a, weights);
    double dp = digital_peak_dbfs(a);
    double tp = true_peak_dbtp(a);
    auto stats = channel_stats(a);
    auto corr = correlation_matrix(a);

    // duplicate_pairs (gates match dsp.py defaults)
    const double CORR_GATE = 0.999, LEVEL_GATE_DB = 0.1, SILENCE_DB = -80.0;
    std::string dup = "[";
    bool first = true;
    for (size_t i = 0; i < a.channels; i++) {
        if (stats[i].rms_dbfs < SILENCE_DB) continue;
        for (size_t j = i + 1; j < a.channels; j++) {
            if (stats[j].rms_dbfs < SILENCE_DB) continue;
            if (corr[i][j] >= CORR_GATE &&
                std::fabs(stats[i].rms_dbfs - stats[j].rms_dbfs) <= LEVEL_GATE_DB) {
                if (!first) dup += ",";
                dup += "[" + std::to_string(i) + "," + std::to_string(j) + "," + jd(corr[i][j]) + "]";
                first = false;
            }
        }
    }
    dup += "]";

    // silent_channels (empty when the whole program is silent — dsp.py rule)
    bool any_active = false;
    for (auto& s : stats) any_active = any_active || (s.rms_dbfs >= SILENCE_DB);
    std::string silent = "[";
    if (any_active) {
        first = true;
        for (size_t c = 0; c < a.channels; c++)
            if (stats[c].rms_dbfs < SILENCE_DB) {
                if (!first) silent += ",";
                silent += std::to_string(c);
                first = false;
            }
    }
    silent += "]";

    double offdiag = 0.0;
    if (a.channels >= 2) {
        double s = 0.0; long n = 0;
        for (size_t i = 0; i < a.channels; i++)
            for (size_t j = 0; j < a.channels; j++)
                if (i != j) { s += std::fabs(corr[i][j]); n++; }
        offdiag = s / (double)n;
    }

    EnergyProfile ep = energy_vector_profile(a, layout, win_s);

    std::string cs = "[";
    for (size_t c = 0; c < a.channels; c++) {
        if (c) cs += ",";
        cs += "{\"rms_dbfs\":" + jd(stats[c].rms_dbfs) +
              ",\"peak_dbfs\":" + jd(stats[c].peak_dbfs) + "}";
    }
    cs += "]";

    std::string layout_json = layout.empty() ? "null" : "\"" + js(layout) + "\"";
    std::string epj = ep.valid
        ? "{\"windows\":" + std::to_string(ep.windows) +
          ",\"median_dev\":" + jd(ep.median_dev) + ",\"p95_dev\":" + jd(ep.p95_dev) + "}"
        : "{\"windows\":0,\"median_dev\":null,\"p95_dev\":null}";

    std::printf("{\n"
                "\"tool\":\"sentinel-dsp\",\"version\":\"0.1.0\",\n"   // kernel JSON-contract version (independent of the Python packages)
                "\"sample_rate\":%u,\"channels\":%zu,\"frames\":%zu,\n"
                "\"variant\":\"%s\",\"layout\":%s,\n"
                "\"integrated_lufs\":%s,\n"
                "\"digital_peak_dbfs\":%s,\n"
                "\"true_peak_dbtp\":%s,\n"
                "\"channel_stats\":%s,\n"
                "\"correlation\":{\"duplicate_pairs\":%s,\"silent_channels\":%s,"
                "\"mean_offdiag\":%s},\n"
                "\"energy_profile\":%s\n"
                "}\n",
                a.sr, a.channels, a.frames,
                variant.c_str(), layout_json.c_str(),
                jd(il).c_str(), jd(dp).c_str(), jd(tp).c_str(),
                cs.c_str(), dup.c_str(), silent.c_str(), jd(offdiag).c_str(),
                epj.c_str());
    return 0;
}
