#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Par {
    double Lx = 10.0, Ly = 10.0, Tmax = 20.0;
    double la1 = 5.0, laT = 0.5, lac = 0.05, ld = 1.0;
    double baR = 2.5, baT = 1.2, bdR = 2.0, bdT = 0.8;
    double area() const { return Lx * Ly; }
};

struct Disk { double x, y, r; };
struct State { std::vector<Disk> active, erk; };
struct Obs { double t, x, y; };
struct Data { std::vector<Obs> obs; std::vector<double> Btrue; };

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double U(std::mt19937_64& g) {
    return std::generate_canonical<double, 53>(g);
}

static double Exp(std::mt19937_64& g, double rate) {
    return std::exponential_distribution<double>(rate)(g);
}

struct Grid {
    int n;
    double dx, dy, cell_area;
    std::vector<double> x, y;

    Grid(int side, const Par& p, uint64_t seed)
        : n(side), dx(p.Lx / side), dy(p.Ly / side),
          cell_area(p.area() / (side * side)), x(side * side), y(side * side) {
        std::mt19937_64 g(seed);
        for (int iy = 0; iy < n; ++iy) for (int ix = 0; ix < n; ++ix) {
            int q = iy * n + ix;
            x[q] = (ix + U(g)) * dx;
            y[q] = (iy + U(g)) * dy;
        }
    }
    int size() const { return n * n; }
};

struct Coverage {
    std::vector<uint16_t> a, p;
    int nD = 0;
    explicit Coverage(int m = 0) : a(m, 0), p(m, 0) {}
};

struct Particle {
    State s;
    Coverage c;
    explicit Particle(int m = 0) : c(m) {}
};

static bool in_disks(const std::vector<Disk>& v, double x, double y) {
    for (const auto& d : v) {
        double dx = x - d.x, dy = y - d.y;
        if (dx * dx + dy * dy <= d.r * d.r) return true;
    }
    return false;
}

static bool in_D(const State& s, double x, double y) {
    return in_disks(s.active, x, y) && !in_disks(s.erk, x, y);
}

static bool in_T(double x, double y, const Par& p) {
    double x1 = p.Lx / 3.0, x2 = 2.0 * p.Lx / 3.0;
    double y1 = p.Ly / 3.0, y2 = 2.0 * p.Ly / 3.0;
    return (x1 <= x && x <= x2) || (x <= x1 && y1 <= y && y <= y2);
}

static double activation_intensity(const State& s, double x, double y, const Par& p) {
    if (in_disks(s.active, x, y)) return p.la1;
    return in_T(x, y, p) ? p.laT : p.lac;
}

static void apply_disk(Coverage& c, const Grid& z, const Disk& d,
                       bool active, int delta) {
    int ix0 = std::max(0, (int)std::floor((d.x - d.r) / z.dx));
    int ix1 = std::min(z.n - 1, (int)std::floor((d.x + d.r) / z.dx));
    int iy0 = std::max(0, (int)std::floor((d.y - d.r) / z.dy));
    int iy1 = std::min(z.n - 1, (int)std::floor((d.y + d.r) / z.dy));
    if (ix0 > ix1 || iy0 > iy1) return;
    for (int iy = iy0; iy <= iy1; ++iy) for (int ix = ix0; ix <= ix1; ++ix) {
        int q = iy * z.n + ix;
        double dx = z.x[q] - d.x, dy = z.y[q] - d.y;
        if (dx * dx + dy * dy > d.r * d.r) continue;
        bool before = c.a[q] > 0 && c.p[q] == 0;
        uint16_t& v = active ? c.a[q] : c.p[q];
        if (delta > 0) ++v;
        else { assert(v > 0); --v; }
        bool after = c.a[q] > 0 && c.p[q] == 0;
        c.nD += int(after) - int(before);
    }
}

static void add_active(Particle& q, const Disk& d, const Grid& z) {
    q.s.active.push_back(d); apply_disk(q.c, z, d, true, +1);
}

static void add_erk(Particle& q, const Disk& d, const Grid& z) {
    q.s.erk.push_back(d); apply_disk(q.c, z, d, false, +1);
}

static void remove_active(Particle& q, size_t i, const Grid& z) {
    apply_disk(q.c, z, q.s.active[i], true, -1);
    q.s.active[i] = q.s.active.back(); q.s.active.pop_back();
}

static void remove_erk(Particle& q, size_t i, const Grid& z) {
    apply_disk(q.c, z, q.s.erk[i], false, -1);
    q.s.erk[i] = q.s.erk.back(); q.s.erk.pop_back();
}

static size_t random_index(std::mt19937_64& g, size_t n) {
    return std::uniform_int_distribution<size_t>(0, n - 1)(g);
}

static Data simulate_data(uint64_t seed, int deaths, const Par& p, const Grid& z) {
    std::mt19937_64 g(seed);
    Particle q(z.size());
    Data out;
    double t = 0.0, B = 0.0;

    while (t < p.Tmax && (int)out.obs.size() < deaths) {
        double ra = p.la1 * p.area();
        double rd = p.ld * p.area();
        double rea = static_cast<double>(q.s.active.size()) * p.baT;
        double rep = static_cast<double>(q.s.erk.size()) * p.bdT;
        double r0 = ra + rd + rea + rep;
        double dt = Exp(g, r0);
        if (t + dt > p.Tmax) break;
        B += q.c.nD * z.cell_area * dt;
        t += dt;
        double v = U(g) * r0;

        if (v < ra) {
            double x = U(g) * p.Lx, y = U(g) * p.Ly;
            if (U(g) <= activation_intensity(q.s, x, y, p) / p.la1)
                add_active(q, {x, y, Exp(g, p.baR)}, z);
        } else if (v < ra + rd) {
            double x = U(g) * p.Lx, y = U(g) * p.Ly;
            if (in_D(q.s, x, y)) {
                out.obs.push_back({t, x, y});
                out.Btrue.push_back(B);
                B = 0.0;
                add_erk(q, {x, y, Exp(g, p.bdR)}, z);
            }
        } else if (v < ra + rd + rea) {
            remove_active(q, random_index(g, q.s.active.size()), z);
        } else {
            remove_erk(q, random_index(g, q.s.erk.size()), z);
        }
    }
    return out;
}

static double simulate_hidden_segment(Particle& q, double dt, std::mt19937_64& g,
                                      const Par& p, const Grid& z) {
    double t = 0.0, B = 0.0;
    while (t < dt) {
        double ra = p.la1 * p.area();
        double rea = static_cast<double>(q.s.active.size()) * p.baT;
        double rep = static_cast<double>(q.s.erk.size()) * p.bdT;
        double r0 = ra + rea + rep;
        double wait = Exp(g, r0);
        if (t + wait >= dt) {
            B += q.c.nD * z.cell_area * (dt - t);
            break;
        }
        B += q.c.nD * z.cell_area * wait;
        t += wait;
        double v = U(g) * r0;
        if (v < ra) {
            double x = U(g) * p.Lx, y = U(g) * p.Ly;
            if (U(g) <= activation_intensity(q.s, x, y, p) / p.la1)
                add_active(q, {x, y, Exp(g, p.baR)}, z);
        } else if (v < ra + rea) {
            remove_active(q, random_index(g, q.s.active.size()), z);
        } else {
            remove_erk(q, random_index(g, q.s.erk.size()), z);
        }
    }
    return B;
}

enum class Weight { Local, Full };
struct FilterResult {
    bool ok = false;
    std::vector<double> meanB;
};

static FilterResult filter(const Data& d, int N, Weight type, uint64_t seed,
                           const Par& p, const Grid& z, double rho = 0.75) {
    std::vector<Particle> x;
    x.reserve(N);
    for (int i = 0; i < N; ++i) x.emplace_back(z.size());
    std::vector<double> w(N, 1.0 / N), nw(N), B(N);
    double previous_time = 0.0;
    std::vector<double> answer;
    answer.reserve(d.obs.size());

    for (size_t k = 0; k < d.obs.size(); ++k) {
        double dt = d.obs[k].t - previous_time;
        previous_time = d.obs[k].t;
        for (int i = 0; i < N; ++i) {
            std::mt19937_64 g(mix64(seed ^ (uint64_t(k + 1) << 48) ^ uint64_t(i + 1)));
            B[i] = simulate_hidden_segment(x[i], dt, g, p, z);
            bool hit = in_D(x[i].s, d.obs[k].x, d.obs[k].y);
            double G = 0.0;
            if (hit) {
                if (type == Weight::Local) {
                    double areaD = x[i].c.nD * z.cell_area;
                    if (areaD > 0.0) G = 1.0 / areaD;
                } else {
                    G = p.ld * std::exp(-p.ld * B[i]);
                }
            }
            nw[i] = w[i] * G;
        }

        double sum = std::accumulate(nw.begin(), nw.end(), 0.0);
        if (!(sum > 0.0) || !std::isfinite(sum)) return {};
        for (double& v : nw) v /= sum;
        answer.push_back(
            std::inner_product(nw.begin(), nw.end(), B.begin(), 0.0));

        if (k + 1 < d.obs.size()) {
            for (int i = 0; i < N; ++i) {
                std::mt19937_64 g(mix64(seed ^ 0xd1b54a32d192ed03ULL ^
                                       (uint64_t(k + 1) << 48) ^ uint64_t(i + 1)));
                add_erk(x[i], {d.obs[k].x, d.obs[k].y, Exp(g, p.bdR)}, z);
            }
            double s2 = std::inner_product(nw.begin(), nw.end(), nw.begin(), 0.0);
            if (1.0 / s2 <= rho * N) {
                std::mt19937_64 g(mix64(seed ^ 0x94d049bb133111ebULL ^ uint64_t(k + 1)));
                std::discrete_distribution<int> pick(nw.begin(), nw.end());
                std::vector<Particle> y;
                y.reserve(N);
                for (int i = 0; i < N; ++i) y.push_back(x[pick(g)]);
                x.swap(y);
                std::fill(w.begin(), w.end(), 1.0 / N);
            } else {
                w = nw;
            }
        }
    }
    return {true, answer};
}

// Empirical approximation of the optimal proposal in Algorithm 3.3:
// draw Mprop prior segments, select one proportionally to its observation
// likelihood G, and multiply the outer weight by their mean likelihood.
static FilterResult filter_empirical_optimal(const Data& d, int N, int Mprop,
                                             uint64_t seed, const Par& p,
                                             const Grid& z, double rho = 0.75) {
    std::vector<Particle> x;
    x.reserve(N);
    for (int i = 0; i < N; ++i) x.emplace_back(z.size());
    std::vector<double> w(N, 1.0 / N), nw(N), B(N);
    double previous_time = 0.0;
    std::vector<double> answer;
    answer.reserve(d.obs.size());

    for (size_t k = 0; k < d.obs.size(); ++k) {
        double dt = d.obs[k].t - previous_time;
        previous_time = d.obs[k].t;

        for (int i = 0; i < N; ++i) {
            Particle selected(z.size());
            double selected_B = 0.0, sum_G = 0.0;
            std::mt19937_64 select_rng(mix64(
                seed ^ 0xbb67ae8584caa73bULL ^
                (uint64_t(k + 1) << 48) ^ uint64_t(i + 1)));

            for (int j = 0; j < Mprop; ++j) {
                Particle candidate = x[i];
                std::mt19937_64 g(mix64(
                    seed ^ 0x3c6ef372fe94f82bULL ^
                    (uint64_t(k + 1) << 48) ^
                    (uint64_t(i + 1) << 16) ^ uint64_t(j + 1)));
                double candidate_B = simulate_hidden_segment(candidate, dt, g, p, z);
                bool hit = in_D(candidate.s, d.obs[k].x, d.obs[k].y);
                double G = hit ? p.ld * std::exp(-p.ld * candidate_B) : 0.0;

                double new_sum_G = sum_G + G;
                if (j == 0 || (G > 0.0 && U(select_rng) * new_sum_G < G)) {
                    selected = std::move(candidate);
                    selected_B = candidate_B;
                }
                sum_G = new_sum_G;
            }

            x[i] = std::move(selected);
            B[i] = selected_B;
            nw[i] = w[i] * (sum_G / static_cast<double>(Mprop));
        }

        double sum = std::accumulate(nw.begin(), nw.end(), 0.0);
        if (!(sum > 0.0) || !std::isfinite(sum)) return {};
        for (double& v : nw) v /= sum;
        answer.push_back(
            std::inner_product(nw.begin(), nw.end(), B.begin(), 0.0));

        if (k + 1 < d.obs.size()) {
            for (int i = 0; i < N; ++i) {
                std::mt19937_64 g(mix64(seed ^ 0xa54ff53a5f1d36f1ULL ^
                                       (uint64_t(k + 1) << 48) ^ uint64_t(i + 1)));
                add_erk(x[i], {d.obs[k].x, d.obs[k].y, Exp(g, p.bdR)}, z);
            }
            double s2 = std::inner_product(nw.begin(), nw.end(), nw.begin(), 0.0);
            if (1.0 / s2 <= rho * N) {
                std::mt19937_64 g(mix64(seed ^ 0x510e527fade682d1ULL ^ uint64_t(k + 1)));
                std::discrete_distribution<int> pick(nw.begin(), nw.end());
                std::vector<Particle> y;
                y.reserve(N);
                for (int i = 0; i < N; ++i) y.push_back(x[pick(g)]);
                x.swap(y);
                std::fill(w.begin(), w.end(), 1.0 / N);
            } else {
                w = nw;
            }
        }
    }
    return {true, answer};
}

static double mean(const std::vector<double>& x) {
    return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

static double mcse(const std::vector<double>& x) {
    if (x.size() < 2) return 0.0;
    double m = mean(x), s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(x.size() - 1) /
                     static_cast<double>(x.size()));
}

int main(int argc, char** argv) {
    const int R = argc > 1 ? std::stoi(argv[1]) : 500;
    const int GRID_SIDE = argc > 2 ? std::stoi(argv[2]) : 40;
    const int M_PROP = argc > 3 ? std::stoi(argv[3]) : 20;
    if (R < 1 || GRID_SIDE < 1 || M_PROP < 1) {
        std::cerr << "Usage: " << argv[0]
                  << " [R>=1] [GRID_SIDE>=1] [M_PROP>=1]\n";
        return 2;
    }
    const int K = 20;
    const std::vector<int> Ns = {100, 250, 500, 1000, 2000, 4000};
    const Par p;
    const Grid grid(GRID_SIDE, p, 20260904ULL);

#ifdef _OPENMP
    omp_set_num_threads(std::min(8, omp_get_max_threads()));
#endif

    std::vector<Data> data(R);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int r = 0; r < R; ++r) {
        for (uint64_t attempt = 0;; ++attempt) {
            Data d = simulate_data(mix64(1234567ULL + uint64_t(r) + attempt * 1000003ULL),
                                   K, p, grid);
            if ((int)d.obs.size() == K) { data[r] = std::move(d); break; }
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "R=" << R << "  K=" << K << "  area_points=" << grid.size()
              << "  ESS_threshold=0.75  M_prop=" << M_PROP << '\n';
    std::cout << "N k algorithm success collapse mean_true_paired mean_PF bias abs_bias paired_MCSE\n";

    for (int N : Ns) {
        std::vector<FilterResult> a3(R), a4(R), a5(R);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int r = 0; r < R; ++r) {
            uint64_t base = mix64(987654321ULL ^ (uint64_t(N) << 32) ^ uint64_t(r + 1));
            a3[r] = filter(data[r], N, Weight::Local, base, p, grid);
            a4[r] = filter(data[r], N, Weight::Full,
                           base ^ 0x6a09e667f3bcc909ULL, p, grid);
            a5[r] = filter_empirical_optimal(
                data[r], N, M_PROP, base ^ 0x9b05688c2b3e6c1fULL, p, grid);
        }
        for (int alg = 3; alg <= 5; ++alg) {
            const auto& out = alg == 3 ? a3 : (alg == 4 ? a4 : a5);
            for (int k = 0; k < K; ++k) {
                std::vector<double> paired_truth, estimates, differences;
                for (int r = 0; r < R; ++r) if (out[r].ok) {
                    double trueB = data[r].Btrue[k];
                    double estimatedB = out[r].meanB[k];
                    paired_truth.push_back(trueB);
                    estimates.push_back(estimatedB);
                    differences.push_back(estimatedB - trueB);
                }
                int success = (int)estimates.size();
                double mt = success ? mean(paired_truth) : NAN;
                double mpf = success ? mean(estimates) : NAN;
                double bias = success ? mean(differences) : NAN;
                const char* name = alg == 3 ? "A3_local" :
                                   (alg == 4 ? "A4_full" : "A5_empirical_optimal");
                std::cout << N << ' ' << (k + 1) << ' '
                          << name << ' '
                          << success << ' ' << (R - success) << ' ' << mt << ' ' << mpf << ' '
                          << bias << ' ' << std::abs(bias) << ' ' << mcse(differences) << '\n';
            }
        }
    }
}
