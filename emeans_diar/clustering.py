import time
import numpy as np
from numba import njit


@njit(fastmath=True)
def _stage1_online_emeans_numba(X, cos_birth, hysteresis_len, alpha_table):
    n, d = X.shape
    max_cap = 64
    S_buf = np.zeros((max_cap, d), dtype=np.float64)
    N_buf = np.zeros(max_cap, dtype=np.float64)
    C_buf = np.zeros((max_cap, d), dtype=np.float64)

    N_buf[0] = 1e-10
    for j in range(d):
        S_buf[0, j] = 1e-10 * X[0, j]
    alpha_init = 0.25
    N_buf[0] = (1.0 - alpha_init) * N_buf[0] + alpha_init * 1.0
    norm0 = 0.0
    for j in range(d):
        S_buf[0, j] = (1.0 - alpha_init) * S_buf[0, j] + alpha_init * X[0, j]
        norm0 += S_buf[0, j] * S_buf[0, j]
    norm0 = np.sqrt(norm0) + 1e-12
    for j in range(d):
        C_buf[0, j] = S_buf[0, j] / norm0
    K_cur = 1

    pending_buf = np.zeros((hysteresis_len, d), dtype=np.float64)
    pending_count = 0

    for i in range(1, n):
        x = X[i]
        best_j = 0
        best_sim = -2.0
        for k in range(K_cur):
            sim = 0.0
            for j in range(d):
                sim += C_buf[k, j] * x[j]
            if sim > best_sim:
                best_sim = sim
                best_j = k

        if best_sim < cos_birth and K_cur < max_cap:
            if pending_count < hysteresis_len:
                for j in range(d):
                    pending_buf[pending_count, j] = x[j]
                pending_count += 1
            if pending_count >= hysteresis_len:
                p_mean = np.zeros(d, dtype=np.float64)
                for r in range(hysteresis_len):
                    for j in range(d):
                        p_mean[j] += pending_buf[r, j]
                norm_pm = 0.0
                for j in range(d):
                    p_mean[j] /= hysteresis_len
                    norm_pm += p_mean[j] * p_mean[j]
                norm_pm = np.sqrt(norm_pm) + 1e-12
                for j in range(d):
                    p_mean[j] /= norm_pm

                coherence = 0.0
                for r in range(hysteresis_len):
                    for j in range(d):
                        coherence += pending_buf[r, j] * p_mean[j]
                coherence /= hysteresis_len
                if hysteresis_len == 1:
                    coherence = 1.0

                max_c_sim = -2.0
                for k in range(K_cur):
                    s_c = 0.0
                    for j in range(d):
                        s_c += C_buf[k, j] * p_mean[j]
                    if s_c > max_c_sim:
                        max_c_sim = s_c

                if coherence > 0.65 and max_c_sim < cos_birth + 0.04:
                    N_buf[K_cur] = 1e-10 + float(hysteresis_len)
                    norm_s = 0.0
                    for j in range(d):
                        s_val = 1e-10 * p_mean[j]
                        for r in range(hysteresis_len):
                            s_val += pending_buf[r, j]
                        S_buf[K_cur, j] = s_val
                        norm_s += s_val * s_val
                    norm_s = np.sqrt(norm_s) + 1e-12
                    for j in range(d):
                        C_buf[K_cur, j] = S_buf[K_cur, j] / norm_s
                    K_cur += 1
                    pending_count = 0
                else:
                    for r in range(hysteresis_len - 1):
                        for j in range(d):
                            pending_buf[r, j] = pending_buf[r + 1, j]
                    pending_count = hysteresis_len - 1
        else:
            pending_count = 0
            alpha = alpha_table[K_cur]
            N_buf[best_j] = (1.0 - alpha) * N_buf[best_j] + alpha * 1.0
            norm_s = 0.0
            for j in range(d):
                S_buf[best_j, j] = (1.0 - alpha) * S_buf[best_j, j] + alpha * x[j]
                norm_s += S_buf[best_j, j] * S_buf[best_j, j]
            norm_s = np.sqrt(norm_s) + 1e-12
            for j in range(d):
                C_buf[best_j, j] = S_buf[best_j, j] / norm_s

    return S_buf[:K_cur], N_buf[:K_cur]


@njit(fastmath=True)
def _leader_follower_numba(X, cos_birth):
    n, d = X.shape
    max_cap = 64
    C_buf = np.zeros((max_cap, d), dtype=np.float64)
    counts = np.zeros(max_cap, dtype=np.float64)
    for j in range(d):
        C_buf[0, j] = X[0, j]
    counts[0] = 1.0
    K_cur = 1
    assigns = np.zeros(n, dtype=np.int64)

    for i in range(1, n):
        x = X[i]
        best_j = 0
        best_sim = -2.0
        for k in range(K_cur):
            sim = 0.0
            for j in range(d):
                sim += C_buf[k, j] * x[j]
            if sim > best_sim:
                best_sim = sim
                best_j = k

        if best_sim < cos_birth and K_cur < max_cap:
            for j in range(d):
                C_buf[K_cur, j] = x[j]
            counts[K_cur] = 1.0
            assigns[i] = K_cur
            K_cur += 1
        else:
            counts[best_j] += 1.0
            eta = 1.0 / counts[best_j]
            norm_c = 0.0
            for j in range(d):
                C_buf[best_j, j] = (1.0 - eta) * C_buf[best_j, j] + eta * x[j]
                norm_c += C_buf[best_j, j] * C_buf[best_j, j]
            norm_c = np.sqrt(norm_c) + 1e-12
            for j in range(d):
                C_buf[best_j, j] /= norm_c
            assigns[i] = best_j

    return C_buf[:K_cur], assigns



@njit(fastmath=True)
def _accum_batch_stats_numba(Xb, assign, k):
    b, d = Xb.shape
    n_j = np.zeros(k, dtype=np.float64)
    s_j = np.zeros((k, d), dtype=np.float64)
    for i in range(b):
        c = assign[i]
        n_j[c] += 1.0
        for j in range(d):
            s_j[c, j] += Xb[i, j]
    return n_j, s_j

# Warmup JIT compilation at import time
_dummy_X = np.ones((8, 4), dtype=np.float64) * 0.5
_dummy_alpha = np.full(65, 0.05, dtype=np.float64)
_ = _stage1_online_emeans_numba(_dummy_X, 0.45, 2, _dummy_alpha)
_ = _leader_follower_numba(_dummy_X, 0.48)
_ = _accum_batch_stats_numba(_dummy_X, np.zeros(8, dtype=np.int64), 2)


def compute_clustering_mse(X: np.ndarray, C: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) as defined in Eq. (1) of arXiv:2611.09501v1."""
    if len(X) == 0 or len(C) == 0:
        return 0.0
    x_norm2 = np.sum(X * X, axis=1, keepdims=True)
    c_norm2 = np.sum(C * C, axis=1, keepdims=True).T
    dists = np.maximum(0.0, x_norm2 + c_norm2 - 2.0 * (X @ C.T))
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def emeans_scaling_law(k: int, n: int, T: float, B_override: int | None = None):
    """
    Formulaic scaling laws from Section 3.3 & Table 2 of Bessa et al. (2026) [arXiv:2611.09501v1]:
      Batch law (Eq. 6): B = 0.035 * k^0.13 * n^0.88 * T^0.81 (rounded to power of 2)
      Step law  (Eq. 5): alpha_0 = min(0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11, 1.0)
    """
    k_eff = max(1, int(k))
    n_eff = max(32, int(n))
    T_eff = max(0.5, float(T))
    if B_override is not None:
        B = int(B_override)
    else:
        b_raw = 0.035 * (k_eff ** 0.13) * (n_eff ** 0.88) * (T_eff ** 0.81)
        p2 = max(5, int(np.round(np.log2(max(16.0, b_raw)))))
        B = int(min(max(32, 2 ** p2), max(32, n_eff // 4)))
    alpha_0 = min(
        0.824 * (k_eff ** -0.46) * (B ** 0.90) * (n_eff ** -0.45) * (T_eff ** -0.11),
        0.85,
    )
    return B, float(max(0.05, alpha_0))


class ElasticEMeansDiarizer:
    """
    Elastic Open-Speaker e-means Diarizer (Stream-eDiar-LLM).
    Adapts e-means (Bessa et al., 2026, arXiv:2611.09501v1) to streaming open-speaker diarization
    at 80ms or 160ms frame resolution.
    """

    def __init__(
        self,
        tau_vad: float = 0.35,
        cos_birth: float = 0.45,
        cos_merge: float = 0.58,
        T_passes: float = 2.0,
        hysteresis_len: int = 4,
        use_cosine_schedule: bool = True,
        use_scaling_law: bool = True,
        use_donor_split_revival: bool = True,
        finalizer: str = "slow_ema",
        use_marginal_vad: bool = True,
        optimizer_type: str = "ema",
        overlap_threshold: float = 1.15,
        seed: int = 42,
    ):
        self.tau_vad = tau_vad
        self.cos_birth = cos_birth
        self.cos_merge = cos_merge
        self.T_passes = T_passes
        self.hysteresis_len = hysteresis_len
        self.use_cosine_schedule = use_cosine_schedule
        self.use_scaling_law = use_scaling_law
        self.use_donor_split_revival = use_donor_split_revival
        self.finalizer = finalizer
        self.use_marginal_vad = use_marginal_vad
        self.optimizer_type = optimizer_type
        self.overlap_threshold = overlap_threshold
        self.seed = seed

    def fit_predict(self, spk_embs: np.ndarray, probs_s: np.ndarray):
        t0 = time.perf_counter()
        sum_p = probs_s.sum(axis=1)
        max_p = probs_s.max(axis=1)
        active_ch = int(np.sum(np.sum(probs_s > 0.50, axis=0) > 20))
        if self.use_marginal_vad:
            if active_ch >= 7:
                vad = (sum_p > self.tau_vad) | (max_p > 0.48)
            else:
                vad = (max_p > 0.48) | (sum_p > 0.54)
        else:
            vad = max_p > 0.50

        active_idx = np.where(vad)[0]
        if len(active_idx) == 0:
            return [[] for _ in range(len(vad))], 0, 0.0, time.perf_counter() - t0

        X = np.ascontiguousarray(spk_embs[active_idx], dtype=np.float64)
        n, d = X.shape

        max_cap = 64
        online_batch_ref = 64
        alpha_table = np.zeros(max_cap + 1, dtype=np.float64)
        for k_idx in range(1, max_cap + 1):
            if self.use_scaling_law:
                _, a0 = emeans_scaling_law(k_idx, max(n, 64), max(1.0, self.T_passes), online_batch_ref)
            else:
                a0 = 0.25
            alpha_table[k_idx] = a0 / max(1.0, online_batch_ref / max(1, k_idx))

        S, N = _stage1_online_emeans_numba(X, float(self.cos_birth), int(self.hysteresis_len), alpha_table)
        K = len(N)

        if self.use_scaling_law:
            B, alpha_0 = emeans_scaling_law(K, n, max(0.5, self.T_passes))
        else:
            B = 64
            alpha_0 = 0.20

        S_slow = S.copy()
        N_slow = N.copy()
        C_direct = S / N[:, None]
        C_direct /= (np.linalg.norm(C_direct, axis=1, keepdims=True) + 1e-12)

        m_adam = np.zeros_like(C_direct)
        v_adam = np.zeros_like(C_direct)

        delta = 1e-2
        pert_pos = np.array([1.0 + delta if i % 2 == 0 else 1.0 - delta for i in range(d)])
        pert_neg = np.array([1.0 - delta if i % 2 == 0 else 1.0 + delta for i in range(d)])

        rng = np.random.default_rng(self.seed)
        num_batches_per_epoch = max(1, int(np.ceil(n / B)))
        M = max(1, int(np.round(self.T_passes * num_batches_per_epoch)))
        eps = 0.10

        step = 0
        full_epochs = int(np.ceil(self.T_passes))

        for ep in range(full_epochs):
            perm = rng.permutation(n)
            for b_start in range(0, n, B):
                if step >= M:
                    break
                b_idx = perm[b_start:b_start + B]
                Xb = X[b_idx]

                if self.use_cosine_schedule:
                    alpha_t = alpha_0 * (eps + (1.0 - eps) * 0.5 * (1.0 + np.cos(np.pi * min(step, M - 1) / max(1, M))))
                else:
                    alpha_t = 0.5 * alpha_0
                gamma_t = 1.0 - alpha_t
                step += 1

                if self.optimizer_type == "ema":
                    C = S / N[:, None]
                    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                else:
                    C_norm = C_direct / (np.linalg.norm(C_direct, axis=1, keepdims=True) + 1e-12)

                sims_b = Xb @ C_norm.T
                assign = np.argmax(sims_b, axis=1)

                n_j, s_j = _accum_batch_stats_numba(Xb, assign, len(N))

                if self.optimizer_type == "ema":
                    N = gamma_t * N + (1.0 - gamma_t) * n_j
                    S = gamma_t * S + (1.0 - gamma_t) * s_j

                    alpha_slow = 0.25 * eps * alpha_0
                    N_slow = (1.0 - alpha_slow) * N_slow + alpha_slow * n_j
                    S_slow = (1.0 - alpha_slow) * S_slow + alpha_slow * s_j
                elif self.optimizer_type == "sgd":
                    lr = 0.5 * alpha_t
                    for j_idx in range(len(N)):
                        if n_j[j_idx] > 0:
                            grad = (n_j[j_idx] * C_direct[j_idx] - s_j[j_idx]) / max(1.0, float(B) / len(N))
                            C_direct[j_idx] = C_direct[j_idx] - lr * grad
                    N = gamma_t * N + (1.0 - gamma_t) * n_j
                    S = C_direct * N[:, None]
                    S_slow, N_slow = S.copy(), N.copy()
                elif self.optimizer_type == "adam":
                    lr = 0.05 * alpha_t
                    for j_idx in range(len(N)):
                        if n_j[j_idx] > 0:
                            grad = (n_j[j_idx] * C_direct[j_idx] - s_j[j_idx]) / max(1.0, float(B) / len(N))
                            m_adam[j_idx] = 0.9 * m_adam[j_idx] + 0.1 * grad
                            v_adam[j_idx] = 0.999 * v_adam[j_idx] + 0.001 * (grad * grad)
                            m_hat = m_adam[j_idx] / (1.0 - 0.9 ** step)
                            v_hat = v_adam[j_idx] / (1.0 - 0.999 ** step)
                            C_direct[j_idx] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    N = gamma_t * N + (1.0 - gamma_t) * n_j
                    S = C_direct * N[:, None]
                    S_slow, N_slow = S.copy(), N.copy()

                if self.use_donor_split_revival and step % 5 == 0 and len(N) > 1 and self.optimizer_type == "ema":
                    k_now = len(N)
                    tau_k = 0.08 / (1.0 + 0.25 * max(0, k_now - 5))
                    dead_mask = N < tau_k * N.mean()
                    if dead_mask.any():
                        C_cur = S / N[:, None]
                        C_cur_n = C_cur / (np.linalg.norm(C_cur, axis=1, keepdims=True) + 1e-12)
                        for j_dead in np.where(dead_mask)[0]:
                            live_idxs = np.where(~dead_mask)[0]
                            if len(live_idxs) == 0:
                                break
                            donor_scores = []
                            for j_live in live_idxs:
                                pts = Xb[assign == j_live]
                                if len(pts) >= 8 and N[j_live] > 1.3 * N.mean():
                                    mean_cos = float(np.mean(pts @ C_cur_n[j_live]))
                                    if mean_cos < 0.62:
                                        donor_scores.append((N[j_live] * (1.0 - mean_cos), j_live))
                            if donor_scores:
                                donor_scores.sort(reverse=True)
                                j_donor = donor_scores[0][1]
                                c_d = S[j_donor] / N[j_donor]
                                N[j_donor] *= 0.5
                                N[j_dead] = N[j_donor]
                                S[j_donor] = (c_d * pert_pos) * N[j_donor]
                                S[j_dead] = (c_d * pert_neg) * N[j_dead]
                                N_slow[j_donor] *= 0.5
                                N_slow[j_dead] = N_slow[j_donor]
                                S_slow[j_donor] = (c_d * pert_pos) * N_slow[j_donor]
                                S_slow[j_dead] = (c_d * pert_neg) * N_slow[j_dead]
                                dead_mask[j_dead] = False
                        keep = ~dead_mask
                        if keep.sum() >= 1 and keep.sum() < len(N):
                            S, N = S[keep], N[keep]
                            S_slow, N_slow = S_slow[keep], N_slow[keep]

                    C_cur = S / N[:, None]
                    C_cur /= (np.linalg.norm(C_cur, axis=1, keepdims=True) + 1e-12)
                    sim_mat = C_cur @ C_cur.T
                    np.fill_diagonal(sim_mat, -1.0)
                    while len(N) > 1 and sim_mat.max() > self.cos_merge:
                        idx_pair = np.unravel_index(np.argmax(sim_mat), sim_mat.shape)
                        i_m, j_m = min(idx_pair), max(idx_pair)
                        N[i_m] += N[j_m]
                        S[i_m] += S[j_m]
                        N_slow[i_m] += N_slow[j_m]
                        S_slow[i_m] += S_slow[j_m]
                        mask_m = np.ones(len(N), dtype=bool)
                        mask_m[j_m] = False
                        S, N = S[mask_m], N[mask_m]
                        S_slow, N_slow = S_slow[mask_m], N_slow[mask_m]
                        C_cur = S / N[:, None]
                        C_cur /= (np.linalg.norm(C_cur, axis=1, keepdims=True) + 1e-12)
                        sim_mat = C_cur @ C_cur.T
                        np.fill_diagonal(sim_mat, -1.0)

        if self.finalizer == "slow_ema":
            C_final = S_slow / N_slow[:, None]
        elif self.finalizer == "last_epoch_mean":
            C_tmp = S / N[:, None]
            C_tmp /= (np.linalg.norm(C_tmp, axis=1, keepdims=True) + 1e-12)
            a_tmp = np.argmax(X @ C_tmp.T, axis=1)
            C_final = np.zeros_like(S)
            for j_idx in range(len(N)):
                pts = X[a_tmp == j_idx]
                C_final[j_idx] = pts.mean(axis=0) if len(pts) > 0 else C_tmp[j_idx]
        else:
            C_final = S / N[:, None]

        C_final /= (np.linalg.norm(C_final, axis=1, keepdims=True) + 1e-12)
        sims_all = X @ C_final.T
        final_assign = np.argmax(sims_all, axis=1)

        if self.use_donor_split_revival:
            counts = np.bincount(final_assign, minlength=len(C_final))
            valid_c = counts >= 8
            if valid_c.sum() > 0 and valid_c.sum() < len(C_final):
                C_final = C_final[valid_c]
                sims_all = X @ C_final.T
                final_assign = np.argmax(sims_all, axis=1)

        mse = compute_clustering_mse(X, C_final)

        frame_spks = [[] for _ in range(len(vad))]
        for idx_pos, t_frame in enumerate(active_idx):
            primary = int(final_assign[idx_pos])
            frame_spks[t_frame].append(primary)
            if sum_p[t_frame] > self.overlap_threshold and C_final.shape[0] >= 2:
                top2 = np.argsort(sims_all[idx_pos])[-2:]
                second = int(top2[0])
                if sims_all[idx_pos, second] > 0.48:
                    frame_spks[t_frame].append(second)

        runtime = time.perf_counter() - t0
        return frame_spks, int(C_final.shape[0]), mse, runtime


class OnlineLeaderFollowerDiarizer:
    """
    Canonical Online Sequential Leader-Follower (MacQueen, 1967) Diarizer Baseline.
    Uses sequential running mean (1 / count) and threshold-based cluster creation without e-means enhancements.
    """

    def __init__(self, tau_vad: float = 0.35, cos_birth: float = 0.48):
        self.tau_vad = tau_vad
        self.cos_birth = cos_birth

    def fit_predict(self, spk_embs: np.ndarray, probs_s: np.ndarray):
        t0 = time.perf_counter()
        vad = probs_s.sum(axis=1) > self.tau_vad
        active_idx = np.where(vad)[0]
        if len(active_idx) == 0:
            return [[] for _ in range(len(vad))], 0, 0.0, time.perf_counter() - t0

        X = np.ascontiguousarray(spk_embs[active_idx], dtype=np.float64)
        C_final, assigns = _leader_follower_numba(X, float(self.cos_birth))
        mse = compute_clustering_mse(X, C_final)

        frame_spks = [[] for _ in range(len(vad))]
        for idx_pos, t_frame in enumerate(active_idx):
            frame_spks[t_frame].append(int(assigns[idx_pos]))
        runtime = time.perf_counter() - t0
        return frame_spks, len(C_final), mse, runtime


class SculleyMiniBatchKMeansDiarizer:
    """
    Sculley (2010) Web-Scale Mini-Batch k-Means Baseline (compared in Section 4.2 of arXiv:2611.09501v1).
    Uses harmonic running-mean step 1 / v[j] per mini-batch without EMA ratio or cosine schedule.
    """

    def __init__(self, tau_vad: float = 0.35, cos_birth: float = 0.45, T_passes: int = 2, batch_size: int = 64, seed: int = 42):
        self.tau_vad = tau_vad
        self.cos_birth = cos_birth
        self.T_passes = T_passes
        self.batch_size = batch_size
        self.seed = seed

    def fit_predict(self, spk_embs: np.ndarray, probs_s: np.ndarray):
        t0 = time.perf_counter()
        vad = probs_s.sum(axis=1) > self.tau_vad
        active_idx = np.where(vad)[0]
        if len(active_idx) == 0:
            return [[] for _ in range(len(vad))], 0, 0.0, time.perf_counter() - t0

        X = spk_embs[active_idx].copy()
        n, d = X.shape

        centroids = [X[0].copy()]
        for i in range(1, n, 4):
            C = np.stack(centroids)
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            if np.max(C @ X[i]) < self.cos_birth and len(centroids) < 24:
                centroids.append(X[i].copy())
        C = np.stack(centroids)
        v_counts = np.ones(len(C), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        for _ in range(self.T_passes):
            perm = rng.permutation(n)
            for b_start in range(0, n, self.batch_size):
                Xb = X[perm[b_start:b_start + self.batch_size]]
                C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                assign = np.argmax(Xb @ C_norm.T, axis=1)
                for x_pt, c_idx in zip(Xb, assign):
                    v_counts[c_idx] += 1.0
                    eta = 1.0 / v_counts[c_idx]
                    C[c_idx] = (1.0 - eta) * C[c_idx] + eta * x_pt

        C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        final_assign = np.argmax(X @ C_norm.T, axis=1)
        mse = compute_clustering_mse(X, C_norm)

        frame_spks = [[] for _ in range(len(vad))]
        for idx_pos, t_frame in enumerate(active_idx):
            frame_spks[t_frame].append(int(final_assign[idx_pos]))
        runtime = time.perf_counter() - t0
        return frame_spks, int(len(np.unique(final_assign))), mse, runtime


class OfflineLloydKMeansDiarizer:
    """
    Full-batch Lloyd k-Means++ @ 25 epochs baseline (Section 4 of arXiv:2611.09501v1).
    Estimates K via cosine eigengap on subsampled affinity matrix and runs 25 full-batch iterations.
    """

    def __init__(self, tau_vad: float = 0.35, epochs: int = 25, seed: int = 42, oracle_k: int | None = None):
        self.tau_vad = tau_vad
        self.epochs = epochs
        self.seed = seed
        self.oracle_k = oracle_k

    def _estimate_k(self, X: np.ndarray) -> int:
        if self.oracle_k is not None:
            return max(1, int(self.oracle_k))
        sub = X[::max(1, len(X) // 400)]
        if len(sub) < 8:
            return 1
        S_mat = np.maximum(0.0, sub @ sub.T)
        np.fill_diagonal(S_mat, 0.0)
        deg = np.sum(S_mat, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        L_norm = np.eye(len(sub)) - (d_inv_sqrt[:, None] * S_mat * d_inv_sqrt[None, :])
        eigvals = np.linalg.eigvalsh(L_norm)
        max_k = min(20, len(eigvals) - 1)
        if float(np.mean(sub @ sub.T)) > 0.72:
            return 1
        gaps = np.diff(eigvals[:max_k + 1])
        k_est = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 1
        return max(1, min(k_est, 20))

    def fit_predict(self, spk_embs: np.ndarray, probs_s: np.ndarray):
        t0 = time.perf_counter()
        vad = probs_s.sum(axis=1) > self.tau_vad
        active_idx = np.where(vad)[0]
        if len(active_idx) == 0:
            return [[] for _ in range(len(vad))], 0, 0.0, time.perf_counter() - t0

        X = spk_embs[active_idx].copy()
        n, d = X.shape
        k = self._estimate_k(X)
        rng = np.random.default_rng(self.seed)

        C = np.empty((k, d), dtype=X.dtype)
        C[0] = X[rng.integers(0, n)]
        for c_id in range(1, k):
            C_prev = C[:c_id] / (np.linalg.norm(C[:c_id], axis=1, keepdims=True) + 1e-12)
            dists = np.maximum(1e-8, 2.0 - 2.0 * np.max(X @ C_prev.T, axis=1))
            probs = dists / np.sum(dists)
            C[c_id] = X[rng.choice(n, p=probs)]

        for _ in range(self.epochs):
            C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            assign = np.argmax(X @ C_norm.T, axis=1)
            for j in range(k):
                pts = X[assign == j]
                if len(pts) > 0:
                    C[j] = pts.mean(axis=0)

        C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        final_assign = np.argmax(X @ C_norm.T, axis=1)
        mse = compute_clustering_mse(X, C_norm)

        frame_spks = [[] for _ in range(len(vad))]
        for idx_pos, t_frame in enumerate(active_idx):
            frame_spks[t_frame].append(int(final_assign[idx_pos]))
        runtime = time.perf_counter() - t0
        return frame_spks, int(len(np.unique(final_assign))), mse, runtime


class SlotConditionedEMeansDiarizer:
    """
    Slot-conditioned elastic e-means (Stream-eDiar-LLM v2).

    The Nemotron fixed-slot head is reliable while #speakers <= #slots, but its slots become impure when
    more speakers talk than slots exist (a slot hosts several voices, or one voice hops between slots).
    This diarizer therefore:
      1. keeps the head's frame-level activity / overlap decisions (thresholded slot posteriors),
      2. runs e-means (hysteresis births + EMA mini-batch refinement, Bessa et al. 2026) on the raw speaker
         embeddings of each slot's *solo* frames, so a slot is only split when its embeddings are
         genuinely multi-modal,
      3. merges clusters across slots only if they are embedding-similar AND rarely co-active
         (cannot-link from simultaneous slot activity), and
      4. labels every active slot at every frame with its best sub-cluster (temporally smoothed).
    Operates on raw (un-fused) embeddings: `raw_inputs = True`.
    """

    raw_inputs = True

    def __init__(
        self,
        act_thr: float = 0.55,
        vad_thr: float = 0.4,
        hysteresis_sec: float = 0.5,
        cos_merge_same: float = 0.62,
        cos_merge_cross: float = 0.62,
        cooc_max: float = 0.12,
        min_slot_sec: float = 2.0,
        min_spk_sec: float = 0.3,
        sat_slots: int = 8,
        rescue_cos: float = 0.55,
        sat_vad_thr: float = 0.15,
        sat_cos_birth: float = 0.3,
        sat_min_sub_frac: float = 0.03,
        sat_min_sub_sec: float = 2.0,
        sat_rescue_cos: float = 0.5,
        min_sub_sec: float = 2.0,
        min_sub_frac: float = 0.1,
        T_passes: float = 2.0,
        smooth_sec: float = 0.4,
        stride_ms: int | None = None,
        seed: int = 42,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ---- e-means core on one slot's solo-frame embeddings -------------------------------------------------
    def _emeans(self, X: np.ndarray, hyst: int, cos_birth: float, min_sub_frac: float, min_sub_sec: float):
        n, d = X.shape
        alpha_table = np.zeros(65, dtype=np.float64)
        for k_idx in range(1, 65):
            _, a0 = emeans_scaling_law(k_idx, max(n, 64), max(1.0, self.T_passes), 64)
            alpha_table[k_idx] = a0 / max(1.0, 64 / max(1, k_idx))
        S, N = _stage1_online_emeans_numba(X, float(cos_birth), int(hyst), alpha_table)
        K = len(N)
        B, alpha_0 = emeans_scaling_law(K, n, max(0.5, self.T_passes))
        rng = np.random.default_rng(self.seed)
        M = max(1, int(round(self.T_passes * int(np.ceil(n / B)))))
        eps, step = 0.10, 0
        S_slow, N_slow = S.copy(), N.copy()
        for _ in range(int(np.ceil(self.T_passes))):
            perm = rng.permutation(n)
            for b0 in range(0, n, B):
                if step >= M:
                    break
                Xb = X[perm[b0:b0 + B]]
                alpha_t = alpha_0 * (eps + (1 - eps) * 0.5 * (1 + np.cos(np.pi * min(step, M - 1) / max(1, M))))
                gamma = 1.0 - alpha_t
                step += 1
                C = S / N[:, None]
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
                assign = np.argmax(Xb @ C.T, axis=1)
                n_j, s_j = _accum_batch_stats_numba(Xb, assign, len(N))
                N = gamma * N + (1 - gamma) * n_j
                S = gamma * S + (1 - gamma) * s_j
                a_slow = 0.25 * eps * alpha_0
                N_slow = (1 - a_slow) * N_slow + a_slow * n_j
                S_slow = (1 - a_slow) * S_slow + a_slow * s_j
        C = S_slow / N_slow[:, None]
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        # prune sub-clusters that are too small in absolute / relative mass (spurious births)
        while len(C) > 1:
            cnt = np.bincount(np.argmax(X @ C.T, axis=1), minlength=len(C))
            thr = max(min_sub_sec * self._fps, min_sub_frac * n)
            small = np.where(cnt < thr)[0]
            if len(small) == 0:
                break
            C = np.delete(C, small[np.argmin(cnt[small])], axis=0)
        # merge near-duplicate sub-clusters inside the slot
        while len(C) > 1:
            sm = C @ C.T
            np.fill_diagonal(sm, -1)
            i, j = np.unravel_index(np.argmax(sm), sm.shape)
            if sm[i, j] <= self.cos_merge_same:
                break
            C[i] = C[i] + C[j]
            C[i] /= np.linalg.norm(C[i]) + 1e-12
            C = np.delete(C, j, axis=0)
        return C

    def fit_predict(self, embs: np.ndarray, probs_s: np.ndarray):
        t0 = time.perf_counter()
        T = len(probs_s)
        self.last_cents, self.last_tags, self.last_saturated = np.zeros((0, embs.shape[1])), np.zeros(0, dtype=int), False
        stride_ms = self.stride_ms or 80
        fps = 1000.0 / stride_ms
        self._fps = fps
        E = np.ascontiguousarray(embs, dtype=np.float64)
        P = probs_s.astype(np.float64)
        n_slots = P.shape[1]

        act = P > self.act_thr
        min_slot = int(self.min_slot_sec * fps)
        hyst = max(2, int(round(self.hysteresis_sec * fps)))
        n_used = int(np.sum(act.sum(axis=0) >= max(min_slot, hyst + 2)))
        saturated = (n_used >= self.sat_slots) or getattr(self, 'force_saturated', False)
        self.last_saturated = bool(n_used >= self.sat_slots)
        vad_thr = self.sat_vad_thr if saturated else self.vad_thr
        rescue_cos = self.sat_rescue_cos if saturated else self.rescue_cos
        cos_birth = self.sat_cos_birth if saturated else -1.0
        min_sub_frac = self.sat_min_sub_frac if saturated else self.min_sub_frac
        min_sub_sec = self.sat_min_sub_sec if saturated else self.min_sub_sec
        none_act = ~act.any(axis=1) & (P.max(axis=1) > vad_thr)
        act[none_act, P[none_act].argmax(axis=1)] = True
        rescued = none_act.copy()
        active_t = act.any(axis=1)
        if not active_t.any():
            return [[] for _ in range(T)], 0, 0.0, time.perf_counter() - t0

        solo = act & (act.sum(axis=1, keepdims=True) == 1)

        # ---- 1. per-slot e-means ------------------------------------------------------------------------
        # capacity-aware births: a fixed-slot head puts a new speaker into a free slot first, so a slot can
        # only host several voices when (almost) all slots are in use (`saturated`, decided above).
        cents, tags = [], []
        for s in range(n_slots):
            idx = np.where(solo[:, s])[0]
            if len(idx) < max(min_slot, hyst + 2):
                # brief slot: single centroid from all its active frames (kept distinct unless cross-merge
                # finds a similar, never-co-active speaker); negligible slots fall through to nearest-centroid
                all_idx = np.where(act[:, s])[0]
                if len(all_idx) >= self.min_spk_sec * fps:
                    m = E[all_idx].mean(axis=0)
                    cents.append(m / (np.linalg.norm(m) + 1e-12))
                    tags.append(s)
                continue
            Cs = self._emeans(np.ascontiguousarray(E[idx]), hyst, cos_birth, min_sub_frac, min_sub_sec)
            for c in Cs:
                cents.append(c)
                tags.append(s)
        if not cents:  # no slot has enough solo evidence: single cluster
            s_best = int(np.argmax(act.sum(axis=0)))
            c = E[act[:, s_best]].mean(axis=0)
            cents, tags = [c / (np.linalg.norm(c) + 1e-12)], [s_best]
        cents = np.array(cents)
        tags = np.array(tags)

        self.last_cents, self.last_tags = cents.copy(), tags.copy()           # model of this fit (used by the live system to anchor permanent ids to slots)
        # ---- 2. cross-slot merge with cannot-link from co-activity ---------------------------------------
        slot_n = act.sum(axis=0).astype(np.float64)
        cooc = (act.T.astype(np.float64) @ act.astype(np.float64))
        denom = np.minimum.outer(slot_n, slot_n) + 1e-9
        cooc_n = cooc / denom
        label = np.arange(len(cents))
        while True:
            groups = {}
            for ci, l in enumerate(label):
                groups.setdefault(l, []).append(ci)
            gl = list(groups)
            if len(gl) < 2:
                break
            gc = {l: (lambda m: m / (np.linalg.norm(m) + 1e-12))(cents[groups[l]].mean(axis=0)) for l in gl}
            best, pair = -1.0, None
            for a in range(len(gl)):
                for b in range(a + 1, len(gl)):
                    la, lb = gl[a], gl[b]
                    sa = {tags[c] for c in groups[la]}
                    sb = {tags[c] for c in groups[lb]}
                    if any(cooc_n[x, y] > self.cooc_max for x in sa for y in sb if x != y):
                        continue
                    if sa & sb:  # same slot already decided distinct by e-means
                        continue
                    sim = float(gc[la] @ gc[lb])
                    if sim > best:
                        best, pair = sim, (la, lb)
            if pair is None or best <= self.cos_merge_cross:
                break
            label[label == pair[1]] = pair[0]

        # ---- 3. per-frame labelling -----------------------------------------------------------------------
        sm = max(1, int(round(self.smooth_sec * fps)))
        frame_lab = [[] for _ in range(T)]
        valid_slots = sorted(set(tags.tolist()))
        for s in range(n_slots):
            t_idx = np.where(act[:, s])[0]
            if len(t_idx) == 0:
                continue
            ci_s = np.where(tags == s)[0] if s in valid_slots else np.arange(len(cents))
            sims = E[t_idx] @ cents[ci_s].T
            if len(ci_s) > 1 and sm > 1:
                ker = np.ones(sm) / sm
                sims = np.stack([np.convolve(sims[:, j], ker, mode="same") for j in range(sims.shape[1])], axis=1)
            if s not in valid_slots:  # weak slot: use its mean embedding against every centroid
                m = E[t_idx].mean(axis=0)
                m /= np.linalg.norm(m) + 1e-12
                best = np.full(len(t_idx), int(np.argmax(cents @ m)))
            else:
                best = ci_s[np.argmax(sims, axis=1)]
            if rescue_cos > 0 and s in valid_slots:
                top = sims.max(axis=1) if sims.shape[1] == 1 else (E[t_idx] @ cents[ci_s].T).max(axis=1)
                ok = ~rescued[t_idx] | (top >= rescue_cos)
            else:
                ok = np.ones(len(t_idx), dtype=bool)
            for t, c, o in zip(t_idx, best, ok):
                if o:
                    frame_lab[t].append(int(label[c]))

        # ---- 4. drop spurious tiny speakers (< min_spk_sec): reassign to nearest big cluster -------------
        cnt = {}
        for labs in frame_lab:
            for l in set(labs):
                cnt[l] = cnt.get(l, 0) + 1
        keep = {l for l, c in cnt.items() if c >= self.min_spk_sec * fps}
        if keep and len(keep) < len(cnt):
            lab_cent = {}
            for l in cnt:
                m = cents[label == l].mean(axis=0)
                lab_cent[l] = m / (np.linalg.norm(m) + 1e-12)
            remap = {l: (l if l in keep else max(keep, key=lambda k: float(lab_cent[l] @ lab_cent[k]))) for l in cnt}
            frame_lab = [sorted({remap[l] for l in labs}) for labs in frame_lab]
        n_pred = len({l for labs in frame_lab for l in labs})
        used = E[active_t]
        mse = compute_clustering_mse(used, cents)
        return frame_lab, n_pred, mse, time.perf_counter() - t0


class OfficialEMeansSlotDiarizer(SlotConditionedEMeansDiarizer):
    """
    v3: slot-conditioned diarizer whose clustering engine is the *official* e-means
    (faiss/contrib/e_means ``NumpyEmeans``, vendored in ``faiss_e_means``; spherical mode, default recipe:
    batch/step scaling laws, cosine-annealed EMA, donor-split revival, last-epoch closing means).

    Official e-means needs a fixed k, so the speaker count per slot is selected on top of it: for
    k = 2..k_max the slot's solo-frame embeddings are clustered, and k is accepted only if every pair of
    centroids is separated (cosine < split_cos) and every cluster is large enough. Capacity-awareness
    (splits only when the head's slots are saturated), cannot-link merging, labelling and the activity
    rules are inherited from `SlotConditionedEMeansDiarizer`.
    """

    def __init__(self, k_max: int = 4, split_cos: float = 0.6, nepochs: float = 5, n_init: int = 3, engine: str = "emeans", lloyd_iters: int = 25, **kw):
        super().__init__(**kw)
        self.k_max, self.split_cos, self.nepochs, self.n_init, self.engine, self.lloyd_iters = k_max, split_cos, nepochs, n_init, engine, lloyd_iters; self.record = None
        self.stat_calls, self.stat_splits, self.stat_time = 0, 0, 0.0      # engine runs / accepted slot splits / seconds spent in the engine

    def _cluster(self, Xf, k, r):
        """clustering engine used by the slot-splitting step: 'emeans' (official), 'lloyd' (25 iterations), 'lloyd5' (5 iterations =
        the epoch budget of e-means), 'minibatch' (Sculley, 5 epochs). All spherical, random-sample init seeded by seed + r."""
        n, d = Xf.shape
        if self.engine == "emeans":
            from .faiss_e_means import NumpyEmeans
            C, _ = NumpyEmeans(d, k, nepochs=self.nepochs, spherical=True, seed=self.seed + r).train(Xf)
            return C
        rng = np.random.RandomState(self.seed + r); C = Xf[rng.permutation(n)[:k]].astype(np.float64).copy()
        if self.engine in ("lloyd", "lloyd5"):
            for _ in range(self.lloyd_iters if self.engine == "lloyd" else 5):
                a = np.argmax(Xf @ C.T, axis=1)
                for j in range(k):
                    if (a == j).any():
                        m = Xf[a == j].mean(axis=0); C[j] = m / (np.linalg.norm(m) + 1e-12)
            return C
        if self.engine == "minibatch":
            v = np.ones(k)
            for _ in range(int(self.nepochs)):
                perm = rng.permutation(n)
                for b0 in range(0, n, 64):
                    Xb = Xf[perm[b0:b0 + 64]]; Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                    for x, c in zip(Xb, np.argmax(Xb @ Cn.T, axis=1)):
                        v[c] += 1; C[c] = (1 - 1 / v[c]) * C[c] + (1 / v[c]) * x
            return C
        raise ValueError(self.engine)

    def _emeans(self, X, hyst, cos_birth, min_sub_frac, min_sub_sec):
        n, d = X.shape
        m = X.mean(axis=0)
        best = (m / (np.linalg.norm(m) + 1e-12))[None, :]
        if cos_birth < 0 or n < 2 * hyst:  # not saturated: a slot is one speaker
            return best
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        thr = max(min_sub_sec * self._fps, min_sub_frac * n)
        for k in range(2, self.k_max + 1):
            cand = None
            for r in range(self.n_init):
                if self.record is not None and r == 0: self.record.append((Xf.copy(), k))        # inputs of the engine calls (for dev/exp_engine_loss.py)
                t_e = time.perf_counter(); C = self._cluster(Xf, k, r); self.stat_time += time.perf_counter() - t_e
                self.stat_calls += 1
                C = C.astype(np.float64)
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
                cnt = np.bincount(np.argmax(X @ C.T, axis=1), minlength=k)
                sm = C @ C.T
                np.fill_diagonal(sm, -1)
                if cnt.min() >= thr and sm.max() < self.split_cos:
                    mse = float(np.mean(1.0 - np.max(X @ C.T, axis=1)))
                    if cand is None or mse < cand[0]:
                        cand = (mse, C)
            if cand is None:
                break
            best = cand[1]
        self.stat_splits += len(best) - 1
        return best


class StreamingOfficialEMeansSlotDiarizer(OfficialEMeansSlotDiarizer):
    """
    Streaming (causal, bounded-latency) simulation of `OfficialEMeansSlotDiarizer`.

    Every `refit_sec` the offline diarizer is re-run on the *prefix* of the session seen so far (no future
    frames are used). Frames older than `commit_lag_sec` are then committed and never changed; labels of a
    refit are mapped to the permanent speaker ids by Hungarian matching against the already committed frames
    (unmatched labels become new speakers). A merge or split discovered later therefore only affects frames
    that are not committed yet, as in a real online system.

    Clustering latency of a frame is between `commit_lag_sec` and `commit_lag_sec + refit_sec`.
    This class makes only the *clustering stage* streaming; slot posteriors and embeddings are inputs.
    """

    def __init__(self, refit_sec: float = 2.0, commit_lag_sec: float = 2.0, **kw):
        super().__init__(**kw)
        self.refit_sec, self.commit_lag_sec = refit_sec, commit_lag_sec

    def fit_predict(self, embs: np.ndarray, probs_s: np.ndarray):
        from scipy.optimize import linear_sum_assignment

        t0 = time.perf_counter()
        T = len(probs_s)
        fps = 1000.0 / (self.stride_ms or 80)
        R = max(1, int(round(self.refit_sec * fps)))
        lag = int(round(self.commit_lag_sec * fps))
        committed = [[] for _ in range(T)]
        next_commit, next_gid = 0, 0
        for tau in list(range(R, T, R)) + [T]:
            end = tau if tau == T else tau - lag
            if end <= next_commit:
                continue
            lab, _, _, _ = super().fit_predict(embs[:tau], probs_s[:tau])
            # contingency between this refit's labels and the permanent ids of already committed frames
            cnt = {}
            for t in range(next_commit):
                for l in lab[t]:
                    for g in committed[t]:
                        cnt[(l, g)] = cnt.get((l, g), 0) + 1
            locs = sorted({l for labs in lab[next_commit:end] for l in labs} | {l for (l, _) in cnt})
            gids = sorted({g for (_, g) in cnt})
            mapping = {}
            if locs and gids:
                M = np.zeros((len(locs), len(gids)))
                for (l, g), c in cnt.items():
                    M[locs.index(l), gids.index(g)] = c
                for r, c in zip(*linear_sum_assignment(-M)):
                    if M[r, c] > 0:
                        mapping[locs[r]] = gids[c]
            for t in range(next_commit, end):
                out = []
                for l in lab[t]:
                    if l not in mapping:
                        mapping[l] = next_gid
                        next_gid += 1
                    out.append(mapping[l])
                committed[t] = sorted(set(out))
            next_commit = end
        return committed, next_gid, 0.0, time.perf_counter() - t0


class OnlineRevisingEMeansSlotDiarizer(OfficialEMeansSlotDiarizer):
    """
    Online diarizer with provisional output and bounded retroactive revision.

    Every `refit_sec` the slot-conditioned official-e-means diarizer is re-run on the prefix seen so far (no
    future frames). A frame is *emitted* (provisional label) at the first refit that is at least `emit_lag_sec`
    after the frame. Until the frame is older than `revise_horizon_sec` its label may be *revised* by later
    refits (e.g. a speaker split, merge or a better centroid); afterwards it is frozen. Labels of every refit are
    mapped to permanent speaker ids by Hungarian matching against the labels already emitted, unmatched labels
    become new speakers.

    output="final"       -> labels after all allowed revisions (what is stored at the end)
    output="provisional" -> label of each frame as first emitted (what a consumer saw in real time)
    revise_horizon_sec=inf -> labels may be revised until the end of the stream.
    Latency of a provisional label is between emit_lag_sec and emit_lag_sec + refit_sec.
    Only the clustering stage is simulated here; slot posteriors and embeddings are inputs.

    Dual embeddings: if `embs` has 2*d columns they are [causal | centred]: the causal embedding of frame t
    (window ending at t) is available immediately, the centred one (window ending at t + look_frames) only
    `look_frames` frames later. At a refit at time tau, frames t < tau - look_frames use the centred embedding
    (this is the retroactive correction of their provisional label), the newest frames use the causal one.
    """

    def __init__(self, refit_sec: float = 2.0, emit_lag_sec: float = 0.0, revise_horizon_sec: float = 30.0,
                 output: str = "final", look_frames: int = 9, record_snapshots: bool = False, **kw):
        super().__init__(**kw)
        self.look_frames = look_frames
        self.record_snapshots, self.snapshots = record_snapshots, []
        self.refit_sec, self.emit_lag_sec = refit_sec, emit_lag_sec
        self.revise_horizon_sec, self.output = revise_horizon_sec, output
        self.refit_times: list = []
        self.refit_meta: list = []                    # per refit: (official e-means runs, accepted slot splits)

    def fit_predict(self, embs: np.ndarray, probs_s: np.ndarray):
        from scipy.optimize import linear_sum_assignment

        t0 = time.perf_counter()
        T = len(probs_s)
        dual = embs.shape[1] % 2 == 0 and embs.shape[1] > 300
        if dual:
            d_ = embs.shape[1] // 2
            e_fast, e_ctr = embs[:, :d_], embs[:, d_:]
        fps = 1000.0 / (self.stride_ms or 80)
        R = max(1, int(round(self.refit_sec * fps)))
        lag = int(round(self.emit_lag_sec * fps))
        H = T + 1 if not np.isfinite(self.revise_horizon_sec) else int(round(self.revise_horizon_sec * fps))
        cur = [[] for _ in range(T)]
        prov = [[] for _ in range(T)]
        emitted, next_gid = 0, 0
        self.refit_times, self.snapshots, self.refit_meta = [], [], []
        for tau in list(range(R, T, R)) + [T]:
            end = tau if tau == T else tau - lag
            if end <= 0:
                continue
            tr = time.perf_counter(); c0, s0 = self.stat_calls, self.stat_splits
            if dual:
                k = max(0, tau - self.look_frames)
                X = np.concatenate([e_ctr[:k], e_fast[k:tau]], axis=0)
            else:
                X = embs[:tau]
            lab, _, _, _ = OfficialEMeansSlotDiarizer.fit_predict(self, X, probs_s[:tau])
            self.refit_times.append(time.perf_counter() - tr); self.refit_meta.append((self.stat_calls - c0, self.stat_splits - s0))
            cnt = {}
            for t in range(emitted):
                for l in lab[t]:
                    for g in cur[t]:
                        cnt[(l, g)] = cnt.get((l, g), 0) + 1
            locs = sorted({l for labs in lab[:end] for l in labs} | {l for (l, _) in cnt})
            gids = sorted({g for (_, g) in cnt})
            mapping = {}
            if locs and gids:
                M = np.zeros((len(locs), len(gids)))
                for (l, g), c in cnt.items():
                    M[locs.index(l), gids.index(g)] = c
                for r, c in zip(*linear_sum_assignment(-M)):
                    if M[r, c] > 0:
                        mapping[locs[r]] = gids[c]
            frozen = max(0, tau - H)
            for t in range(frozen, end):
                out = []
                for l in lab[t]:
                    if l not in mapping:
                        mapping[l] = next_gid
                        next_gid += 1
                    out.append(mapping[l])
                out = sorted(set(out))
                cur[t] = out
                if t >= emitted:
                    prov[t] = out
            emitted = max(emitted, end)
            if self.record_snapshots:  # (time of refit, primary label of every frame so far; -1 = silence)
                self.snapshots.append((tau, [c[0] if c else -1 for c in cur[:tau]]))
        return (prov if self.output == "provisional" else cur), next_gid, 0.0, time.perf_counter() - t0
