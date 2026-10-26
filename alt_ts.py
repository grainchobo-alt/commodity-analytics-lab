"""대안 시계열 예측 방법론 백테스트 — 롱숏 탭과 같은 원자재 19개 · 같은 매매 규칙

  공통: 매주 금요일, 각 모델이 '향후 4주(20영업일) 로그수익률'을 예측 → 두 가지 방식으로 매매 (편도 5bp, 사전 목표 변동성 10%, 평가 2012-07–)
    방향(시계열): 예측 부호 ÷ 품목 26주 변동성 → 품목마다 롱 또는 숏 (52주 모멘텀 추세추종과 같은 틀)
    순위 롱숏(횡단면): 예측 ÷ 변동성 상위 1/3 롱 · 하위 1/3 숏, 4주 보유, 달러 중립 (롱숏 탭 ⑨와 같은 틀)
  모델
    Prophet 방식 (자체 구현): 로그가격 = 구간별 선형 추세(변곡점 20개, L1로 꺾임 억제) + 연간 푸리에 계절성(3차), 직전 3년 주간 자료로 매주 적합
                              → 4주 뒤 적합값 − 지금 적합값. (prophet 패키지 미설치 → 같은 수식을 Lasso로 구현)
    Chronos-Bolt-small · Chronos-2 (Amazon) · TimesFM-2.5 (Google): 사전학습 파운데이션 모델 zero-shot, 직전 512영업일 로그가격 → 20영업일 뒤 중앙값
    소형 신경망 5종 (19개 품목 자료로 매년 1월 재학습, 직전 52주 입력, 목표 4주 뒤 수익률, 학습 끝 4주 purge):
      PatchTST 방식 · iTransformer 방식 · TFT 방식 · Neural ODE 방식 · 시공간 GNN 방식 — 원 논문 구조를 줄인 CPU용 구현 (패키지 미설치)
  지표: Sharpe(비용 후) · 2017 전후 · 방향 적중률 · IC(예측과 실제 4주 수익률의 품목 간 순위 상관, 주별 평균)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ls_strategies as LS

ANN = 52
H_W = 4
EVAL = "2012-01-01"
FMS = ("Chronos-Bolt-small", "Chronos-2", "TimesFM-2.5")
BASE = "기준 · 52주 모멘텀"


def weekly_px(L: pd.DataFrame, start: str = "2001-01-01") -> pd.DataFrame:
    P = LS.weekly(L, LS.BASKET, start)
    return P[[c for c in LS.BASKET if c in P]]


def signal_book(pred: pd.DataFrame, P: pd.DataFrame, mode: str = "dir", cost_bp: float = 5.0, tgt: float = 10.0) -> pd.Series:
    """예측 4주 수익률 → 비중 → 목표 변동성 10% → 주간 손익. mode: dir = 부호 ÷ 변동성, xs = 순위 상·하위 1/3 롱숏."""
    import ls_extra as LX
    R = LS._ret(P)
    pred = pred.reindex(index=P.index, columns=P.columns)
    vol = R.rolling(26, min_periods=13).std()
    if mode == "dir":
        W = np.sign(pred) / vol
        W = W.div(W.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        W = LX.xs_weights((pred / vol).where(pred.notna()), hold=H_W, inv_vol=1 / vol)
    return LS._book(LS._vol_target(W, R, tgt), R, cost_bp)


def eval_pred(pred: pd.DataFrame, P: pd.DataFrame) -> dict:
    lp = np.log(P)
    fwd = (lp.shift(-H_W) - lp).reindex(index=pred.index, columns=pred.columns)
    m = pred.notna() & fwd.notna()
    hit = float((np.sign(pred[m]) == np.sign(fwd[m])).sum().sum() / m.sum().sum()) if m.values.any() else np.nan
    ic = pred.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1).mean()
    return {"방향 적중률": hit, "IC (주별 순위 상관)": float(ic)}


# ============================================================================
# Prophet 방식 (자체 구현)
# ============================================================================
def prophet_like(P: pd.DataFrame, win: int = 156, n_cp: int = 20, alpha: float = 2e-4, start: str = EVAL) -> pd.DataFrame:
    from sklearn.linear_model import Lasso
    lp = np.log(P)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)

    def four(dates):
        a = 2 * np.pi * pd.DatetimeIndex(dates).dayofyear.values / 365.25
        return np.column_stack([f(k * a) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    cps = np.linspace(0, 0.8, n_cp + 1)[1:]                        # Prophet 기본값처럼 앞 80% 구간에만 변곡점
    tf = np.arange(win + H_W) / win
    for e in np.flatnonzero(P.index >= pd.Timestamp(start)):
        if e < win:
            continue
        s = slice(e - win + 1, e + 1)
        dts = list(P.index[s]) + [P.index[e] + pd.Timedelta(weeks=k) for k in range(1, H_W + 1)]   # 미래 날짜는 달력으로만
        Xf = np.column_stack([tf] + [np.maximum(tf - c, 0) for c in cps] + [four(dts)])
        X = Xf[:win]
        for c in P.columns:
            y = lp[c].values[s]
            if np.isnan(y).sum() > win * 0.2:
                continue
            ok = ~np.isnan(y)
            mdl = Lasso(alpha=alpha, max_iter=5000).fit(X[ok], y[ok] - np.nanmean(y))
            yh = mdl.predict(Xf)
            out.iloc[e, out.columns.get_loc(c)] = yh[win - 1 + H_W] - yh[win - 1]
    return out


# ============================================================================
# 파운데이션 모델 (forecast_lab 캐시 사용)
# ============================================================================
def fm_pred(name: str, L: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    import forecast_lab as FL
    series = {i: np.log(L[i].dropna()) for i in P.columns}
    dates = {i: FL.fridays(s.index, start=EVAL, end_pad=0) for i, s in series.items()}
    q = FL.fm_forecast(name, "ls19", series, dates, 20)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for i in P.columns:
        if i not in q or not len(q[i]):
            continue
        s = series[i]
        lab = dates[i].to_period("W-FRI").to_timestamp(how="end").normalize()
        last = s.reindex(dates[i]).values
        v = pd.Series(q[i][:, -1, 4] - last, index=lab)            # 20영업일 뒤 중앙값 − 현재 로그가격
        v = v[~v.index.duplicated(keep="last")]
        out[i] = v.reindex(out.index)
    return out


# ============================================================================
# 소형 신경망 (PatchTST · iTransformer · TFT · Neural ODE · 시공간 GNN 방식) — 같은 학습 틀, 매년 1월 재학습
# ============================================================================
NN_KINDS = {"patch": "PatchTST 방식 (4주 패치 → 트랜스포머)", "itrans": "iTransformer 방식 (품목 = 토큰, 품목 간 어텐션)",
            "tft": "TFT 방식 (변수 선택 + LSTM + 어텐션, 달력 변수)", "node": "Neural ODE 방식 (ODE-RNN, RK4로 은닉 상태 연속 진화)",
            "gnn": "시공간 GNN 방식 (GRU 시계열 + 상관 그래프 합성곱)"}
EXACT_NN = {"node_x": "Neural ODE (torchdiffeq dopri5 ODE-RNN, 정식)", "gnn_x": "시공간 GNN (torch_geometric GCN + GRU, 정식)"}
CROSS = {"itrans", "gnn", "gnn_x"}                                            # 19개 품목을 한 표본으로 동시에 보는 모델


def _windows(R: pd.DataFrame):
    """주간 수익률(대략 표준화)과 4주 앞 누적 수익률 목표."""
    r = R.fillna(0.0).values / 5.0
    fwd = (R.rolling(H_W).sum().shift(-H_W)).values / 10.0
    return r, fwd


def nn_pred(P: pd.DataFrame, kind: str = "patch", lookback: int = 52, epochs: int = 8, seed: int = 0,
            first_year: int = 2012, d: int = 32, stride: int = 2) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(4)
    R = LS._ret(P)
    r, fwd = _windows(R)
    T, N = r.shape
    idx = P.index
    woy = 2 * np.pi * idx.dayofyear.values / 365.25
    cal = np.column_stack([np.sin(woy), np.cos(woy)])               # TFT의 '미리 아는 변수' (달력)

    class PatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(4, d); self.pos = nn.Parameter(torch.zeros(lookback // 4, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True), 2)
            self.head = nn.Linear(d * (lookback // 4), 1)

        def forward(self, x):                                       # x: (B, lookback, 3) — 수익률만 사용
            z = self.emb(x[..., 0].reshape(x.shape[0], lookback // 4, 4)) + self.pos
            return self.head(self.enc(z).flatten(1)).squeeze(-1)

    class ITrans(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(lookback, d)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True), 2)
            self.head = nn.Linear(d, 1)

        def forward(self, x, A=None):                               # x: (B, N, lookback) → (B, N)
            return self.head(self.enc(self.emb(x))).squeeze(-1)

    class GRN(nn.Module):                                           # TFT의 게이트 잔차 블록
        def __init__(self, i, o):
            super().__init__()
            self.l1, self.l2, self.g, self.skip = nn.Linear(i, d), nn.Linear(d, o), nn.Linear(d, o), nn.Linear(i, o)
            self.ln = nn.LayerNorm(o)

        def forward(self, x):
            h = self.l1(x).relu()
            return self.ln(self.skip(x) + self.l2(h) * torch.sigmoid(self.g(h)))

    class TFT(nn.Module):
        def __init__(self, nv=3):
            super().__init__()
            self.var_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(nv)])
            self.vsn = GRN(nv, nv)
            self.lstm = nn.LSTM(d, d, batch_first=True)
            self.post = GRN(d, d)
            self.att = nn.MultiheadAttention(d, 4, batch_first=True)
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):                                       # x: (B, lookback, 3)
            w = torch.softmax(self.vsn(x), -1)                      # 시점별 변수 선택 가중치
            e = torch.stack([m(x[..., k:k + 1]) for k, m in enumerate(self.var_emb)], -1)
            z = self.post(self.lstm((e * w[:, :, None, :]).sum(-1))[0])
            a, _ = self.att(z[:, -1:], z, z)                        # 마지막 시점이 과거 전체를 참조
            return self.head(self.ln(a[:, 0] + z[:, -1])).squeeze(-1)

    class ODERNN(nn.Module):
        def __init__(self, h=16):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, h))
            self.cell = nn.GRUCell(1, h)
            self.head = nn.Linear(h, 1)
            self.h = h

        def forward(self, x):                                       # 관측 사이 dh/dt = f(h)를 RK4로 1주 적분 → 관측으로 갱신
            h = torch.zeros(x.shape[0], self.h)
            for t in range(x.shape[1]):
                k1 = self.f(h); k2 = self.f(h + k1 / 2); k3 = self.f(h + k2 / 2); k4 = self.f(h + k3)
                h = self.cell(x[:, t, :1], h + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            return self.head(h).squeeze(-1)

    class STGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, d, batch_first=True)
            self.self1, self.nb1 = nn.Linear(d, d), nn.Linear(d, d)
            self.self2, self.nb2 = nn.Linear(d, d), nn.Linear(d, d)
            self.head = nn.Linear(d, 1)

        def forward(self, x, A):                                    # x: (B, N, lookback), A: (N, N) 행 정규화 인접 행렬
            B_ = x.shape[0]
            h = self.gru(x.reshape(B_ * N, lookback, 1))[1][-1].view(B_, N, d)
            h = (self.self1(h) + A @ self.nb1(h)).relu()
            h = (self.self2(h) + A @ self.nb2(h)).relu()
            return self.head(h).squeeze(-1)

    class ODERNNExact(nn.Module):                                   # torchdiffeq 적응 스텝(dopri5) — Rubanova 외 2019 ODE-RNN
        def __init__(self, h=16):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, h))
            self.cell = nn.GRUCell(1, h)
            self.head = nn.Linear(h, 1)
            self.h = h
            self.tt = torch.tensor([0.0, 1.0])

        def forward(self, x):
            from torchdiffeq import odeint
            h = torch.zeros(x.shape[0], self.h)
            for t in range(x.shape[1]):
                h = odeint(lambda _, z: self.f(z), h, self.tt, method="dopri5", rtol=1e-3, atol=1e-4)[-1]
                h = self.cell(x[:, t, :1], h)
            return self.head(h).squeeze(-1)

    class GCNGRU(nn.Module):                                        # torch_geometric GCNConv × 2 (정규화 인접 + 자기 루프) + GRU 시간 인코더 (T-GCN 계열)
        def __init__(self):
            super().__init__()
            from torch_geometric.nn import GCNConv
            self.gru = nn.GRU(1, d, batch_first=True)
            self.g1, self.g2 = GCNConv(d, d), GCNConv(d, d)
            self.head = nn.Linear(d, 1)

        def forward(self, x, A):
            from torch_geometric.utils import dense_to_sparse
            ei, ew = dense_to_sparse(A)
            B_ = x.shape[0]
            h = self.gru(x.reshape(B_ * N, x.shape[-1], 1))[1][-1].view(B_, N, d)
            h = self.g1(h, ei, ew).relu() + h
            h = self.g2(h, ei, ew).relu() + h
            return self.head(h).squeeze(-1)

    def adjacency(t_end):                                         # 학습 끝 직전 156주 |상관| 상위 4개 이웃 (그 해 고정)
        C = np.nan_to_num(np.abs(np.corrcoef(r[max(0, t_end - 155):t_end + 1].T))); np.fill_diagonal(C, 0)
        M = np.zeros_like(C)
        for i in range(N):
            nb = np.argsort(C[i])[-4:]; M[i, nb] = C[i, nb]
        return torch.tensor(M / M.sum(1, keepdims=True).clip(1e-9), dtype=torch.float32)

    def feats(t):                                                   # 품목별 입력 (N, lookback, 3) = 수익률 + 달력 2개
        x = r[t - lookback + 1:t + 1].T[:, :, None]
        c = np.broadcast_to(cal[t - lookback + 1:t + 1][None], (N, lookback, 2))
        return np.concatenate([x, c], -1)

    def cross(t):
        return r[t - lookback + 1:t + 1].T                         # (N, lookback), t 시점까지 (t 포함)

    make = {"patch": PatchTST, "itrans": ITrans, "tft": TFT, "node": ODERNN, "gnn": STGNN, "node_x": ODERNNExact, "gnn_x": GCNGRU}[kind]
    ep = max(4, epochs // 2) if kind in ("node", "node_x") else epochs         # ODE-RNN은 순차 계산이라 에폭 절반
    out = pd.DataFrame(np.nan, index=idx, columns=P.columns)
    for yr in sorted(set(idx.year)):
        if yr < first_year:
            continue
        te = [t for t in np.flatnonzero(idx.year == yr) if t >= lookback]
        tr_end = np.flatnonzero(idx < pd.Timestamp(f"{yr}-01-01"))
        if len(tr_end) < lookback + 104 or not te:
            continue
        tr_t = np.arange(lookback, tr_end[-1] - H_W + 1)[::-1][::stride][::-1]   # 목표(4주 뒤)가 학습 구간 안에서 끝나는 시점만 (purge), 2주 간격 표본
        A = adjacency(tr_end[-1]) if kind in ("gnn", "gnn_x") else None
        model = make()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        if kind in CROSS:
            X_ = torch.tensor(np.stack([cross(t) for t in tr_t]), dtype=torch.float32)
            Y_ = torch.tensor(np.nan_to_num(fwd[tr_t]), dtype=torch.float32)
            run = lambda xb: model(xb, A)                           # noqa: E731
        else:
            Xs = np.stack([feats(t) for t in tr_t]).reshape(-1, lookback, 3)
            Ys = fwd[tr_t].reshape(-1); ok = np.isfinite(Ys)
            X_, Y_ = torch.tensor(Xs[ok], dtype=torch.float32), torch.tensor(Ys[ok], dtype=torch.float32)
            run = model
        n = len(X_)
        for _ in range(ep):
            model.train()
            perm = torch.randperm(n)
            for k in range(0, n, 256):
                b = perm[k:k + 256]
                loss = ((run(X_[b]) - Y_[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if kind in CROSS:
                pr = run(torch.tensor(np.stack([cross(t) for t in te]), dtype=torch.float32))
            else:
                pr = model(torch.tensor(np.stack([feats(t) for t in te]).reshape(-1, lookback, 3), dtype=torch.float32)).view(len(te), N)
            out.iloc[te] = pr.numpy()
    return out


# ============================================================================
# 정식 패키지 — prophet (Meta) · neuralforecast (Nixtla: PatchTST · iTransformer · TFT)
# ============================================================================
def _prophet_item(lp: pd.Series, ends: list, win: int) -> dict:
    import logging
    from prophet import Prophet
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).setLevel(logging.ERROR)
    out = {}
    for e in ends:
        y = lp.iloc[e - win + 1:e + 1].dropna()
        if len(y) < win * 0.8:
            continue
        m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=3, uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        f = m.predict(pd.DataFrame({"ds": [lp.index[e], lp.index[e] + pd.Timedelta(weeks=H_W)]}))
        out[lp.index[e]] = float(f.yhat.iloc[1] - f.yhat.iloc[0])
    return out


def prophet_exact(P: pd.DataFrame, win: int = 156, start: str = EVAL, n_jobs: int = 4) -> pd.DataFrame:
    """prophet 기본 설정(변곡점 25개 · 앞 80% · changepoint_prior 0.05) + 연간 계절 3차, 직전 3년 주간 로그가격으로 매주 적합."""
    from joblib import Parallel, delayed
    lp = np.log(P)
    ends = [e for e in np.flatnonzero(P.index >= pd.Timestamp(start)) if e >= win]
    res = Parallel(n_jobs=n_jobs)(delayed(_prophet_item)(lp[c], ends, win) for c in P.columns)
    return pd.DataFrame({c: pd.Series(r) for c, r in zip(P.columns, res)}).reindex(P.index)


NF_MODELS = {"PatchTST": "PatchTST (neuralforecast, 정식)", "iTransformer": "iTransformer (neuralforecast, 정식)", "TFT": "TFT (neuralforecast, 정식)"}


def nf_pred(P: pd.DataFrame, name: str, first_year: int = 2012, input_size: int = 52, max_steps: int = 300, seed: int = 1) -> pd.DataFrame:
    """매년: 그해 첫 결정일까지 자료로 학습(refit=False 교차검증) → 그해 매주 금요일 '그때까지 자료'로 4주 앞 로그가격 예측 → 예측 − 현재."""
    import logging
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, iTransformer, TFT
    from neuralforecast.losses.pytorch import MSE
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR); logging.getLogger("lightning").setLevel(logging.ERROR)
    lp = np.log(P).ffill()
    woy = 2 * np.pi * lp.index.dayofyear.values / 365.25
    long = lp.reset_index(names="ds").melt(id_vars="ds", var_name="unique_id", value_name="y").dropna()
    cal = pd.DataFrame({"ds": lp.index, "s1": np.sin(woy), "c1": np.cos(woy)})
    long = long.merge(cal, on="ds")
    first = long.groupby("unique_id").ds.min()
    long = long[long.unique_id.isin(first[first <= pd.Timestamp(f"{first_year - 5}-12-31")].index)]   # 첫 시험 연도 전 5년 이상 자료가 있는 품목만 (19개 모두 해당)
    kw = dict(h=H_W, input_size=input_size, max_steps=max_steps, loss=MSE(), random_seed=seed, enable_progress_bar=False,
              enable_model_summary=False, logger=False, accelerator="cpu")
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for yr in sorted(set(P.index.year)):
        if yr < first_year:
            continue
        wk = P.index[P.index.year == yr]
        end = P.index[min(P.index.get_loc(wk[-1]) + H_W, len(P) - 1)]
        n_win = int(((P.index >= wk[0]) & (P.index <= P.index[P.index.get_loc(end) - H_W])).sum())
        if n_win < 1:
            continue
        if name == "PatchTST":
            m = PatchTST(patch_len=4, stride=4, hidden_size=64, n_heads=4, encoder_layers=2, scaler_type="robust", **kw)
        elif name == "iTransformer":
            m = iTransformer(n_series=long.unique_id.nunique(), hidden_size=64, n_heads=4, e_layers=2, d_ff=128, **kw)
        else:
            m = TFT(hidden_size=32, n_head=4, futr_exog_list=["s1", "c1"], scaler_type="robust", **kw)
        nf = NeuralForecast(models=[m], freq="W-FRI")
        cv = nf.cross_validation(df=long[long.ds <= end], n_windows=n_win, step_size=1, refit=False)
        col = [c for c in cv.columns if c.startswith(name)][0]
        cv = cv[cv.ds == cv.cutoff + pd.Timedelta(weeks=H_W)]
        cur = lp.stack().rename("y0").reset_index().rename(columns={"level_0": "cutoff", "level_1": "unique_id"})
        cur.columns = ["cutoff", "unique_id", "y0"]
        cv = cv.merge(cur, on=["cutoff", "unique_id"])
        pv = (cv.assign(p=cv[col] - cv.y0).pivot(index="cutoff", columns="unique_id", values="p"))
        pv = pv[pv.index.year == yr]
        out.loc[pv.index, pv.columns] = pv.values
    return out


# ============================================================================
# 전체 실행
# ============================================================================
def all_preds(L: pd.DataFrame, fms=FMS, progress=None, cache: dict | None = None, exact: bool = True, on_done=None) -> tuple[pd.DataFrame, dict]:
    """모델별 예측 표 {이름: date×id}. cache에 이미 있는 모델은 다시 계산하지 않음."""
    P = weekly_px(L)
    preds = {k: v for k, v in (cache or {}).items() if " · 시드 " not in k}
    jobs = [("Prophet 방식 (추세 변곡 + 연간 계절, 자체 구현)", lambda: prophet_like(P))]
    jobs += [(lbl, lambda k=k: nn_pred(P, k)) for k, lbl in NN_KINDS.items()]
    if exact:                                                       # 정식 패키지 판 (prophet · neuralforecast · torchdiffeq · torch_geometric)
        jobs += [("Prophet (Meta prophet 패키지, 정식)", lambda: prophet_exact(P))]
        jobs += [(lbl, lambda k=k: nf_pred(P, k)) for k, lbl in NF_MODELS.items()]
        jobs += [(lbl, lambda k=k: nn_pred(P, k, **(dict(lookback=26, stride=4, epochs=6) if k == "node_x" else {}))) for k, lbl in EXACT_NN.items()]
    jobs += [(f"{nm} (zero-shot 파운데이션)", lambda nm=nm: fm_pred(nm, L, P)) for nm in fms]
    for lbl, fn in jobs:
        if lbl in preds:
            continue
        if progress: progress(lbl)
        try:
            preds[lbl] = fn()
            if on_done: on_done(preds)                              # 모델 하나 끝날 때마다 캐시 저장 (중간에 멈춰도 다시 계산 안 함)
        except Exception as e:                                      # 파운데이션 모델 미설치·캐시 없음 등
            print("실패", lbl, e)
    lp = np.log(P)
    preds[BASE] = lp - lp.shift(52)
    models = [k for k in preds if k != BASE and not k.startswith("모델 합의")]
    preds["모델 합의 (기준 제외 모델들 부호 평균)"] = sum(np.sign(preds[k].reindex(index=P.index, columns=P.columns)).fillna(0) for k in models) / len(models)
    return P, preds


def run_alt(P: pd.DataFrame, preds: dict, cost_bp: float = 5.0, split: str = "2017-01-01") -> dict:
    books = {(k, md): signal_book(v, P, md, cost_bp) for k, v in preds.items() for md in ("dir", "xs")}
    D = pd.DataFrame(books)
    D = D[D.index >= pd.Timestamp(EVAL) + pd.DateOffset(weeks=26)].fillna(0.0)
    rows = []
    for k in preds:
        ev = eval_pred(preds[k].reindex(index=D.index), P)
        row = {"모델": k, **ev}
        for md, nm in (("dir", "방향"), ("xs", "순위 롱숏")):
            r = D[(k, md)]
            m = LS.metrics(r)
            row.update({f"{nm} Sharpe": m["Sharpe"], f"{nm} ~{split[:4]}": LS.metrics(r[r.index < split])["Sharpe"],
                        f"{nm} {split[:4]}~": LS.metrics(r[r.index >= split])["Sharpe"], f"{nm} t값": m["Sharpe"] * np.sqrt(len(r) / ANN),
                        f"{nm} 모멘텀 상관": float(r.corr(D[(BASE, md)]))})
        rows.append(row)
    D.columns = [f"{k} | {'방향' if md == 'dir' else '순위 롱숏'}" for k, md in D.columns]
    return dict(table=pd.DataFrame(rows), returns=D)


# ============================================================================
# 보완 분석 ① 정식 vs 경량 짝 비교 · ② 시드 흔들림 · ③ IC 높은 모델을 롱숏 혼합 25% 후보로 (구간 이동 선택)
# ============================================================================
PAIRS = {"Prophet": ("Prophet 방식 (추세 변곡 + 연간 계절, 자체 구현)", "Prophet (Meta prophet 패키지, 정식)"),
         "PatchTST": (NN_KINDS["patch"], NF_MODELS["PatchTST"]), "iTransformer": (NN_KINDS["itrans"], NF_MODELS["iTransformer"]),
         "TFT": (NN_KINDS["tft"], NF_MODELS["TFT"]), "Neural ODE": (NN_KINDS["node"], EXACT_NN["node_x"]),
         "시공간 GNN": (NN_KINDS["gnn"], EXACT_NN["gnn_x"])}
SEED_JOBS = {NN_KINDS["patch"]: ("nn", "patch"), NN_KINDS["itrans"]: ("nn", "itrans"), NN_KINDS["tft"]: ("nn", "tft"), NN_KINDS["gnn"]: ("nn", "gnn"),
             NF_MODELS["PatchTST"]: ("nf", "PatchTST"), NF_MODELS["iTransformer"]: ("nf", "iTransformer"), NF_MODELS["TFT"]: ("nf", "TFT"),
             EXACT_NN["gnn_x"]: ("nn", "gnn_x")}                     # Neural ODE 2종은 학습이 너무 느려 시드 반복 제외
SEEDS = (1, 2)                                                       # 기본 실행 = 시드 0 (neuralforecast는 1) → 추가 2개


def pair_table(P: pd.DataFrame, preds: dict, table: pd.DataFrame) -> pd.DataFrame:
    T = table.set_index("모델")
    rows = []
    for fam, (lite, exact) in PAIRS.items():
        if lite not in preds or exact not in preds:
            continue
        a, b = preds[lite].reindex(index=P.index, columns=P.columns), preds[exact].reindex(index=P.index, columns=P.columns)
        m = a.notna() & b.notna()
        x, y = a[m].stack(), b[m].stack()
        row = {"모델": fam, "예측 상관 (경량 vs 정식)": float(x.corr(y)), "부호 일치율": float((np.sign(x) == np.sign(y)).mean())}
        for tag, k in (("경량", lite), ("정식", exact)):
            for c in ("방향 Sharpe", "순위 롱숏 Sharpe", "IC (주별 순위 상관)", "방향 적중률"):
                row[f"{tag} · {c}"] = T.loc[k, c]
        rows.append(row)
    return pd.DataFrame(rows)


def seed_preds(P: pd.DataFrame, preds: dict, seeds=SEEDS, progress=None, on_done=None) -> dict:
    """SEED_JOBS 모델을 다른 시드로 다시 학습 → preds[f'{모델} · 시드 {s}'] (캐시에 있으면 건너뜀)."""
    for lbl, (src, k) in SEED_JOBS.items():
        if lbl not in preds:
            continue
        for s in seeds:
            key = f"{lbl} · 시드 {s}"
            if key in preds:
                continue
            if progress: progress(key)
            try:
                if src == "nf":
                    preds[key] = nf_pred(P, k, seed=s + 1)
                else:
                    preds[key] = nn_pred(P, k, seed=s)
                if on_done: on_done(preds)
            except Exception as e:
                print("실패", key, e)
    return preds


def seed_table(P: pd.DataFrame, preds: dict, cost_bp: float = 5.0) -> pd.DataFrame:
    D0 = pd.Timestamp(EVAL) + pd.DateOffset(weeks=26)
    rows = []
    for lbl in SEED_JOBS:
        runs = ([preds[lbl]] if lbl in preds else []) + [preds[f"{lbl} · 시드 {s}"] for s in SEEDS if f"{lbl} · 시드 {s}" in preds]
        if len(runs) < 2:
            continue
        row = {"모델": lbl, "시드 수": len(runs)}
        for md, nm in (("dir", "방향"), ("xs", "순위 롱숏")):
            sh = [LS.metrics(signal_book(p, P, md, cost_bp).loc[D0:])["Sharpe"] for p in runs]
            avg = sum(p.reindex(index=P.index, columns=P.columns) for p in runs) / len(runs)
            row.update({f"{nm} Sharpe 평균": float(np.mean(sh)), f"{nm} 최소": float(np.min(sh)), f"{nm} 최대": float(np.max(sh)),
                        f"{nm} 시드 평균 예측": LS.metrics(signal_book(avg, P, md, cost_bp).loc[D0:])["Sharpe"]})
        ics = [eval_pred(p.loc[D0:], P)["IC (주별 순위 상관)"] for p in runs]
        row.update({"IC 평균": float(np.mean(ics)), "IC 최소": float(np.min(ics)), "IC 최대": float(np.max(ics))})
        rows.append(row)
    return pd.DataFrame(rows)


ALT_FOLDS = [("2017-01-01", "2020-12-31"), ("2021-01-01", "2026-12-31")]
ALT_TRAIN = "2012-07-01"


def mix_candidates(L: pd.DataFrame, P: pd.DataFrame, preds: dict, ic_min: float = 0.02, cost_bp: float = 5.0) -> dict:
    """학습 구간(2012-07–2016) IC > ic_min 인 모델만 → 'MinVar 50 + 모멘텀 25 + 모델(순위 롱숏) 25' 후보.
    기존 혼합 후보(가치·따라잡기 등) 전부와 함께 구간 이동 선택 (시험 2017–20 · 2021–26, 학습은 2012-07부터 — 모델 예측이 2012부터라)."""
    import ls_extra as LX
    X = LX.run_extra(L, cost_bp)
    R_ = X["returns"]
    mvb, tsm = R_["기준 · MinVar 롱 (⑥)"], R_["기준 · 52주 모멘텀 (④)"]
    base_key = "혼합 · MinVar 70 + 모멘텀 30 (기존)"

    def mix(parts):
        m = sum(w * s.reindex(mvb.index).fillna(0) for w, s in parts)
        pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        return m * (10.0 / pv).clip(upper=4.0).fillna(1.0)
    ic_tr = {}
    for k, v in preds.items():
        if k == BASE or k.startswith("모델 합의") or " · 시드 " in k:
            continue
        ic_tr[k] = eval_pred(v.loc[ALT_TRAIN:"2016-12-31"], P)["IC (주별 순위 상관)"]
    picks = [k for k, v in sorted(ic_tr.items(), key=lambda kv: -kv[1]) if v > ic_min]
    cands = {k: v.loc[ALT_TRAIN:] for k, v in X["cands"].items()}
    new = {}
    for k in picks:
        nm = f"혼합 · MinVar 50 + 모멘텀 25 + {k.split(' (')[0]} 25"
        cands[nm] = mix([(0.5, mvb), (0.25, tsm), (0.25, signal_book(preds[k], P, "xs", cost_bp))]).loc[ALT_TRAIN:]
        new[nm] = k
    ws = LX.walk_select(cands, base_key, folds=ALT_FOLDS, train_start=ALT_TRAIN)
    sh = lambda r: float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan
    base = cands[base_key]
    rows = []
    for k in [base_key, "혼합 · MinVar 50 + 모멘텀 25 + V1 가치 25"] + list(new):
        if k not in cands:
            continue
        r = cands[k]; d = (r - base).dropna()
        rows.append({"후보": k, "학습 IC (2012-07–2016)": ic_tr.get(new.get(k), np.nan),
                     "Sharpe 2012-07–": sh(r), "Sharpe ~2016": sh(r[:"2016-12-31"]), "Sharpe 2017~": sh(r["2017-01-01":]),
                     "기존 대비 연 수익 차 (%p)": float(d.mean() * ANN), "차이 t값": float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan})
    ic_df = pd.DataFrame([{"모델": k, "학습 IC (2012-07–2016)": v, "후보 포함": v > ic_min} for k, v in sorted(ic_tr.items(), key=lambda kv: -kv[1])])
    return dict(ic=ic_df, cands=pd.DataFrame(rows), walk=ws["table"], summary=ws["summary"], n_cands=len(cands))
