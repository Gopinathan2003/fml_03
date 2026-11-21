"""
SPAM or HAM classifier -- everything (features, Naive Bayes, Logistic Regression,
cross-validation, tuning) is implemented from scratch with numpy.

Usage
-----
Train  : python spam_classifier.py train --data data      (data/spam/*, data/ham/*)
Predict: python spam_classifier.py predict [--test test]   (reads test/email*.txt)

From Python: from spam_classifier import predict; predict("test")
predict() returns {filename: 1 (spam) or 0 (ham)} and writes predictions.txt.
"""
import os, re, sys, glob, pickle, argparse, html
from email import message_from_string
import numpy as np

MODEL_FILE = "model.pkl"
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z]{2,20}|[!$]")


# ----------------------------------------------------------------------------
# 1. Reading / cleaning an email
# ----------------------------------------------------------------------------
def read_email(path):
    """Return (text, is_html) for one email file; the Subject is prepended to the body."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    subject, body, is_html = "", [], False
    try:
        msg = message_from_string(raw)
        subject = str(msg.get("Subject", "") or "")
        parts = msg.walk() if msg.is_multipart() else [msg]
        for p in parts:
            ctype = p.get_content_type()
            if ctype in ("text/plain", "text/html"):
                payload = p.get_payload(decode=True)
                if payload is None:
                    continue
                txt = payload.decode(p.get_content_charset() or "utf-8", errors="ignore")
                if ctype == "text/html":
                    is_html = True
                    txt = html.unescape(TAG_RE.sub(" ", txt))
                body.append(txt)
    except Exception:
        pass
    text = " ".join(body).strip()
    if not text:                       # no headers / unparsable: treat whole file as text
        text = raw
    if "<html" in raw.lower() or "<body" in raw.lower():
        is_html = True
    return subject + " . " + subject + " . " + text, is_html   # subject counted twice


def tokenize(text):
    """Lower-case, replace URLs/emails/numbers by placeholders, split into words."""
    t = URL_RE.sub(" urltoken ", text)
    t = MAIL_RE.sub(" emailtoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    return TOKEN_RE.findall(t.lower())


DENSE_NAMES = ["log_len", "frac_upper", "n_exclaim", "n_dollar", "n_url", "is_html"]


def dense_features(text, is_html):
    """Hand-crafted numeric features (all squashed with log1p so they are on one scale)."""
    letters = [c for c in text if c.isalpha()]
    frac_up = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    return np.array([np.log1p(len(text)), frac_up, np.log1p(text.count("!")),
                     np.log1p(text.count("$")), np.log1p(len(URL_RE.findall(text))),
                     1.0 if is_html else 0.0])


def load_docs(paths):
    toks, dens = [], []
    for p in paths:
        text, h = read_email(p)
        toks.append(tokenize(text))
        dens.append(dense_features(text, h))
    return toks, np.array(dens)


# ----------------------------------------------------------------------------
# 2. Vectorisation: bag-of-words counts, TF-IDF, standardised dense features
# ----------------------------------------------------------------------------
class Vectorizer:
    def __init__(self, max_features=3000, min_df=3):
        self.max_features, self.min_df = max_features, min_df

    def fit(self, toks, dens):
        df = {}
        for d in toks:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        items = [(w, c) for w, c in df.items() if c >= self.min_df]
        items.sort(key=lambda x: -x[1])
        self.vocab = {w: i for i, (w, _) in enumerate(items[: self.max_features])}
        n = len(toks)
        idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            idf[i] = np.log((1 + n) / (1 + df[w])) + 1.0
        self.idf = idf
        self.mu, self.sd = dens.mean(0), dens.std(0) + 1e-8
        return self

    def counts(self, toks):
        X = np.zeros((len(toks), len(self.vocab)), dtype=np.float32)
        for r, d in enumerate(toks):
            for w in d:
                j = self.vocab.get(w)
                if j is not None:
                    X[r, j] += 1
        return X

    def tfidf(self, C, dens):
        X = np.log1p(C) * self.idf                         # sub-linear tf * idf
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # L2 normalise rows
        D = (dens - self.mu) / self.sd
        return np.hstack([X, D]).astype(np.float32)


# ----------------------------------------------------------------------------
# 3. Algorithm 1: Multinomial Naive Bayes (with Laplace/Lidstone smoothing alpha)
# ----------------------------------------------------------------------------
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, C, y):
        self.logprior = np.array([np.log(np.mean(y == k)) for k in (0, 1)])
        self.loglik = np.zeros((2, C.shape[1]))
        for k in (0, 1):
            cnt = C[y == k].sum(0) + self.alpha
            self.loglik[k] = np.log(cnt / cnt.sum())
        return self

    def decision(self, C):                 # log-odds of spam
        s = C @ self.loglik.T + self.logprior
        return s[:, 1] - s[:, 0]

    def predict(self, C):
        return (self.decision(C) > 0).astype(int)


# ----------------------------------------------------------------------------
# 4. Algorithm 2: L2-regularised Logistic Regression (mini-batch gradient descent
#    with momentum)
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegression:
    def __init__(self, lr=0.5, lam=1e-4, epochs=40, batch=64, seed=0):
        self.lr, self.lam, self.epochs, self.batch, self.seed = lr, lam, epochs, batch, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        vw, vb, mom = np.zeros(d), 0.0, 0.9
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            lr = self.lr / (1 + 0.05 * ep)                  # learning-rate decay
            for s in range(0, n, self.batch):
                bi = idx[s: s + self.batch]
                err = sigmoid(X[bi] @ w + b) - y[bi]
                gw = X[bi].T @ err / len(bi) + self.lam * w   # L2 penalty on w only
                gb = err.mean()
                vw, vb = mom * vw - lr * gw, mom * vb - lr * gb
                w, b = w + vw, b + vb
        self.w, self.b = w, b
        return self

    def decision(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision(X) > 0).astype(int)


# ----------------------------------------------------------------------------
# 5. Metrics, splitting, tuning
# ----------------------------------------------------------------------------
def metrics(y, p):
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0)); tn = np.sum((y == 0) & (p == 0))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def kfold_indices(y, k, seed=0):
    """Stratified k-fold split written from scratch."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(f) for f in folds]


def cross_val(make_model, Xfun, y, k=5):
    """Mean accuracy / F1 over k folds. Xfun(train_idx, val_idx)->(Xtr, Xva)."""
    folds = kfold_indices(y, k)
    accs, f1s = [], []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, Xva = Xfun(tr, va)
        m = make_model().fit(Xtr, y[tr])
        r = metrics(y[va], m.predict(Xva))
        accs.append(r["acc"]); f1s.append(r["f1"])
    return np.mean(accs), np.mean(f1s)


# ----------------------------------------------------------------------------
# 6. Training driver
# ----------------------------------------------------------------------------
def list_files(folder):
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if not f.startswith(".") and f != "cmds"]
    return sorted(out)


def train(data_dir="data", model_file=MODEL_FILE, k=5, holdout=0.2, seed=0):
    spam_files, ham_files = list_files(os.path.join(data_dir, "spam")), list_files(os.path.join(data_dir, "ham"))
    paths = spam_files + ham_files
    y = np.array([1] * len(spam_files) + [0] * len(ham_files))
    print(f"Loaded {len(spam_files)} spam and {len(ham_files)} ham emails")
    toks, dens = load_docs(paths)

    # hold-out test split (never used for tuning) to report final performance
    rng = np.random.RandomState(seed)
    te = np.concatenate([rng.permutation(np.where(y == c)[0])[: int(holdout * np.sum(y == c))] for c in (0, 1)])
    tr = np.setdiff1d(np.arange(len(y)), te)
    ytr, yte = y[tr], y[te]

    def build(idx_fit, idx_apply_list):
        vec = Vectorizer().fit([toks[i] for i in idx_fit], dens[idx_fit])
        outs = []
        for idx in idx_apply_list:
            C = vec.counts([toks[i] for i in idx])
            outs.append((C, vec.tfidf(C, dens[idx])))
        return vec, outs

    # ---- hyper-parameter search with k-fold CV on the training part ----------
    # (vocabulary / idf are computed once on the whole training part for speed; the
    #  labels are never used by them, so there is no label leakage.)
    vec0, ((Ctr, Xtr),) = build(tr, [tr])
    results = []
    print("\nNaive Bayes: k-fold CV over alpha")
    for a in (0.01, 0.03, 0.1, 0.3, 1.0):
        acc, f1 = cross_val(lambda: NaiveBayes(a), lambda t, v: (Ctr[t], Ctr[v]), ytr, k)
        print(f"  alpha={a:<5} acc={acc:.4f} f1={f1:.4f}")
        results.append((acc, f1, "nb", dict(alpha=a)))
    print("\nLogistic Regression: k-fold CV over (lr, lambda)")
    for lr in (0.3, 1.0):
        for lam in (1e-5, 1e-4, 1e-3):
            acc, f1 = cross_val(lambda: LogisticRegression(lr, lam), lambda t, v: (Xtr[t], Xtr[v]), ytr, k)
            print(f"  lr={lr:<4} lambda={lam:<7} acc={acc:.4f} f1={f1:.4f}")
            results.append((acc, f1, "lr", dict(lr=lr, lam=lam)))
    best = max(results, key=lambda r: (r[0], r[1]))
    print(f"\nBest by CV: {best[2]} {best[3]} (acc={best[0]:.4f})")

    # ---- final check on the hold-out set, then refit on ALL data ------------
    def make(kind, hp):
        return NaiveBayes(**hp) if kind == "nb" else LogisticRegression(**hp)

    vec, ((Cte, Xte),) = build(tr, [te])
    model = make(best[2], best[3]).fit(Ctr if best[2] == "nb" else Xtr, ytr)
    r = metrics(yte, model.predict(Cte if best[2] == "nb" else Xte))
    print("Hold-out test:", {k_: (round(float(v), 4) if k_ in ("acc", "prec", "rec", "f1") else int(v)) for k_, v in r.items()})

    allidx = np.arange(len(y))
    vec, ((Call, Xall),) = build(allidx, [allidx])
    model = make(best[2], best[3]).fit(Call if best[2] == "nb" else Xall, y)
    with open(model_file, "wb") as f:
        pickle.dump(dict(vec=vec, model=model, kind=best[2]), f)
    print(f"Saved final model (trained on all {len(y)} emails) to {model_file}")


# ----------------------------------------------------------------------------
# 7. Prediction on the ./test folder
# ----------------------------------------------------------------------------
def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def predict(test_dir="test", model_file=MODEL_FILE, out_file="predictions.txt"):
    """Read every file in test_dir; return {filename: 1 (spam) / 0 (ham)}."""
    with open(model_file, "rb") as f:
        M = pickle.load(f)
    vec, model, kind = M["vec"], M["model"], M["kind"]
    names = sorted([n for n in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, n))], key=_natural_key)
    toks, dens = load_docs([os.path.join(test_dir, n) for n in names])
    C = vec.counts(toks)
    pred = model.predict(C if kind == "nb" else vec.tfidf(C, dens))
    result = {n: int(p) for n, p in zip(names, pred)}
    with open(out_file, "w") as f:
        for n, p in result.items():
            f.write(f"{n} {p}\n")
            print(n, p)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--test", default="test")
    a = ap.parse_args()
    train(a.data) if a.mode == "train" else predict(a.test)