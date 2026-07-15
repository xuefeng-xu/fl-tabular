import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    figsize = (2, 2)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    y = np.concatenate([np.ones(50), np.zeros(50)])
    shift = np.array([[2.5, 2.5]])
    x1 = x[:50, :] - shift
    x2 = x[50:, :] + shift

    clf = LogisticRegression(random_state=0)
    clf.fit(x, y)
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T

    fig1, ax1 = plt.subplots(figsize=figsize, layout="constrained")
    ax1.scatter(x1[:, 0], x1[:, 1], s=12, marker="x", label="A")
    ax1.scatter(x2[:, 0], x2[:, 1], s=12, marker="o", label="B")
    ax1.legend(loc="lower right")
    c = -b / w2
    m = -w1 / w2

    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    ax1.plot(xd, yd, "k", lw=1, ls="--")

    s1 = StandardScaler()
    s2 = StandardScaler()
    x1p = s1.fit_transform(x1)
    x2p = s2.fit_transform(x2)

    fig2, ax2 = plt.subplots(figsize=figsize, layout="constrained")
    ax2.scatter(x1p[:, 0], x1p[:, 1], s=12, marker="x", label="A")
    ax2.scatter(x2p[:, 0], x2p[:, 1], s=12, marker="o", label="B")
    ax2.legend(loc="lower right")

    img_path = Path(__file__).parent / "img" / "demo"
    img_path.mkdir(parents=True, exist_ok=True)
    fig1.savefig(img_path / "linearly-separable.pdf")
    fig2.savefig(img_path / "improper-scaling.pdf")
    plt.show()
