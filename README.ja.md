# Black-Scholes オプションプライサー

`Python 3.11+` · `numpy / scipy` · `pytest` · `black + ruff` · `MIT`

プロダクション品質のヨーロピアン・オプション価格計算ライブラリです。Black-Scholes-Merton
の解析解、全 Greeks、Newton-Raphson 法によるインプライド・ボラティリティの逆算、そして
解析解を独立に検証するモンテカルロ・エンジンを備えています。

> English version: [README.md](README.md)

---

## なぜこのプロジェクトか（クオンツ選考での意義）

オプション価格計算は、確率解析・数値計算・規律あるソフトウェア工学の交点に位置し、まさに
クオンツ／リスク部門が求めるスキルセットそのものです。本リポジトリは小さくも完結した構成
を意図しています。解析解を導出し、それを2通りの独立した方法（数値微分による Greeks 検証
**および** モンテカルロの収束）で正しさを示し、素朴な実装が破綻するエッジケース（満期ゼロ、
ボラゼロ、ディープ ITM/OTM のインプライド・ボラ）を処理し、テスト・型ヒント・lint・
ドキュメントを同梱しています。単に Black-Scholes 式が書けることではなく、他者がリスク
システムで信頼して使える数値コンポーネントを「出荷できる」ことを示すのが狙いです。

---

## 1分で分かる理論

Black-Scholes-Merton モデルでは、リスク中立測度の下で原資産は幾何ブラウン運動に従います。

```
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

ヨーロピアン・コール／プット価格は次の閉形式を持ちます。

```
Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
```

ここで `N` は標準正規分布の累積分布関数です。**Greeks** はこの価格の解析的偏微分です
（Delta `∂V/∂S`、Gamma `∂²V/∂S²`、Vega `∂V/∂σ`、Theta `-∂V/∂t`、Rho `∂V/∂r`）。
**インプライド・ボラティリティ** は価格の写像を逆に解き、市場プレミアムを再現する `sigma`
を求めます。モデルは単一のフラットなボラを仮定するため、満期価格を厳密な対数正規分布から
直接シミュレートしてペイオフの平均を割り引くモンテカルロ推定量は、同じ値に収束しなければ
ならず、独立した検証手段となります。

---

## ディレクトリ構成

```
quant-portfolio/
├── src/
│   └── pricing/
│       ├── black_scholes.py   # 価格・Greeks・インプライド・ボラ（ベクトル化）
│       └── monte_carlo.py     # GBM モンテカルロ + 標準誤差 / 信頼区間
├── tests/
│   └── test_black_scholes.py  # パリティ・MC vs 解析解・数値微分 Greeks・エッジケース
├── notebooks/
│   └── black_scholes_demo.ipynb
├── requirements.txt
└── README.md / README.ja.md
```

---

## インストール

```bash
git clone https://github.com/<your-username>/quant-portfolio.git
cd quant-portfolio

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

---

## 使い方

```python
from src.pricing.black_scholes import bs_price, greeks, implied_volatility, OptionType
from src.pricing.monte_carlo import mc_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# 解析解による価格
price = bs_price(S, K, T, r, sigma, OptionType.CALL)
print(f"Call price: {price:.4f}")        # Call price: 10.4506

# リスク・レポート一式（価格 + 全 Greeks）
report = greeks(S, K, T, r, sigma, OptionType.CALL)
print(report)
# {'price': 10.4506, 'delta': 0.6368, 'gamma': 0.0188,
#  'vega': 37.524, 'theta': -0.0176, 'rho': 53.232}

# 市場価格からインプライド・ボラを逆算
iv = implied_volatility(price, S, K, T, r, OptionType.CALL)
print(f"Implied vol: {iv:.4f}")          # Implied vol: 0.2000

# モンテカルロによる独立検証（seed で再現性を担保）
mc = mc_price(S, K, T, r, sigma, OptionType.CALL, n_paths=200_000, seed=42)
print(f"MC price: {mc.price:.4f} ± {mc.std_error:.4f} (95% CI {mc.ci_95})")
```

価格・Greeks の各関数は**ベクトル化**されています。任意の引数に NumPy 配列を渡せば、
グリッド全体を一括で計算できます。

```python
import numpy as np
spots = np.linspace(80, 120, 5)
bs_price(spots, K, T, r, sigma, OptionType.CALL)   # -> 5 個の価格の配列
```

---

## テストの実行

```bash
pytest -q
```

テストは正しさの4本柱を網羅しています。

- **プット・コール・パリティ** — モデル非依存の無裁定恒等式 `C - P = S e^{-qT} - K e^{-rT}`。
- **モンテカルロ vs 解析解** — 数標準誤差以内での一致。
- **Greeks vs 数値微分** — 解析微分がバンプ価格と一致すること。
- **エッジケース** — 満期ゼロ／ボラゼロで本質的価値を返す（NaN を出さない）、
  負の入力は弾く。

### コード品質

```bash
black src tests      # フォーマット
ruff check src tests # lint
```

コードベースは **black** でフォーマットされ、**ruff** の警告ゼロで通過します。

---

## 結果サンプル

`notebooks/black_scholes_demo.ipynb` を実行すると以下の図を再現できます。

1. **価格 vs 原資産価格** — 本質的価値ペイオフに対するコール／プットのプレミアム。
2. **Greeks vs 原資産価格** — Delta・Gamma・Vega・Theta・Rho のマネーネス別の挙動。
3. **モンテカルロの収束** — パス数増加に伴い MC 推定値と縮小する 95% 信頼区間が
   解析解へ収束する様子。
4. **ボラティリティ・スマイル** — 作為的なスキューで価格付けし、インプライド・ボラ
   ソルバーでストライク別に逆算して再現。

---

## ライセンス

MIT。
