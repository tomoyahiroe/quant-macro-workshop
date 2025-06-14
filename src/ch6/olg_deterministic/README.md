# モデル

## 個人の動学的最適化問題

個人は以下の最適化問題を $j=1$期に解き切って、死ぬまでの消費と貯蓄を決定する。

$$
\max_{\{c_j, a_{j+1}\}^J_{j=1}}\sum_{j=1}^J \beta^{j-1}u(c_j)\\\begin{aligned}&subject \;to\\ &c_j + a_{j+1}=\left\{\,\begin{aligned} &(1+r)a_j+(1-\tau)\theta_j w,&for\ j = 1,\cdots,j^{R}-1\\&(1+r)a_j + p,&for\ j=j^R,\cdots,J\ \ \ \ \ \  \end{aligned}\right. \\&a_{J+1}\ge0\\&a_1=0\end{aligned}
$$

- 各期に年齢 $j = 1,\cdots,J$の個人が $\mu_j$人存在する
    - $\mu_j = 1$
- $c_j$は各期の消費
- $a_j$は $j$期の期初に個人が持っている資産
- 個人は、
    - $j^R-1$期までは働く（勤労期）
        - 労働所得から定率で年金保険料 $\tau$を支払う
    - $j^R$期以降は公的年金 $p$ をもらって生活をする（引退期）
- 労働生産性 $\theta_j$は各年代において確定的（確率的に変動しない）
- 完全競争市場を仮定しているので、個人にとって価格 $r,w$は所与
- 個人の初期の資産 $a_1$はゼロ
- 個人は負債を残して死ぬことはできない $a_{j+1}\ge0$

上記の最適化問題の一階条件（First Order Condition; FOC）を求めて式をまとめると、次のオイラー方程式と呼ばれる、2期間の消費の比率を表す式が導出できる。

$$
u^\prime(c_j)=\beta(1+r)u^\prime(c_{j+1})
$$

ここで、相対的リスク回避度一定（constant relative risk aversion; CRRA）型効用関数を仮定する。

$$
u(c) = \frac{c^{1-\gamma}}{1-\gamma}
$$

このCRRA型効用関数をオイラー方程式に代入し式変形をすると、次のように、個人の消費成長率 $g_c$が導出できる。

$$
\frac{c_{j+1}}{c_j} = [\beta(1+r)]^{\frac{1}{\gamma}}\equiv1+g_c
$$

これにより、任意の期の消費 $c_j$を $c_1$の関数として表すことができる。

$$
c_{j} = (1+g_c)^{j-1}c_1
$$

このように、第一期の消費量が解析的に求まるため、消費成長率 $g_c$を用いれば全年代での消費水準が求まる。また、予算制約式を用いれば、全年代の資産水準も求められる。

家計は企業に資本ストックを貸し出し、一単位につきレンタル料 $r$を受け取る。資本が減耗した部分は、企業に補填してもらう。

資本ストック供給 $K^s$、労働供給$L^s$はそれぞれ次のように計算できる。

$$
\left\{\begin{aligned}K^s&=\sum_{j=1}^{J}a_j \mu_j\\L^s&=\sum_{j=1}^{J^R-1}\theta_j\mu_j\end{aligned}\right.
$$

## 企業の利潤最大化問題

企業は、毎期 $r,w$を所与として、利潤を最大化する生産要素の需要量 $K^d,L^d$を決定する。また、個人から供給された資本の減耗分は、企業が補填する。

$$
\max_{\{K^d,L^d\}} F(K^d,L^d)-\delta K^d- rK^d -wL^d
$$

一階条件は次の通り。

$$
\left\{
\begin{aligned}r &= \frac{\partial F(K^d,L^d)}{\partial K^d} - \delta\\w &= \frac{\partial F(K^d,L^d)}{\partial L^d} \end{aligned}
\right.
$$

## 政府の意思決定

政府は個人の労働所得に、定率 $\tau$の年金保険料（労働所得税）を課す。一方で引退期には、一括で毎期 $p$ の公的年金を支給する。

政府は国債や貨幣を発行することはせず、均衡財政の制約の下で分配を行う。よって、政府の予算制約式は次のように定義される。

$$
\sum_{j=1}^{j^R -1} \tau \theta_j w \mu_j = \sum_{j=j^R}^{J}p\mu_j
$$

ここで、政府は勤労世代の平均賃金の半分を年金として支給するとする。この年金の平均所得代替率を $\psi$で表し、半分の場合は、 $\psi = 0.5$となる。例えば、このモデルの経済において、平均賃金が年収400万円だったら、引退期に該当する人には毎年 $p = 200$万円支給される。

つまり、先に所得代替率$\psi$が外生的に設定され、それを実現するように、内生的に年金保険料率$\tau$と、年金支給額$p$が決定される。

政府は、この方針を達成できるだけの年金保険料率 $\tau$を計算し、個人に課す。

これを式で表すと次のようになる。

- 平均賃金

$$
\bar{w} = \frac{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}{\sum_{j=1}^{j^R -1}\mu_j}
$$

- 公的年金

$$
p = \psi\bar{w}
$$

- 上記の $p$ を達成するために必要な保険料率

$$
\tau\sum_{j=1}^{j^R -1}w\theta_j \mu_j = p\sum_{j=j^R}^{J}\mu_j
$$

$$
\tau = \dfrac{\psi \bar{w}\sum_{j=j^R}^{J}\mu_j}{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}
$$

平均賃金$\bar{w}$を代入すると

$p,\tau$ともに人口構造と賃金（企業の需要変化のみで変化）に依存しており、家計の行動変化とは無関係に設定される。

## 均衡の定義

- 家計が効用最大化をしている → 政策関数（最適な資産パス？）
    
    $$
    u^\prime(c_j)=\beta(1+r)u^\prime(c_{j+1}),\ j = 1,\dots,J-1
    $$
    
    $$
    a_{J+1} = 0
    $$
    
- 企業が利潤最大化をしている

$$
\left\{
\begin{aligned}r &= \frac{\partial F(K^d,L^d)}{\partial K^d} - \delta\\w &= \frac{\partial F(K^d,L^d)}{\partial L^d} \end{aligned}
\right.
$$

- 政府の予算制約式を満たしている
    
    $$
    \bar{w} = \frac{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}{\sum_{j=1}^{j^R -1}\mu_j}
    $$
    
    $$
    p = \psi\bar{w}
    $$
    

$$
\sum_{j=1}^{j^R -1} \tau \theta_j w \mu_j = \sum_{j=j^R}^{J}p\mu_j
$$

- Market Clearing Conditionを満たす
    - 資本ストック市場
        
        $$
        K_s = K_d
        $$
        
    - 労働市場
        
        $$
        L_s = L_d
        $$
        
    - 財市場
        
        $$
        Y = \delta K + C
        $$
        

## 均衡の性質

オイラー方程式によって、異時点間の消費の比率を求めることができた。次に、予算制約式を用いて、消費の水準（消費量）を求めたい。

異時点間の予算制約式を一つの式にまとめていくと、次の式が得られる。

[異時点間の予算制約（政府あり）の導出](https://www.notion.so/1db32ebc809980f9965be0b6d3d04dbd?pvs=21)

$$
\small\sum_{j=1}^{J}\frac{c_j}{(1+r)^{j-1}}=(1-\tau)\sum_{j=1}^{j^R -1}\frac{w\theta_j}{(1+r)^{j-1}}+\sum_{j=j^R}^{J}\frac{p}{(1+r)^{j-1}} \;\dots\;J_{J}
$$

先ほどの $c_{j} = (1+g_c)^{j-1}c_1$を代入する。

$$
\small c_1\sum_{j=1}^{J}\frac{(1+g_c)^{j-1}}{(1+r)^{j-1}}=(1-\tau)\sum_{j=1}^{j^R -1}\frac{w\theta_j}{(1+r)^{j-1}}+\sum_{j=j^R}^{J}\frac{p}{(1+r)^{j-1}} \;\dots\;J_{J}
$$

$$
\sum_{j=1}^{J}\frac{(1+g_c)^{j-1}c_1}{(1+r)^{j-1}}=Y\\\Rightarrow c_1=\frac{Y}{\sum_{j=1}^{J}\frac{(1+g_c)^{j-1}}{(1+r)^{j-1}}}
$$

[確認事項]

$\sum_{j=1}^{J}\frac{c_j}{(1+r)^{j-1}}$と $\sum_{j=1}^J\mu_j c_j$ が一致するかどうかを確認する

## アーカイブ

数値計算上は、金利 $r$（北尾先生のコードでは資産パス）を所与として、 $K_s = K_d$が計算される。

# カリブレーション

- 割引因子$\beta = 0.98$
- 効用関数のパラメータ$\gamma = 1.0$
- 生産関数のパラメータ$\alpha = 0.4$
- 資本減耗率$\delta = 0.08$
- 年金の平均所得代替率$\psi = 0.5$
- 各年代の人口$\mu_j =1$
- 各年代の労働生産性$\theta_j = 1$
- モデルの期間$J = 65$
- 引退期の始まり$j^R = 46$
- 初期の資産$a_1 = 0$

# アルゴリズム

## 教科書の記述から考えたアルゴリズム

1. モデルのパラメータとアルゴリズムのパラメータを設定
    - 割引因子$\beta = 0.98$
    - 相対的リスク回避度$\gamma = 1.0$
    - 資本の所得分配率$\alpha = 0.4$
    - 資本減耗率$\delta = 0.08$
    - 年金の平均所得代替率$\psi = 0.5$
    - モデルの期間$J = 65$
    - 勤労期初期の年齢 $Jw = 20 (work)$
    - 引退期の初期 $Jr = 46(retire)$
    - 初期の資産$a_1 = 0$
    - 収束の基準値 $tol = 1.0 \times 10^{-5} (= 1e^{-5})$
    - 各年代の人口 $\forall j, \mu_j =1$
    - 各年代の労働生産性 $\forall j, \theta_j = 1$
2. 均衡に関連するクラス $Equilibrium$ を定義
    - 資産パス（資産の政策関数？） $aprime\_path$
    - 消費パス（消費の政策関数？） $c\_path$
    - 総資本需要 $K_d$
    - 総資本供給 $K_s$
    - 総労働需要 $L_d$
    - 総労働供給 $L_s$
    - 総消費 $C$
    - 生産量 $Y$
    - 均衡金利 $r^\star$
    - 均衡賃金 $w^\star$
    - 均衡資本ストック $K^\star$
    - 均衡労働量 $L^\star$
    - 年金保険料率　$\tau$
    - 年金給付額 $p$
    - 消費の成長率 $gc$
3. 以下の処理を収束の基準を満たすまで繰り返す
    1.  $K_d$を当て推量する
    2. $K_d$ と企業の利潤最大化問題、個人の総労働供給 $L_s$から、価格 $w,r$を求める
    （と $Y$も求める）
        - 労働需要
            
            $$
            L^{d} = L^{s} = \sum_{j=1}^{J} \theta_j\mu_j
            $$
            
        - 均衡金利
            
            $$
            r = \alpha \left(\frac{K_d}{L_d}\right)^{\alpha -1} -\delta
            $$
            
        - 均衡賃金
            
            $$
            w = (1-\alpha)\left(\frac{K_d}{L_d}\right)^{\alpha}
            $$
            
        - 生産量
            
            $$
            Y = K_d^\alpha L_s^{1-\alpha}
            $$
            
    3. 保険料率 $\tau$ と公的年金の支給額 $p$を求める
        - 平均賃金
            
            $$
            \bar{w} = \frac{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}{\sum_{j=1}^{j^R -1}\mu_j}
            $$
            
        - 公的年金
            
            $$
            p = \psi\bar{w}
            $$
            
        - 上記の $p$ を達成するために必要な保険料率
            
            $$
            \tau = \dfrac{\psi \bar{w}\sum_{j=j^R}^{J}\mu_j}{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}
            $$
            
    4. 個人の消費と資産の政策関数を求める
        - 初期の消費
        （分母は等比数列の和で計算可能）
            
            $$
            c_1=\frac{Y}{\sum_{j=1}^{J}\frac{(1+g_c)^{j-1}}{(1+r)^{j-1}}}
            $$
            
            $$
            \sum_{j=1}^{J}\frac{(1+g_c)^{j-1}}{(1+r)^{j-1}} = \frac{1 - (\frac{1+g_c}{1+r})^J}{1-(\frac{1+g_c}{1+r})}
            $$
            
        - 2期以降の消費
            
            $$
            c_{j} = (1+g_c)^{j-1}c_1
            $$
            
        - 上記を同じ配列に格納し、**c_path** と命名する
        - **c_path**と予算制約式を用いて順々に計算し、**aprime_path**を作成する
    5. 資産の政策関数から総資本供給 $K_s$を計算する
        
        $$
        K_s = \sum_{j=1}^{J} a_j \mu_j
        $$
        
4. 所与の均衡金利から計算された総資本需要と総資本供給の差分を取り、収束の基準より小さければ、均衡条件を満たしたとみなす。収束の基準より大きければ、所与とする資本ストック供給 $K_s$の値を更新し、ステップ３を繰り返す

## 北尾先生のコードのアルゴリズム

### めも

[chapter6/Julia/6_2_OLG_Deterministic/olg.jl at master · quant-macro-book/chapter6](https://github.com/quant-macro-book/chapter6/blob/master/Julia/6_2_OLG_Deterministic/olg.jl)

- 資産パスをオイラー方程式誤差がゼロになるように計算
    - $a_1 = a_{J+1} = 0$は常に満たすようにする
- 資産パスから、 $K_s(=K_d)$を計算し、この値を用いて、 $r,w$も計算
- そうしたら、 $\tau$も計算できて $p$も計算できる
- これらの値を使って予算制約式を満たすように消費パスを計算
- イテレーションすらしてない求根法を使ってはいるか…

### アルゴリズム

1. カリブレーション
    - 割引因子$\beta = 0.98$
    - 相対的リスク回避度$\gamma = 1.0$
    - 資本の所得分配率$\alpha = 0.4$
    - 資本減耗率$\delta = 0.08$
    - 年金の平均所得代替率$\psi = 0.5$
    - モデルの期間$J = 65$
    - 勤労期初期の年齢 $Jw = 20 (work)$
    - 引退期の初期 $Jr = 46(retire)$
    - 初期の資産$a_1 = 0$
    - 収束の基準値 $tol = 1.0 \times 10^{-5} (= 1e^{-5})$
    - 各年代の人口 $\forall j, \mu_j =1$
    - 各年代の労働生産性 $\forall j, \theta_j = 1$
2. 資産パスの初期値を設定
    1. 初期の資産 $a_1$と $J+1$期の資産 $a_{J+1}$は $0$
    2. １期ごとに $0.01$ずつ単調に増加するようなguess
3. 資産パスを初期値からずらしながら以下のa〜fを計算して、 $EulerError$を求め、それが最小になるような資産パスを求める
    1. 資産パスの合計から総資本供給（と総資本需要）を計算
        
        $$
        K_d =K_s = \sum_{j = 1}^{J}a_{j}
        $$
        
    2. 均衡金利を計算
        
        $$
        r = \alpha \left(\frac{K_d}{L_d}\right)^{\alpha-1} - \delta
        $$
        
    3. 均衡賃金を計算
        
        $$
        w = (1-\alpha)\left(\frac{K_d}{L_d}\right)^{\alpha}
        $$
        
    4. 年金支給額を計算
        
        $$
        \bar{w} = \frac{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}{\sum_{j=1}^{j^R -1}\mu_j}
        $$
        
        $$
        p = \psi \bar{w}
        $$
        
    5. 上記の $p$ を達成するために必要な保険料率を計算
        
        $$
        \tau = \dfrac{\psi \bar{w}\sum_{j=j^R}^{J}\mu_j}{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}
        $$
        
    6. 各期のオイラー方程式誤差を計算（繰り返し $j = 1,\cdots, J-1$）
        
        $$
        c_j = (1+r)a_j + (1-\tau)\theta_j w + p
        \\c_{j+1} = (1+r)a_{j+1} + (1-\tau)\theta_{j+1} w + p
        \\EulerError_j = u^\prime(c_j) - \beta(1+r)u^\prime(c_{j+1})
        $$
        
    7.  $EulerError$を出力
4. 最適な資産パスの総和から資本供給 $K_s$を求める
    
    $$
    K_d =K_s = \sum_{j = 1}^{J}a_{j}
    $$
    
5. 次の式から、均衡金利と均衡賃金と生産量を求める
    
    $$
    K_d = K_s
    $$
    
    $$
    L^{d} = L^{s} = \sum_{j=1}^{J} \theta_j\mu_j
    $$
    
    $$
    r = \alpha \left(\frac{K_d}{L_d}\right)^{\alpha-1} - \delta
    $$
    
    $$
    w = (1-\alpha)\left(\frac{K_d}{L_d}\right)^{\alpha}
    $$
    
    $$
    Y = F(K^{d}, L^{d})
    $$
    
6. 保険料率 $\tau$ と公的年金の支給額 $p$を求める
    - 平均賃金
        
        $$
        \bar{w} = \frac{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}{\sum_{j=1}^{j^R -1}\mu_j}
        $$
        
    - 公的年金
        
        $$
        p = \psi\bar{w}
        $$
        
    - 上記の $p$ を達成するために必要な保険料率
        
        $$
        \tau = \dfrac{\psi \bar{w}\sum_{j=j^R}^{J}\mu_j}{\sum_{j=1}^{j^R -1}w\theta_j \mu_j}
        $$
        
7. 予算制約式から消費のパスを計算する（北尾先生のコードでは、 $\tau$と $p$は配列になっていて、勤労期は $p$は０,引退期は $\tau$が０となっている）

$$
c_j = w^\star\theta_j(1-\tau) + (1+r^\star)a_j + p - a_{j+1}
$$

# コード

### 北尾先生のコード、求根法

[olg_deterministic_kitao.ipynb](attachment:f3c6fbcd-85fb-491c-8240-6daa6066f553:olg_deterministic_kitao.ipynb)

# シナリオ分析

[シナリオ分析](https://www.notion.so/20c32ebc80998040bc60fc2f8517caf3?pvs=21)