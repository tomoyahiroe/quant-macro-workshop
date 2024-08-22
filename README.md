# 定量的マクロ経済学と数値計算(勉強会)

勉強会で書いたコードと Codon を使った計算の例をまとめています。

## ディレクトリ構成

```
.
├── README.md
├── requirement.txt
└── src
    ├── ch2
    │   ├── grid_search.ipynb
    │   └── grid_search.py
    └── codon
        ├── fib.py
        └── grid_search.py
```
(2024/08/23現在)

## 環境構築

### Python のインストール

1. Python 自体のバージョン管理ツールである pyenv のインストール

mac ユーザは、Homebrew を使って pyenv をインストールします。

```{terminal}
brew install pyenv
```

2. Pyenv を使用して Python のインストール

3.10.5 以上が好ましいです。私のバージョンは python 3.12.4 です。

```{terminal}
pyenv install 3.12.4
pyenv global 3.12.4
```

### 仮想環境の構築

**macOS**
```{terminal}
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```{terminal}
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Codon の使い方

### Coden のインストール

0. `.sh`ファイルを実行できる環境を用意する

WindowsOSの場合は, コマンドプロンプトで `.sh`ファイルを実行できません. Cygwinをインストールしたり, Git Bash をインストールする方法が考えられます. 

1. `install.sh` を実行

```{terminal}
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

### Codon での '.py'ファイルの実行方法

```{terminal}
codon run --release filename.py
```

### Codon を使用する際の注意点

pythonで実装されているライブラリを使用する場合、次のように import します。

```{python}
from python import numpy
```

### 計測

**フィボナッチ数列の n 番目の数を計算するコード** (`src/codon/fib.py`)

およそ、72倍の速さで計算できる。

```{terminal}
>> codon run -release fib.py
Computed fib(40) = 102334155 in 0.277365 seconds.
>> python3 fib.py
Computed fib(40) = 102334155 in 19.99923300743103 seconds.
```

```{python}
>>> 19.99923300743103 / 0.277365
72.10438594426489
```

**gird search の計算時間の比較** (`src/codon/grid_search.py`)

およそ、9倍の速さで計算できる。

```{python}
>> python src/codon/grid_search.py 
Computed policy function =
[0.025, 0.07500000000000001, 0.1, 0.15, 0.175, 0.22499999999999998, 0.24999999999999997, 0.27499999999999997, 0.325, 0.35000000000000003]
in 0.00010895729064941406 seconds.

>> codon run --release src/codon/grid_search.py
Computed policy function =
[0.025, 0.075, 0.1, 0.15, 0.175, 0.225, 0.25, 0.275, 0.325, 0.35]
in 1.21593e-05 seconds.
```

```{python}
>>> 0.00010895729064941406 / 1.21593e-05
8.960819343992998
```