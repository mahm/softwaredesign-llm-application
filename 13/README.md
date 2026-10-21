# 第13回 Google Colabサンプル

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応に合わせ、Google Colabではランタイムバージョン「2026.07」を使用してください。
Python 3.12.13で動作します。紙面に記載したPython 3.10.7とは異なります。

[サンプルNotebook](https://colab.research.google.com/drive/1xYAkuD75rbNNiNSKcQmmRb1N06hL_jFn?usp=sharing)を開き、「ランタイム」→「ランタイムのタイプを変更」から、ランタイムバージョンを「2026.07」に設定します。
Notebook先頭のパッケージ導入セルを実行する前に、次のコードセルを追加して実行してください。

```python
%pip install --upgrade pip==26.2.2 certifi==2026.10.19

import os
from pathlib import Path
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["SSL_CERT_DIR"] = str(Path(certifi.where()).parent)
```

続けて、Notebookの各セルを先頭から実行してください。
