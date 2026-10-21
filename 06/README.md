# 第06回 サンプルコード

## 証明書検証に関する更新（2026年10月22日）

[eMudhraの中間CA侵害](https://emudhra.com/en/post/certificate-security-incident)への対応として、Pythonを紙面に記載した3.10.13から3.11.17へ更新しました。
HTTPS通信に使用するCA証明書も、certifi 2026.10.19へ更新しています。

Python 3.11.17で以下の環境構築手順を実行してください。
依存パッケージの導入後、サンプルを起動するシェルで次の環境変数を設定してください。

```bash
export SSL_CERT_FILE="$(python -m certifi)"
export SSL_CERT_DIR="$(dirname "$SSL_CERT_FILE")"
```

## 環境構築方法

```
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install --upgrade "pip>=26.2.2"
$ python -m pip install -r requirements.txt
```
