# [PWS Cup 2025](https://www.iwsec.org/pws/2025/cup25.html) 用サンプルデータ・サンプルコード共有ディレクトリ

※ サンプルデータやサンプルコードは随時追加や更新されますので、更新日時を確認して最新版を利用するようにしてください。予備戦開始（8/20）後に追加または更新した場合はチーム代表者に連絡します。
## 重要な変更
- (8/29, 13:01) ↓の正規化方法が不適切で誤差が0になるべき入出力で誤差が0.5になる問題を修正 [PR#21](https://github.com/pwscup/pwscup2025-scripts/pull/21)
- (8/28, 15:16) evaluation/LR_asthma_diff.py のcoefに関する採点で正規化が行われていなかったため、正規化処理を追加 [PR#19](https://github.com/pwscup/pwscup2025-scripts/pull/19)

## ![PWS Cup 2025 の基本的な流れ](PWSCUP2025flow.pdf)

<img width="1050" height="567" alt="image" src="https://github.com/user-attachments/assets/859c85d5-c340-488e-bf68-3a58edc2e981" />

## Installation

```bash
pip install -r requirements.txt
```

