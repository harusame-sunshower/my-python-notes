# fix_seed()

**目的**  
seed値固定＆決定的アルゴリズムを使うときのプログラム。

---
**仕様**

- os、python、numpyの乱数を固定
- cudnnの最適アルゴリズム使用の無効化  
torch.backends.cudnn.benchmarkが最適なアルゴリズムを使用する機能。
- cudaの決定的アルゴリズムの使用

---
