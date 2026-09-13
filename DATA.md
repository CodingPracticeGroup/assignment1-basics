# 作业 1 数据集准备记录（TinyStories / OpenWebText）

本文记录本仓库的数据**从哪来、怎么下载、怎么校验**，便于复现。
（数据集的“知识点/概念”部分——例如 tokenizer 训练集与预训练集的分布匹配——见概念仓库
`Stanford-CS336/Spring2026/assignments/assignment1/01_assignment_1.md`。）

## 1. 准备结果

已下载并解压到本仓库 `data/`（该目录已被 `.gitignore` 忽略，不会提交）。

| 文件 | 字节数 | 人类可读 | 用途 |
| :-- | --: | --: | :-- |
| `data/TinyStoriesV2-GPT4-train.txt` | 2,227,753,162 | 2.1G | §2 训练 BPE；§7.2 TinyStories 训练 / 消融 |
| `data/TinyStoriesV2-GPT4-valid.txt` | 22,502,601 | 22M | 验证 |
| `data/owt_train.txt` | 11,920,511,059 | 12G | §7.4 OpenWebText 训练、leaderboard |
| `data/owt_valid.txt` | 289,998,753 | 277M | OpenWebText 验证 |

合计约 **14G**。

## 2. 数据来源

两者都是 Hugging Face 上的公开**单文件纯文本**数据集：

- **TinyStories** (R. Eldan et al., 2023)：<https://huggingface.co/datasets/roneneldan/TinyStories>
- **OpenWebText sample** (A. Gokaslan et al., 2019；课程提供的子样本)：
  <https://huggingface.co/datasets/stanford-cs336/owt-sample>

## 3. 下载命令

```sh
mkdir -p data && cd data

# TinyStories
wget -c --tries=5 --timeout=60 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -c --tries=5 --timeout=60 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText（先下 .gz 再解压）
wget -c --tries=5 --timeout=60 \
  https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz
wget -c --tries=5 --timeout=60 \
  https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

cd ..
```

`wget -c` 支持断点续传：中断后重跑同一条命令即可。
下载前可先看大小：`curl -sIL <url> | grep -i content-length`。

下载时 `.gz` 的原始大小：`owt_train.txt.gz` 4,591,240,837 字节；
`owt_valid.txt.gz` 111,785,382 字节。

## 4. 校验

```sh
cd data
ls -l
head -c 160 TinyStoriesV2-GPT4-train.txt   # 抽查内容
wc -l TinyStoriesV2-GPT4-valid.txt          # 157831
sha256sum *.txt
```

本次实测 SHA256：

```
6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443  TinyStoriesV2-GPT4-train.txt
6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d  TinyStoriesV2-GPT4-valid.txt
bbeb7f291a981ecfd5cf44b84d0f654b9e96c53dff99f2556b7d2cccaf8c1918  owt_train.txt
2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660  owt_valid.txt
```

行数：TinyStories valid = 157,831；OpenWebText valid = 2,301,018。
抽查 `head` 内容正常（TinyStories 为童话文本，OWT 为网页文本）。

## 5. 提交与忽略

- `.gitignore` 已加入 `data/` 与 `wandb/`，14G 语料不会误入 git。
- `make_submission.sh` 使用 `-x '*.txt'` 排除文本文件，因此数据集不会进提交 zip。
- 训练产物（`wandb/`、`*.pt` / `*.pth` checkpoint）同样建议不要提交。

## 6. 低资源 / 快速迭代建议

- handout §2.5：在 TinyStories 上训练 BPE 约 ≤30 分钟（CPU）、≤30GB RAM。
- 想先跑通流程，可只用子集，例如：
  ```sh
  head -c 200000000 data/TinyStoriesV2-GPT4-train.txt > data/tiny_train_200MB.txt
  ```
- §7.4 的 OpenWebText 训练量较大；资源有限时可先在 TinyStories 上完成 §7.2 / §7.3 的实验。

## 7. 本次实际执行记录

- 磁盘（下载前）：`/` 剩余约 1.1T；下载后 `data/` 约 14G。
- 顺序：TinyStories train → valid → `owt_train.txt.gz` → `owt_valid.txt.gz` → `gunzip`。
- 命令均在仓库根目录 `assignment1-basics/` 下执行，数据落在 `assignment1-basics/data/`。
