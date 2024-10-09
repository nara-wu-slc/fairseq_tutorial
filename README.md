# Lab4SLC fairseq チュートリアル

```
# 作業ディレクトリに異動 (/path/to/dir は好きなところで良い、相対パスでもOK)
cd /path/to/dir

# 本チュートリアル用のディレクトリを作成して移動
mkdir -p fairseq_tutorial
cd fairseq_tutorial

# 本チュートリアル用のPython仮想環境を /path/to/dir/fairseq_tutorial/.venv に作成
python3 -m venv .venv

# 仮想環境を有効化
source .venv/bin/activate
```

とりあえず動かすだけならPyPI標準のfairseqをpipで入れる（最新版ではない）
```
# pipでfairseqをインストール
pip3 install fairseq
```

```
# sentencepieceをインストール
pip3 install sentencepiece

# tensorboardXをインストール
pip3 install tensorboardX
```

```
# データ格納用のディレクトリを作成しておく
mkdir -p data spm

# curlでKFTTをダウンロードして展開（そんなに大きくないので圧縮ファイルを手元に残さない方向で）
# data/kftt-data-1.0 にデータが格納される
curl http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz | tar zxf -C data
```

```
# sentencepieceのモデルを学習
python3

# Pythonのプロンプト内で以下を実行
import sentencepiece as spm
spm.SentencePieceTrainer.train(input=['data/kftt-data-1.0/data/orig/kyoto-train.en', 'data/kftt-data-1.0/data/orig/kyoto-train.ja'], model_prefix="spm/spm", bos_id=0, pad_id=1, eos_id=2, unk_id=3)
exit

# fairseq用の辞書ファイルを生成
cut -f1 spm/spm.vocab | tail -n +5 | sed "s/$/ 100/g" > spm/spm.dict

# トークン化したファイルを保存するためのディレクトリを作成
mkdir -p data/kftt-data-1.0/data/spm

# 学習したsentencepieceのモデルでテキストデータをトークン化
python3

# Pythonのプロンプト内で以下を実行
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='spm/spm.model')
for data in ['kyoto-train', 'kyoto-dev', 'kyoto-test']:
    for lang in ['en', 'ja']:
        with open('data/kftt-data-1.0/data/orig/' + data + '.' + lang, 'rt') as rf, open('data/kftt-data-1.0/data/spm/' + data + '.' + lang, 'wt') as wf:
            for line in rf:
                _ = wf.write(' '.join(sp.encode(line.rstrip("\r"), out_type=str))+"\n")

exit
```

```
# fairseq用データ形式の変換（前処理）
fairseq-preprocess \
        --source-lang ja \
        --target-lang en \
        --trainpref data/kftt-data-1.0/data/spm/kyoto-train \
        --validpref data/kftt-data-1.0/data/spm/kyoto-dev \
        --testpref data/kftt-data-1.0/data/spm/kyoto-test \
        --destdir data/kftt-data-1.0/data/spm.bin \
        --joined-dictionary \
        --srcdict spm/spm.dict \
        --bpe sentencepiece

# fairseqモデルの学習
# CUDA_VISIBLE_DEVICES=0 は1枚目のGPUだけを使う、というおまじない
env CUDA_VISIBLE_DEVICES=0 fairseq-train data/kftt-data-1.0/data/spm.bin \
        --fp16 \
        --arch transformer \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 4000 \
        --warmup-init-lr 1e-07 \
        --dropout 0.1 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --save-dir data/kftt-data-1.0/model \
        --log-format tqdm \
        --log-interval 100 \
        --max-tokens 8000 \
        --max-epoch 100 \
        --patience 5 \
        --seed 3921 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok space \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --tensorboard-logdir runs
```