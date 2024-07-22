# melodyGPT
A GPT based model that is trained with song chords text.

## Dataset

### melodyGPT-song-chords-text-1 ([huggingface](https://huggingface.co/datasets/lluccardoner/melodyGPT-song-chords-text-1))
The initial dataset is created by aggregating the chords of each song given by the [Chords and Lyrics Dataset](https://www.kaggle.com/datasets/eitanbentora/chords-and-lyrics-dataset). 
You can see in the dataset folder notebooks with the code used to do so.
Also, the special characters that are not chords are analysed.
For the simplicity of this project, we will start with the dataset without any processing.

Import from Huggingface:
```python
from datasets import load_dataset
dataset = load_dataset("lluccardoner/melodyGPT-song-chords-text-1", split="train")
dataset.to_pandas()
```

## Tokenization

We will train a custom tokenizer and compare it to the [original tokenizer](https://tiktokenizer.vercel.app/?model=gpt2) used by GPT-2 of OpenAI ([GitHub](https://github.com/openai/tiktoken)).
The idea is that the tokens learned will be more representative of the chords representations.

Here is a super nice video about how [GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) is done.

### melodyGPT-song-chords-tokenizer-1 ([huggingface](https://huggingface.co/lluccardoner/melodyGPT-song-chords-tokenizer-1))

```python
from transformers import AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
chords_gpt2_tokenizer = AutoTokenizer.from_pretrained("lluccardoner/melodyGPT-song-chords-tokenizer-1")

print(gpt2_tokenizer.vocab_size) # 50257
print(chords_gpt2_tokenizer.vocab_size) # 19972 

example_chords = "Intro: Adim G7/13 Em Bb (4x) G#dim Bm/C F#m Ab|---------------------------------| (Bridge) C G Em7 Asus4"

tokens = gpt2_tokenizer.tokenize(example_chords)
# tokens: ['Int', 'ro', ':', 'ĠAd', 'im', 'ĠG', '7', '/', '13', 'ĠEm', 'ĠB', 'b', 'Ġ(', '4', 'x', ')', 'ĠG', '#', 'dim', 'ĠB', 'm', '/', 'C', 'ĠF', '#', 'm', 'ĠAb', '|', '--------------------------------', '-|', 'Ġ(', 'Bridge', ')', 'ĠC', 'ĠG', 'ĠEm', '7', 'ĠAsus', '4']

new_tokens = chords_gpt2_tokenizer.tokenize(example_chords)
# new_tokens: ['Intro', ':', 'ĠAdim', 'ĠG', '7', '/', '13', 'ĠEm', 'ĠBb', 'Ġ(', '4', 'x', ')', 'ĠG', '#', 'dim', 'ĠBm', '/', 'C', 'ĠF', '#', 'm', 'ĠAb', '|---------------------------------|', 'Ġ(', 'Bridge', ')', 'ĠC', 'ĠG', 'ĠEm', '7', 'ĠAsus', '4']
```

## Model

For the model we will train a GPT2 model of 124M parameters. 
For that we will follow this [lecture](https://www.youtube.com/watch?v=l8pRSuU81PU&list=LL&index=1&t=15s) by Andrej Karpathy.

### GPT2

The pretrained GPT2 model does not perform good on predicting the next chords.

With the given code from the generate script:
```python
# Pachelbel Canon in D chords: "D A Bm F#m G D G A"
chords = "D A Bm F#m"

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2 = GPT.from_pretrained('gpt2')
generate(chords, gpt2_tokenizer, gpt2)
```

The output is the following:
```text
> D A Bm F#m F#m, C Cm %D A B - E A B E C - - # E - E
> D A Bm F#m F#m F#m F#m F#m F#m F#m F#m F#
> D A Bm F#m G#m G#m G#m G#m G#m G#m G#m G#
> D A Bm F#m Sb Ss Sc S Tm Vb W E R R Tt W Bw W F 1 3
> D A Bm F#m F#z %.1 F#t B M %.0% f(A A C B m F
```

### MelodyGPT v1

A GPT2 model trained with the melodyGPT-song-chords-text-1 dataset using the GPT2 tokenizer.

Trained on Google Colab [notebook](https://colab.research.google.com/drive/16R157wRI70YnJOGBOzRmr9VL3V7CYYPR?usp=sharing) 
with a Python 3.10.12 runtime with one T4 GPU.

Training was done with 16,323,255 train tokens and 1,813,694 validation tokens.

Micro batch size:
* `B = 4`
* `T = 1024`

LR scheduler:
* `max_steps = 31`
* `warmup_steps = 1`

Final loss:
* `train_loss: 3.588122`
* `val_loss: 3.594247`

I was using `torch.compile` but the cast to `bfloat16` was not working.

Metrics plot of training (validation loss computed each 5 steps):

![metrics_plot_melodyGPT_v1.png](assets%2Fmetrics_plot_melodyGPT_v1.png)

### MelodyGPT v2

A GPT2 model trained with the melodyGPT-song-chords-text-1 dataset using the trained melodyGPT-song-chords-tokenizer-1.

### TODO

* Encountered problems when adding the changes to use [bfloat16](https://github.com/karpathy/build-nanogpt/commit/177e4cd5b4cc05df4bb637ed1a9e55911d6f1e2c).
* Encountered problems when adding [torch compile](https://github.com/karpathy/build-nanogpt/commit/fb8bd6efd1bd7c4c894c9256f3bf41420efd1cb2) due to the python version 3.12

## Other

Use a virtual environment with Jupyter Notebook [blog](https://janakiev.com/blog/jupyter-virtual-envs/).