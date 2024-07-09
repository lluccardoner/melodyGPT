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

## Other

Use a virtual environment with Jupyter Notebook [blog](https://janakiev.com/blog/jupyter-virtual-envs/).