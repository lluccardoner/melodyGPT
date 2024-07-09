from gpt import GPT

if __name__ == "__main__":
    model = GPT.from_pretrained('gpt2')
    print("didn't crash yay!")
