from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=False)
inputs = tokenizer(["with the advent of von neumann probes "], return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"], max_length=50, repetition_penalty=12.)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])

def get_tokens_as_tuple(word):
  return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])

biased_ids = model.generate(inputs["input_ids"], max_length=50, repetition_penalty=12., top_k=1)
print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
