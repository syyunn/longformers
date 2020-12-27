import torch
import torch
from transformers import LongformerTokenizerFast
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

tokenizer = LongformerTokenizer.from_pretrained(
    "valhalla/longformer-base-4096-finetuned-squadv1"
)
model = LongformerForQuestionAnswering.from_pretrained(
    "valhalla/longformer-base-4096-finetuned-squadv1"
)

# # Comment out for inference on CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print ("device ",device)
# model = model.to(device)

text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
question = "What has Huggingface done ?"
encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]

outputs = model(input_ids, attention_mask=attention_mask)
start_scores = outputs["start_logits"]
end_scores = outputs["end_logits"]
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
print(answer)
if __name__ == "__main__":
    pass
