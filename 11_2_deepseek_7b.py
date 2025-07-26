#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install transformers datasets torch peft accelerate bitsandbytes')


# In[3]:


get_ipython().system('pip uninstall keras -y  # Remove Keras 3 if installed')
get_ipython().system('pip install tf-keras    # Install TensorFlow-compatible Keras')


# In[4]:


get_ipython().system('pip install trl')


# In[5]:


get_ipython().system('pip install triton')


# In[6]:


get_ipython().system('pip uninstall bitsandbytes -y')
get_ipython().system('pip install bitsandbytes')


# In[7]:


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="../Datos/chatbot_data.json", split="train")

# Model & Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Quantization (4-bit + CPU offload)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load model with offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# Training Arguments (without max_seq_length)
training_args = TrainingArguments(
    output_dir="./deepseek-chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=50,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# Formatting function
def formatting_func(example):
    # Adjust this based on your dataset structure
    # For simple text:
    return {"text": example["messages"]}

    # If your data has chat format (user/assistant messages):
    # text = ""
    # for message in example["messages"]:
    #     text += f"{message['role']}: {message['content']}\n"
    # return {"text": text.strip()}

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_func,  # Replaces dataset_text_field
)

# Train!
trainer.train()
trainer.save_model("deepseek-chatbot-finetuned")


# In[8]:


from transformers import pipeline

# Load fine-tuned model
model_path = "deepseek-chatbot-finetuned"
chatbot = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    device="cuda",
)

# Chat example
prompt = """<|system|>You are a helpful assistant.</s>
<|user|>How do I make coffee?</s>
<|assistant|>"""

output = chatbot(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(output[0]['generated_text'])


# In[ ]:




