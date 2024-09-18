#!/usr/bin/env python
# coding: utf-8

# In[1]:

import subprocess

# subprocess.check_call(['pip', 'install', 'transformers==4.41.0'])
# get_ipython().system('pip install transformers==4.44.0')
# !pip install -e '.[dev]'
# 4.44.0


# In[1]:
subprocess.check_call(['pip', 'install', '-q', 'pypdf'])
subprocess.check_call(['pip', 'install', '-q', 'python-dotenv'])
subprocess.check_call(['pip', 'install', 'llama-index==0.10.12'])
# subprocess.check_call(['pip', 'install', '-q', 'gradio'])
subprocess.check_call(['pip', 'install', 'einops'])
subprocess.check_call(['pip', 'install', 'accelerate'])

# get_ipython().system('pip install -q pypdf')
# get_ipython().system('pip install -q python-dotenv')
# get_ipython().system('pip install  llama-index==0.10.12')
# get_ipython().system('pip install -q gradio')
# get_ipython().system('pip install einops')
# get_ipython().system('pip install accelerate')


# In[2]:


# get_ipython().system('pip install llama-index-llms-huggingface --upgrade')


# In[3]:


# get_ipython().system('pip install fastembed')


# In[4]:


# !pip install tokenizers --upgrade
# get_ipython().system('pip install transformers -U #4.44.2')


# In[9]:


# !pip show transformers


# In[5]:


# get_ipython().system('pip install huggingface-hub --upgrade')
# get_ipython().system('pip install llama-index-embeddings-fastembed')

subprocess.check_call(['pip', 'install', 'llama-index-llms-huggingface --upgrade'])
subprocess.check_call(['pip', 'install', 'fastembed'])
# subprocess.check_call(['pip', 'install', 'tokenizers', '--upgrade'])
subprocess.check_call(['pip', 'install', 'transformers', '--upgrade'])
subprocess.check_call(['pip', 'install', 'huggingface-hub --upgrade'])
subprocess.check_call(['pip', 'install', 'llama-index-embeddings-fastembed'])
# !pip show llama-index-llms-huggingface
# !pip show huggingface_hub


# In[6]:


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

documents = SimpleDirectoryReader("data").load_data()


# In[6]:





# In[7]:


from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512


# In[8]:


from llama_index.core import PromptTemplate


system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."


# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")




# In[ ]:


# from huggingface_hub import  notebook_login
# notebook_login()


# In[9]:


# get_ipython().system('huggingface-cli login --token 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')
subprocess.run(['huggingface-cli', 'login', '--token', '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'], check=True)



# In[10]:


import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"

)

# empty string token to identify the end of the line/sentence
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    # device_map="cpu",
    # stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

Settings.llm = llm
Settings.chunk_size = 1024


# In[11]:


index = VectorStoreIndex.from_documents(documents)


# In[ ]:


print(index)


# In[12]:


query_engine = index.as_query_engine()

while True:
    query = input()
    if query == "exit":
        break
    response = query_engine.query(query)
    print(response)


# chat_engine = index.as_chat_engine()

# while True:
#     query = input()
#     if query == "exit":
#         break
#     response = chat_engine.chat(query)
#     print(response)


# In[ ]:

