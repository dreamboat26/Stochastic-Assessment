import torch
import transformers
import gradio as gr
from ragatouille import RAGPretrainedModel
from huggingface_hub import InferenceClient
import re
from datetime import datetime
import json
import os
import arxiv
from utils import get_md_text_abstract, search_cleaner, get_arxiv_live_search

retrieve_results = 10 
show_examples = False
llm_models_to_choose = ['mistralai/Mixtral-8x7B-Instruct-v0.1','mistralai/Mistral-7B-Instruct-v0.3', 'google/gemma-2-2b-it', 'None']

token = os.getenv("HF_TOKEN")

generate_kwargs = dict(
    temperature = None,
    max_new_tokens = 512,
    top_p = None,
    do_sample = False,
    )

## RAG Model
RAG = RAGPretrainedModel.from_index("colbert/indexes/arxiv_colbert")

try:
  gr.Info("Setting up retriever, please wait...")
  rag_initial_output = RAG.search("what is Mistral?", k = 1)
  gr.Info("Retriever working successfully!")
    
except:
  gr.Warning("Retriever not working!")

## Header
mark_text = '# 🔍 Search Results\n'
header_text = "# ArXiv CS RAG \n"

try:
  with open("README.md", "r") as f:
      mdfile = f.read()
  date_pattern = r'Index Last Updated : \d{4}-\d{2}-\d{2}'
  match = re.search(date_pattern, mdfile)
  date = match.group().split(': ')[1]
  formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%d %b %Y')
  header_text += f'Index Last Updated: {formatted_date}\n'
  index_info = f"Semantic Search - up to {formatted_date}"  
except:
  index_info = "Semantic Search"

database_choices = [index_info,'Arxiv Search - Latest - (EXPERIMENTAL)']

## Arxiv API
arx_client = arxiv.Client()
is_arxiv_available = True
check_arxiv_result = get_arxiv_live_search("What is Mistral?", arx_client, retrieve_results)
if len(check_arxiv_result) == 0:
  is_arxiv_available = False
  print("Arxiv search not working, switching to default search ...")
  database_choices = [index_info]



## Show examples (disabled)
if show_examples:
    with open("sample_outputs.json", "r") as f:
      sample_outputs = json.load(f)
    output_placeholder = sample_outputs['output_placeholder']
    md_text_initial = sample_outputs['search_placeholder']
    
else:
    output_placeholder = None 
    md_text_initial = ''


def rag_cleaner(inp):
    rank = inp['rank']
    title = inp['document_metadata']['title']
    content = inp['content']
    date = inp['document_metadata']['_time']
    return f"{rank}. <b> {title} </b> \n Date : {date} \n Abstract: {content}"

def get_prompt_text(question, context, formatted = True, llm_model_picked = 'mistralai/Mistral-7B-Instruct-v0.3'):
    if formatted:
      sys_instruction = f"Context:\n {context} \n Given the following scientific paper abstracts, take a deep breath and lets think step by step to answer the question. Cite the titles of your sources when answering, do not cite links or dates."
      message = f"Question: {question}"
        
      if 'mistralai' in llm_model_picked:
          return f"<s>" + f"[INST] {sys_instruction}" +  f" {message}[/INST]"
          
      elif 'gemma' in llm_model_picked:
          return f"<bos><start_of_turn>user\n{sys_instruction}" +  f" {message}<end_of_turn>\n"
          
    return f"Context:\n {context} \n Given the following info, take a deep breath and lets think step by step to answer the question: {question}. Cite the titles of your sources when answering.\n\n"

def get_references(question, retriever, k = retrieve_results):
    rag_out = retriever.search(query=question, k=k)
    return rag_out

def get_rag(message):
    return get_references(message, RAG)

with gr.Blocks(theme = gr.themes.Soft()) as demo:
    header = gr.Markdown(header_text)
    
    with gr.Group():
      msg = gr.Textbox(label = 'Search', placeholder = 'What is Mistral?')
        
      with gr.Accordion("Advanced Settings", open=False):
        with gr.Row(equal_height = True):
          llm_model = gr.Dropdown(choices = llm_models_to_choose, value = 'mistralai/Mistral-7B-Instruct-v0.3', label = 'LLM Model')
          llm_results = gr.Slider(minimum=4, maximum=10, value=5, step=1, interactive=True, label="Top n results as context")
          database_src = gr.Dropdown(choices = database_choices, value = index_info, label = 'Search Source')
          stream_results = gr.Checkbox(value = True, label = "Stream output", visible = False)

    output_text = gr.Textbox(show_label = True, container = True, label = 'LLM Answer', visible = True, placeholder = output_placeholder)
    input = gr.Textbox(show_label = False, visible = False)
    gr_md = gr.Markdown(mark_text + md_text_initial)

    def update_with_rag_md(message, llm_results_use = 5, database_choice = index_info, llm_model_picked = 'mistralai/Mistral-7B-Instruct-v0.3'):
        prompt_text_from_data = ""
        database_to_use = database_choice
        if database_choice == index_info:
          rag_out = get_rag(message)
        else:
          arxiv_search_success = True
          try:
            rag_out = get_arxiv_live_search(message, arx_client, retrieve_results)
            if len(rag_out) == 0:
              arxiv_search_success = False 
          except:
            arxiv_search_success = False
 

          if not arxiv_search_success:
            gr.Warning("Arxiv Search not working, switching to semantic search ...")
            rag_out = get_rag(message)
            database_to_use = index_info 

        md_text_updated = mark_text
        for i in range(retrieve_results):
          rag_answer = rag_out[i]
          if i < llm_results_use:
            md_text_paper, prompt_text = get_md_text_abstract(rag_answer, source = database_to_use, return_prompt_formatting = True)
            prompt_text_from_data += f"{i+1}. {prompt_text}"
          else:
            md_text_paper = get_md_text_abstract(rag_answer, source = database_to_use)
          md_text_updated += md_text_paper
        prompt = get_prompt_text(message, prompt_text_from_data, llm_model_picked = llm_model_picked)
        return md_text_updated, prompt

    def ask_llm(prompt, llm_model_picked = 'mistralai/Mistral-7B-Instruct-v0.3', stream_outputs = False):
       model_disabled_text = "LLM Model is disabled"
       output = ""
        
       if llm_model_picked == 'None':
          if stream_outputs:
              for out in model_disabled_text:
                output += out
                yield output
              return output 
          else:
              return model_disabled_text
              
       client = InferenceClient(llm_model_picked, token = token)
       try:
           stream = client.text_generation(prompt, **generate_kwargs,  stream=stream_outputs, details=False, return_full_text=False)
           
       except:
           gr.Warning("LLM Inference rate limit reached, try again later!")
           return ""
       
       if stream_outputs:
           for response in stream:
              output += response
              yield output
           return output
       else:
           return stream


    msg.submit(update_with_rag_md, [msg, llm_results,  database_src, llm_model], [gr_md, input]).success(ask_llm, [input, llm_model, stream_results], output_text)

demo.queue().launch()