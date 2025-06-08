import datetime 
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import arxiv

def get_md_text_abstract(rag_answer, source = ['Arxiv Search', 'Semantic Search'][1], return_prompt_formatting = False):
    if 'Semantic Search' in source:
      title = rag_answer['document_metadata']['title'].replace('\n','')
      #score = round(rag_answer['score'], 2)
      date = rag_answer['document_metadata']['_time']  
      paper_abs = rag_answer['content']
      authors = rag_answer['document_metadata']['authors'].replace('\n','')
      doc_id = rag_answer['document_id']
      paper_link = f'''https://arxiv.org/abs/{doc_id}'''
      download_link = f'''https://arxiv.org/pdf/{doc_id}'''

    elif 'Arxiv' in source:
      title = rag_answer.title
      date = rag_answer.updated.strftime('%d %b %Y')
      paper_abs = rag_answer.summary.replace('\n',' ') + '\n'
      authors = ', '.join([author.name for author in rag_answer.authors])
      paper_link = rag_answer.links[0].href
      download_link = rag_answer.links[1].href
      
    else:
      raise Exception

    paper_title = f'''### {date} | [{title}]({paper_link}) | [⬇️]({download_link})\n'''
    authors_formatted = f'*{authors}*' + ' \n\n'

    md_text_formatted = paper_title + authors_formatted + paper_abs +  '\n---------------\n'+ '\n'
    if return_prompt_formatting:
        prompt_formatted =  f"<b> {title} </b> \n Abstract: {paper_abs}"
        return md_text_formatted, prompt_formatted

    return md_text_formatted

def remove_punctuation(text):
    punct_str = string.punctuation
    punct_str = punct_str.replace("'", "")
    return text.translate(str.maketrans("", "", punct_str))

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def search_cleaner(text):
    new_text = text.lower()
    new_text = remove_stopwords(new_text)
    new_text = remove_punctuation(new_text)
    return new_text


q = '(cat:cs.CV OR cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.NE OR cat:cs.RO)'


def get_arxiv_live_search(query, client, max_results = 10):
  clean_text = search_cleaner(query)
  search = arxiv.Search(
    query = clean_text + " AND "+q,
    max_results = max_results,
    sort_by = arxiv.SortCriterion.Relevance
  )
  results = client.results(search)
  all_results = list(results)
  return all_results
