import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import concurrent.futures
from django.core.management.base import BaseCommand

def query_arxiv(keyword, max_results):
    print(f'Querying arXiv for keyword: {keyword} with max results: {max_results}')
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{keyword}&start=0&max_results={max_results}'
    query_url = base_url + search_query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        response = requests.get(query_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.RequestException as e:
        print(f'Error during requests to {query_url} : {str(e)}')
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f'Error parsing XML: {str(e)}')
        return []
    
    entries = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        link = entry.find('{http://www.w3.org/2005/Atom}id').text
        published = entry.find('{http://www.w3.org/2005/Atom}published').text

        # Convert published date to YYYY-MM-DD format
        published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')

        entries.append({
            'title': title,
            'summary': summary,
            'link': link,
            'published': published_date
        })
    
    print(f'Found {len(entries)} entries for keyword: {keyword}')
    return entries

def parallel_query_arxiv(discipline, max_results):
    print(f'Starting parallel query for disciplines: {discipline}')
    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_keyword = {executor.submit(query_arxiv, keyword, max_results): keyword for keyword in discipline}
        for future in concurrent.futures.as_completed(future_to_keyword):
            keyword = future_to_keyword[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f'Completed query for keyword: {keyword}')
            except Exception as e:
                print(f'Error with keyword {keyword}: {str(e)}')
    print(f'Collected total {len(all_results)} results from all keywords')
    return all_results

def generate_embeddings(model, summaries):
    print(f'Generating embeddings for {len(summaries)} summaries')
    embeddings = model.encode(summaries)
    print(f'Generated {len(embeddings)} embeddings')
    return embeddings

def save_to_file(data, output_filepath):
    print(f'Saving results to file: {output_filepath}')
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Results saved to {output_filepath}')


class Command(BaseCommand):
    help = 'Populate the database with arXiv data for NLP'

    def handle(self, *args, **kwargs):
        discipline = ['Machine Learning', 'NLP', 'Natural Language Processing']
        max_results = 20
        output_file = 'papers_project_dev/papers/data/natural_language_processing_dataset.json'

        print('Loading SentenceTransformer model...')
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print('Starting arXiv query...')
        all_results = parallel_query_arxiv(discipline, max_results)

        if all_results:
            summaries = [result['summary'] for result in all_results]
            embeddings = generate_embeddings(model, summaries)

            for i, result in enumerate(all_results):
                result['embedding'] = embeddings[i].tolist()

            save_to_file(all_results, output_file)
        else:
            print('No results found.')