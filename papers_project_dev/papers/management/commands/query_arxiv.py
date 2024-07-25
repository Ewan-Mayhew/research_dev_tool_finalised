import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from datetime import datetime
from django.core.management.base import BaseCommand
import concurrent.futures
import os

def query_arxiv(keyword, max_results):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{keyword}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    query_url = base_url + search_query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    response = requests.get(query_url, headers=headers)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        
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
        
        return entries
    else:
        print(f'Error: {response.status_code}')
        return []

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def load_existing_embeddings(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def find_similar_embeddings(existing_embeddings, new_results, new_embeddings, threshold=0.2):
    matches = []
    for i, new_embed in enumerate(new_embeddings):
        for existing in existing_embeddings:
            similarity = cosine_similarity(np.array(existing['embedding']), new_embed)
            if similarity >= threshold:
                arxiv_id = new_results[i]['link'].split('/')[-1]
                pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                matches.append({
                    'new_title': new_results[i]['title'],
                    'new_link': new_results[i]['link'],
                    'new_summary': new_results[i]['summary'],
                    'pdf_link': pdf_link,
                    'similarity': similarity,
                    'existing_title': existing.get('title', 'Unknown Title'),
                    'existing_link': existing.get('link', 'Unknown Link'),
                    'published': new_results[i]['published']
                })
    return matches

def remove_duplicates(matches):
    seen_links = set()
    unique_matches = []
    for match in matches:
        if match['new_link'] not in seen_links:
            unique_matches.append(match)
            seen_links.add(match['new_link'])
    return unique_matches

def filter_by_date_range(matches, start_date, end_date=None):
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')  # Use current date if end date not provided

    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    filtered_matches = [match for match in matches if start_date_dt <= datetime.strptime(match['published'], '%Y-%m-%d') <= end_date_dt]
    return filtered_matches

def save_matches_to_file(matches, output_filepath):
    # Sort matches by similarity score in descending order
    matches_sorted = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    with open(output_filepath, 'w') as f:
        json.dump(matches_sorted, f, indent=4)
    print(f'Results saved to {output_filepath}')

def parallel_query_arxiv(discipline, max_results):
    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_keyword = {executor.submit(query_arxiv, keyword, max_results): keyword for keyword in discipline}
        for future in concurrent.futures.as_completed(future_to_keyword):
            results = future.result()
            all_results.extend(results)
    return all_results

def batch_encode(model, summaries, batch_size=32):
    embeddings = []
    for i in range(0, len(summaries), batch_size):
        batch_summaries = summaries[i:i+batch_size]
        batch_embeddings = model.encode(batch_summaries)
        embeddings.extend(batch_embeddings)
    return embeddings

# Main execution flow in the Command class
def main(discipline, max_results, existing_embeddings_file, output_file, threshold=0.2, start_date=None, end_date=None):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    existing_embeddings = load_existing_embeddings(existing_embeddings_file)

    all_results = parallel_query_arxiv(discipline, max_results)

    if all_results:
        summaries = [result['summary'] for result in all_results]
        new_embeddings = batch_encode(model, summaries)
        matches = find_similar_embeddings(existing_embeddings, all_results, new_embeddings, threshold)
        unique_matches = remove_duplicates(matches)
        filtered_matches = filter_by_date_range(unique_matches, start_date, end_date)

        save_matches_to_file(filtered_matches, output_file)
    else:
        print('No results found.')

class Command(BaseCommand):
    help = 'Run the arXiv query and save the results'

    def handle(self, *args, **kwargs):
        discipline = ['machine learning', 'deep learning', 'computer vision', 'nlp']
        max_results = 10000
        existing_embeddings_file = '/Users/ewan.mayhew/Jaid/papers_project_dev/papers/data/arxiv_summaries_embeddings.json'
        output_file = '/Users/ewan.mayhew/Jaid/papers_project_dev/papers/data/papers.json'
        start_date = '2024-01-01'
        end_date = None
        main(discipline, max_results, existing_embeddings_file, output_file, start_date=start_date, end_date=end_date)
