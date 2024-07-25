import json
import os
from xml.etree import ElementTree as ET

import requests
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer

def fetch_summary(arxiv_id):
    base_url = 'http://export.arxiv.org/api/query?id_list='
    query_url = base_url + arxiv_id
    try:
        response = requests.get(query_url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            return summary
        else:
            print(f'Error fetching {arxiv_id}: Status code {response.status_code}')
            return None
    except Exception as e:
        print(f'Exception fetching {arxiv_id}: {e}')
        return None

class Command(BaseCommand):
    help = 'Fetches summaries and embeddings for arXiv papers.'

    def add_arguments(self, parser):
        parser.add_argument('arxiv_links', nargs='+', type=str)

    def handle(self, *args, **options):
        arxiv_links = options['arxiv_links']
        model = SentenceTransformer('all-MiniLM-L6-v2')

        results = []
        for link in arxiv_links:
            arxiv_id = link.split('/')[-1]
            summary = fetch_summary(arxiv_id)
            if summary:
                embedding = model.encode(summary)
                results.append({
                    'link': link,
                    'summary': summary,
                    'embedding': embedding.tolist()
                })

        output_file = os.path.join(os.getcwd(), 'arxiv_summaries_embeddings.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        self.stdout.write(self.style.SUCCESS(f'Results saved to {output_file}'))