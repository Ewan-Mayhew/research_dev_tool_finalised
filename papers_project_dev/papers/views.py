from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.db import transaction
from django.core.management import call_command
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
import os
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import concurrent.futures
import logging
from urllib.parse import quote

from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from .models import Paper
from .forms import PaperUploadForm, NoteForm, ArxivLinksForm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_DIR = '/Users/ewan.mayhew/Jaid/papers_project_dev/papers/data'  # Update this path if necessary
@login_required
def paper_list(request):
    papers = Paper.objects.all()
    return render(request, 'papers/paper_list.html', {'papers': papers})

@login_required
def paper_detail(request, pk):
    paper = get_object_or_404(Paper, pk=pk)
    if request.method == 'POST':
        form = NoteForm(request.POST, instance=paper)
        if form.is_valid():
            form.save()
            return redirect('paper_detail', pk=paper.pk)
    else:
        form = NoteForm(instance=paper)
    return render(request, 'papers/paper_detail.html', {'paper': paper, 'form': form})

@login_required
def upload_json(request):
    if request.method == 'POST':
        try:
            json_file_path = os.path.join(os.path.dirname(__file__), 'data', 'papers.json')
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

            with transaction.atomic():
                Paper.objects.all().delete()  # Be cautious with this in production!
                for item in data:
                    Paper.objects.create(
                        title=item['new_title'],
                        link=item['new_link'],
                        summary=item['new_summary'],
                        pdf_link=item['pdf_link'],
                        publication_date=item.get('published'),
                        similarity=item.get('similarity', 0)  # Default similarity to 0 if not present
                    )
            messages.success(request, 'Papers uploaded successfully.')
            return redirect('paper_list')
        except FileNotFoundError:
            messages.error(request, 'JSON file not found.')
        except json.JSONDecodeError:
            messages.error(request, 'Invalid JSON file.')
        except Exception as e:
            messages.error(request, f'Error uploading papers: {e}')
    return render(request, 'papers/upload_json.html')

@login_required
def run_arxiv_script(request):
    if request.method == 'POST':
        try:
            call_command('query_arxiv')  # Make sure this command is correctly defined
            messages.success(request, 'arXiv query completed successfully.')
            return redirect('paper_list')
        except Exception as e:
            messages.error(request, f'Error running arXiv query: {e}')
    return render(request, 'papers/run_arxiv_script.html')

@login_required
def search_papers(request):
    query = request.GET.get('query')
    papers = Paper.objects.filter(title__icontains=query) if query else Paper.objects.all()
    return render(request, 'papers/paper_list.html', {'papers': papers})

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
            return None
    except Exception as e:
        return None

@login_required
def upload_papers(request):
    if request.method == 'POST':
        if 'find_similar_papers' in request.POST:
            form = ArxivLinksForm(request.POST)
            if form.is_valid():
                links = form.cleaned_data['links'].split()
                model = SentenceTransformer('all-MiniLM-L6-v2')
                results = []
                for link in links:
                    arxiv_id = link.strip().split('/')[-1]
                    summary = fetch_summary(arxiv_id)
                    if summary:
                        embedding = model.encode(summary)
                        results.append({
                            'link': link,
                            'summary': summary,
                            'embedding': embedding.tolist()
                        })

                output_file = os.path.join(DATASET_DIR, 'arxiv_summaries_embeddings.json')
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)

                messages.success(request, 'Results saved successfully.')
                return redirect('paper_list')  # Adjust the redirect as needed
        elif 'run_arxiv_query' in request.POST:
            try:
                call_command('query_arxiv')
                messages.success(request, 'arXiv query completed successfully.')
                return redirect('paper_list')
            except Exception as e:
                messages.error(request, f'Error running arXiv query: {e}')
        elif 'upload_json' in request.POST:
            try:
                json_file_path = os.path.join(os.path.dirname(__file__), 'data', 'papers.json')
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)

                with transaction.atomic():
                    Paper.objects.all().delete()
                    for item in data:
                        Paper.objects.create(
                            title=item['new_title'],
                            link=item['new_link'],
                            summary=item['new_summary'],
                            pdf_link=item['pdf_link'],
                            publication_date=item.get('published'),
                            similarity=item.get('similarity', 0)
                        )
                messages.success(request, 'Papers uploaded successfully.')
                return redirect('paper_list')
            except FileNotFoundError:
                messages.error(request, 'JSON file not found.')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid JSON file.')
            except Exception as e:
                messages.error(request, f'Error uploading papers: {e}')
    else:
        form = ArxivLinksForm()
    return render(request, 'papers/upload_papers.html', {'form': form})

@login_required
def average_paper(request):
    return render(request, 'average_paper.html', {'message': 'Average paper feature coming soon.'})

def query_arxiv(keyword, max_results):
    # Replace underscores with '%20'
    formatted_keyword = keyword.replace("_", "%20")
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{quote(formatted_keyword)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    query_url = base_url + search_query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(query_url, headers=headers)
    logging.debug(f"Query URL: {query_url}")
    logging.debug(f"Response status code: {response.status_code}")
    logging.debug(f"Response content: {response.content}")

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

        logging.debug(f"Parsed {len(entries)} entries from response")
        return entries
    else:
        logging.error(f"Error fetching papers: {response.status_code}")
        return []

def parallel_query_arxiv(discipline, max_results):
    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_keyword = {executor.submit(query_arxiv, keyword, max_results): keyword for keyword in discipline}
        for future in concurrent.futures.as_completed(future_to_keyword):
            results = future.result()
            all_results.extend(results)
    return all_results

def generate_embeddings(model, summaries):
    embeddings = model.encode(summaries)
    return embeddings

def save_to_file(data, output_filepath):
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logging.debug(f"Results saved to {output_filepath}")

def fetch_and_process_papers(request):
    discipline = request.GET.get('discipline', 'machine learning')
    max_results = 1000 # Adjust as needed
    output_file = os.path.join(DATASET_DIR, 'previous_weeks_papers.json')

    logging.debug(f"Fetching papers for discipline: {discipline}")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_results = parallel_query_arxiv([discipline], max_results)

    if all_results:
        summaries = [result['summary'] for result in all_results]
        embeddings = generate_embeddings(model, summaries)

        for i, result in enumerate(all_results):
            result['embedding'] = embeddings[i].tolist()

        save_to_file(all_results, output_file)
        logging.debug(f"Fetched and processed {len(all_results)} papers for discipline: {discipline}")
        return HttpResponse(f"Papers fetched and processed successfully for discipline: {discipline}.")
    else:
        logging.debug(f"No papers were fetched this week for discipline: {discipline}")
        return HttpResponse(f"No papers were fetched this week for discipline: {discipline}.")

@login_required
def trends_closest_view(request):
    try:
        discipline = request.GET.get('discipline', 'machine_learning')
        dataset_file = f'/Users/ewan.mayhew/Jaid/papers_project_dev/papers/data/previous_weeks_papers.json'
        
        if not os.path.exists(dataset_file):
            return HttpResponse(f"No dataset found for discipline: {discipline}")

        with open(dataset_file, 'r') as file:
            papers = json.load(file)

        embeddings = np.array([np.array(paper['embedding']) for paper in papers])
        mean_embedding = np.mean(embeddings, axis=0)
        papers_with_distances = [(paper, distance.cosine(mean_embedding, np.array(paper['embedding']))) for paper in papers]
        top_papers = sorted(papers_with_distances, key=lambda x: x[1])[:5]
        context = {'papers': [paper for paper, _ in top_papers]}
        return render(request, 'papers/trends_closest.html', context)
    except Exception as e:
        return render(request, 'papers/trends_closest.html', {'error': str(e)})

@login_required
def trends_scatter_view(request):
    try:
        comparison_discipline = request.GET.get('comparison_discipline', 'machine_learning')
        dataset_file = f'/Users/ewan.mayhew/Jaid/papers_project_dev/papers/data/{comparison_discipline}_dataset.json'
        
        if not os.path.exists(dataset_file):
            return HttpResponse(f"No dataset found for discipline: {comparison_discipline}")

        # Load comparison dataset
        with open(dataset_file, 'r') as file:
            group1 = json.load(file)
        
        # Load previous week's papers
        with open(os.path.join(DATASET_DIR, 'previous_weeks_papers.json'), 'r') as file:
            group2 = json.load(file)

        # Extract embeddings, titles, and URLs
        embeddings1 = np.array([item['embedding'] for item in group1])
        titles1 = [item['title'] for item in group1]
        urls1 = [item['link'] for item in group1]  # Assuming 'link' contains the URL
        embeddings2 = np.array([item['embedding'] for item in group2])
        titles2 = [item['title'] for item in group2]
        urls2 = [item['link'] for item in group2]  # Assuming 'link' contains the URL

        # Apply t-SNE to all embeddings combined
        tsne = TSNE(n_components=2, random_state=42)
        all_embeddings = np.concatenate([embeddings1, embeddings2])
        projections = tsne.fit_transform(all_embeddings)

        # Apply jitter to spread out the points
        jitter_strength = 0.5  # Adjust this value to control the amount of jitter
        projections += np.random.normal(0, jitter_strength, projections.shape)

        # Split the projections back into two groups
        projections1 = projections[:len(embeddings1)]
        projections2 = projections[len(embeddings1):]

        # Create a Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projections1[:, 0], y=projections1[:, 1],
            mode='markers',
            marker=dict(color='red'),
            text=titles1,
            customdata=urls1,  # Add URLs to customdata
            name=f'{comparison_discipline.capitalize()} dataset'
        ))
        fig.add_trace(go.Scatter(
            x=projections2[:, 0], y=projections2[:, 1],
            mode='markers',
            marker=dict(color='blue'),
            text=titles2,
            customdata=urls2,  # Add URLs to customdata
            name='Last week\'s arXiv papers'
        ))

        # Update plot settings
        fig.update_layout(
            title='2D Visualization of Paper Embeddings from Two Groups with Jitter',
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            hovermode='closest',
            legend_title_text='Group',
            height=800,  # Increased height
            width=1200   # Increased width
        )

        # Convert the plot to HTML
        plot_html = fig.to_html(full_html=False)

        # Render the template with the plot
        return render(request, 'papers/trends_scatter.html', {'plot_html': plot_html})
    except Exception as e:
        return render(request, 'papers/trends_scatter.html', {'error': str(e)})

@login_required
def fetch_and_trends_view(request):
    return render(request, 'papers/fetch_and_trends.html')
