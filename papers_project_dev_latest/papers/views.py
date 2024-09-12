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

# Directory to store datasets

@login_required
def paper_list(request):
    """
    View to list all papers.
    Only accessible to logged-in users.
    """
    papers = Paper.objects.all()
    return render(request, 'papers/paper_list.html', {'papers': papers})

@login_required
def paper_detail(request, pk):
    """
    View to display the details of a specific paper.
    Allows users to edit notes associated with the paper.
    Only accessible to logged-in users.
    """
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
    """
    View to upload papers from a JSON file.
    Only accessible to logged-in users.
    """
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
    """
    View to run a script that queries arXiv.
    Only accessible to logged-in users.
    """
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
    """
    View to search for papers by title.
    Only accessible to logged-in users.
    """
    query = request.GET.get('query')
    papers = Paper.objects.filter(title__icontains=query) if query else Paper.objects.all()
    return render(request, 'papers/paper_list.html', {'papers': papers})

def fetch_summary(arxiv_id):
    """
    Fetch the summary of a paper from arXiv using its ID.
    """
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

@csrf_exempt

def upload_papers(request):
    """
    View to handle multiple functionalities related to papers:
    - Finding similar papers using arXiv links.
    - Running the arXiv query script.
    - Uploading papers from a JSON file.
    """
    form = ArxivLinksForm()
    
    if request.method == 'POST':
        print("POST request received")  # Debugging
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

                relative_output_file = 'papers/data/arxiv_summaries_embeddings.json'
                
                # Convert to absolute path
                output_file = os.path.abspath(relative_output_file)

                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)

                messages.success(request, 'Results saved successfully.')
                return redirect('paper_list')  # Adjust the redirect as needed
            else:
                messages.error(request, 'Invalid form submission.')
                return redirect('upload_papers')

        elif 'upload_json' in request.POST:
            try:
                print("Uploading JSON file...")  # Debugging
                json_file_path = os.path.join(os.path.dirname(__file__), 'data', 'papers.json')
                # Convert to absolute path
                json_file_path = os.path.abspath(json_file_path)
                
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
                return redirect('upload_papers')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid JSON file.')
                return redirect('upload_papers')
            except Exception as e:
                messages.error(request, f'Error uploading papers: {e}')
                return redirect('upload_papers')

    else:
        print("GET request received")  # Debugging
    return render(request, 'papers/upload_papers.html', {'form': form})

@csrf_exempt
def run_arxiv_query(request):
    if request.method == 'POST':
        try:
            print("Triggering arXiv query command...")  # Debugging
            call_command('query_arxiv')
            messages.success(request, 'arXiv query completed successfully.')
            return redirect('paper_list')
        except Exception as e:
            print(f"Error running arXiv query: {e}")  # Debugging
            messages.error(request, f'Error running arXiv query: {e}')
            return redirect('upload_papers')
    return render(request, 'papers/upload_papers.html')

@login_required
def average_paper(request):
    """
    Placeholder view for the average paper feature.
    Only accessible to logged-in users.
    """
    return render(request, 'average_paper.html', {'message': 'Average paper feature coming soon.'})

def query_arxiv(keyword, max_results):
    """
    Query arXiv for papers based on a keyword.
    """
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
    """
    Query arXiv for papers in parallel using multiple threads.
    """
    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_keyword = {executor.submit(query_arxiv, keyword, max_results): keyword for keyword in discipline}
        for future in concurrent.futures.as_completed(future_to_keyword):
            results = future.result()
            all_results.extend(results)
    return all_results

def generate_embeddings(model, summaries):
    """
    Generate embeddings for a list of summaries using the provided model.
    """
    embeddings = model.encode(summaries)
    return embeddings

def save_to_file(data, output_filepath):
    """
    Save data to a file in JSON format.
    """
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logging.debug(f"Results saved to {output_filepath}")

def fetch_and_process_papers(request):
    """
    Fetch and process papers for a given discipline.
    """
    discipline = request.GET.get('discipline', 'machine learning')
    max_results = 2000  # This number was chosen as it is approximately the number of papers published in a fortnight.
    relative_output_file = 'papers/data/previous_weeks_papers.json'
    
    # Convert to absolute path
    output_file = os.path.abspath(relative_output_file)



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
    """
    View to display the papers closest to the average embedding.
    Only accessible to logged-in users.
    """
    try:
        discipline = request.GET.get('discipline', 'machine_learning')

        relative_output_file = 'papers/data/previous_weeks_papers.json'
    
    # Convert to absolute path
        dataset_file = os.path.abspath(relative_output_file)
        
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

import os
import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from django.shortcuts import render
from django.http import HttpResponse

def trends_scatter_view(request):
    """
    View to display a scatter plot of paper embeddings.
    Only accessible to logged-in users.
    """
    try:
        comparison_discipline = request.GET.get('comparison_discipline', 'machine_learning')
        dataset_file_relative = f'papers/data/{comparison_discipline}_dataset.json'
        dataset_file = os.path.abspath(dataset_file_relative)
        
        if not os.path.exists(dataset_file):
            return HttpResponse(f"No dataset found for discipline: {comparison_discipline}")

        # Load comparison dataset
        with open(dataset_file, 'r') as file:
            group1 = json.load(file)
        
        # Load previous week's papers
        relative_output_file = 'papers/data/previous_weeks_papers.json'
        output_file = os.path.abspath(relative_output_file)
        with open(output_file, 'r') as file:
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

        # Enhanced Plotly figure
        fig = go.Figure()

        # First group
        fig.add_trace(go.Scatter(
            x=projections1[:, 0], y=projections1[:, 1],
            mode='markers',
            marker=dict(
                color='rgba(31, 119, 180, 0.7)',  # Blue color with transparency
                size=10,
                line=dict(
                    color='rgba(31, 119, 180, 1)',
                    width=1
                )
            ),
            text=titles1,
            customdata=urls1,  # Add URLs to customdata
            hovertemplate='<b>%{text}</b><br>URL: %{customdata}<extra></extra>',  # Custom hover info
            name=f'{comparison_discipline.capitalize()} dataset'
        ))

        # Second group
        fig.add_trace(go.Scatter(
            x=projections2[:, 0], y=projections2[:, 1],
            mode='markers',
            marker=dict(
                color='rgba(255, 127, 14, 0.7)',  # Orange color with transparency
                size=10,
                line=dict(
                    color='rgba(255, 127, 14, 1)',
                    width=1
                )
            ),
            text=titles2,
            customdata=urls2,  # Add URLs to customdata
            hovertemplate='<b>%{text}</b><br>URL: %{customdata}<extra></extra>',  # Custom hover info
            name='Last week\'s arXiv papers'
        ))

        # Update layout with enhanced aesthetics
        fig.update_layout(
            title=dict(
                text='2D Visualization of Paper Embeddings from Two Groups with Jitter',
                x=0.5,
                xanchor='center',
                font=dict(
                    size=24,
                    color='#333'
                )
            ),
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            hovermode='closest',
            legend=dict(
                title='Group',
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=100, b=0),  # No left or right margin
            height=800,  # Height of the plot
            paper_bgcolor='rgba(255,255,255,1)',  # White background
            plot_bgcolor='rgba(245, 246, 249, 1)',  # Light grey background for the plot
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',  # Light grey gridlines
                zerolinecolor='rgba(200, 200, 200, 0.5)'
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',  # Light grey gridlines
                zerolinecolor='rgba(200, 200, 200, 0.5)'
            ),
            dragmode='lasso',  # Default to lasso selection
            showlegend=True,  # Keep the legend
            modebar_remove=['select', 'lasso2d', 'zoom', 'zoomIn', 'zoomOut', 'pan', 'autoScale', 'resetScale']  # Remove unwanted tools
        )

        # Convert the plot to HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Render the template with the plot
        return render(request, 'papers/trends_scatter.html', {'plot_html': plot_html})
    except Exception as e:
        return render(request, 'papers/trends_scatter.html', {'error': str(e)})


@login_required
def fetch_and_trends_view(request):
    """
    View to render the fetch and trends page.
    Only accessible to logged-in users.
    """
    return render(request, 'papers/fetch_and_trends.html')

from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.management import call_command, CommandError

def run_populators(request):
    """
    View to run populator commands that fill out the canonical databases.
    Only accessible to logged-in users.
    """
    template_path = 'papers/paper_list.html'

    if request.method == 'POST':
        print("POST request received")  # Debugging
        try:
            call_command('populator_loss')
            call_command('populator_ML')
            call_command('populator_NLP')
            call_command('populator_TableRecognition')
            call_command('populator_computer_vision')
            messages.success(request, 'All populators ran successfully!')
        except CommandError as e:
            print(f"Command error occurred: {str(e)}")  # Debugging
            messages.error(request, f'Command error: {str(e)}')
        except Exception as e:
            print(f"Error occurred: {str(e)}")  # Debugging
            messages.error(request, f'An error occurred: {str(e)}')
        return redirect('paper_list')
    else:
        print("GET request received")  # Debugging
    return render(request, template_path)


def about(request):
    return render(request, os.path.abspath(r'papers/templates/papers/about.html'))

from django.shortcuts import render
import subprocess
import sys
def run_code_view(request):
    output = ""
    if request.method == "POST":
        code = request.POST.get("code")
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout if result.stdout else result.stderr
        except Exception as e:
            output = str(e)
    return render(request, os.path.abspath(r'papers\templates\papers\code_runner.html'), {'output': output})

import openai
from django.shortcuts import render, redirect, get_object_or_404
from .models import Paper

# Initialize the OpenAI API key (ensure you have this in your settings)
openai.api_key = 'sk-...AW8A'

def implement_ideas(request, paper_id):
    paper = get_object_or_404(Paper, id=paper_id)

    if request.method == "POST":
        # Call ChatGPT API to generate implementation code
        prompt = f"Implement the ideas discussed in the paper: {paper.title}\nSummary: {paper.summary}"

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # Or another model of your choice
                prompt=prompt,
                max_tokens=1500  # Adjust as necessary
            )
            implementation_code = response['choices'][0]['text']
        except Exception as e:
            implementation_code = f"Error generating code: {str(e)}"

        return render(request, 'papers/paper_detail.html', {
            'paper': paper,
            'implementation_code': implementation_code
        })

    return redirect('paper_detail', paper_id=paper.id)
