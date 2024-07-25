from django import forms
from .models import Paper

class PaperUploadForm(forms.Form):
    json_file = forms.FileField()

class NoteForm(forms.ModelForm):
    class Meta:
        model = Paper
        fields = ['notes']

class ArxivQueryForm(forms.Form):
    keywords = forms.CharField(label='Keywords', max_length=100)
    max_results = forms.IntegerField(label='Max Results', min_value=1)
    start_date = forms.DateField(label='Start Date', widget=forms.SelectDateWidget)
    end_date = forms.DateField(label='End Date', required=False, widget=forms.SelectDateWidget)
    threshold = forms.FloatField(label='Similarity Threshold', min_value=0.0, max_value=1.0, initial=0.7)


class ArxivLinksForm(forms.Form):
    links = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Enter arXiv links separated by new lines'}), label='ArXiv Links')
