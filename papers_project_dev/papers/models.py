from django.db import models
import json

class Paper(models.Model):
    title = models.CharField(max_length=200)
    link = models.URLField(unique=True)
    summary = models.TextField()
    pdf_link = models.URLField()
    notes = models.TextField(blank=True, null=True)
    publication_date = models.DateField(null=True, blank=True)
    similarity = models.FloatField(null=True, blank=True)
    embedding = models.TextField(blank=True, null=True)  # Store embeddings as JSON serialized strings

    def __str__(self):
        return self.title

    def get_embedding_as_list(self):
        return json.loads(self.embedding) if self.embedding else []

    def set_embedding_from_list(self, embedding_list):
        self.embedding = json.dumps(embedding_list)