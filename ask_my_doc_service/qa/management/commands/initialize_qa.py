from django.core.management.base import BaseCommand
from django.core.cache import cache
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


class Command(BaseCommand):
    help = 'Initialize the QA system and cache the necessary objects.'

    def handle(self, *args, **kwargs):
        doc_path = 'ask_my_doc_service/py_exercise.docx'

        # Load Microsoft Word document
        loader = UnstructuredWordDocumentLoader(doc_path)
        document = loader.load()

        # Split the text of the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(document)

        # Print debug information
        self.stdout.write(self.style.SUCCESS('Documents split into chunks:'))
        for idx, doc in enumerate(docs):
            self.stdout.write(f'Chunk {idx + 1}: {doc[:50]}...')  # Print the first 50 characters of each chunk

        # the embeddings represent text into vector representations. Her we are using the OpenAI model
        # to convert text into vectors. we are storing those embeddings in a vectorial DB called Chroma
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(docs, embeddings)

        # Cache the docsearch object
        # cache.set('docsearch', docsearch)

        # Cache the documents and embeddings separately
        try:
            cache.set('docs', docs)
            cache.set('embeddings', embeddings)
            self.stdout.write(self.style.SUCCESS('QA system initialized and cached successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error caching data: {e}'))
