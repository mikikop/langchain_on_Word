"""Key Steps:
Extraction: The text content is extracted from the .docx file.
Text Splitting: The text is split into smaller chunks to handle large documents effectively.
Embeddings Creation: Embeddings for the text chunks are created using OpenAI's embeddings model.
Embeddings Storage: These embeddings are stored in a Chroma vector store for efficient similarity searches.
Question Answering Chain: The OpenAI instance is used to create a question-answering chain,
which can then process the question based on the document content.
Result Generation: The chain runs the similarity search and generates a response to the question,
which is then returned as an API response."""


import os.path
from django.conf import settings
from langchain_community.llms.openai import OpenAI
from .serializers import QuestionSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


class QuestionAnswerView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']

            doc_path = 'ask_my_doc_service/py_exercise.docx'

            if not os.path.exists(doc_path):
                return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

            try:
                # Load Microsoft Word document
                loader = UnstructuredWordDocumentLoader(doc_path)
                document = loader.load()

                # needing to split the text of the document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(document)

                # the embeddings represent text into vector representations. Her we are using the OpenAI model
                # to convert text into vectors. we are storing those embeddings in a vectorial DB called Chroma
                embeddings = OpenAIEmbeddings()
                docsearch = Chroma.from_documents(docs, embeddings)

                # the initialization of an instance of the model. Here OpenAI
                # Lower values (closer to 0) make the output more deterministic and focused,
                # while higher values (closer to 1) make it more random and creative.
                llm = OpenAI(temperature=0.7, api_key=settings.OPENAI_API_KEY)

                # creates the chain of question-answer and specifies which llm to use
                # the chain_type will determinate on what kind of text the model was trained (for example "medical",
                # "legal", "technical",...)
                chain = load_qa_chain(llm, chain_type="stuff")

                # run the process of answering the question based on similarity search (to identify relevant sections
                # or passages)
                answer = chain.run(input_documents=docsearch.similarity_search(question), question=question)

                return Response({'answer': answer}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
