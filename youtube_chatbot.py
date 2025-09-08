import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set environment variable
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "Your_api_key")

class YouTubeChatbot:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.vector_store = None
        self.chain = None
        
    def get_transcript(self, video_id):
        """Extract transcript from YouTube video"""
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id, languages=["en"])
            transcript = " ".join(snippet.text for snippet in transcript_list)
            return transcript
        except TranscriptsDisabled:
            return f"No transcripts available for video ID: {video_id}"
    
    def create_vector_store(self, transcript):
        """Create vector store from transcript"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.create_documents([transcript])
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
    def setup_chain(self):
        """Setup the RAG chain"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            Provide the answer in points rather than a single sentence.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def initialize_chatbot(self, video_id):
        """Initialize chatbot with video transcript"""
        transcript = self.get_transcript(video_id)
        if "No transcripts available" in transcript:
            return transcript
        
        self.create_vector_store(transcript)
        self.setup_chain()
        return "Chatbot initialized successfully!"
    
    def ask_question(self, question):
        """Ask a question to the chatbot"""
        if self.chain is None:
            return "Please initialize the chatbot first with a video ID."
        
        return self.chain.invoke(question)

# Usage example
if __name__ == "__main__":
    chatbot = YouTubeChatbot()
    
    # Initialize with video ID
    video_id = "Gfr50f6ZBvo"
    init_result = chatbot.initialize_chatbot(video_id)
    print(init_result)
    
    # Ask questions
    question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
    answer = chatbot.ask_question(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
