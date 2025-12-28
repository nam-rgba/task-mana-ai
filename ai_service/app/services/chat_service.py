import logging
import os
from typing import Optional, List
from dotenv import load_dotenv
from upstash_redis import Redis
# LangChain Imports
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory

# Service Imports
from app.services.vector_store import VectorStoreService
from app.services.models_loader import ModelsLoader

load_dotenv()
log = logging.getLogger(__name__)

class ChatService:
    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        self.llm = ModelsLoader.llm()
        self.vector_store = vector_store or VectorStoreService()
        self.retriever = self.vector_store.guides_retriever(k=3)

        self.redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not self.redis_url or not self.redis_token:
            raise RuntimeError("Missing Upstash Redis env vars")

        # 1. Define Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a helpful assistant. 
                - If the question is answerable from the provided context, answer using the context.
                - If the question is general, answer naturally.
                - If not answerable from context, say "I'm sorry, I don't have that information."
                - Response in the same language as the question.
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        # 2. Define Context Formatter
        def format_docs(docs):
            return "\n\n".join(
                f"[Page {doc.metadata.get('page','N/A')}]\n{doc.page_content}" 
                for doc in docs
            )

        # 3. Define the Core Chain (RAG + LLM)
        # We process the context retrieval BEFORE the history injection to keep it clean
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(self.retriever.invoke(x["question"]))
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 4. Create History Factory
        def get_session_history(session_id: str):
            return UpstashRedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url,
                token=self.redis_token,
                ttl=86400  # 24 hours
            )

        # 5. Define Trimmer (To fix "Context Window" without deleting DB data)
        # This keeps the last 10 messages for the AI, but keeps older ones in Redis
        trimmer = trim_messages(
            max_tokens=2000, # Or max_messages=10
            strategy="last",
            token_counter=self.llm, 
            include_system=True,
            start_on="human"
        )

        # 6. Wrap with Message History
        self.chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def ask(self, question: str, session_id: str) -> str:
        try:
            # Invoking the chain handles:
            # 1. Fetching history from Redis (Optimized)
            # 2. Retrieving Docs
            # 3. Generating Response
            # 4. Appending new Question + Answer to Redis automatically
            response = self.chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            log.error(f"Error in ask: {str(e)}")
            # Fallback for the "dict replace" error: 
            # If history is corrupted, clear it and retry once
            if "replace" in str(e) or "dict" in str(e):
                log.warning(f"Corrupted history detected for {session_id}, clearing...")
                self.clear_memory(session_id)
                return "I encountered an error with your session history. I have reset it. Please ask again."
            raise e

    def clear_memory(self, session_id: str):
        """2-step: LangChain clear + Redis cleanup"""
        try:
            # Step 1: LangChain history clear (xóa key chính)
            history = UpstashRedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url,
                token=self.redis_token,
            )
            history.clear()
            
            # # Step 2: Direct Redis cleanup tất cả message_store keys
            # redis = Redis(url=self.redis_url, token=self.redis_token)
            # pattern = f"message_store:{session_id}:*"
            # keys = redis.keys(pattern)
            
            # if keys:
            #     deleted = redis.delete(*keys)
            #     log.info(f"Cleaned {deleted} leftover keys for {session_id}")
            # else:
            #     log.info(f"No leftover keys for {session_id}")
                
        except Exception as e:
            log.error(f"Clear failed for {session_id}: {e}")

    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        history = UpstashRedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url,
            token=self.redis_token
        )
        return history.messages