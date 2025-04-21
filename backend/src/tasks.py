import logging
import time
from agent import ai_agent_handle
import os
import json
from brain import (
    detect_route,
    detect_user_intent,
    gen_doc_prompt,
    get_embedding,
    openai_chat_complete,
)
from celery import shared_task
from configs import DEFAULT_COLLECTION_NAME
from database import get_celery_app
from models import get_conversation_messages, update_chat_conversation
from rerank import rerank_documents
from splitter import split_document
from summarizer import summarize_text
from utils import setup_logging
from vectorize import add_vector, search_vector

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "final_chunk")


def follow_up_question(history, question):
    user_intent = detect_user_intent(history, question)
    logger.info(f"User intent: {user_intent}")
    return user_intent


@shared_task()
def bot_rag_answer_message(history, question):
    # Follow-up question
    new_question = follow_up_question(history, question)
    # Embedding text
    vector = get_embedding(new_question)
    logger.info(f"Get vector: {new_question}")

    # Search documents
    top_docs = search_vector(DEFAULT_COLLECTION_NAME, vector, 8)
    logger.info(f"Top docs: {top_docs}")

    # Rerank documents
    ranked_docs = rerank_documents(top_docs, new_question)

    openai_messages = history + [
        {"role": "user", "content": gen_doc_prompt(ranked_docs)},
        {"role": "user", "content": question},
    ]

    logger.info(f"Openai messages: {openai_messages}")

    assistant_answer = openai_chat_complete(openai_messages)

    logger.info(f"Bot RAG reply: {assistant_answer}")
    return assistant_answer


def index_single_node(
    title: str, content: str, collection_name=DEFAULT_COLLECTION_NAME
):
    id = time.time_ns()
    vector = get_embedding(title + ". " + content)
    add_vector_status = add_vector(
        collection_name=collection_name,
        vectors={
            id: {"vector": vector, "payload": {"title": title, "content": content}}
        },
    )
    logger.info(f"Add vector status: {add_vector_status}")
    return add_vector_status


def get_summarized_response(response):
    output = summarize_text(response)
    logger.info("Summarized response: %s", output)
    return output


@shared_task()
def bot_route_answer_message(history, question):
    # detect the route
    route = detect_route(history, question)
    if route == "legal":
        return bot_rag_answer_message(history, question)
    # elif route == "research":
    #     return ai_agent_handle(question)
    else:
        # return deepseek_chat_complete(history, question)
        return ai_agent_handle(question)


@shared_task()
def llm_handle_message(bot_id, user_id, question):
    logger.info("Start handle message")
    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)
    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]
    # Bot generation
    response = bot_route_answer_message(history, question)
    logger.info(f"Chatbot response: {response}")
    # Summarize response
    summarized_response = get_summarized_response(response)
    # Save response to history
    update_chat_conversation(bot_id, user_id, summarized_response, False)
    # Return response
    return {"role": "assistant", "content": response}
