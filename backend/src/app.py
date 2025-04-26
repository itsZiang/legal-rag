import logging
import time
from typing import Dict, Optional

from celery.result import AsyncResult
from database import get_db
from fastapi import Depends, FastAPI, HTTPException
from indexing import indexing
from models import ChatConversation, insert_document
from pydantic import BaseModel
from sqlalchemy.orm import Session
from tasks import index_single_node, llm_handle_message
from utils import setup_logging
from vectorize import create_collection

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI()


class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botFinancial"
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    if data.sync_request:
        response = llm_handle_message(bot_id, user_id, user_message)
        return response
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > 60:  # 60 seconds timeout
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "task_result": task_result.result,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(0.5)  # sleep for 0.5 seconds before retrying
        else:
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/collection/create")
async def create_vector_collection(data: Dict):
    collection_name = data.get("collection_name")
    create_status = create_collection(collection_name)
    logging.info(f"Create collection {collection_name} status: {create_status}")
    return {"status": create_status is not None}


@app.post("/document/create")
async def create_document(data: Dict[str, str]):
    title = data.get("title")
    content = data.get("content")
    create_status = insert_document(title, content)
    logging.info(f"Create document status: {create_status}")
    index_status = index_single_node(title, content)
    return {"status": create_status is not None, "index_status": index_status}


@app.post("/document/index")
async def indexing_multiple_nodes():
    index_status = indexing()
    logging.info(f"Index documents status: {index_status}")
    return {"status": index_status is not None}


@app.post("/conversation/delete")
async def delete_all_conversations(db: Session = Depends(get_db)):
    db.query(ChatConversation).delete()
    db.commit()
    return {"status": "all conversations deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=2, log_level="info")
