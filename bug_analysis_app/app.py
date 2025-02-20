from flask import Flask, render_template, request, jsonify
import os
import json
import pandas as pd
from typing import List
from alibabacloud_devops20210625.client import Client as devops20210625Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_devops20210625 import models as devops_20210625_models
import requests
from bs4 import BeautifulSoup
import chromadb
from openai import AzureOpenAI
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Configuration (from the original notebook)
config = open_api_models.Config(
    access_key_id=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"),
    access_key_secret=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
)
config.endpoint = "devops.cn-hangzhou.aliyuncs.com"
client = devops20210625Client(config)
runtime = util_models.RuntimeOptions()
headers = {}
organization_id = "6189f099041d450d2c253abc"
project_id = "0c593b861aafc8b8546d67dd65"

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="docs")


# Azure Embedding Function (from the original notebook)
def get_azure_embeddings(texts):
    logging.info(f"Fetching embeddings from Azure for {len(texts)} texts.")
    url = "https://esat-us.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ.get("AZURE_API_KEY_XM"),
    }
    data = {"input": texts}
    response = requests.post(url, headers=headers, json=data, timeout=15)
    response.raise_for_status()  # Raise an exception for bad status codes
    return [item["embedding"] for item in response.json()["data"]]


# Function to get work item info (from the original notebook)
def get_work_item_info(workitem_id, organization_id):
    logging.info(f"Getting detail info for workitem: {workitem_id}")
    retry_count = 0
    while retry_count < 3:
        try:
            work_item = client.get_work_item_info(
                workitem_id=workitem_id, organization_id=organization_id
            )
            return work_item.body.to_map()["workitem"]
        except Exception as e:
            retry_count += 1
            logging.error(f"An error occurred: {e}, retrying... ({retry_count}/3)")
            time.sleep(15)
            if retry_count == 3:
                logging.error("Max retries reached. Failing.")
                raise


# Function to clean HTML (from the original notebook)
def get_clean_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text(strip=True)
    return clean_text


# Function to fetch and process workitems (adapted from the original notebook)
def fetch_and_process_workitems():
    logging.info("Fetching and processing workitems...")
    condition = {
        "spaceType": "Project",
        "spaceIdentifier": project_id,
        "category": "Bug",
        "toPage": 1,
        "pageSize": 25,
        "conditions": '{"conditionGroups":[[{"fieldIdentifier":"workitemType","operator":"CONTAINS","value":["77c180ff4e5da530ca9b8e332e"],"toValue":null,"className":"workitemType","format":"list"}]]}',
        "searchType": "LIST",
    }

    items_array = []
    next_token = None

    while True:
        req = devops_20210625_models.ListWorkitemsRequest(
            space_type="Project",
            category="Bug",
            conditions=condition.get("conditions"),
            max_results=100,
            search_type="LIST",
            space_identifier=project_id,
            next_token=next_token,
        )
        try:
            res = client.list_workitems(organization_id=organization_id, request=req)
            result = res.body.to_map()
            current_items = result["workitems"]
            items_array.extend(current_items)
            next_token = result.get("nextToken")
            if not next_token or len(next_token) <= 10:
                break
        except Exception as e:
            logging.error(f"Error fetching workitems: {e}")
            return []  # Return empty list on error

    work_item_info_list = []
    for item in items_array:
        try:
            workitem = get_work_item_info(item["identifier"], organization_id)
            comments = client.get_workitem_comment_list(
                organization_id=organization_id, workitem_id=item["identifier"]
            ).body.to_map()["commentList"]
            workitem["comments"] = comments
            work_item_info_list.append(workitem)
        except Exception as e:
            logging.error(f"Error getting workitem details or comments: {e}")
            # Continue to the next item even if one fails

    documents = []
    for item in work_item_info_list:
        subject = item["subject"]
        document = (
            get_clean_text_from_html(json.loads(item["document"]).get("htmlValue"))
            if item["document"]
            else ""
        )
        comments = []
        for comment in item["comments"]:
            comments.append(
                get_clean_text_from_html(json.loads(comment["content"])["htmlValue"])
            )
        comments = "\\n".join(comments)
        documents.append(
            f"""## 问题标题:{subject}

## 问题补充描述:{document}

## 问题评论(含答案):{comments}

"""
        )
    return documents


# Function to add documents to ChromaDB (adapted from original notebook)
def add_documents_to_chromadb(documents):
    logging.info("Adding documents to ChromaDB...")
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        try:
            batch_embeddings = get_azure_embeddings(batch_docs)
            collection.add(
                ids=[str(i + j) for j in range(len(batch_docs))],
                embeddings=batch_embeddings,
                documents=batch_docs,
            )
        except Exception as e:
            logging.error(f"Error adding batch to chromadb: {e}")
            # Decide if you want to continue or stop on error. Here, we continue.


# Function to perform RAG (adapted from the original notebook)
def perform_rag(prompt):
    logging.info(f"Performing RAG for prompt: {prompt}")
    try:
        embedding = get_azure_embeddings([prompt])[0]
        results = collection.query(query_embeddings=[embedding], n_results=5)

        data = ""
        for index, doc in enumerate(results["documents"][0]):
            data = f"{data}- [问题{index + 1}]{doc}\n\n"

        system_prompt = f"""你是一个客服问题解答助手，负责根据用户提供的新问题，以及之前的历史记录来推理问题的原因并提供建议。

请根据以下历史记录，回答用户的新问题。你的回答应包括以下几个部分：

1. **问题分析**：根据新问题，明确问题涉及的内容，找出关键词、产品或券类型等关键信息。
2. **相关历史记录**：从历史记录中找到与新问题相似或相关的案例，分析其中的原因。
3. **可能原因**：基于历史记录，推理出新问题可能的原因。例如，券无法使用可能是因为商品类别限制、订单类型选择错误等。
4. **解决方案**：给出针对问题的解决方案，包括可能的操作步骤或检查项。说明用户需要做什么或需要联系谁来解决问题。

### 历史记录：
{data}
"""
        question = f"""新问题：
{prompt}

请根据历史记录回答新问题，找出问题的原因，并提供建议。"""

        openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY_XM"),
            azure_endpoint="https://esat-us.openai.azure.com",
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
        )
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.9,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        answer = completion.choices[0].message.content
        return answer, results["documents"][0]  # Return answer and source documents

    except Exception as e:
        logging.error(f"Error during RAG: {e}")
        return "An error occurred while processing your request.", []


# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]
        answer_markdown, references = perform_rag(question)
        print(f"answer_markdown:{answer_markdown}")
        return render_template(
            "index.html", question=question, answer=answer_markdown, references=references
        )
    return render_template("index.html")


@app.route("/refresh_data", methods=["POST", "GET"])
def refresh_data():
    """Endpoint to manually refresh data from the API."""
    try:
        documents = fetch_and_process_workitems()
        if documents:
            add_documents_to_chromadb(documents)
            return jsonify(
                {"status": "success", "message": "Data refreshed successfully!"}
            )
        else:
            return jsonify({"status": "error", "message": "Failed to fetch data."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Removed data loading on startup
    app.run(debug=True, port=5400)
