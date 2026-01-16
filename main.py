# ============================================================================
# Import
# ============================================================================

# Python標準ライブラリ
import os
import json
import logging
from datetime import datetime, timedelta, timezone

# サードパーティーライブラリ
import functions_framework
import pandas as pd
import requests

# Google Cloudサービス
from google.cloud import storage, bigquery
import vertexai
from vertexai.generative_models import GenerativeModel

# Flaskウェブフレームワーク
from flask import Request, jsonify

# ============================================================================
# 環境変数
# ============================================================================

# Slack Webhook URL
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# Google Cloud Storage URI
CUSTOMER_ATTRIBUTE_ANALYSIS_PROMPT_URI = os.environ.get(
    "CUSTOMER_ATTRIBUTE_ANALYSIS_PROMPT_URI"
)
CUSTOMER_APPROACH_RECOMMENDATION_PROMPT_URI = os.environ.get(
    "CUSTOMER_APPROACH_RECOMMENDATION_PROMPT_URI"
)
SQL_GCS_URI = os.environ.get("SQL_GCS_URI")

# Project
PROJECT_ID = os.environ.get("PROJECT_ID")
USE_LEGACY_SQL = os.environ.get("USE_LEGACY_SQL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)


# ============================================================================
# 外部との接続
# ============================================================================
def post_to_slack(message: str, webhook_url: str):
    """Slackにテキストメッセージを送信"""
    payload = {"text": message}

    res = requests.post(webhook_url, json=payload)

    if res.status_code != 200:
        logging.error(f"Slack webhook error: {res.text}")


def run_query(sql: str, project: str) -> dict:
    """BigQuery に SQL を投げて完了を待つ。戻り値は job 情報の dict"""
    bd_client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = USE_LEGACY_SQL

    query_job = bd_client.query(sql, job_config=job_config)

    return query_job.to_dataframe()


def read_text_from_gcs(uri: str) -> str:
    """引数で与えられた Google Cloud Storage URI からテキストを返す"""
    if not uri or not uri.startswith("gs://"):
        raise ValueError("The Google Cloud Storage URI must begin with 'gs://.'")

    # Google Cloud Storage URI を分解して、バケット名とファイル名を取得
    _, _, path = uri.partition("gs://")
    bucket_name, _, file_name = path.partition("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if not blob.exists():
        raise FileNotFoundError(f"{uri} not found")

    return blob.download_as_text()


def call_gemini_with_payload(
    prompt_template: str,
    placeholder_token: str,
    payload,
    project: str,
) -> str:
    """prompt_template 内の placeholder_token を payload の文字列表現で置き換えて Gemini を呼び出す。
    - payload が dict/list の場合: JSON文字列にして埋め込む
    - payload が str の場合: そのまま埋め込む
    """
    vertexai.init(project=project, location="asia-northeast1")
    model = GenerativeModel("gemini-2.5-pro")

    if isinstance(payload, str):
        payload_text = payload
    else:
        payload_text = json.dumps(payload, ensure_ascii=False)

    prompt = prompt_template.replace(placeholder_token, payload_text)

    response = model.generate_content(prompt)
    return response.text


def build_gemini_payload_from_df(df: dict) -> dict:
    # 実行年月・日時を取得する
    JST = timezone(timedelta(hours=+9), "JST")
    target_month = datetime.now(JST).strftime("%Y-%m")
    customers: list[dict] = []

    for _, row in df.iterrows():
        # --- months: [{year_month, count}] -> {"2025-09": 3, ...} ---
        months_dict: dict[str, int] = {}
        months_raw = row.get("months")
        if isinstance(months_raw, (list, tuple)):
            for m in months_raw:
                # BigQuery Row or dict の両方に対応
                year_month = (
                    m.get("year_month") if isinstance(m, dict) else m["year_month"]
                )
                count = m.get("count") if isinstance(m, dict) else m["count"]
                months_dict[str(year_month)] = int(count)

        # --- inquiry_type_count: [{inquiry_type, count}] -> dict ---
        inquiry_type_count: dict[str, int] = {}
        itypes_raw = row.get("inquiry_type_count")
        if isinstance(itypes_raw, (list, tuple)):
            for t in itypes_raw:
                inquiry_type = (
                    t.get("inquiry_type") if isinstance(t, dict) else t["inquiry_type"]
                )
                count = t.get("count") if isinstance(t, dict) else t["count"]
                inquiry_type_count[str(inquiry_type)] = int(count)

        # --- channel_count: [{channel, count}] -> dict ---
        channel_count: dict[str, int] = {}
        channels_raw = row.get("channel_count")
        if isinstance(channels_raw, (list, tuple)):
            for c in channels_raw:
                channel = c.get("channel") if isinstance(c, dict) else c["channel"]
                count = c.get("count") if isinstance(c, dict) else c["count"]
                channel_count[str(channel)] = int(count)

        # --- inquiries: list[struct] -> list[dict]（ISO文字列に整形） ---
        inquiries_list: list[dict] = []
        for i in row["inquiries"]:
            # BigQuery Row or dict の両対応
            def _get(obj, key):
                return obj.get(key) if isinstance(obj, dict) else obj[key]

            inq_ts = _get(i, "inquired_at")
            if hasattr(inq_ts, "isoformat"):
                inq_ts = inq_ts.isoformat()

            inquiries_list.append(
                {
                    "inquired_at": inq_ts,
                    "inquiry_year_month": _get(i, "inquiry_year_month"),
                    "channel": _get(i, "channel"),
                    "inquiry_type": _get(i, "inquiry_type"),
                    "subject": _get(i, "subject"),
                    "details": _get(i, "details"),
                    "agent_response": _get(i, "agent_response"),
                    "response_action": _get(i, "response_action"),
                    "next_action_type": _get(i, "next_action_type"),
                    "is_resolved": bool(_get(i, "is_resolved")),
                }
            )

        # --- 日付系を ISO 文字列に ---
        registered_at = row.get("registered_at")
        if hasattr(registered_at, "isoformat"):
            registered_at = registered_at.isoformat()

        last_inquired_at = row.get("last_inquired_at")
        if hasattr(last_inquired_at, "isoformat"):
            last_inquired_at = last_inquired_at.isoformat()

        # --- 顧客オブジェクトを構築 ---
        customer_obj = {
            "customer_id": str(row["customer_id"]),
            "customer_name": row.get("customer_name"),
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "prefecture": row.get("prefecture"),
            "registered_at": registered_at,
            "inquiry_summary": {
                "total_inquiries_last_3m": int(row["total_inquiries_last_3m"]),
                "months": months_dict,
                "inquiry_type_count": inquiry_type_count,
                "channel_count": channel_count,
                "resolved_count": int(row["resolved_count"]),
                "unresolved_count": int(row["unresolved_count"]),
                "last_inquired_at": last_inquired_at,
            },
            "inquiries": inquiries_list,
        }

        customers.append(customer_obj)

    payload = {
        "run_context": {
            "target_month": target_month,
            "lookback_months": 3,
        },
        "customers": customers,
    }

    return payload


@functions_framework.http
def execute_from_gcs(request: Request):
    """関数のエントリポイント"""
    try:
        if not SQL_GCS_URI:
            return (
                jsonify({"error": "Failed to retrieve the Google Cloud Storage URL."}),
                400,
            )
        logging.info("Reading SQL from [%s] ...", SQL_GCS_URI)

        # Google Cloud Storage からSQLを取得
        sql = read_text_from_gcs(SQL_GCS_URI)

        if not PROJECT_ID:
            return jsonify({"error": "Failed to retrieve the Project ID."}), 400
        logging.info("Executing SQL [len=%d chars] ...", len(sql))

        # BigQuery 上でクエリを実行し、データフレームを取得
        df = run_query(sql, project=PROJECT_ID)
        payload = build_gemini_payload_from_df(df)

        # Step1: 顧客属性分析
        if not CUSTOMER_ATTRIBUTE_ANALYSIS_PROMPT_URI:
            return (
                jsonify(
                    {
                        "error": "Failed to retrieve the Customer attribute analysis prompt."
                    }
                ),
                400,
            )
        logging.info(
            "Reading Prompt from [%s] ...", CUSTOMER_ATTRIBUTE_ANALYSIS_PROMPT_URI
        )

        # Google Cloud Storage から顧客属性を定義するためのGemini用プロンプトを取得
        customer_attribute_analysis_prompt = read_text_from_gcs(
            CUSTOMER_ATTRIBUTE_ANALYSIS_PROMPT_URI
        )

        customer_attribute_analysis_result = call_gemini_with_payload(
            prompt_template=customer_attribute_analysis_prompt,
            placeholder_token="{payload_json}",
            payload=payload,
            project=PROJECT_ID,
        )

        if not CUSTOMER_APPROACH_RECOMMENDATION_PROMPT_URI:
            return (
                jsonify(
                    {
                        "error": "Failed to retrieve the Customr approach recommendation prompt."
                    }
                ),
                400,
            )
        logging.info(
            "Reading Prompt from [%s] ...", CUSTOMER_APPROACH_RECOMMENDATION_PROMPT_URI
        )

        # Google Cloud Storage から顧客へのアプローチを出力させるためのGemini用プロンプトを取得
        customer_approach_recommendation_prompt = read_text_from_gcs(
            CUSTOMER_APPROACH_RECOMMENDATION_PROMPT_URI
        )

        customer_approach_recommendation_result = call_gemini_with_payload(
            prompt_template=customer_approach_recommendation_prompt,
            placeholder_token="{attributes_json}",
            payload=customer_attribute_analysis_result,
            project=PROJECT_ID,
        )

        slack_message = (
            f"AI提案（テキスト出力）:\n{customer_approach_recommendation_result}"
        )

        if not SLACK_WEBHOOK_URL:
            return jsonify({"error": "Failed to retrieve the SLACK WEBHOOK URL."}), 400
        logging.info("Sending Message to [%s] ...", SLACK_WEBHOOK_URL)
        # Slackの指定のチャンネルにGeminiの出力結果を送信
        post_to_slack(slack_message, SLACK_WEBHOOK_URL)

        return jsonify({"message": "Query executed", "rows": len(df)}), 200

    except Exception as e:
        logging.exception("Failed to execute query")
        return jsonify({"error": str(e)}), 500
