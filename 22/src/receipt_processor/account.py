"""
勘定科目提案機能
"""

from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.receipt_processor.models import AccountInfo, ReceiptOCRResult


def format_prompt(inputs: Dict[str, Any]) -> ChatPromptTemplate:
    """
    OCR結果からプロンプトテンプレートを生成する

    Parameters:
    -----------
    inputs: Dict[str, Any]
        OCR結果とヒント情報を含む辞書

    Returns:
    --------
    ChatPromptTemplate
        適切に設定されたChatPromptTemplate
    """
    ocr_result = inputs["ocr_result"]
    hint = inputs.get("hint", "")

    # 品目情報のテキスト化
    items_text = ""
    if ocr_result.items:
        items_text = "品目リスト:\n"
        for item in ocr_result.items:
            items_text += f"- {item.name}: {item.price}円\n"

    # その他情報のテキスト化
    other_info_text = ""
    if ocr_result.other_info:
        other_info_text = "その他情報:\n"
        for info in ocr_result.other_info:
            other_info_text += f"- {info.key}: {info.value}\n"

    # フィードバックセクションの制御
    feedback_section = ""
    if hint:
        feedback_section = f"""
【ユーザーフィードバック】
{hint}
"""

    # システムプロンプト
    system_prompt = """\
あなたは日本の会計士です。領収書の情報から最適な勘定科目を提案してください。
中小企業の一般的な勘定科目を使用し、適切な補助科目、取引先情報、摘要も含めてください。
なぜその勘定科目が適切かの理由も必ず含めてください。

使用可能な主な勘定科目の例:
- 旅費交通費: 交通機関の利用料、出張費など
- 通信費: 電話料金、インターネット料金など
- 消耗品費: 事務用品、日用品など
- 会議費: 会議での飲食代など
- 接待交際費: 取引先との会食、贈答品など
- 広告宣伝費: 広告費、販促物など
- 新聞図書費: 書籍、雑誌、新聞代など
- 水道光熱費: 電気代、ガス代、水道代など
- 地代家賃: オフィス賃料など
- 雑費: 他の科目に当てはまらない少額の経費
""".strip()

    # ユーザープロンプト
    user_prompt = f"""\
以下の領収書情報から、最適な勘定科目情報を提案してください。

【領収書情報】
日付: {ocr_result.date}
金額: {ocr_result.amount}円
店舗/発行者: {ocr_result.shop_name}
{items_text}
{other_info_text}
詳細: {ocr_result.raw_text}
{feedback_section}
""".strip()

    # ChatPromptTemplateを作成して返す
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )


def suggest_account_info(
    ocr_result: ReceiptOCRResult,
    hint: str | None = None,
    model_name: str = "claude-3-7-sonnet-20250219",
) -> AccountInfo:
    """
    OCR結果から適切な勘定科目情報を提案する

    Parameters:
    -----------
    ocr_result: ReceiptOCRResult
        OCR処理結果の構造化データ
    hint: str | None
        ユーザーからの修正ヒント（あれば）
    model_name: str
        使用するClaudeモデル名

    Returns:
    --------
    AccountInfo
        提案された勘定科目情報
    """
    # LLMの初期化
    llm = ChatAnthropic(
        model_name=model_name,
        temperature=0.2,
        timeout=None,
        stop=None,
        max_retries=3,
    )

    # プロンプト生成用のRunnableLambda
    prompt_generator = RunnableLambda(format_prompt)

    # チェーンの構築
    account_chain = prompt_generator | llm.with_structured_output(
        AccountInfo,
        mode="function_calling",
    )

    # 勘定科目情報を生成
    account_info: AccountInfo = account_chain.invoke(
        {"ocr_result": ocr_result, "hint": hint}
    )  # type: ignore

    return account_info
