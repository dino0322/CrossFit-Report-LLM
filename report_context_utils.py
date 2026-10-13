
from __future__ import annotations

from typing import Dict, List


def detect_page_categories(page: Dict) -> List[str]:
    """
    한 페이지에 여러 시각화/표 요소가 동시에 있을 수 있으므로
    단일 라벨 대신 multi-label로 카테고리를 수집한다.
    """
    full_text = "\n".join(
        [
            (page.get("text") or ""),
            (page.get("vision_extraction") or ""),
        ]
    ).lower()

    categories: List[str] = []

    if "performance metrics" in full_text or "baseline models performance" in full_text:
        categories.append("metrics")

    if "confusion matrix" in full_text:
        categories.append("confusion")

    if "roc curve" in full_text or "roc curves" in full_text:
        categories.append("roc")

    if "precision-recall" in full_text or "pr curve" in full_text or "pr curves" in full_text:
        categories.append("pr")

    if "calibration curve" in full_text or "calibration" in full_text:
        categories.append("calibration")

    if "lime explanation" in full_text or "lime" in full_text:
        categories.append("lime")

    if "shap summary" in full_text or "shap" in full_text:
        categories.append("shap")

    # 이번 deadlift PDF처럼 보조 통계 시각화(boxplot 등)가 같이 붙는 경우 대응
    if "boxplot" in full_text or "box plot" in full_text or "box-plot" in full_text:
        categories.append("stat")

    if not categories:
        categories.append("other")

    # 순서 안정화 + 중복 제거
    ordered = []
    seen = set()
    for cat in categories:
        if cat not in seen:
            ordered.append(cat)
            seen.add(cat)
    return ordered


def classify_page(page: Dict) -> str:
    """
    기존 구조 호환용.
    단일 문자열이 필요한 곳에서는 multi-label 결과를 ' + '로 연결해 반환한다.
    """
    categories = detect_page_categories(page)
    if categories == ["other"]:
        return "other"
    return " + ".join(categories)


def has_category(page: Dict, category: str) -> bool:
    return category in detect_page_categories(page)


def select_template_pages(pages: List[Dict]) -> List[Dict]:
    """
    공통 템플릿 선택 규칙:
    - metrics 페이지는 최대 2개
    - confusion / roc / pr / calibration / lime / shap / stat 는 각 1개씩
    - 한 페이지가 여러 카테고리를 동시에 가져도 페이지 중복 추가는 방지
    """
    selected: List[Dict] = []
    selected_nums = set()

    metrics_pages = [p for p in pages if has_category(p, "metrics")][:2]
    for p in metrics_pages:
        if p["page_num"] not in selected_nums:
            selected.append(p)
            selected_nums.add(p["page_num"])

    for category in ["confusion", "roc", "pr", "calibration", "lime", "shap", "stat"]:
        for p in pages:
            if has_category(p, category) and p["page_num"] not in selected_nums:
                selected.append(p)
                selected_nums.add(p["page_num"])
                break

    return selected


def merge_selected_and_retrieved(template_pages: List[Dict], retrieved_pages: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()

    for p in template_pages + retrieved_pages:
        if p["page_num"] not in seen:
            merged.append(p)
            seen.add(p["page_num"])

    merged.sort(key=lambda x: x["page_num"])
    return merged


def build_context_with_visuals(pages: List[Dict]) -> str:
    """
    기존 구조는 유지하되, 페이지가 multi-label일 수 있음을 context에 반영한다.
    """
    text_blocks: List[str] = []

    for item in pages:
        page_num = item["page_num"]
        category_text = classify_page(item)
        text = item.get("text", "").strip() or "[No extractable text on this page]"
        vision = item.get("vision_extraction", "").strip() or "[No vision extraction available]"

        text_blocks.append(
            f"[Page {page_num} | {category_text}]\n"
            f"[Extracted Text]\n{text}\n\n"
            f"[Vision Extraction]\n{vision}"
        )

    return "\n\n".join(text_blocks)
