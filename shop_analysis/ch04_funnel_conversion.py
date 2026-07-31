"""
4장. 퍼널과 전환 분석
문제 08: 세션 퍼널 구축과 단계별 이탈
문제 09: 요일×시간대 전환 패턴과 광고 시간 최적화
문제 10: 행동 로그와 주문 데이터 대사

실행: python ch04_funnel_conversion.py
"""

import pandas as pd
import numpy as np
from shop_analysis.common_order_items import net_ledger

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

LOG_COLS = ["session_id", "event_type", "event_time", "customer_id"]
LOG_DTYPE = {"session_id": "int32", "event_type": "category"}


def load_logs():
    logs = pd.read_csv(
        "data/web_logs.csv",
        usecols=LOG_COLS,
        dtype=LOG_DTYPE,
        parse_dates=["event_time"],
    )
    return logs


# ============================================================
# 문제 08. 세션 퍼널 구축과 단계별 이탈
# ============================================================
def session_funnel(logs):
    reach = logs.drop_duplicates(["session_id", "event_type"])
    mat = pd.crosstab(reach["session_id"], reach["event_type"])
    mat = (mat > 0).astype(int)

    n_sessions = len(mat)
    n_view = mat["view"].sum() if "view" in mat else 0
    n_cart = mat["cart"].sum() if "cart" in mat else 0
    n_purchase = mat["purchase"].sum() if "purchase" in mat else 0

    funnel = pd.Series(
        {"전체 세션": n_sessions, "view 도달": n_view, "cart 도달": n_cart, "purchase 도달": n_purchase}
    )
    view_to_cart = n_cart / n_view
    cart_to_purchase = n_purchase / n_cart
    view_to_purchase = n_purchase / n_view

    # cart를 거치지 않고 바로 purchase한 세션(직접구매)도 규칙으로 명시해 별도 집계
    direct_purchase = ((mat.get("purchase", 0) == 1) & (mat.get("cart", 0) == 0)).sum()

    return funnel, view_to_cart, cart_to_purchase, view_to_purchase, direct_purchase


# ============================================================
# 문제 09. 요일×시간대 전환 패턴과 광고 시간 최적화
# ============================================================
def hourly_conversion(logs):
    session_start = logs.groupby("session_id")["event_time"].min().rename("start_time")
    session_purchase = (
        logs.groupby("session_id")["event_type"].apply(lambda s: (s == "purchase").any())
    ).rename("converted")

    sess = pd.concat([session_start, session_purchase], axis=1).reset_index()
    sess["hour"] = sess["start_time"].dt.hour
    sess["dow"] = sess["start_time"].dt.dayofweek  # 0=월요일

    hourly = sess.groupby("hour").agg(세션수=("session_id", "count"), 전환율=("converted", "mean"))

    pivot_sessions = sess.pivot_table(index="dow", columns="hour", values="session_id", aggfunc="count", fill_value=0)
    pivot_conv = sess.pivot_table(index="dow", columns="hour", values="converted", aggfunc="mean")

    # 최소 표본 기준(예: 30세션 미만) 이하 셀은 신뢰 불가로 마스킹
    pivot_conv_masked = pivot_conv.where(pivot_sessions >= 30)

    return hourly, pivot_sessions, pivot_conv_masked


# ============================================================
# 문제 10. 행동 로그와 주문 데이터 대사
# ============================================================
def log_order_reconciliation(logs, ledger):
    # web 채널 유효 주문만 대상 (로그는 사이트 행동이므로 store/app과 대조 무의미)
    web_orders = ledger[(ledger["channel"] == "web") & ledger["is_net"]].copy()
    web_orders["date"] = web_orders["order_datetime"].dt.date
    order_daily = web_orders.groupby("date").agg(
        주문건수=("order_id", "nunique"), 주문고객수=("customer_id", "nunique")
    )

    log_purchase = logs[logs["event_type"] == "purchase"].dropna(subset=["customer_id"]).copy()
    log_purchase["date"] = log_purchase["event_time"].dt.date
    log_daily = log_purchase.groupby("date").agg(
        구매이벤트수=("session_id", "count"), 구매고객수=("customer_id", "nunique")
    )

    total_compare = pd.DataFrame({
        "로그_purchase이벤트": [len(log_purchase)],
        "로그_고유고객": [log_purchase["customer_id"].nunique()],
        "주문_유효주문수": [web_orders["order_id"].nunique()],
        "주문_고유고객": [web_orders["customer_id"].nunique()],
    })

    monthly = pd.concat([
        order_daily.resample("D").sum() if False else order_daily,  # placeholder no-op
    ])
    order_daily.index = pd.to_datetime(order_daily.index)
    log_daily.index = pd.to_datetime(log_daily.index)
    order_monthly = order_daily.resample("ME").sum()
    log_monthly = log_daily.resample("ME").sum()
    monthly_compare = pd.concat([order_monthly, log_monthly], axis=1)

    # 고객x일 단위 병합으로 유실/과다 규모 추정
    order_cust_day = web_orders[["customer_id", "date"]].drop_duplicates()
    order_cust_day["date"] = pd.to_datetime(order_cust_day["date"])
    log_cust_day = log_purchase[["customer_id", "date"]].drop_duplicates()
    log_cust_day["date"] = pd.to_datetime(log_cust_day["date"])

    merged = order_cust_day.merge(log_cust_day, on=["customer_id", "date"], how="outer", indicator=True)
    only_order = (merged["_merge"] == "left_only").sum()
    only_log = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()

    n_cust_null_log = logs["customer_id"].isna().sum()

    return total_compare, monthly_compare, only_order, only_log, both, n_cust_null_log


if __name__ == "__main__":
    logs = load_logs()
    ledger = net_ledger()

    print("=" * 70)
    print("문제 08. 세션 퍼널 구축과 단계별 이탈")
    print("=" * 70)
    funnel, v2c, c2p, v2p, direct = session_funnel(logs)
    print("\n[퍼널 도달 표]")
    print(funnel.to_string())
    print(f"\nview->cart 전환율: {v2c:.1%}")
    print(f"cart->purchase 전환율: {c2p:.1%}")
    print(f"view->purchase(정규경로) 전체 전환율: {v2p:.1%}")
    print(f"cart 없이 바로 구매한 세션(직접구매): {direct:,}건")
    print(
        "\n[병목 진단] view->cart 단계가 상대적으로 낮다면 상품 상세페이지 개선이,\n"
        "cart->purchase가 낮다면 결제 단계 이탈(배송비/결제수단) 개선이 시급하다.\n"
        "[실험 아이디어] 1) 장바구니 담기 후 24시간 리마인드 푸시  2) 결제 페이지 원클릭 결제 도입"
    )

    print("\n" + "=" * 70)
    print("문제 09. 요일×시간대 전환 패턴과 광고 시간 최적화")
    print("=" * 70)
    hourly, pivot_sessions, pivot_conv = hourly_conversion(logs)
    print("\n[시간대별 세션수·전환율]")
    print(hourly.to_string())
    print("\n[요일x시간대 세션수 피벗] (0=월요일)")
    print(pivot_sessions.to_string())
    print("\n[요일x시간대 전환율 피벗] (표본 30 미만 마스킹)")
    print(pivot_conv.round(3).to_string())
    top_traffic_hour = hourly["세션수"].idxmax()
    top_conv_hour = hourly["전환율"].idxmax()
    print(f"\n[대조] 트래픽 최고 시간대: {top_traffic_hour}시 / 전환율 최고 시간대: {top_conv_hour}시")
    if top_traffic_hour == top_conv_hour:
        print("-> 두 시간대가 일치하므로 단순 트래픽 기준 입찰과 전환 기준 입찰의 결론이 같다.")
    else:
        print("-> 두 시간대가 다르므로, 트래픽만 보고 입찰하면 전환이 낮은 시간에 예산을 쓰게 될 위험이 있다.")

    print("\n" + "=" * 70)
    print("문제 10. 행동 로그와 주문 데이터 대사")
    print("=" * 70)
    total_compare, monthly_compare, only_order, only_log, both, n_cust_null_log = log_order_reconciliation(logs, ledger)
    print("\n[전체 대조: web 채널 한정]")
    print(total_compare.to_string(index=False))
    print("\n[월별 대조]")
    print(monthly_compare.to_string())
    print(f"\n[고객x일 단위 대사] 주문에만 있음: {only_order:,}건 / 로그에만 있음: {only_log:,}건 / 둘 다 있음: {both:,}건")
    print(f"customer_id 결측 로그(대사 불가, 별도 계상): {n_cust_null_log:,}건")
    match_rate = both / (both + only_order) if (both + only_order) > 0 else float("nan")
    print(f"\n[유실률 추정] 주문 기준 로그 매칭률: {match_rate:.1%}")
    print(
        "[로그 기반 지표 사용 가이드라인]\n"
        "  1) 로그 전환율은 반드시 web 채널 주문과만 대조해야 하며, store·app 성과 판단에 쓰지 않는다.\n"
        "  2) customer_id 결측 로그(익명 방문)는 대사가 불가능하므로 전환율 분모에서 그 비중을 항상 함께 보고한다.\n"
        "  3) 로그 단독 수치(구매 이벤트 수)는 추세 참고용으로만 쓰고, 정확한 매출·주문 건수는 항상 주문 시스템을 기준으로 삼는다."
    )
