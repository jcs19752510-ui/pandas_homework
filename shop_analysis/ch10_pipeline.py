"""
10장. 대용량 로그 마케팅 파이프라인
문제 27: 100만 행 로그 메모리 진단과 최적화
문제 28: 청크 집계로 일별 활성 고객 산출
문제 29: 로그 기반 상품 관심 스코어 파이프라인
문제 30: 월간 마케팅 지표 자동화 파이프라인

실행: python ch10_pipeline.py
"""

import time
import pandas as pd
import numpy as np
from shop_analysis.common_order_items import build_ledger, clean_category

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")


# ============================================================
# 문제 27. 100만 행 로그 메모리 진단과 최적화
# ============================================================
def memory_diagnosis():
    t0 = time.perf_counter()
    raw = pd.read_csv("data/web_logs.csv")
    t_raw = time.perf_counter() - t0
    mem_raw = raw.memory_usage(deep=True)
    mem_raw_total = mem_raw.sum()

    # 단계1: usecols로 필요한 열만
    t0 = time.perf_counter()
    step1 = pd.read_csv("data/web_logs.csv", usecols=["customer_id", "event_time", "event_type", "product_id", "session_id"])
    t_step1 = time.perf_counter() - t0
    mem_step1 = step1.memory_usage(deep=True).sum()

    # 단계2: 숫자열 다운캐스트
    step2 = step1.copy()
    step2["session_id"] = pd.to_numeric(step2["session_id"], downcast="integer")
    step2["product_id"] = pd.to_numeric(step2["product_id"], downcast="float")
    step2["customer_id"] = pd.to_numeric(step2["customer_id"], downcast="float")
    mem_step2 = step2.memory_usage(deep=True).sum()

    # 단계3: 저카디널리티 문자열 category화
    step3 = step2.copy()
    step3["event_type"] = step3["event_type"].astype("category")  # 저카디널리티(4종) -> 유리
    mem_step3 = step3.memory_usage(deep=True).sum()

    # category가 불리한 열의 예시: session_id처럼 거의 매행 다른 값을 가진 열
    session_as_category_mem = step2["session_id"].astype("category").memory_usage(deep=True)
    session_as_int_mem = step2["session_id"].memory_usage(deep=True)

    summary = pd.DataFrame({
        "메모리(MB)": [mem_raw_total, mem_step1, mem_step2, mem_step3],
        "적재시간(초)": [t_raw, t_step1, None, None],
    }, index=["0.원본 read_csv", "1.usecols", "2.+다운캐스트", "3.+category"])
    summary["메모리(MB)"] = summary["메모리(MB)"] / 1e6

    return raw.dtypes, mem_raw, summary, session_as_category_mem, session_as_int_mem


def standard_loader(path="data/web_logs.csv"):
    """팀 배포용 표준 적재 함수"""
    df = pd.read_csv(
        path,
        usecols=["customer_id", "event_time", "event_type", "product_id", "session_id"],
        dtype={"event_type": "category"},
        parse_dates=["event_time"],
    )
    df["session_id"] = pd.to_numeric(df["session_id"], downcast="integer")
    df["customer_id"] = pd.to_numeric(df["customer_id"], downcast="float")
    df["product_id"] = pd.to_numeric(df["product_id"], downcast="float")
    return df


# ============================================================
# 문제 28. 청크 집계로 일별 활성 고객 산출
# ============================================================
def chunked_dau(path="data/web_logs.csv", chunksize=100_000):
    daily_customers = {}
    for chunk in pd.read_csv(
        path, usecols=["customer_id", "event_time", "event_type"],
        parse_dates=["event_time"], chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["customer_id"])  # 익명 방문은 DAU 분모 제외
        chunk["date"] = chunk["event_time"].dt.date
        for date, grp in chunk.groupby("date"):
            daily_customers.setdefault(date, set()).update(grp["customer_id"].tolist())

    dau = pd.Series({d: len(s) for d, s in daily_customers.items()}).sort_index().rename("DAU")
    return dau


def full_load_dau_check(path="data/web_logs.csv"):
    """전량 적재 결과와 대조해 청크 결과의 정확성을 검증"""
    df = pd.read_csv(path, usecols=["customer_id", "event_time"], parse_dates=["event_time"])
    df = df.dropna(subset=["customer_id"])
    df["date"] = df["event_time"].dt.date
    return df.groupby("date")["customer_id"].nunique().sort_index().rename("DAU_전량적재")


def chunksize_benchmark(path="data/web_logs.csv"):
    results = {}
    for cs in [50_000, 200_000]:
        t0 = time.perf_counter()
        _ = chunked_dau(path, chunksize=cs)
        results[cs] = time.perf_counter() - t0
    return results


# ============================================================
# 문제 29. 로그 기반 상품 관심 스코어 파이프라인
# ============================================================
def interest_score_pipeline(weight=None, recency_boost=1.5, chunksize=200_000, top_n=20):
    if weight is None:
        weight = {"view": 1, "cart": 3, "purchase": 5}

    agg = {}  # (product_id, is_recent) -> weighted count accumulator, per event type

    for chunk in pd.read_csv(
        "data/web_logs.csv",
        usecols=["product_id", "event_type", "event_time"],
        parse_dates=["event_time"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["product_id"])  # search 이벤트(product_id 결측) 제외
        chunk["is_recent"] = chunk["event_time"].dt.month == 6
        grouped = chunk.groupby(["product_id", "event_type", "is_recent"], observed=True).size()
        for (pid, etype, recent), cnt in grouped.items():
            key = (pid, etype, recent)
            agg[key] = agg.get(key, 0) + cnt

    rows = []
    for (pid, etype, recent), cnt in agg.items():
        w = weight.get(etype, 0)
        boost = recency_boost if recent else 1.0
        rows.append({"product_id": pid, "score_contrib": cnt * w * boost})

    score_df = pd.DataFrame(rows).groupby("product_id")["score_contrib"].sum().rename("관심스코어")

    products = pd.read_csv("data/products.csv").drop_duplicates()
    products["category"] = clean_category(products["category"])

    top = score_df.sort_values(ascending=False).head(top_n).reset_index()
    top = top.merge(products[["product_id", "product_name", "category", "price"]], on="product_id", how="left")

    return top, weight


# ============================================================
# 문제 30. 월간 마케팅 지표 자동화 파이프라인
# ============================================================
def load_and_clean():
    return build_ledger()


def metric_net_revenue(ledger, month):
    net = ledger[ledger["is_net"] & (ledger["order_datetime"].dt.to_period("M") == month)]
    val = net["line_amount"].sum()
    assert val >= 0, "월 순매출은 음수가 될 수 없다"
    return val


def metric_channel_aov(ledger, month):
    net = ledger[ledger["is_net"] & (ledger["order_datetime"].dt.to_period("M") == month)]
    orders = net.groupby(["channel", "order_id"])["line_amount"].sum().reset_index()
    aov = orders.groupby("channel")["line_amount"].mean()
    assert aov.notna().all(), "채널별 AOV에 결측이 있으면 안 된다"
    return aov


def metric_new_customer_share(ledger, month):
    net = ledger[ledger["is_net"]]
    first_month = net.groupby("customer_id")["order_datetime"].min().dt.to_period("M")
    this_month = net[net["order_datetime"].dt.to_period("M") == month]
    is_new = this_month["customer_id"].map(first_month) == month
    total = this_month["line_amount"].sum()
    new_rev = this_month.loc[is_new, "line_amount"].sum()
    share = new_rev / total if total > 0 else np.nan
    assert 0 <= share <= 1, "신규 매출 비중은 0~1 사이여야 한다"
    return share


def metric_m1_retention(ledger, month):
    """전월(m-1) 첫 구매 코호트가 당월(m)에 재구매한 비율. (당월까지 데이터만으로 항상 계산 가능)"""
    net = ledger[ledger["is_net"]]
    prev_month = month - 1
    first_month = net.groupby("customer_id")["order_datetime"].min().dt.to_period("M")
    prev_cohort = first_month[first_month == prev_month].index
    if len(prev_cohort) == 0:
        return np.nan
    this_month_customers = set(
        net[net["order_datetime"].dt.to_period("M") == month]["customer_id"].unique()
    )
    retained = sum(1 for c in prev_cohort if c in this_month_customers)
    rate = retained / len(prev_cohort)
    assert 0 <= rate <= 1, "리텐션율은 0~1 사이여야 한다"
    return rate


def metric_funnel_conversion(month):
    logs = pd.read_csv(
        "data/web_logs.csv", usecols=["session_id", "event_type", "event_time"],
        parse_dates=["event_time"],
    )
    logs = logs[logs["event_time"].dt.to_period("M") == month]
    reach = logs.drop_duplicates(["session_id", "event_type"])
    mat = pd.crosstab(reach["session_id"], reach["event_type"])
    if "view" not in mat or "purchase" not in mat or mat["view"].sum() == 0:
        return np.nan
    rate = (mat["purchase"] > 0).sum() / (mat["view"] > 0).sum()
    assert 0 <= rate <= 1, "퍼널 전환율은 0~1 사이여야 한다"
    return rate


def run_monthly_report(month_str, save=True):
    month = pd.Period(month_str, freq="M")
    ledger = load_and_clean()

    result = {
        "월": str(month),
        "월순매출": metric_net_revenue(ledger, month),
        "채널별AOV": metric_channel_aov(ledger, month).to_dict(),
        "신규매출비중": metric_new_customer_share(ledger, month),
        "M+1리텐션(전월코호트)": metric_m1_retention(ledger, month),
        "퍼널전환율": metric_funnel_conversion(month),
    }

    if save:
        out = pd.DataFrame([{
            "월": result["월"],
            "월순매출": result["월순매출"],
            "신규매출비중": result["신규매출비중"],
            "M+1리텐션": result["M+1리텐션(전월코호트)"],
            "퍼널전환율": result["퍼널전환율"],
        }])
        out.to_csv(f"monthly_report_{month}.csv", index=False, encoding="utf-8-sig")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("문제 27. 100만 행 로그 메모리 진단과 최적화")
    print("=" * 70)
    dtypes, mem_raw, summary, cat_mem, int_mem = memory_diagnosis()
    print("\n[원본 열별 dtype]")
    print(dtypes.to_string())
    print("\n[단계별 메모리·시간 절감 요약]")
    print(summary.to_string())
    print(f"\n[category 유불리 실측] session_id(고카디널리티)를 category로 바꾸면 "
          f"{int_mem/1e6:.2f}MB -> {cat_mem/1e6:.2f}MB로 오히려 {'증가' if cat_mem > int_mem else '감소'}함"
          f" -> 고카디널리티 열은 category가 불리하다.")

    print("\n" + "=" * 70)
    print("문제 28. 청크 집계로 일별 활성 고객 산출")
    print("=" * 70)
    dau_chunk = chunked_dau()
    dau_full = full_load_dau_check()
    match = (dau_chunk.sort_index().values == dau_full.sort_index().values).all()
    print(f"\n[청크 vs 전량적재 DAU 일치 여부] {match}")
    print(dau_chunk.head(10).to_string())
    bench = chunksize_benchmark()
    print("\n[청크 크기별 처리시간(초)]")
    for cs, t in bench.items():
        print(f"  chunksize={cs:>7,}: {t:.2f}초")
    print("[권장 청크 크기] 처리시간이 더 짧은 쪽을 권장하되, 메모리 여유가 있다면 큰 청크가 보통 더 빠르다.")

    print("\n" + "=" * 70)
    print("문제 29. 로그 기반 상품 관심 스코어 파이프라인")
    print("=" * 70)
    top20, weight = interest_score_pipeline()
    print(f"\n[가중치 가정] {weight} (6월 이벤트는 1.5배 가중)")
    print("\n[상위 20 상품 관심 스코어]")
    print(top20[["product_id", "product_name", "category", "관심스코어"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("문제 30. 월간 마케팅 지표 자동화 파이프라인")
    print("=" * 70)
    for m in ["2024-05", "2024-06"]:
        result = run_monthly_report(m)
        print(f"\n[{m} 리포트]")
        for k, v in result.items():
            print(f"  {k}: {v}")
    # 재실행 재현성 확인
    r1 = run_monthly_report("2024-06", save=False)
    r2 = run_monthly_report("2024-06", save=False)
    same = r1["월순매출"] == r2["월순매출"]
    print(f"\n[재현성 검증] 동일 인자 재실행시 같은 값 반환: {same}")
    print(
        "\n[운영 수칙] 각 지표 함수 내부의 assert가 실패하면(예: 리텐션율이 0~1 범위를 벗어남)\n"
        "파이프라인을 즉시 중단하고 원본 데이터 오염 여부를 먼저 점검한다. 절대 이상치를 조용히 넘기지 않는다."
    )
