# -*- coding: utf-8 -*-
"""
6장. 종합 심화 분석
  문제 26: 채널 x 상태 조합별 매출 요약 (groupby + agg 여러개)
  문제 27: 월별 x 채널별 매출 피벗 + 결측 처리(fill_value=0)
  문제 28: 고객별 누적 매출(LTV 유사 지표) 계산
  문제 29: 채널별 재구매 고객 비율 분석
  문제 30: 위 지표들을 하나의 요약 DataFrame으로 concat(axis=1)
"""
import pandas as pd
from common import build_net_sales_ledger


def problem_26(gross_ledger: pd.DataFrame = None):
    print("\n########## 문제 26: 채널 x 상태 조합별 매출 요약 ##########")
    if gross_ledger is None:
        gross_ledger = build_net_sales_ledger()["gross_ledger"]

    summary = gross_ledger.groupby(["channel", "status"]).agg(
        건수=("order_id", "count"),
        합계금액=("line_amount", "sum"),
        평균금액=("line_amount", "mean"),
    ).round(0)
    print("\n[채널 x 상태별 건수/합계/평균]")
    print(summary.to_string())
    return summary


def problem_27(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 27: 월별 x 채널별 매출 피벗 (결측 0 처리) ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    df = net_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M").astype(str)

    pt = pd.pivot_table(
        data=df, index="month", columns="channel",
        values="line_amount", aggfunc="sum", fill_value=0,
    )
    print("\n[월별 x 채널별 순매출]")
    print(pt.to_string())
    return pt


def problem_28(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 28: 고객별 누적 매출(LTV 유사 지표) ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    ltv = net_ledger.groupby("customer_id").agg(
        누적매출=("line_amount", "sum"),
        구매횟수=("order_id", "nunique"),
    ).sort_values("누적매출", ascending=False)

    print("\n[누적매출 상위 10명]")
    print(ltv.head(10).to_string())
    print(f"\n전체 고객 평균 누적매출: {ltv['누적매출'].mean():,.0f}")
    return ltv


def problem_29(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 29: 채널별 재구매 고객 비율 ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    # 고객의 첫 구매 채널을 기준으로 재구매(2회 이상 구매) 여부 판단
    order_level = net_ledger.drop_duplicates(subset="order_id")[
        ["order_id", "customer_id", "channel", "order_datetime"]
    ].sort_values("order_datetime")

    first_channel = order_level.groupby("customer_id").first()["channel"]
    purchase_count = order_level.groupby("customer_id")["order_id"].nunique()

    cust_summary = pd.concat([first_channel.rename("첫구매채널"), purchase_count.rename("구매횟수")], axis=1)
    cust_summary["재구매여부"] = cust_summary["구매횟수"] >= 2

    repeat_rate = cust_summary.groupby("첫구매채널")["재구매여부"].mean().round(3)
    print("\n[채널별 재구매 고객 비율]")
    print(repeat_rate)
    return repeat_rate


def problem_30(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 30: 종합 요약 리포트 (axis=1 concat) ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    df = net_ledger.copy()

    combo = df.groupby("channel")["order_id"].nunique().rename("주문수")
    sales = df.groupby("channel")["line_amount"].sum().rename("순매출")
    aov = (sales / combo).rename("AOV")
    cust_cnt = df.groupby("channel")["customer_id"].nunique().rename("구매고객수")

    report = pd.concat([combo, sales, aov, cust_cnt], axis=1).round(0)
    report["매출비중"] = (report["순매출"] / report["순매출"].sum()).round(3)

    print("\n[채널별 종합 요약 리포트]")
    print(report.to_string())

    top_channel = report["순매출"].idxmax()
    print(f"\n결론: 순매출 기준 최우수 채널은 '{top_channel}' 입니다.")
    return report


if __name__ == "__main__":
    data = build_net_sales_ledger()
    problem_26(data["gross_ledger"])
    problem_27(data["net_ledger"])
    problem_28(data["net_ledger"])
    problem_29(data["net_ledger"])
    problem_30(data["net_ledger"])
