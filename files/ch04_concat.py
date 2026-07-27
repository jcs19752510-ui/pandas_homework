# -*- coding: utf-8 -*-
"""
4장. concat으로 데이터 이어붙이기
  문제 16: 월별(1월, 2월) 주문 데이터 concat(axis=0) + ignore_index
  문제 17: concat 시 컬럼 불일치 처리 (기본 vs join='inner')
  문제 18: 서로 다른 지표(Series)를 axis=1로 합쳐 월별 대사표 만들기
  문제 19: concat과 SQL UNION ALL의 차이 - drop_duplicates로 UNION 재현
  문제 20: keys 옵션으로 출처를 구분한 concat 결과 분석
"""
import pandas as pd
from common import build_net_sales_ledger


def problem_16(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 16: 1월-2월 주문 데이터 concat(axis=0) ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    df = net_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M")

    jan = df[df["month"] == "2024-01"]
    feb = df[df["month"] == "2024-02"]

    j_f = pd.concat([jan, feb], axis=0, ignore_index=True)
    print(f"1월 행 수: {len(jan)}, 2월 행 수: {len(feb)}")
    print(f"1월-2월 합친 행 수: {j_f.shape[0]} (= {len(jan)} + {len(feb)} 이어야 정상)")
    assert j_f.shape[0] == len(jan) + len(feb)
    print("\n[마지막 5행 확인 - 2월 데이터가 잘 붙었는지 검증]")
    print(j_f.tail()[["order_id", "month", "line_amount"]].to_string(index=False))
    return j_f


def problem_17(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 17: concat 시 컬럼 불일치 처리 ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    df = net_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M")
    jan = df[df["month"] == "2024-01"][["order_id", "customer_id", "line_amount"]]
    # feb 는 컬럼 하나가 더 있는 상황을 재현 (channel 추가)
    feb = df[df["month"] == "2024-02"][["order_id", "customer_id", "line_amount", "channel"]]

    default_concat = pd.concat([jan, feb], ignore_index=True)
    inner_concat = pd.concat([jan, feb], join="inner", ignore_index=True)

    print("[기본 concat] - 없는 컬럼은 NaN으로 채워짐")
    print(default_concat.head(3).to_string(index=False))
    print("\nchannel 컬럼 결측 건수(1월분):", default_concat["channel"].isna().sum())

    print("\n[join='inner' concat] - 공통 컬럼만 남음")
    print(inner_concat.columns.tolist())
    return default_concat, inner_concat


def problem_18(gross_ledger: pd.DataFrame = None):
    print("\n########## 문제 18: 여러 지표(Series)를 axis=1로 합쳐 대사표 만들기 ##########")
    if gross_ledger is None:
        gross_ledger = build_net_sales_ledger()["gross_ledger"]

    df = gross_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M")

    total_amount = df.groupby("month")["line_amount"].sum().rename("총매출")
    order_count = df.groupby("month")["order_id"].nunique().rename("주문수")
    avg_amount = df.groupby("month")["line_amount"].mean().rename("평균라인금액")

    summary = pd.concat([total_amount, order_count, avg_amount], axis=1).round(0)
    print("\n[월별 대사표] (axis=1 concat)")
    print(summary.to_string())
    return summary


def problem_19(orders: pd.DataFrame = None):
    print("\n########## 문제 19: concat(UNION ALL) vs drop_duplicates(UNION) ##########")
    if orders is None:
        orders = build_net_sales_ledger()["orders"]

    df = orders.copy()  # order_id가 행마다 유일한 주문 단위 데이터 사용
    df["month"] = df["order_datetime"].dt.to_period("M")
    jan = df[df["month"] == "2024-01"][["order_id", "customer_id"]]
    feb = df[df["month"] == "2024-02"][["order_id", "customer_id"]]

    # 일부러 겹치는 고객 행을 하나 추가해 중복 상황을 재현
    overlap_row = jan.iloc[[0]]
    feb_with_overlap = pd.concat([feb, overlap_row], ignore_index=True)

    union_all = pd.concat([jan, feb_with_overlap], ignore_index=True)  # SQL UNION ALL과 동일
    union_distinct = union_all.drop_duplicates()  # SQL UNION과 동일

    print("UNION ALL 방식 (중복 제거 없음) 행 수:", union_all.shape[0])
    print("UNION 방식 (drop_duplicates) 행 수   :", union_distinct.shape[0])
    assert union_all.shape[0] - union_distinct.shape[0] == 1
    return union_all, union_distinct


def problem_20(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 20: keys 옵션으로 출처 구분하기 ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger()["net_ledger"]

    df = net_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M")
    jan = df[df["month"] == "2024-01"][["order_id", "line_amount"]]
    feb = df[df["month"] == "2024-02"][["order_id", "line_amount"]]

    tagged = pd.concat([jan, feb], keys=["1월", "2월"])
    print("\n[출처가 표시된 MultiIndex 결과 - 상위 3행씩]")
    print(tagged.groupby(level=0).head(3).to_string())

    # keys를 활용해 월별로 다시 꺼내오기
    print("\n1월 데이터만 다시 추출 (tagged.loc['1월']):")
    print(tagged.loc["1월"].head(3).to_string(index=False))
    return tagged


if __name__ == "__main__":
    data = build_net_sales_ledger()
    problem_16(data["net_ledger"])
    problem_17(data["net_ledger"])
    problem_18(data["gross_ledger"])
    problem_19(data["orders"])
    problem_20(data["net_ledger"])
