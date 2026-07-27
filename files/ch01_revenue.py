# -*- coding: utf-8 -*-
"""
1장. 매출 성과 진단과 채널 리포트
  문제 01: 신뢰할 수 있는 매출 원장 만들기
  문제 02: 상반기 성장 기여 분해 브리핑
  문제 03: 채널 x 상태 분포 확인 (groupby + size, MultiIndex)
  문제 04: 채널별 취소율 계산 (unstack)
  문제 05: 카테고리 x 채널 매출 피벗테이블 (margins)
"""
import pandas as pd
import numpy as np
from common import build_net_sales_ledger


def problem_01():
    print("\n########## 문제 01: 신뢰할 수 있는 매출 원장 만들기 ##########")
    result = build_net_sales_ledger(verbose=True)
    gross = result["gross_ledger"]
    net = result["net_ledger"]

    gross_m = gross.copy()
    gross_m["month"] = gross_m["order_datetime"].dt.to_period("M")
    net_m = net.copy()
    net_m["month"] = net_m["order_datetime"].dt.to_period("M")

    monthly_gross = gross_m.groupby("month")["line_amount"].sum().rename("총매출")
    monthly_net = net_m.groupby("month")["line_amount"].sum().rename("순매출")
    monthly_excluded = (monthly_gross - monthly_net).rename("취소_반품_제외액")
    monthly_valid_orders = net_m.groupby("month")["order_id"].nunique().rename("유효_주문수")

    reconciliation = pd.concat(
        [monthly_gross, monthly_net, monthly_excluded, monthly_valid_orders], axis=1
    )
    print("\n[월별 대사표] (총매출/순매출/취소·반품 제외액/유효 주문수)")
    print(reconciliation.round(0).to_string())
    return result


def problem_02(net_ledger: pd.DataFrame = None):
    print("\n########## 문제 02: 상반기 성장 기여 분해 브리핑 ##########")
    if net_ledger is None:
        net_ledger = build_net_sales_ledger(verbose=False)["net_ledger"]

    df = net_ledger.copy()
    df["month"] = df["order_datetime"].dt.to_period("M")

    monthly_sales = df.groupby("month")["line_amount"].sum()
    monthly_customers = df.groupby("month")["customer_id"].nunique()
    monthly_orders = df.groupby("month")["order_id"].nunique()

    decomposition = pd.DataFrame({
        "구매고객수": monthly_customers,
        "고객당_주문수": monthly_orders / monthly_customers,
        "AOV": monthly_sales / monthly_orders,
        "순매출": monthly_sales,
    })
    decomposition["검산_순매출"] = (
        decomposition["구매고객수"] * decomposition["고객당_주문수"] * decomposition["AOV"]
    )
    decomposition["검산_오차"] = (decomposition["검산_순매출"] - decomposition["순매출"]).abs()
    assert (decomposition["검산_오차"] < 1e-6).all(), "분해 항등식 검산 실패"

    print("\n[월별 성장 분해표]")
    print(decomposition.round(2).to_string())

    jan, jun = decomposition.iloc[0], decomposition.iloc[-1]
    growth = {
        "구매고객수_변화율": jun["구매고객수"] / jan["구매고객수"] - 1,
        "고객당_주문수_변화율": jun["고객당_주문수"] / jan["고객당_주문수"] - 1,
        "AOV_변화율": jun["AOV"] / jan["AOV"] - 1,
    }
    print("\n[1월 대비 6월 요인별 변화율]")
    for k, v in growth.items():
        print(f"  {k}: {v:+.1%}")

    log_growth = {k: np.log(1 + v) for k, v in growth.items()}
    total_log = sum(log_growth.values())
    contribution = {k: v / total_log for k, v in log_growth.items()}
    print("\n[요인별 성장 기여 비중] (로그 분해, 합=100%)")
    for k, v in contribution.items():
        print(f"  {k}: {v:.1%}")

    main_driver = max(growth, key=growth.get)
    print(f"\n결론: 상반기 성장의 주된 동력은 '{main_driver}' 입니다.")
    return decomposition


def problem_03(orders: pd.DataFrame = None):
    print("\n########## 문제 03: 채널 x 상태 분포 확인 ##########")
    if orders is None:
        orders = build_net_sales_ledger()["orders"]

    combo = orders.groupby(["channel", "status"]).size()  # 멀티인덱스 / value
    print("\n[채널 x 상태 별 주문건수 (MultiIndex Series)]")
    print(combo)
    return combo


def problem_04(orders: pd.DataFrame = None):
    print("\n########## 문제 04: 채널별 취소율 계산 ##########")
    if orders is None:
        orders = build_net_sales_ledger()["orders"]

    combo = orders.groupby(["channel", "status"]).size()
    table = combo.unstack("status", fill_value=0)  # 행=channel, 열=status
    table["총주문수"] = table.sum(axis=1)
    table["취소율"] = table["canceled"] / table["총주문수"]
    table["반품율"] = table["returned"] / table["총주문수"]

    print("\n[채널별 상태 분포 + 취소율/반품율]")
    print(table.round(3).to_string())

    worst_channel = table["취소율"].idxmax()
    print(f"\n결론: 취소율이 가장 높은 채널은 '{worst_channel}' 입니다.")
    return table


def problem_05(gross_ledger: pd.DataFrame = None):
    print("\n########## 문제 05: 카테고리 x 채널 매출 피벗테이블 ##########")
    if gross_ledger is None:
        gross_ledger = build_net_sales_ledger()["gross_ledger"]

    pd.set_option('display.float_format', '{:,.0f}'.format)
    pt = pd.pivot_table(
        data=gross_ledger,
        index="category",
        columns="channel",
        values="line_amount",
        aggfunc="sum",
        margins=True,
        margins_name="합",
        fill_value=0,
    )
    print("\n[카테고리 x 채널 매출 피벗테이블]")
    print(pt)
    return pt


if __name__ == "__main__":
    result = problem_01()
    problem_02(result["net_ledger"])
    problem_03(result["orders"])
    problem_04(result["orders"])
    problem_05(result["gross_ledger"])
