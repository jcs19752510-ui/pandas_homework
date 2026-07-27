"""
generate_shop_data.py
쇼핑몰 실습 데이터셋 생성 스크립트 (재구성판)

주의: 원본 교재/문제집에 동봉된 스크립트가 아니라,
문제집(pandas_shop_marketing.pdf)과 교재 스크린샷에 명시된
스키마 · 행수 · 결측/중복 건수 · 오염 패턴을 최대한 재현하여
새로 작성한 대체 스크립트입니다. seed=42로 고정되어 있어
이 스크립트로 생성한 데이터는 재실행해도 항상 동일합니다.
다만 원본 교재의 정확한 생성 로직과는 다르므로,
책에 적힌 최종 집계 수치(예: 상반기 순매출 466억 원)와는
차이가 있을 수 있습니다. 구조·정제 로직 학습에는 문제 없습니다.
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 공통 인적 요소
# ------------------------------------------------------------------
SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배")
GIVEN1 = list("민서지우현재도윤성하은수영준")
GIVEN2 = list("준서연아민호진영경수아빈")
CITIES = ["서울", "부산", "인천", "대구", "대전", "광주", "울산", "수원", "성남", "고양"]


def rand_name():
    return random.choice(SURNAMES) + random.choice(GIVEN1) + random.choice(GIVEN2)


def rand_date(start, end):
    start_ts = pd.Timestamp(start).value
    end_ts = pd.Timestamp(end).value
    return pd.Timestamp(rng.integers(start_ts, end_ts))


# ------------------------------------------------------------------
# 1) customers.csv  (5,000행)
#    결측: gender 304 / birth_date 384 / email 248
#    완전중복 행 50건
# ------------------------------------------------------------------
N_CUST = 5000
N_CUST_UNIQUE = N_CUST - 50  # 50건은 완전중복으로 채움 (총합 5,000행 고정)
customer_ids = rng.choice(np.arange(1, 6000), size=N_CUST_UNIQUE, replace=False)
names = [rand_name() for _ in range(N_CUST_UNIQUE)]

# gender 표기 불일치: 여/남/F/M 혼재
gender_pool = ["여", "남", "F", "M"]
genders = [random.choice(gender_pool) for _ in range(N_CUST_UNIQUE)]

birth_dates = [rand_date("1955-01-01", "2005-12-31").strftime("%Y-%m-%d") for _ in range(N_CUST_UNIQUE)]
signup_dates = [rand_date("2020-01-01", "2024-06-30").strftime("%Y-%m-%d") for _ in range(N_CUST_UNIQUE)]
cities = [random.choice(CITIES) for _ in range(N_CUST_UNIQUE)]
emails = [f"user{cid}@example.com" for cid in customer_ids]

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "name": names,
    "gender": genders,
    "birth_date": birth_dates,
    "signup_date": signup_dates,
    "city": cities,
    "email": emails,
})

# 결측 주입
idx = rng.choice(customers.index, size=304, replace=False)
customers.loc[idx, "gender"] = np.nan
idx = rng.choice(customers.index, size=384, replace=False)
customers.loc[idx, "birth_date"] = np.nan
idx = rng.choice(customers.index, size=248, replace=False)
customers.loc[idx, "email"] = np.nan

# 완전 중복 행 50건 추가(총 5,000행 고정) 후 섞기
dup_rows = customers.sample(n=50, random_state=SEED)
customers = pd.concat([customers, dup_rows], ignore_index=True)
assert len(customers) == N_CUST
customers = customers.sample(frac=1, random_state=SEED).reset_index(drop=True)

customers.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
print(f"[1/5] customers.csv 완료: {len(customers):,}행")

# ------------------------------------------------------------------
# 2) products.csv (500행)
#    category: 도서/전자/식품/뷰티/의류/가구/스포츠/완구/문구/주방
#              + 앞뒤 공백 오염("가구 " 등)
#    price: object 타입(콤마/원/0/음수/결측 오염) - 문제집 13번 대상
#    cost: 결측 없음
#    완전중복 3행
# ------------------------------------------------------------------
N_PROD = 500
N_PROD_UNIQUE = N_PROD - 3  # 3건은 완전중복으로 채움(총합 500행 고정)
CATS = ["가구", "전자", "식품", "뷰티", "의류", "스포츠", "완구", "문구", "주방", "도서"]
PROD_ADJ = ["프리미엄", "베이직", "플러스", "슬림", "클래식", "미니", "프로", "스탠다드"]
PROD_NOUN = {
    "가구": ["원목 테이블", "소파", "침대 프레임", "책상"],
    "전자": ["노트북", "무선이어폰", "모니터", "청소기"],
    "식품": ["원두커피", "홍삼", "견과류 세트", "올리브오일"],
    "뷰티": ["수분크림", "선크림", "립스틱", "샴푸"],
    "의류": ["티셔츠", "패딩", "청바지", "니트"],
    "스포츠": ["요가매트", "런닝화", "덤벨세트", "자전거"],
    "완구": ["블록세트", "인형", "보드게임", "퍼즐"],
    "문구": ["다이어리", "만년필", "노트세트", "샤프"],
    "주방": ["프라이팬", "칼세트", "밀폐용기", "텀블러"],
    "도서": ["에세이", "자기계발서", "소설", "요리책"],
}

product_ids = np.arange(1, N_PROD_UNIQUE + 1)
categories_clean = [random.choice(CATS) for _ in range(N_PROD_UNIQUE)]
product_names = [f"{random.choice(PROD_ADJ)} {random.choice(PROD_NOUN[c])}" for c in categories_clean]

base_price = rng.integers(3000, 90000, size=N_PROD_UNIQUE)
cost = (base_price * rng.uniform(0.35, 0.7, size=N_PROD_UNIQUE)).round(0)

# category에 앞뒤 공백 오염 주입 (약 15% 행)
categories = categories_clean.copy()
contam_idx = rng.choice(N_PROD_UNIQUE, size=int(N_PROD_UNIQUE * 0.15), replace=False)
for i in contam_idx:
    categories[i] = " " + categories[i] if rng.random() < 0.5 else categories[i] + " "

# price를 문자열(object)로 오염: 정상/콤마/원표기/0/음수/결측
price_str = []
for p in base_price:
    r = rng.random()
    if r < 0.55:
        price_str.append(str(int(p)))
    elif r < 0.75:
        price_str.append(f"{int(p):,}")
    elif r < 0.90:
        price_str.append(f"{int(p)}원")
    elif r < 0.94:
        price_str.append("0")
    elif r < 0.97:
        price_str.append(str(-int(p)))
    else:
        price_str.append(np.nan)

products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": product_names,
    "category": categories,
    "price": price_str,
    "cost": cost,
})

dup_rows = products.sample(n=3, random_state=SEED)
products = pd.concat([products, dup_rows], ignore_index=True)
assert len(products) == N_PROD
products = products.sample(frac=1, random_state=SEED).reset_index(drop=True)

products.to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
print(f"[2/5] products.csv 완료: {len(products):,}행")

products_clean_price = base_price  # 후속 주문 생성에 사용(오염 전 정상가)
products_id_price = dict(zip(product_ids, base_price))
products_id_cat = dict(zip(product_ids, categories_clean))

# ------------------------------------------------------------------
# 3) orders.csv (200,000행)
#    order_datetime: 빈 문자열 1,933건 + 기간 밖(2025년) 40건
#    status: delivered/canceled/returned/shipped/pending
#    channel: store/web/app
# ------------------------------------------------------------------
N_ORD = 200000
order_ids = np.arange(1, N_ORD + 1)
order_customer_ids = rng.choice(customer_ids, size=N_ORD, replace=True)

# 2024년 상반기 내 균등 분포 + 약간의 성장 추세(월별 가중)
month_weights = np.array([0.12, 0.13, 0.15, 0.16, 0.18, 0.26])
months = rng.choice(np.arange(1, 7), size=N_ORD, p=month_weights / month_weights.sum())
days_in_month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30}
order_datetimes = []
for m in months:
    d = rng.integers(1, days_in_month[m] + 1)
    h = rng.integers(0, 24)
    mi = rng.integers(0, 60)
    se = rng.integers(0, 60)
    order_datetimes.append(pd.Timestamp(2024, m, int(d), int(h), int(mi), int(se)))

channels = rng.choice(["store", "web", "app"], size=N_ORD, p=[0.35, 0.40, 0.25])
statuses = rng.choice(
    ["delivered", "shipped", "pending", "canceled", "returned"],
    size=N_ORD,
    p=[0.55, 0.20, 0.15, 0.06, 0.04],
)

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": order_customer_ids,
    "order_datetime": order_datetimes,
    "channel": channels,
    "status": statuses,
})
orders["order_datetime"] = orders["order_datetime"].astype(str)

# 빈 문자열 오염 1,933건
idx = rng.choice(orders.index, size=1933, replace=False)
orders.loc[idx, "order_datetime"] = ""

# 기간 밖(2025년) 이상값 40건
idx = rng.choice(orders.index.difference(idx), size=40, replace=False)
bad_dates = [
    pd.Timestamp(2025, int(rng.integers(1, 13)), int(rng.integers(1, 28))).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(40)
]
orders.loc[idx, "order_datetime"] = bad_dates

orders.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
print(f"[3/5] orders.csv 완료: {len(orders):,}행")

order_id_to_dt = dict(zip(order_ids, order_datetimes))

# ------------------------------------------------------------------
# 4) order_items.csv (500,000행)
#    unit_price 결측 15,113건 / 완전중복 120행
# ------------------------------------------------------------------
N_ITEMS = 500000
N_ITEMS_UNIQUE = N_ITEMS - 120  # 120건은 완전중복으로 채움(총합 500,000행 고정)
item_order_ids = rng.choice(order_ids, size=N_ITEMS_UNIQUE, replace=True)
item_product_ids = rng.choice(product_ids, size=N_ITEMS_UNIQUE, replace=True)
quantity = rng.integers(1, 6, size=N_ITEMS_UNIQUE)
unit_price = np.array([products_id_price[p] for p in item_product_ids], dtype=float)
discount = rng.choice(
    [0.0, 0.05, 0.1, 0.15, 0.2, 0.23, 0.3, 0.45],
    size=N_ITEMS_UNIQUE,
    p=[0.15, 0.15, 0.2, 0.15, 0.15, 0.08, 0.07, 0.05],
)

order_items = pd.DataFrame({
    "order_item_id": np.arange(1, N_ITEMS_UNIQUE + 1),
    "order_id": item_order_ids,
    "product_id": item_product_ids,
    "quantity": quantity,
    "unit_price": unit_price,
    "discount": discount,
})

idx = rng.choice(order_items.index, size=15113, replace=False)
order_items.loc[idx, "unit_price"] = np.nan

dup_rows = order_items.sample(n=120, random_state=SEED)  # order_item_id까지 완전히 동일하게 복제
order_items = pd.concat([order_items, dup_rows], ignore_index=True)
assert len(order_items) == N_ITEMS
order_items = order_items.sample(frac=1, random_state=SEED).reset_index(drop=True)

order_items.to_csv(os.path.join(OUT_DIR, "order_items.csv"), index=False)
print(f"[4/5] order_items.csv 완료: {len(order_items):,}행")

# ------------------------------------------------------------------
# 5) web_logs.csv (1,000,000행)
#    event_type: view/search/cart/purchase
#    product_id: search는 결측
#    customer_id: 일부 결측(익명 방문)
# ------------------------------------------------------------------
N_LOGS = 1_000_000
session_ids = rng.integers(100000, 999999, size=N_LOGS // 3)
log_session = rng.choice(session_ids, size=N_LOGS, replace=True)
log_customer = rng.choice(customer_ids, size=N_LOGS, replace=True).astype(float)
na_cust_idx = rng.choice(N_LOGS, size=int(N_LOGS * 0.12), replace=False)
log_customer[na_cust_idx] = np.nan

event_types = rng.choice(
    ["view", "search", "cart", "purchase"],
    size=N_LOGS,
    p=[0.55, 0.20, 0.15, 0.10],
)

log_months = rng.choice(np.arange(1, 7), size=N_LOGS, p=month_weights / month_weights.sum())
log_times = []
for m in log_months:
    d = rng.integers(1, days_in_month[m] + 1)
    h = rng.integers(0, 24)
    mi = rng.integers(0, 60)
    se = rng.integers(0, 60)
    log_times.append(pd.Timestamp(2024, m, int(d), int(h), int(mi), int(se)))

log_product = rng.choice(product_ids, size=N_LOGS).astype(float)
search_idx = np.where(event_types == "search")[0]
log_product[search_idx] = np.nan

web_logs = pd.DataFrame({
    "log_id": np.arange(1, N_LOGS + 1),
    "customer_id": log_customer,
    "event_time": log_times,
    "event_type": event_types,
    "product_id": log_product,
    "session_id": log_session,
})

web_logs.to_csv(os.path.join(OUT_DIR, "web_logs.csv"), index=False)
print(f"[5/5] web_logs.csv 완료: {len(web_logs):,}행")

print(f"\n모든 CSV 생성 완료: {OUT_DIR}")
