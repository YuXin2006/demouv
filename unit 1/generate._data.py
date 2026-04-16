import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子
np.random.seed(42)
data_size = 1000 # 你可以改为 100000 挑战更高性能

# --- 1. 构建产品维表 (Product Dim Table) ---
products = {
    'pid': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'product_name': ['旗舰手机', '联名卫衣', '人体工学椅', '机械键盘', '咖啡机'],
    'category': ['数码', '服装', '家具', '数码', '家电'],
    'base_price': [5999, 299, 1299, 499, 2500]
}
df_products = pd.DataFrame(products)

# --- 2. 构建核心销售表 (Fact Table) ---
start_date = datetime(2023, 1, 1)
data = {
    'order_id': [f'ORD{i:06d}' for i in range(data_size)],
    'timestamp': [start_date + timedelta(minutes=np.random.randint(0, 525600)) for _ in range(data_size)],
    'pid': np.random.choice(products['pid'], data_size),
    'user_id': np.random.randint(10000, 10500, data_size), # 模拟重复购买
    'actual_price': np.nan, # 稍后计算
    'quantity': np.random.randint(1, 5, data_size),
    'region': np.random.choice(['华北', '华东', '华南', '西南'], data_size, p=[0.2, 0.4, 0.2, 0.2]),
    'payment_method': np.random.choice(['支付宝', '微信', '信用卡', None], data_size, p=[0.4, 0.4, 0.15, 0.05])
}

df_sales = pd.DataFrame(data)

# 模拟价格波动：实际价格 = 基准价 * 随机折扣 (0.8-1.1)
price_map = df_products.set_index('pid')['base_price'].to_dict()
df_sales['actual_price'] = df_sales['pid'].map(price_map) * np.random.uniform(0.8, 1.1, data_size)
df_sales['actual_price'] = df_sales['actual_price'].round(2)

# --- 3. 写入文件 ---
df_sales.to_csv('large_sales_data.csv', index=False)
df_products.to_csv('product_info.csv', index=False)

print(f"成功生成 {data_size} 条数据的销售表 'large_sales_data.csv' 和产品表 'product_info.csv'")