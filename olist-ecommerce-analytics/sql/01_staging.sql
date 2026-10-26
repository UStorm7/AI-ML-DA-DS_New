-- Staging layer: typed, cleaned copies of the raw CSVs.
-- {raw} is replaced with the raw data directory by src/build_warehouse.py

CREATE OR REPLACE TABLE stg_orders AS
SELECT
    order_id,
    customer_id,
    order_status,
    CAST(order_purchase_timestamp      AS TIMESTAMP) AS purchased_at,
    CAST(order_approved_at             AS TIMESTAMP) AS approved_at,
    CAST(order_delivered_carrier_date  AS TIMESTAMP) AS shipped_at,
    CAST(order_delivered_customer_date AS TIMESTAMP) AS delivered_at,
    CAST(order_estimated_delivery_date AS TIMESTAMP) AS estimated_delivery_at
FROM read_csv_auto('{raw}/olist_orders_dataset.csv', header = true);

CREATE OR REPLACE TABLE stg_customers AS
SELECT customer_id, customer_unique_id, customer_zip_code_prefix,
       customer_city, customer_state
FROM read_csv_auto('{raw}/olist_customers_dataset.csv', header = true);

CREATE OR REPLACE TABLE stg_order_items AS
SELECT order_id, order_item_id, product_id, seller_id,
       CAST(price AS DOUBLE) AS price, CAST(freight_value AS DOUBLE) AS freight_value
FROM read_csv_auto('{raw}/olist_order_items_dataset.csv', header = true);

CREATE OR REPLACE TABLE stg_payments AS
SELECT order_id, payment_sequential, payment_type, payment_installments,
       CAST(payment_value AS DOUBLE) AS payment_value
FROM read_csv_auto('{raw}/olist_order_payments_dataset.csv', header = true);

-- Some orders have several reviews; keep the most recent answer per order.
CREATE OR REPLACE TABLE stg_reviews AS
SELECT order_id, review_score, review_comment_message,
       CAST(review_answer_timestamp AS TIMESTAMP) AS answered_at
FROM read_csv_auto('{raw}/olist_order_reviews_dataset.csv', header = true)
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC) = 1;

CREATE OR REPLACE TABLE stg_products AS
SELECT p.product_id,
       COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
       p.product_weight_g, p.product_photos_qty
FROM read_csv_auto('{raw}/olist_products_dataset.csv', header = true) p
LEFT JOIN read_csv_auto('{raw}/product_category_name_translation.csv', header = true) t
       USING (product_category_name);

CREATE OR REPLACE TABLE stg_sellers AS
SELECT seller_id, seller_state
FROM read_csv_auto('{raw}/olist_sellers_dataset.csv', header = true);
