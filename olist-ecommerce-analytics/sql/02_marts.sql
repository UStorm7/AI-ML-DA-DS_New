-- Mart layer: one row per order, the grain most analyses need.

CREATE OR REPLACE TABLE fct_orders AS
WITH items AS (
    SELECT order_id,
           COUNT(*)                     AS n_items,
           COUNT(DISTINCT seller_id)    AS n_sellers,
           SUM(price)                   AS item_value,
           SUM(freight_value)           AS freight_value,
           ANY_VALUE(seller_id)         AS main_seller_id,
           ANY_VALUE(product_id)        AS main_product_id
    FROM stg_order_items
    GROUP BY order_id
),
payments AS (
    SELECT order_id,
           SUM(payment_value)           AS payment_value,
           MAX(payment_installments)    AS installments,
           ARG_MAX(payment_type, payment_value) AS main_payment_type
    FROM stg_payments
    GROUP BY order_id
)
SELECT
    o.order_id,
    c.customer_unique_id                         AS customer_id,
    c.customer_state,
    o.order_status,
    o.purchased_at,
    DATE_TRUNC('month', o.purchased_at)          AS order_month,
    o.delivered_at,
    o.estimated_delivery_at,
    DATE_DIFF('hour', o.purchased_at, o.delivered_at) / 24.0          AS delivery_days,
    DATE_DIFF('hour', o.estimated_delivery_at, o.delivered_at) / 24.0 AS days_late,
    o.delivered_at > o.estimated_delivery_at     AS is_late,
    i.n_items, i.n_sellers, i.item_value, i.freight_value,
    p.payment_value, p.installments, p.main_payment_type,
    pr.category                                  AS main_category,
    s.seller_state                               AS main_seller_state,
    s.seller_state = c.customer_state            AS same_state,
    r.review_score
FROM stg_orders o
JOIN stg_customers c   USING (customer_id)
LEFT JOIN items i      USING (order_id)
LEFT JOIN payments p   USING (order_id)
LEFT JOIN stg_products pr ON pr.product_id = i.main_product_id
LEFT JOIN stg_sellers  s  ON s.seller_id  = i.main_seller_id
LEFT JOIN stg_reviews  r  USING (order_id);

-- Delivered orders inside the well-covered window (data before 2017 and after
-- Aug 2018 is sparse / truncated).
CREATE OR REPLACE VIEW fct_delivered AS
SELECT *
FROM fct_orders
WHERE order_status = 'delivered'
  AND delivered_at IS NOT NULL
  AND purchased_at >= TIMESTAMP '2017-01-01'
  AND purchased_at <  TIMESTAMP '2018-09-01';
