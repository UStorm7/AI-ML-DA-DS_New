-- Reference analysis queries. Each block is named "-- name: <key>" so
-- src/queries.py can load them individually.

-- name: monthly_kpis
SELECT order_month,
       COUNT(*)                                    AS orders,
       COUNT(DISTINCT customer_id)                 AS customers,
       ROUND(SUM(payment_value), 2)                AS gmv,
       ROUND(AVG(payment_value), 2)                AS aov,
       ROUND(AVG(is_late::INT) * 100, 2)           AS late_pct,
       ROUND(AVG(review_score), 3)                 AS avg_review
FROM fct_delivered
GROUP BY order_month
ORDER BY order_month;

-- name: cohort_retention
WITH firsts AS (
    SELECT customer_id, MIN(order_month) AS cohort_month
    FROM fct_delivered GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT f.customer_id, f.cohort_month,
           DATE_DIFF('month', f.cohort_month, d.order_month) AS month_number
    FROM fct_delivered d JOIN firsts f USING (customer_id)
)
SELECT cohort_month, month_number, COUNT(*) AS active_customers
FROM activity
GROUP BY cohort_month, month_number
ORDER BY cohort_month, month_number;

-- name: review_by_delay_bucket
SELECT CASE
         WHEN days_late <= -10 THEN '1: 10+ days early'
         WHEN days_late <= -5  THEN '2: 5-10 days early'
         WHEN days_late <= 0   THEN '3: 0-5 days early'
         WHEN days_late <= 3   THEN '4: 0-3 days late'
         WHEN days_late <= 7   THEN '5: 3-7 days late'
         ELSE                       '6: 7+ days late'
       END                                         AS delay_bucket,
       COUNT(*)                                    AS orders,
       ROUND(AVG(review_score), 3)                 AS avg_review,
       ROUND(AVG((review_score <= 2)::INT) * 100, 2) AS pct_negative
FROM fct_delivered
WHERE review_score IS NOT NULL
GROUP BY delay_bucket
ORDER BY delay_bucket;

-- name: late_rate_by_state
SELECT customer_state,
       COUNT(*)                                    AS orders,
       ROUND(AVG(is_late::INT) * 100, 2)           AS late_pct,
       ROUND(AVG(delivery_days), 2)                AS avg_delivery_days,
       ROUND(AVG(review_score), 3)                 AS avg_review
FROM fct_delivered
GROUP BY customer_state
HAVING COUNT(*) >= 300
ORDER BY late_pct DESC;

-- name: category_performance
SELECT main_category                               AS category,
       COUNT(*)                                    AS orders,
       ROUND(SUM(payment_value), 0)                AS gmv,
       ROUND(AVG(review_score), 3)                 AS avg_review,
       ROUND(AVG(is_late::INT) * 100, 2)           AS late_pct,
       ROUND(AVG(freight_value / NULLIF(item_value, 0)) * 100, 1) AS freight_pct_of_price
FROM fct_delivered
GROUP BY main_category
HAVING COUNT(*) >= 500
ORDER BY gmv DESC;
