-- ============================================================
--  AI-Driven Customer Analytics Platform — MySQL Schema
--  Database: customer_analytics
-- ============================================================

CREATE DATABASE IF NOT EXISTS customer_analytics
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE customer_analytics;

-- ── 1. Main customer transactions table ─────────────────────
DROP TABLE IF EXISTS customer_transactions;
CREATE TABLE customer_transactions (
    id                     INT AUTO_INCREMENT PRIMARY KEY,
    customer_id            VARCHAR(20)   NOT NULL,
    age                    TINYINT       NOT NULL,
    gender                 VARCHAR(10)   NOT NULL,
    item_purchased         VARCHAR(50)   NOT NULL,
    category               VARCHAR(30)   NOT NULL,
    purchase_amount        DECIMAL(8,2)  NOT NULL,
    location               VARCHAR(50),
    size                   VARCHAR(5),
    color                  VARCHAR(20),
    season                 VARCHAR(15)   NOT NULL,
    review_rating          DECIMAL(3,2),
    subscription_status    VARCHAR(5)    NOT NULL,
    payment_method         VARCHAR(30)   NOT NULL,
    shipping_type          VARCHAR(30),
    discount_applied       VARCHAR(5),
    promo_code_used        VARCHAR(5),
    previous_purchases     TINYINT       NOT NULL,
    frequency_of_purchases VARCHAR(25),
    -- Engineered features
    age_group              VARCHAR(20),
    loyalty_tier           VARCHAR(15),
    purchase_frequency_days INT,
    subscription_bin       TINYINT,
    discount_bin           TINYINT,
    cluster_id             TINYINT,
    customer_segment       VARCHAR(20),
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category     (category),
    INDEX idx_season       (season),
    INDEX idx_gender       (gender),
    INDEX idx_subscription (subscription_status),
    INDEX idx_cluster      (cluster_id),
    INDEX idx_loyalty      (loyalty_tier)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 2. Statistical analysis results ─────────────────────────
DROP TABLE IF EXISTS statistical_results;
CREATE TABLE statistical_results (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    test_name     VARCHAR(80)   NOT NULL,
    variable_1    VARCHAR(40),
    variable_2    VARCHAR(40),
    statistic     DECIMAL(12,6),
    p_value       DECIMAL(12,8),
    degrees_of_freedom INT,
    is_significant TINYINT(1),
    notes         TEXT,
    run_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_test (test_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 3. Customer segments / clusters ─────────────────────────
DROP TABLE IF EXISTS customer_segments;
CREATE TABLE customer_segments (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id           TINYINT NOT NULL UNIQUE,
    segment_name         VARCHAR(25) NOT NULL,
    customer_count       INT,
    avg_age              DECIMAL(5,2),
    avg_purchase_amount  DECIMAL(8,2),
    avg_review_rating    DECIMAL(4,3),
    avg_previous_purchases DECIMAL(5,2),
    subscription_rate    DECIMAL(5,4),
    discount_rate        DECIMAL(5,4),
    description          TEXT,
    strategy             TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 4. ML model performance log ─────────────────────────────
DROP TABLE IF EXISTS ml_model_results;
CREATE TABLE ml_model_results (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    model_name     VARCHAR(60)   NOT NULL,
    model_type     VARCHAR(30),
    target         VARCHAR(40),
    r2_score       DECIMAL(8,6),
    mae            DECIMAL(10,4),
    rmse           DECIMAL(10,4),
    f1_score       DECIMAL(8,6),
    accuracy       DECIMAL(8,6),
    auc_roc        DECIMAL(8,6),
    cv_folds       TINYINT,
    cv_r2_mean     DECIMAL(8,6),
    cv_r2_std      DECIMAL(8,6),
    hyperparameters JSON,
    notes          TEXT,
    run_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_type (model_type),
    INDEX idx_target     (target)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 5. SHAP feature importance ───────────────────────────────
DROP TABLE IF EXISTS shap_feature_importance;
CREATE TABLE shap_feature_importance (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    model_name      VARCHAR(60)  NOT NULL,
    feature_name    VARCHAR(50)  NOT NULL,
    mean_shap_value DECIMAL(10,6),
    rank_position   TINYINT,
    run_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model   (model_name),
    INDEX idx_feature (feature_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 6. Revenue summary view ──────────────────────────────────
CREATE OR REPLACE VIEW vw_revenue_by_category AS
    SELECT
        category,
        COUNT(*)                          AS total_orders,
        ROUND(SUM(purchase_amount), 2)    AS total_revenue,
        ROUND(AVG(purchase_amount), 2)    AS avg_purchase,
        ROUND(MIN(purchase_amount), 2)    AS min_purchase,
        ROUND(MAX(purchase_amount), 2)    AS max_purchase,
        ROUND(AVG(review_rating), 3)      AS avg_rating
    FROM customer_transactions
    GROUP BY category
    ORDER BY total_revenue DESC;

CREATE OR REPLACE VIEW vw_revenue_by_season AS
    SELECT
        season,
        COUNT(*)                          AS total_orders,
        ROUND(SUM(purchase_amount), 2)    AS total_revenue,
        ROUND(AVG(purchase_amount), 2)    AS avg_purchase
    FROM customer_transactions
    GROUP BY season
    ORDER BY avg_purchase DESC;

CREATE OR REPLACE VIEW vw_segment_summary AS
    SELECT
        cs.segment_name,
        cs.customer_count,
        cs.avg_purchase_amount,
        cs.subscription_rate,
        ROUND(cs.avg_purchase_amount * cs.customer_count, 2) AS estimated_total_revenue
    FROM customer_segments cs
    ORDER BY cs.avg_purchase_amount DESC;

-- ── Sample queries for verification ─────────────────────────
-- SELECT * FROM vw_revenue_by_category;
-- SELECT * FROM vw_revenue_by_season;
-- SELECT * FROM vw_segment_summary;
-- SELECT COUNT(*) AS total_rows FROM customer_transactions;
