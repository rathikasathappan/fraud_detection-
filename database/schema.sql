-- Merchant Fraud Detection Schema
-- Supports high-throughput cold-start onboarding risk scoring & network graph history

CREATE DATABASE IF NOT EXISTS merchant_fraud_db;
USE merchant_fraud_db;

-- 1. Merchants Table
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id VARCHAR(64) PRIMARY KEY,
    business_category VARCHAR(64) NOT NULL,
    business_age_days INT NULL DEFAULT NULL,
    registration_date DATETIME NOT NULL,
    phone_verified TINYINT(1) NULL DEFAULT NULL,
    email_verified TINYINT(1) NULL DEFAULT NULL,
    document_verified TINYINT(1) NULL DEFAULT NULL,
    bank_account_id VARCHAR(64) NOT NULL,
    device_id VARCHAR(64) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    transaction_count INT NOT NULL DEFAULT 0,
    fraud_label TINYINT(1) NULL,
    data_source ENUM('synthetic', 'company', 'api', 'test') NOT NULL DEFAULT 'synthetic',
    in_training_set TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reg_date (registration_date),
    INDEX idx_category (business_category),
    INDEX idx_bank (bank_account_id),
    INDEX idx_device (device_id),
    INDEX idx_ip (ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Relationships Table (Graph Links)
CREATE TABLE IF NOT EXISTS relationships (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    merchant_id VARCHAR(64) NOT NULL,
    entity_id VARCHAR(64) NOT NULL,
    entity_type ENUM('bank_account', 'device', 'ip_address', 'phone') NOT NULL,
    relationship VARCHAR(64) NOT NULL,
    timestamp DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_m_id (merchant_id),
    INDEX idx_entity (entity_id, entity_type),
    INDEX idx_rel_time (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Risk Assessments Table
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id VARCHAR(64) PRIMARY KEY,
    merchant_id VARCHAR(64) NOT NULL,
    risk_score FLOAT NOT NULL,
    decision ENUM('Approve', 'Review', 'Hold') NOT NULL,
    reasons JSON NOT NULL,
    graph_evidence JSON NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    data_completeness VARCHAR(16) NULL DEFAULT NULL,
    evaluated_at DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_assessment_merchant (merchant_id),
    INDEX idx_decision (decision),
    INDEX idx_evaluated_at (evaluated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    merchant_id VARCHAR(64) NOT NULL,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    alert_type VARCHAR(64) NOT NULL,
    details TEXT NOT NULL,
    status ENUM('OPEN', 'ACKNOWLEDGED', 'RESOLVED') NOT NULL DEFAULT 'OPEN',
    created_at DATETIME NOT NULL,
    INDEX idx_alert_merchant (merchant_id),
    INDEX idx_alert_status (status),
    INDEX idx_alert_severity (severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Users Table (Role-Based Access Control)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(128) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('Admin', 'Investigator', 'Analyst') NOT NULL DEFAULT 'Analyst',
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    last_login DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. Investigation Cases
CREATE TABLE IF NOT EXISTS investigation_cases (
    case_id VARCHAR(64) PRIMARY KEY,
    alert_id VARCHAR(64) NOT NULL UNIQUE,
    merchant_id VARCHAR(64) NOT NULL,
    investigator_id INT NULL,
    status ENUM('Open', 'In review', 'Closed') NOT NULL DEFAULT 'Open',
    decision ENUM('Confirm fraud', 'False positive', 'Needs more info') NULL,
    notes TEXT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    INDEX idx_case_merchant (merchant_id),
    INDEX idx_case_status (status),
    CONSTRAINT fk_case_alert FOREIGN KEY (alert_id) REFERENCES alerts(alert_id),
    CONSTRAINT fk_case_merchant FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id),
    CONSTRAINT fk_case_investigator FOREIGN KEY (investigator_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. Investigation Audit Log
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    username VARCHAR(64) NOT NULL,
    action TEXT NOT NULL,
    merchant_id VARCHAR(64) NULL,
    created_at DATETIME NOT NULL,
    INDEX idx_audit_merchant (merchant_id),
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_created (created_at),
    CONSTRAINT fk_audit_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT fk_audit_merchant FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

