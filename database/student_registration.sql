CREATE DATABASE student_db;

USE student_db;

CREATE TABLE students (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    gender ENUM('MALE', 'FEMALE', 'OTHER') NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    enrollment_no VARCHAR(20) UNIQUE NOT NULL,
    contact VARCHAR(15) NOT NULL,
    course VARCHAR(50) NOT NULL,
    department VARCHAR(50) NOT NULL,
    address TEXT NOT NULL,
    state VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    image_path VARCHAR(255),
    face_embedding BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_email CHECK (email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'),
    CONSTRAINT chk_contact CHECK (contact REGEXP '^[0-9]{10}$'),
    CONSTRAINT chk_enrollment CHECK (enrollment_no REGEXP '^[0-9]{2}[A-Z]{4,5}[0-9]{3}$')
);