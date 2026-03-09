-- Q1: Readmission rate and cost by diagnosis

    SELECT
        a.diagnosis,
        COUNT(*)                         AS total_cases,
        ROUND(AVG(a.total_cost), 2)      AS avg_cost,
        ROUND(AVG(a.length_of_stay), 2)  AS avg_los,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM admissions a
    GROUP BY a.diagnosis
    ORDER BY readmit_rate_pct DESC


-- Q2: Average cost and readmission by insurance type

    SELECT
        p.insurance,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.total_cost), 2)     AS avg_cost,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY p.insurance
    ORDER BY avg_cost DESC


-- Q3: Readmission rate by age group

    SELECT
        CASE
            WHEN p.age BETWEEN 18 AND 40 THEN '18-40'
            WHEN p.age BETWEEN 41 AND 60 THEN '41-60'
            WHEN p.age BETWEEN 61 AND 80 THEN '61-80'
            ELSE '81+'
        END AS age_group,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.length_of_stay), 2) AS avg_los,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY age_group
    ORDER BY age_group


-- Q4: Cost and readmission disparities by race/ethnicity

    SELECT
        p.race,
        COUNT(*)                        AS patients,
        ROUND(AVG(a.total_cost), 2)     AS avg_cost,
        ROUND(100.0 * SUM(a.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
    FROM patients p
    JOIN admissions a ON p.patient_id = a.patient_id
    GROUP BY p.race
    ORDER BY readmit_rate_pct DESC

