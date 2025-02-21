use news_db;

select
    *
from
    news
order by
    id desc
limit
    10;

CREATE TABLE
    news_with_summaries (
        id INT AUTO_INCREMENT PRIMARY KEY,
        content TEXT NOT NULL,
        summarized_content TEXT NOT NULL,
        fine_tuned_summary TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );