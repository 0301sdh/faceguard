-- Week 10에 만든 DB는 상태 칸이 ENUM(DONE, FAILED, PROCESSING)이라 QUEUED를 저장할 수 없다 → 글자로 바꿈 (이미 글자면 그대로)
ALTER TABLE protection_job ALTER COLUMN status SET DATA TYPE VARCHAR(20);
