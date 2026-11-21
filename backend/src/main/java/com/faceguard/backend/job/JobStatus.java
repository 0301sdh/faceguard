package com.faceguard.backend.job;

public enum JobStatus {
    PROCESSING,  // AI 서버가 노이즈를 만드는 중
    DONE,        // 보호된 사진 저장 완료
    FAILED       // 실패 (이유는 errorMessage)
}
