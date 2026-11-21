package com.faceguard.backend.common;

import org.springframework.http.HttpStatus;

/** 사용자에게 그대로 보여 줄 오류 (상태 코드 + 메시지). */
public class ApiException extends RuntimeException {

    private final HttpStatus status;

    public ApiException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public HttpStatus getStatus() {
        return status;
    }
}
