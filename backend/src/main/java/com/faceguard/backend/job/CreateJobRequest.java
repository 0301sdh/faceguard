package com.faceguard.backend.job;

import jakarta.validation.constraints.NotNull;

public record CreateJobRequest(@NotNull(message = "보호할 사진 번호가 필요합니다") Long imageId) {
}
