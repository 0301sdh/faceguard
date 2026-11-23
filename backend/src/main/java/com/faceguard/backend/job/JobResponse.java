package com.faceguard.backend.job;

import java.time.Instant;

public record JobResponse(Long id, JobStatus status, Long originalImageId, Long protectedImageId,
                          String protectedImageUrl, String errorMessage, Integer progress, int attempts, Double aiSeconds,
                          Instant createdAt, Instant startedAt, Instant finishedAt) {

    public static JobResponse from(ProtectionJob job, Integer progress) {
        Long protectedId = job.getProtectedImage() == null ? null : job.getProtectedImage().getId();
        return new JobResponse(job.getId(), job.getStatus(), job.getOriginalImage().getId(), protectedId,
                protectedId == null ? null : "/api/images/" + protectedId, job.getErrorMessage(), progress,
                job.getAttempts(), job.getAiSeconds(), job.getCreatedAt(), job.getStartedAt(), job.getFinishedAt());
    }
}
