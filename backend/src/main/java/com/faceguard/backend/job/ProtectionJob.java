package com.faceguard.backend.job;

import java.time.Instant;

import com.faceguard.backend.image.Image;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;

/** 보호 작업 하나: 어떤 사진을 보호했고, 결과가 무엇인지 */
@Entity
public class ProtectionJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(optional = false)
    private Image originalImage;

    @ManyToOne
    private Image protectedImage;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private JobStatus status;

    private String errorMessage;
    private Double aiSeconds;        // AI 서버가 알려 준 처리 시간
    private Instant createdAt;
    private Instant finishedAt;

    protected ProtectionJob() {
    }

    public ProtectionJob(Image originalImage) {
        this.originalImage = originalImage;
        this.status = JobStatus.PROCESSING;
        this.createdAt = Instant.now();
    }

    public void succeed(Image protectedImage, Double aiSeconds) {
        this.protectedImage = protectedImage;
        this.aiSeconds = aiSeconds;
        this.status = JobStatus.DONE;
        this.finishedAt = Instant.now();
    }

    public void fail(String errorMessage) {
        this.errorMessage = errorMessage;
        this.status = JobStatus.FAILED;
        this.finishedAt = Instant.now();
    }

    public Long getId() { return id; }
    public Image getOriginalImage() { return originalImage; }
    public Image getProtectedImage() { return protectedImage; }
    public JobStatus getStatus() { return status; }
    public String getErrorMessage() { return errorMessage; }
    public Double getAiSeconds() { return aiSeconds; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getFinishedAt() { return finishedAt; }
}
