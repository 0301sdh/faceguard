package com.faceguard.backend.job;

import java.time.Instant;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import com.faceguard.backend.image.Image;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;

/** 보호 작업 하나: 어떤 사진을 보호했고, 지금 상태와 결과가 무엇인지 */
@Entity
public class ProtectionJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(optional = false)
    private Image originalImage;

    @ManyToOne
    private Image protectedImage;

    // DB의 ENUM 형식은 상태를 추가할 때마다 표를 고쳐야 하므로 글자(VARCHAR)로 저장
    @Enumerated(EnumType.STRING)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    @Column(nullable = false, length = 20)
    private JobStatus status;

    @Column(columnDefinition = "integer default 0 not null")   // Week 10에 만든 기록에는 0이 들어감
    private int attempts;            // AI 서버에 보낸 횟수

    private String errorMessage;
    private Double aiSeconds;        // AI 서버가 알려 준 처리 시간
    private Instant createdAt;
    private Instant startedAt;       // 처리를 시작한 시각
    private Instant finishedAt;

    protected ProtectionJob() {
    }

    public ProtectionJob(Image originalImage) {
        this.originalImage = originalImage;
        this.status = JobStatus.QUEUED;
        this.createdAt = Instant.now();
    }

    /** 처리 시작 (재시도할 때는 처음 시작한 시각을 유지) */
    public void start() {
        this.status = JobStatus.PROCESSING;
        if (this.startedAt == null) {
            this.startedAt = Instant.now();
        }
    }

    public void countAttempt() {
        this.attempts++;
    }

    public void requeue() {
        this.status = JobStatus.QUEUED;
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
    public int getAttempts() { return attempts; }
    public String getErrorMessage() { return errorMessage; }
    public Double getAiSeconds() { return aiSeconds; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getStartedAt() { return startedAt; }
    public Instant getFinishedAt() { return finishedAt; }
}
