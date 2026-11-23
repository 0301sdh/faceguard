package com.faceguard.backend.job;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import com.faceguard.backend.ai.AiClient;
import com.faceguard.backend.image.Image;
import com.faceguard.backend.image.ImageService;

/**
 * 뒤에서 작업을 한 장씩 순서대로 처리한다.
 * AI 서버가 한 번에 한 장만 처리하므로 작업자(스레드)도 1개다. 줄(대기열)은 이 스레드의 할 일 목록이다.
 */
@Component
public class JobWorker implements SmartInitializingSingleton, DisposableBean {

    private static final Logger log = LoggerFactory.getLogger(JobWorker.class);

    private final ProtectionJobRepository repository;
    private final ImageService imageService;
    private final AiClient aiClient;
    private final int maxAttempts;
    private final long retryDelayMillis;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "job-worker");
        t.setDaemon(true);   // 서버를 끌 때 AI 응답을 기다리느라 붙잡혀 있지 않게
        return t;
    });
    private volatile boolean stopping = false;

    public JobWorker(ProtectionJobRepository repository, ImageService imageService, AiClient aiClient,
                     @Value("${faceguard.job.max-attempts}") int maxAttempts,
                     @Value("${faceguard.job.retry-delay-millis}") long retryDelayMillis) {
        this.repository = repository;
        this.imageService = imageService;
        this.aiClient = aiClient;
        this.maxAttempts = maxAttempts;
        this.retryDelayMillis = retryDelayMillis;
    }

    public void enqueue(Long jobId) {
        executor.execute(() -> process(jobId));
    }

    /** 서버가 요청을 받기 전에, 지난번에 끝내지 못한 작업을 다시 줄에 세운다 */
    @Override
    public void afterSingletonsInstantiated() {
        List<ProtectionJob> unfinished =
                repository.findByStatusInOrderByIdAsc(List.of(JobStatus.QUEUED, JobStatus.PROCESSING));
        for (ProtectionJob job : unfinished) {
            job.requeue();
            repository.save(job);
            enqueue(job.getId());
        }
        if (!unfinished.isEmpty()) {
            log.info("끝내지 못한 작업 {}개를 다시 줄에 세웠습니다", unfinished.size());
        }
    }

    @Override
    public void destroy() {
        stopping = true;
        executor.shutdownNow();   // 처리 중이던 작업은 DB에 남아 있어 다음에 켤 때 이어서 함
    }

    private void process(Long jobId) {
        try {
            ProtectionJob job = repository.findById(jobId).orElse(null);
            if (job == null || job.getStatus() != JobStatus.QUEUED) {
                return;
            }
            job.start();
            job = repository.save(job);
            run(job);
        } catch (Exception e) {
            if (stopping) {
                return;   // 서버가 꺼지는 중에 난 오류는 실패로 적지 않음 (다음에 이어서 처리)
            }
            log.error("작업 {} 처리 중 오류", jobId, e);
            failUnexpected(jobId);
        }
    }

    private void run(ProtectionJob job) throws InterruptedException {
        Image original = job.getOriginalImage();
        while (true) {
            // 서버가 도중에 꺼져도 횟수가 남게, 보내기 전에 먼저 기록
            job.countAttempt();
            job = repository.save(job);
            try {
                AiClient.AiResult result = aiClient.protect(imageService.pathOf(original), job.progressId());
                Image protectedImage = imageService.saveProtected(original, result.png());
                job.succeed(protectedImage, result.processingSeconds());
                repository.save(job);
                return;
            } catch (AiClient.AiException e) {
                if (stopping) {
                    return;
                }
                if (!e.isRetryable() || job.getAttempts() >= maxAttempts) {
                    job.fail(e.getMessage());
                    repository.save(job);
                    return;
                }
                log.warn("작업 {} 시도 {}/{} 실패: {}. 다시 시도합니다", job.getId(), job.getAttempts(), maxAttempts, e.getMessage());
                Thread.sleep(retryDelayMillis * job.getAttempts());   // 시도할수록 더 오래 쉼
            }
        }
    }

    /** 예상하지 못한 오류(파일 없음, 디스크 가득 참 등)에도 작업이 PROCESSING으로 남지 않게 */
    private void failUnexpected(Long jobId) {
        try {
            repository.findById(jobId).ifPresent(job -> {
                job.fail("서버 내부 오류로 처리하지 못했습니다");
                repository.save(job);
            });
        } catch (Exception e) {
            log.error("작업 {}의 실패를 기록하지 못했습니다", jobId, e);
        }
    }
}
