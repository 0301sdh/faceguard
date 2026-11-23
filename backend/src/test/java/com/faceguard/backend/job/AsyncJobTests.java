package com.faceguard.backend.job;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import com.faceguard.backend.image.Image;
import com.faceguard.backend.image.ImageService;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/** 가짜 AI 서버로 비동기 작업의 순서, 재시도, 실패, 이어서 하기를 확인한다 */
@SpringBootTest
@ActiveProfiles("test")
class AsyncJobTests {

    interface Behavior {
        void answer(HttpExchange exchange, int hit) throws IOException, InterruptedException;
    }

    static final HttpServer ai;
    static volatile Behavior behavior;
    static final AtomicInteger hits = new AtomicInteger();
    static final AtomicInteger running = new AtomicInteger();
    static final AtomicInteger maxRunning = new AtomicInteger();
    static final List<String> progressAsked = new CopyOnWriteArrayList<>();
    static volatile int progressStatus = 200;
    static final List<String> received = new CopyOnWriteArrayList<>();

    static {
        try {
            ai = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        ai.setExecutor(Executors.newCachedThreadPool());
        ai.createContext("/protect", exchange -> {
            maxRunning.accumulateAndGet(running.incrementAndGet(), Math::max);
            try {
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1);
                received.add(body);
                behavior.answer(exchange, hits.incrementAndGet());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                running.decrementAndGet();
                exchange.close();
            }
        });
        ai.createContext("/progress/", exchange -> {
            byte[] b = "{\"progress\":42}".getBytes(StandardCharsets.UTF_8);
            progressAsked.add(exchange.getRequestURI().getPath());
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(progressStatus, b.length);
            exchange.getResponseBody().write(b);
            exchange.close();
        });
        ai.start();
    }

    @DynamicPropertySource
    static void aiUrl(DynamicPropertyRegistry registry) {
        registry.add("faceguard.ai-server-url", () -> "http://127.0.0.1:" + ai.getAddress().getPort());
    }

    @AfterAll
    static void stopAi() {
        ai.stop(0);
    }

    @Autowired ProtectionJobService service;
    @Autowired ProtectionJobRepository repository;
    @Autowired ImageService imageService;
    @Autowired JobWorker worker;

    @BeforeEach
    void reset() {
        hits.set(0);
        maxRunning.set(0);
        received.clear();
        progressAsked.clear();
        progressStatus = 200;
        behavior = (ex, hit) -> reply(ex, 200, "PNG");
    }

    @Test
    void answersImmediatelyAndFinishesInBackground() throws Exception {
        behavior = (ex, hit) -> { Thread.sleep(700); reply(ex, 200, "PNG"); };
        long t0 = System.nanoTime();
        ProtectionJob job = service.create(upload("a").getId());
        assertThat(Duration.ofNanos(System.nanoTime() - t0)).isLessThan(Duration.ofMillis(500));
        assertThat(job.getStatus()).isEqualTo(JobStatus.QUEUED);

        ProtectionJob done = await(job.getId());
        assertThat(done.getStatus()).isEqualTo(JobStatus.DONE);
        assertThat(done.getAttempts()).isEqualTo(1);
        assertThat(Files.readString(imageService.pathOf(imageService.get(done.getProtectedImage().getId())))).isEqualTo("PNG");
    }

    @Test
    void reportsProgressWhileProcessing() throws Exception {
        behavior = (ex, hit) -> { Thread.sleep(700); reply(ex, 200, "PNG"); };
        ProtectionJob job = service.create(upload("a").getId());
        assertThat(service.progressOf(job)).isEqualTo(0);      // 대기 중

        ProtectionJob processing = awaitStarted(job.getId());
        assertThat(service.progressOf(processing)).isEqualTo(42);
        assertThat(progressAsked).containsExactly("/progress/" + job.getId() + "-1");
        assertThat(received.get(0)).contains("progress_id").contains(job.getId() + "-1");   // AI 서버에 보낸 이름과 같아야 함

        progressStatus = 404;                                  // 진행률을 못 받아도 조회는 성공
        assertThat(service.progressOf(processing)).isNull();

        assertThat(service.progressOf(await(job.getId()))).isEqualTo(100);
    }

    @Test
    void processesOneAtATimeInOrder() throws Exception {
        behavior = (ex, hit) -> { Thread.sleep(150); reply(ex, 200, "PNG"); };
        List<ProtectionJob> jobs = List.of(service.create(upload("photo-1").getId()),
                service.create(upload("photo-2").getId()), service.create(upload("photo-3").getId()));
        for (ProtectionJob job : jobs) {
            assertThat(await(job.getId()).getStatus()).isEqualTo(JobStatus.DONE);
        }
        assertThat(maxRunning.get()).isEqualTo(1);
        assertThat(received).hasSize(3);
        for (int i = 0; i < 3; i++) {
            assertThat(received.get(i)).contains("photo-" + (i + 1));
        }
    }

    @Test
    void doesNotRetryWhenThereIsNoFace() throws Exception {
        behavior = (ex, hit) -> reply(ex, 422, "{}");
        ProtectionJob done = await(service.create(upload("x").getId()).getId());
        assertThat(done.getStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(done.getErrorMessage()).isEqualTo("얼굴을 찾지 못했습니다");
        assertThat(hits.get()).isEqualTo(1);
    }

    @Test
    void retriesWhenConnectionDropsThenSucceeds() throws Exception {
        behavior = (ex, hit) -> {
            if (hit == 1) {
                return;   // 첫 번째만 응답 없이 연결을 끊음
            }
            reply(ex, 200, "PNG");
        };
        ProtectionJob done = await(service.create(upload("x").getId()).getId());
        assertThat(done.getStatus()).isEqualTo(JobStatus.DONE);
        assertThat(done.getAttempts()).isEqualTo(2);
    }

    @Test
    void givesUpAfterMaxAttempts() throws Exception {
        behavior = (ex, hit) -> { };   // 항상 응답 없이 닫음
        ProtectionJob done = await(service.create(upload("x").getId()).getId());
        assertThat(done.getStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(done.getErrorMessage()).isEqualTo("AI 서버에 연결할 수 없습니다");
        assertThat(done.getAttempts()).isEqualTo(3);
        assertThat(hits.get()).as("한 번 시도에 요청도 한 번만 나가야 함").isEqualTo(3);
    }

    @Test
    void doesNotRetryWhenTooSlow() throws Exception {
        behavior = (ex, hit) -> { Thread.sleep(3000); };   // 제한 시간 2초보다 늦음
        ProtectionJob done = await(service.create(upload("x").getId()).getId());
        assertThat(done.getStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(done.getErrorMessage()).isEqualTo("AI 서버의 응답이 너무 늦습니다");
        assertThat(hits.get()).isEqualTo(1);
    }

    @Test
    void failsClearlyWhenPhotoFileIsMissingAndKeepsWorking() throws Exception {
        Image image = upload("x");
        Files.delete(imageService.pathOf(image));
        ProtectionJob failed = await(service.create(image.getId()).getId());
        assertThat(failed.getStatus()).isEqualTo(JobStatus.FAILED);
        assertThat(failed.getErrorMessage()).isEqualTo("저장된 사진 파일을 찾을 수 없습니다");
        assertThat(await(service.create(upload("y").getId()).getId()).getStatus()).isEqualTo(JobStatus.DONE);
    }

    @Test
    void resumesUnfinishedJobsAfterRestart() throws Exception {
        // 서버가 꺼지면서 PROCESSING, QUEUED로 남은 작업
        ProtectionJob halfway = new ProtectionJob(upload("x"));
        halfway.start();
        halfway = repository.save(halfway);
        ProtectionJob waiting = repository.save(new ProtectionJob(upload("y")));

        worker.afterSingletonsInstantiated();   // 서버를 켤 때 실행되는 부분

        assertThat(await(halfway.getId()).getStatus()).isEqualTo(JobStatus.DONE);
        assertThat(await(waiting.getId()).getStatus()).isEqualTo(JobStatus.DONE);
        assertThat(hits.get()).isEqualTo(2);
    }

    private Image upload(String marker) {
        byte[] head = {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF};   // JPEG 앞부분
        byte[] tail = marker.getBytes(StandardCharsets.ISO_8859_1);
        byte[] data = new byte[head.length + tail.length];
        System.arraycopy(head, 0, data, 0, head.length);
        System.arraycopy(tail, 0, data, head.length, tail.length);
        return imageService.saveUpload(new MockMultipartFile("file", "x.jpg", "image/jpeg", data));
    }

    private ProtectionJob await(Long id) throws InterruptedException {
        long end = System.currentTimeMillis() + 15_000;
        while (System.currentTimeMillis() < end) {
            ProtectionJob job = service.get(id);
            if (job.getStatus() == JobStatus.DONE || job.getStatus() == JobStatus.FAILED) {
                return job;
            }
            Thread.sleep(20);
        }
        throw new AssertionError("작업이 끝나지 않음: " + id + " " + service.get(id).getStatus());
    }

    private ProtectionJob awaitStarted(Long id) throws InterruptedException {
        for (int i = 0; i < 250; i++) {
            ProtectionJob job = service.get(id);
            if (job.getAttempts() == 1 && hits.get() == 1) {
                return job;
            }
            Thread.sleep(20);
        }
        throw new AssertionError("작업이 시작되지 않음: " + id);
    }

    private static void reply(HttpExchange ex, int status, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, b.length);
        ex.getResponseBody().write(b);
    }
}
