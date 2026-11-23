package com.faceguard.backend.ai;

import java.net.SocketTimeoutException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

/** AI 서버(FastAPI, Week 9)의 POST /protect 를 부른다. */
@Component
public class AiClient {

    private final RestClient client;
    private final RestClient progressClient;

    public AiClient(@Value("${faceguard.ai-server-url}") String baseUrl,
                    @Value("${faceguard.ai-timeout-seconds}") long timeoutSeconds) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(5));
        factory.setReadTimeout(Duration.ofSeconds(timeoutSeconds));   // 노이즈 계산이 오래 걸림
        this.client = RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();

        SimpleClientHttpRequestFactory quick = new SimpleClientHttpRequestFactory();   // 진행률은 못 받아도 그만이라 짧게
        quick.setConnectTimeout(Duration.ofSeconds(1));
        quick.setReadTimeout(Duration.ofSeconds(2));
        this.progressClient = RestClient.builder().baseUrl(baseUrl).requestFactory(quick).build();
    }

    /** 사진 파일 → 보호된 사진(PNG). 실패하면 사용자에게 보여 줄 메시지를 담은 AiException */
    public AiResult protect(Path photo, String progressId) {
        if (!Files.isReadable(photo)) {
            throw new AiException("저장된 사진 파일을 찾을 수 없습니다", false);
        }
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("photo", new FileSystemResource(photo));   // AI 서버의 필드 이름과 같아야 함
        parts.add("progress_id", progressId);
        AiResult result;
        try {
            ResponseEntity<byte[]> response = client.post().uri("/protect")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(parts)
                    .retrieve()
                    .toEntity(byte[].class);
            String seconds = response.getHeaders().getFirst("X-Processing-Seconds");
            result = new AiResult(response.getBody(), seconds == null ? null : Double.valueOf(seconds));
        } catch (RestClientResponseException e) {
            // AI 서버가 답은 했지만 실패 (400 사진 아님, 413 너무 큼, 422 얼굴 없음, 500 내부 오류) → 다시 해도 같음
            String message = switch (e.getStatusCode().value()) {
                case 422 -> "얼굴을 찾지 못했습니다";
                case 400 -> "AI 서버가 사진을 읽지 못했습니다";
                case 413 -> "사진이 너무 큽니다";
                default -> "AI 서버 오류 (" + e.getStatusCode().value() + ")";
            };
            throw new AiException(message, false);
        } catch (RestClientException e) {
            if (isReadTimeout(e)) {
                // AI 서버가 아직 계산 중일 수 있음 → 또 보내면 같은 일이 쌓이므로 다시 시도하지 않음
                throw new AiException("AI 서버의 응답이 너무 늦습니다", false);
            }
            throw new AiException("AI 서버에 연결할 수 없습니다", true);   // 꺼져 있거나 연결이 끊김 → 다시 시도
        }
        if (result.png() == null || result.png().length == 0) {
            throw new AiException("AI 서버가 빈 결과를 보냈습니다", false);
        }
        return result;
    }

    /** AI 서버가 지금 몇 퍼센트인지 (0 ~ 99). 아직 시작하지 않았거나 물어볼 수 없으면 null */
    public Integer progress(String progressId) {
        try {
            ProgressBody body = progressClient.get().uri("/progress/{id}", progressId).retrieve().body(ProgressBody.class);
            return body == null ? null : body.progress();
        } catch (RestClientException e) {
            return null;   // 진행률을 못 받았다고 작업 조회까지 실패하면 안 됨
        }
    }

    private record ProgressBody(Integer progress) {
    }

    private static boolean isReadTimeout(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof SocketTimeoutException) {
                return !String.valueOf(t.getMessage()).toLowerCase().contains("connect");
            }
        }
        return false;
    }

    public record AiResult(byte[] png, Double processingSeconds) {
    }

    public static class AiException extends RuntimeException {

        private final boolean retryable;   // 다시 시도하면 나아질 수 있는 실패인가

        public AiException(String message, boolean retryable) {
            super(message);
            this.retryable = retryable;
        }

        public boolean isRetryable() {
            return retryable;
        }
    }
}
