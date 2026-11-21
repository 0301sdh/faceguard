package com.faceguard.backend.job;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

import com.faceguard.backend.ai.AiClient;
import com.faceguard.backend.common.ApiException;
import com.faceguard.backend.image.Image;
import com.faceguard.backend.image.ImageKind;
import com.faceguard.backend.image.ImageService;

@Service
public class ProtectionJobService {

    private final ProtectionJobRepository repository;
    private final ImageService imageService;
    private final AiClient aiClient;

    public ProtectionJobService(ProtectionJobRepository repository, ImageService imageService, AiClient aiClient) {
        this.repository = repository;
        this.imageService = imageService;
        this.aiClient = aiClient;
    }

    /**
     * 작업을 만들고 AI 서버가 끝날 때까지 기다린다 (Week 10: 동기 방식).
     * Week 11에서 작업 번호를 먼저 돌려주고 뒤에서 처리하는 비동기 방식으로 바꾼다.
     */
    public ProtectionJob createAndRun(Long imageId) {
        Image original = imageService.get(imageId);
        if (original.getKind() != ImageKind.ORIGINAL) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "이미 보호된 사진입니다: " + imageId);
        }
        // 작업 기록을 먼저 남김 (AI 처리 중에도 조회 가능)
        ProtectionJob job = repository.save(new ProtectionJob(original));
        try {
            AiClient.AiResult result = aiClient.protect(imageService.pathOf(original));
            Image protectedImage = imageService.saveProtected(original, result.png());
            job.succeed(protectedImage, result.processingSeconds());
        } catch (AiClient.AiException e) {
            job.fail(e.getMessage());
        }
        return repository.save(job);
    }

    public ProtectionJob get(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "작업을 찾을 수 없습니다: " + id));
    }
}
