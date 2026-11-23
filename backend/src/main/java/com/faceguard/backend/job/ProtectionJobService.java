package com.faceguard.backend.job;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

import com.faceguard.backend.common.ApiException;
import com.faceguard.backend.image.Image;
import com.faceguard.backend.image.ImageKind;
import com.faceguard.backend.image.ImageService;

@Service
public class ProtectionJobService {

    private final ProtectionJobRepository repository;
    private final ImageService imageService;
    private final JobWorker worker;

    public ProtectionJobService(ProtectionJobRepository repository, ImageService imageService, JobWorker worker) {
        this.repository = repository;
        this.imageService = imageService;
        this.worker = worker;
    }

    /** 작업 기록을 QUEUED로 남기고 줄에 세운 뒤 바로 돌아온다. 실제 처리는 JobWorker가 뒤에서 한다 */
    public ProtectionJob create(Long imageId) {
        Image original = imageService.get(imageId);
        if (original.getKind() != ImageKind.ORIGINAL) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "이미 보호된 사진입니다: " + imageId);
        }
        ProtectionJob job = repository.save(new ProtectionJob(original));   // DB 저장이 끝난 뒤에 줄을 세움
        worker.enqueue(job.getId());
        return job;
    }

    public ProtectionJob get(Long id) {
        return repository.findById(id)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "작업을 찾을 수 없습니다: " + id));
    }
}
