package com.faceguard.backend.job;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;

@Tag(name = "보호 작업")
@RestController
@RequestMapping("/api/protection-jobs")
public class ProtectionJobController {

    private final ProtectionJobService service;

    public ProtectionJobController(ProtectionJobService service) {
        this.service = service;
    }

    /** 사진 번호로 보호 작업 만들기. 지금은 AI 처리가 끝날 때까지 기다렸다가 결과를 돌려줌 */
    @Operation(summary = "보호 작업 만들기", description = "AI 서버가 노이즈를 넣을 때까지 기다린다 (Mac CPU 약 40초). 결과 사진은 protectedImageUrl")
    @PostMapping
    public ResponseEntity<JobResponse> create(@Valid @RequestBody CreateJobRequest request) {
        ProtectionJob job = service.createAndRun(request.imageId());
        return ResponseEntity.status(HttpStatus.CREATED).body(JobResponse.from(job));
    }

    /** 작업 상태와 결과 확인 */
    @Operation(summary = "작업 상태 확인")
    @GetMapping("/{id}")
    public JobResponse get(@PathVariable Long id) {
        return JobResponse.from(service.get(id));
    }
}
