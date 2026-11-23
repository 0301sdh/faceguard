package com.faceguard.backend.job;

import java.net.URI;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;

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

    /** 사진 번호로 보호 작업 만들기. 작업 번호를 바로 돌려주고 처리는 뒤에서 한다 (202 = 접수됨) */
    @Operation(summary = "보호 작업 만들기", description = "작업 번호를 바로 돌려준다(QUEUED). 처리(Mac CPU 약 40초)가 끝났는지는 작업 상태 확인으로 본다")
    @ApiResponse(responseCode = "202", description = "접수됨")
    @PostMapping
    public ResponseEntity<JobResponse> create(@Valid @RequestBody CreateJobRequest request) {
        ProtectionJob job = service.create(request.imageId());
        return ResponseEntity.accepted()
                .location(URI.create("/api/protection-jobs/" + job.getId()))
                .body(JobResponse.from(job));
    }

    /** 작업 상태와 결과 확인 */
    @Operation(summary = "작업 상태 확인")
    @GetMapping("/{id}")
    public JobResponse get(@PathVariable Long id) {
        return JobResponse.from(service.get(id));
    }
}
