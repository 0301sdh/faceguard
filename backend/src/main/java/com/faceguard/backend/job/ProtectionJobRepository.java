package com.faceguard.backend.job;

import java.util.Collection;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface ProtectionJobRepository extends JpaRepository<ProtectionJob, Long> {

    /** 아직 끝나지 않은 작업을 만든 순서대로 (서버를 다시 켰을 때 이어서 하려고) */
    List<ProtectionJob> findByStatusInOrderByIdAsc(Collection<JobStatus> statuses);
}
